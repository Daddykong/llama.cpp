#include "models.h"
#include "llama-memory-recurrent.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// b65: per-layer threshold file reader ("<layer> <tag> <thr>" lines)
static std::vector<float> b65_read_thr(const char * env) {
    std::vector<float> v(256, -1.0f);
    const char * f = getenv(env);
    if (f && *f) {
        if (FILE * fp = fopen(f, "r")) {
            int il; char tag[32]; float t;
            while (fscanf(fp, "%d %31s %f", &il, tag, &t) == 3) {
                if (il >= 0 && il < 256) v[il] = t;
            }
            fclose(fp);
        }
    }
    return v;
}

// b65 LLAMA_B65_MERGE=1: weights that read the same input (ffn gate|up, linear-attention qkv|z, attention q|k|v) and
// sit back to back in one weight buffer (same type and row length, contiguous, no padding between them) are used as
// ONE weight tensor spanning all of them: one matvec dispatch instead of two or three, outputs taken as views. The
// merged tensor aliases the loaded data (no copy, no extra VRAM). It is used in every graph (prompt, decode, MTP),
// so a weight is never also read alone: the Vulkan row reorder is keyed by the first tensor's offset and is per
// row, so reordering the merged span is the same as reordering each part. Off with LoRA adapters and with the
// movable-weight experiments (LLAMA_W_SPARSE / LLAMA_VRAM_BUDGET_MB), which move tensors one by one.
static bool b65_merge_on() {
    static const bool on = getenv("LLAMA_B65_MERGE") != nullptr && atoi(getenv("LLAMA_B65_MERGE")) != 0 &&
                           getenv("LLAMA_W_SPARSE") == nullptr && getenv("LLAMA_VRAM_BUDGET_MB") == nullptr;
    return on;
}

static ggml_tensor * b65_merged_w(ggml_context * ctx, const llama_adapter_loras * loras, const char * kind, int il,
                                  std::initializer_list<const ggml_tensor *> ws) {
    if (!b65_merge_on() || (loras && !loras->empty())) {
        return nullptr;
    }
    const ggml_tensor * a = *ws.begin();
    int64_t rows = 0;
    const char * expect = a ? (const char *) a->data : nullptr;
    bool ok = a != nullptr && a->buffer != nullptr && expect != nullptr;
    for (const ggml_tensor * w : ws) {
        if (!ok) break;
        ok = w && w->type == a->type && w->ne[0] == a->ne[0] && w->ne[2] == 1 && w->ne[3] == 1 && w->view_src == nullptr &&
             ggml_is_contiguous(w) && w->buffer == a->buffer && (const char *) w->data == expect;
        if (ok) {
            rows += w->ne[1];
            expect += ggml_nbytes(w);
        }
    }
    static int n_ok = 0, n_fail = 0;
    if (!ok) {
        if (n_fail++ < 4) {
            LLAMA_LOG_INFO("b65 merge: %s layer %d not merged (weights not back to back in one buffer)\n", kind, il);
        }
        return nullptr;
    }
    if (n_ok++ == 0) {
        LLAMA_LOG_INFO("b65 merge: same-input weights merged into single matvecs (first: %s layer %d, %lld rows)\n", kind, il, (long long) rows);
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, a->type, a->ne[0], rows);
    t->data   = a->data;
    t->buffer = a->buffer;
    ggml_format_name(t, "b65_merged_%s-%d", kind, il);
    return t;
}

// [ne0, T] (or any 2-D view with row stride nb1) as [ne0a, ne0/ne0a, T]: a reshape when contiguous, else a strided view
static ggml_tensor * b65_as_3d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (ggml_is_contiguous(t)) {
        return ggml_reshape_3d(ctx, t, ne0, ne1, ne2);
    }
    return ggml_view_3d(ctx, t, ne0, ne1, ne2, ggml_row_size(t->type, ne0), t->nb[1], 0);
}

static ggml_tensor * b65_as_4d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    if (ggml_is_contiguous(t)) {
        return ggml_reshape_4d(ctx, t, ne0, ne1, ne2, ne3);
    }
    // t is [ne0*ne1, ne2*ne3] with row stride nb1
    return ggml_view_4d(ctx, t, ne0, ne1, ne2, ne3, ggml_row_size(t->type, ne0), t->nb[1], t->nb[1] * ne2, 0);
}

// b65 merged attention projection: q (query + gate) | k | v in one matvec; returns false when not merged.
// Outputs are 2-D views [n, T] with the merged row stride (contiguous for one token).
static bool b65_qkv_merged(ggml_context * ctx, const llama_adapter_loras * loras, const llama_layer & layer,
                           ggml_tensor * cur, int il, ggml_tensor *& q, ggml_tensor *& k, ggml_tensor *& v) {
    if (layer.wqkv || !layer.wq || layer.wq_s || layer.wk_s || layer.wv_s || layer.wq_b || layer.wk_b || layer.wv_b) {
        return false;
    }
    ggml_tensor * w = b65_merged_w(ctx, loras, "qkv", il, { layer.wq, layer.wk, layer.wv });
    if (!w) {
        return false;
    }
    ggml_tensor * out = ggml_mul_mat(ctx, w, cur);   // [q + k + v, T]
    const int64_t nq = layer.wq->ne[1], nk = layer.wk->ne[1], nv = layer.wv->ne[1];
    q = ggml_view_2d(ctx, out, nq, out->ne[1], out->nb[1], 0);
    k = ggml_view_2d(ctx, out, nk, out->ne[1], out->nb[1], ggml_row_size(out->type, nq));
    v = ggml_view_2d(ctx, out, nv, out->ne[1], out->nb[1], ggml_row_size(out->type, nq + nk));
    return true;
}

// b65 merged FFN: gate|up in one matvec, then swiglu on the [2*n_ff, T] result (first half gated); nullptr if not merged
static ggml_tensor * b65_ffn_merged(ggml_context * ctx, const llama_adapter_loras * loras, const llama_layer & layer,
                                    ggml_tensor * cur, int il) {
    // not with the sparse-FFN tensors: their paths read ffn_gate / ffn_up alone in some graphs
    if (layer.ffn_gate_s || layer.ffn_up_s || layer.ffn_down_s || layer.ffn_gate_b || layer.ffn_up_b || layer.ffn_down_b ||
        layer.ffn_down_t || layer.ffn_sk_gate) {
        return nullptr;
    }
    ggml_tensor * w = b65_merged_w(ctx, loras, "ffn_gate_up", il, { layer.ffn_gate, layer.ffn_up });
    if (!w) {
        return nullptr;
    }
    ggml_tensor * gu = ggml_mul_mat(ctx, w, cur);    // [2*n_ff, T]: gate rows then up rows
    ggml_tensor * h  = ggml_swiglu(ctx, gu);         // silu(gate) * up
    return ggml_mul_mat(ctx, layer.ffn_down, h);
}

// b65: per-layer thresholds for the sparse down projection (LLAMA_DS_THR_FILE), read once
static const std::vector<float> & b65_ds_thr() {
    static std::vector<float> thr = [] {
        std::vector<float> v(256, -1.0f);
        const char * f = getenv("LLAMA_DS_THR_FILE");
        if (f && *f) {
            if (FILE * fp = fopen(f, "r")) {
                int il; char tag[32]; float t;
                while (fscanf(fp, "%d %31s %f", &il, tag, &t) == 3) {
                    if (il >= 0 && il < 256) v[il] = t;
                }
                fclose(fp);
            }
        }
        return v;
    }();
    return thr;
}

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // optional projection of the embeddings output
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_embd_out() }, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   { hparams.n_embd_out() },         TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // b65: ffn_up created right after ffn_gate so the two sit back to back in the weight buffer (LLAMA_B65_MERGE)
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_down_t = create_tensor(tn(LLM_TENSOR_FFN_DOWN_T, "weight", il), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_sk_gate  = create_tensor(tn(LLM_TENSOR_FFN_SK_GATE,  "weight", il), {n_embd / 32, n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_sk_up    = create_tensor(tn(LLM_TENSOR_FFN_SK_UP,    "weight", il), {n_embd / 32, n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_sk_scale = create_tensor(tn(LLM_TENSOR_FFN_SK_SCALE, "weight", il), {2, n_ff}, TENSOR_NOT_REQUIRED);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto & lyr = model.layers[il];
    if (!lyr.wqkv_s && !lyr.wqkv_gate_s) {
        if (ggml_tensor * w = b65_merged_w(ctx0, loras, "qkvz", il, { lyr.wqkv, lyr.wqkv_gate })) {
            ggml_tensor * out = ggml_mul_mat(ctx0, w, input);   // [qkv + z, T]
            cb(out, "linear_attn_qkvz", il);
            const int64_t n_qkv = lyr.wqkv->ne[1];
            ggml_tensor * qkv_mixed = ggml_view_2d(ctx0, out, n_qkv, out->ne[1], out->nb[1], 0);
            qkv_mixed = ggml_is_contiguous(qkv_mixed) ? ggml_reshape_3d(ctx0, qkv_mixed, n_qkv, n_seq_tokens, n_seqs)
                : ggml_view_3d(ctx0, out, n_qkv, n_seq_tokens, n_seqs, out->nb[1], out->nb[1] * n_seq_tokens, 0);
            cb(qkv_mixed, "linear_attn_qkv_mixed", il);
            ggml_tensor * z = ggml_view_2d(ctx0, out, lyr.wqkv_gate->ne[1], out->ne[1], out->nb[1], ggml_row_size(out->type, n_qkv));
            cb(z, "z", il);
            return { qkv_mixed, z };
        }
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = nullptr, * Kcur = nullptr, * Vcur = nullptr;
    if (!b65_qkv_merged(ctx0, loras, model.layers[il], cur, il, Qcur_full, Kcur, Vcur)) {
        auto r = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
        Qcur_full = r.q; Kcur = r.k; Vcur = r.v;
    }
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // b65: token stride from the tensor (a merged q|k|v output has a longer row than q alone)
    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        Qcur_full->nb[1], 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = b65_as_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        Qcur_full->nb[1],
        ggml_element_size(Qcur_full) * n_embd_head);
    // b65: LLAMA_GATE_NOCONT=1 applies the sigmoid straight to the strided view (Vulkan unary ops take strided input;
    // the sigmoid output is contiguous), dropping one CONT dispatch per full-attention layer
    static const bool gate_nocont = getenv("LLAMA_GATE_NOCONT") != nullptr && atoi(getenv("LLAMA_GATE_NOCONT")) != 0;
    if (!gate_nocont) {
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    }
    cb(gate, "gate_reshaped", il);

    Vcur = b65_as_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    if (gate_nocont) {
        // b65: 3-D multiply right after the sigmoid (fusable), flattened afterwards
        ggml_tensor * cur3 = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
        ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
        cb(gate_sigmoid, "gate_sigmoid", il);
        cur = ggml_mul(ctx0, cur3, gate_sigmoid);
        cur = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head, n_tokens);
    } else {
        ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
        cb(gate_sigmoid, "gate_sigmoid", il);
        cur = ggml_mul(ctx0, cur, gate_sigmoid);
    }
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // b65: LLAMA_GDN_GATES=1 computes beta and the decay gate in one op for small steps
    static const bool gdn_gates = getenv("LLAMA_GDN_GATES") != nullptr && atoi(getenv("LLAMA_GDN_GATES")) != 0;
    const auto & lyr = model.layers[il];
    ggml_tensor * beta;
    ggml_tensor * gate;
    if (gdn_gates && cur->ne[1] <= 8 && ggml_n_dims(cur) <= 2 && !lyr.ssm_beta_s && !lyr.ssm_alpha_s &&
        lyr.ssm_beta->type == lyr.ssm_alpha->type && (lyr.ssm_beta->type == GGML_TYPE_F16 || lyr.ssm_beta->type == GGML_TYPE_F32) &&
        lyr.ssm_dt->type == GGML_TYPE_F32 && lyr.ssm_a->type == GGML_TYPE_F32 &&
        ggml_nelements(lyr.ssm_dt) == num_v_heads && ggml_nelements(lyr.ssm_a) == num_v_heads) {
        ggml_tensor * gg = ggml_gdn_gates(ctx0, cur, lyr.ssm_beta, lyr.ssm_alpha, lyr.ssm_dt, lyr.ssm_a);
        const int64_t T = cur->ne[1];
        beta = ggml_reshape_4d(ctx0, ggml_view_2d(ctx0, gg, num_v_heads, T, gg->nb[1], 0), 1, num_v_heads, n_seq_tokens, n_seqs);
        gate = ggml_reshape_3d(ctx0, ggml_view_2d(ctx0, gg, num_v_heads, T, gg->nb[1], gg->nb[2]), num_v_heads, n_seq_tokens, n_seqs);
        cb(beta, "beta_sigmoid", il);
        cb(gate, "gate", il);
    } else {
    beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);
    }

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = b65_as_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    // b65: sparse gate/up (sign-sketch predictor + masked matvecs) for small steps, then sparse or dense down
    {
        const auto & layer = model.layers[il];
        static const std::vector<float> sk_thr = b65_read_thr("LLAMA_SK_THR_FILE");
        static const int sk_max_tok = getenv("LLAMA_SK_MAX_TOK") ? atoi(getenv("LLAMA_SK_MAX_TOK")) : 1;
        static const int ds_chunks2 = getenv("LLAMA_DS_CHUNKS") ? atoi(getenv("LLAMA_DS_CHUNKS")) : 48;
        const float st = il < 256 ? sk_thr[il] : -1.0f;
        if (layer.ffn_sk_gate && layer.ffn_sk_up && layer.ffn_sk_scale && st >= 0.0f && cur->ne[1] <= sk_max_tok && cur->ne[1] <= 4) {
            ggml_tensor * s    = ggml_sign_score(ctx0, cur, layer.ffn_sk_gate, layer.ffn_sk_up, layer.ffn_sk_scale);
            cb(s, "ffn_sk_score", il);
            ggml_tensor * gate = ggml_mul_mat_masked(ctx0, layer.ffn_gate, cur, s, st);
            ggml_tensor * up   = ggml_mul_mat_masked(ctx0, layer.ffn_up,   cur, s, st);
            ggml_tensor * h    = ggml_swiglu_split(ctx0, gate, up);
            cb(h, "ffn_swiglu", il);
            const float dt = il < 256 ? b65_ds_thr()[il] : -1.0f;
            if (layer.ffn_down_t) {
                // skipped neurons have h == 0 exactly; with no down threshold, skip only those
                ggml_tensor * p = ggml_mul_mat_sparse_t(ctx0, layer.ffn_down_t, h, dt >= 0.0f ? dt : 1e-30f, ds_chunks2);
                p = ggml_cont(ctx0, ggml_permute(ctx0, p, 1, 0, 2, 3));
                p = ggml_sum_rows(ctx0, p);
                cur = ggml_reshape_2d(ctx0, p, layer.ffn_down_t->ne[0], h->ne[1]);
            } else {
                cur = build_lora_mm(layer.ffn_down, h);
            }
            cb(cur, "ffn_out", il);
            return cur;
        }
    }

    // b65: activation-sparse down projection for small steps (decode, MTP verify) when configured
    {
        const auto & layer = model.layers[il];
        static const int ds_max_tok = getenv("LLAMA_DS_MAX_TOK") ? atoi(getenv("LLAMA_DS_MAX_TOK")) : 1; // default 1: the MTP 3-token verify needs ~90% of neurons (union), and the kernel reads rows per token, so verify stays dense
        static const int ds_chunks  = getenv("LLAMA_DS_CHUNKS")  ? atoi(getenv("LLAMA_DS_CHUNKS"))  : 48;
        const float thr = il < 256 ? b65_ds_thr()[il] : -1.0f;
        if (layer.ffn_down_t && thr >= 0.0f && cur->ne[1] <= ds_max_tok && !layer.ffn_up_s && !layer.ffn_gate_s) {
            ggml_tensor * up   = build_lora_mm(layer.ffn_up, cur);
            ggml_tensor * gate = build_lora_mm(layer.ffn_gate, cur);
            ggml_tensor * h    = ggml_swiglu_split(ctx0, gate, up);
            cb(h, "ffn_swiglu", il);
            ggml_tensor * p = ggml_mul_mat_sparse_t(ctx0, layer.ffn_down_t, h, thr, ds_chunks); // [n_embd, chunks, T]
            p = ggml_cont(ctx0, ggml_permute(ctx0, p, 1, 0, 2, 3));                            // [chunks, n_embd, T]
            p = ggml_sum_rows(ctx0, p);                                                        // [1, n_embd, T]
            cur = ggml_reshape_2d(ctx0, p, layer.ffn_down_t->ne[0], h->ne[1]);
            cb(cur, "ffn_out", il);
            return cur;
        }
    }

    if (ggml_tensor * m = b65_ffn_merged(ctx0, loras, model.layers[il], cur, il)) {
        cur = m;
    } else {
    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * Qcur_full = nullptr, * Kcur = nullptr, * Vcur = nullptr;
    if (!b65_qkv_merged(ctx0, loras, layer, cur, il, Qcur_full, Kcur, Vcur)) {
        auto r = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
        Qcur_full = r.q; Kcur = r.k; Vcur = r.v;
    }
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            Qcur_full->nb[1],
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            Qcur_full->nb[1],
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = b65_as_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = b65_as_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    if (ggml_tensor * m = b65_ffn_merged(ctx0, loras, layer, cur, il)) {
        cur = m;
    } else {
    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    // b65 [mtp]: LLAMA_MTP_DRAFT_VOCAB=<K>: the draft head scores only token ids < K (the first, most frequent BPE ids;
    // code and English stay below ~99K for this vocab) and gives the rest -1e30, so each draft reads K/n_vocab of the
    // head. Only the drafts change: the main model still verifies every token with the full head (lossless output).
    static const int64_t draft_vocab = getenv("LLAMA_MTP_DRAFT_VOCAB") ? atoll(getenv("LLAMA_MTP_DRAFT_VOCAB")) : 0;
    const int64_t n_vocab_head = head_w->ne[1];
    if (draft_vocab > 0 && draft_vocab < n_vocab_head && head_s == nullptr && cur->ne[1] > 0) {
        ggml_tensor * hw = ggml_view_2d(ctx0, head_w, head_w->ne[0], draft_vocab, head_w->nb[1], 0);
        ggml_tensor * small = build_lora_mm(hw, cur, nullptr);                                   // [K, n_out]
        ggml_tensor * padded = ggml_pad(ctx0, small, n_vocab_head - draft_vocab, 0, 0, 0);        // zeros past K
        ggml_tensor * tail = ggml_cont(ctx0, ggml_view_2d(ctx0, padded, n_vocab_head - draft_vocab, padded->ne[1],
                                                          padded->nb[1], draft_vocab * ggml_element_size(padded)));
        tail = ggml_scale_bias(ctx0, tail, 0.0f, -1e30f);
        cur = ggml_concat(ctx0, small, tail, 0);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
