// b65: attention-KV range round trip. usage: test-b65-slice <model.gguf> [n_prompt=2048] [n_gpu_layers=99] [fa=on|off|auto]
// fa=off also covers the transposed V cache (the llama-server runs of X33 use -fa off)
// Works for hybrid models (recurrent state cannot be rolled back): the prompt is decoded once into seq 0 and copied to
// seq 1 and seq 2 before anything else. ref = next-token logits on seq 0. seq 1: save + drop positions [n/4, 3n/4)
// of the attention cache, logits must change. seq 2: save + drop + load back, logits must match ref. seq 2 keeps the
// first 16 positions of the range (like llama-server's kept turn heads): the load must replace them, not duplicate them.
// Checks: saving the range of seq 2 again after the load gives the same bytes (exact KV + positions round trip; catches
// duplicated or missing cells), and the logits after the restore are far closer to ref than with the hole (catches a
// load that loses the rest of the sequence). Not 0: the restored cells sit at other cell indices, so the attention sums
// run in another order. seq 3, an untouched copy evaluated at the same point, is printed for scale.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::vector<float> eval_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq, int n_vocab) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = seq; b.logits[0] = 1;
    std::vector<float> v;
    if (llama_decode(ctx, b) == 0) {
        const float * l = llama_get_logits_ith(ctx, 0);
        v.assign(l, l + n_vocab);
    }
    llama_batch_free(b);
    return v;
}

static double max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || b.empty()) return 1e9;
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s model.gguf [n_prompt] [n_gpu_layers] [fa=on|off|auto]\n", argv[0]); return 2; }
    const int n = argc > 2 ? atoi(argv[2]) : 2048;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 3 ? atoi(argv[3]) : 99;   // partial offload on smaller cards
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("SLICE_FAIL load\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 3 * (n + 64); cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 4; cp.kv_unified = true;
    if (argc > 4) {
        const std::string fa = argv[4];
        cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("SLICE_FAIL context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(n);
    for (int i = 0; i < n; ++i) toks[i] = 100 + (i * 7919) % 20000;
    for (int i0 = 0; i0 < n - 1; i0 += 512) {
        const int m = std::min(512, n - 1 - i0);
        llama_batch b = llama_batch_init(m, 0, 1);
        b.n_tokens = m;
        for (int j = 0; j < m; ++j) { b.token[j] = toks[i0 + j]; b.pos[j] = i0 + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 0; }
        if (llama_decode(ctx, b) != 0) { printf("SLICE_FAIL decode\n"); return 1; }
        llama_batch_free(b);
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    llama_memory_seq_cp(mem, 0, 2, -1, -1);
    llama_memory_seq_cp(mem, 0, 3, -1, -1);
    const llama_pos p0 = n / 4, p1 = 3 * n / 4;
    std::vector<float> ref = eval_one(ctx, toks[n - 1], n - 1, 0, n_vocab);
    const size_t b1 = llama_b65_seq_range_save(ctx, 1, p0, p1, "/tmp/b65-slice-1.kv");
    const bool d1 = llama_b65_seq_range_drop(ctx, 1, p0, p1);
    std::vector<float> holed = eval_one(ctx, toks[n - 1], n - 1, 1, n_vocab);
    const size_t b2 = llama_b65_seq_range_save(ctx, 2, p0, p1, "/tmp/b65-slice-2.kv");
    const bool d2 = llama_b65_seq_range_drop(ctx, 2, p0 + 16, p1);
    const bool l2 = llama_b65_seq_range_load(ctx, 2, "/tmp/b65-slice-2.kv");
    const size_t b3 = llama_b65_seq_range_save(ctx, 2, p0, p1, "/tmp/b65-slice-3.kv");
    auto slurp = [](const char * f) { std::ifstream in(f, std::ios::binary); return std::vector<char>(std::istreambuf_iterator<char>(in), {}); };
    const bool same = b3 == b2 && slurp("/tmp/b65-slice-2.kv") == slurp("/tmp/b65-slice-3.kv");
    std::vector<float> back = eval_one(ctx, toks[n - 1], n - 1, 2, n_vocab);
    std::vector<float> ctrl = eval_one(ctx, toks[n - 1], n - 1, 3, n_vocab);
    const double d_hole = max_diff(ref, holed), d_back = max_diff(ref, back), d_ctrl = max_diff(ref, ctrl);
    printf("saved %zu / %zu bytes, dropped %d %d, loaded %d, saved again after the load: %s; max |logit diff| with the hole %.4f, "
           "after restore %.6f (untouched copy %.6f)\n", b1, b2, d1, d2, l2, same ? "same bytes" : "DIFFERENT", d_hole, d_back, d_ctrl);
    const bool ok = b1 > 0 && b2 > 0 && d1 && d2 && l2 && same && d_hole > 1e-3 && d_back < std::max(1e-2, d_hole / 8);
    printf("%s\n", ok ? "SLICE_OK" : "SLICE_FAIL");
    remove("/tmp/b65-slice-1.kv"); remove("/tmp/b65-slice-2.kv"); remove("/tmp/b65-slice-3.kv");
    llama_free(ctx);
    llama_model_free(model);
    return ok ? 0 : 1;
}
