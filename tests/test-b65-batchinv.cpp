// b65: batch invariance of the decode kernels. usage: test-b65-batchinv [backend name, default: first GPU]
// For each case the first token's outputs with n = 1 are compared bit for bit with those of the same token computed
// together with 1-3 others (n = 2, 3, 4), as in MTP verify. Prints one line per case and BATCHINV_OK / BATCHINV_DIFF.
// Environment switches of the backend (GGML_VK_BATCH_INVARIANT, GGML_VK_*_REPACK, ...) apply as usual.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> rnd(size_t n, uint32_t seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto & x : v) x = d(g);
    return v;
}

// out = mul_mat(a, b): a [k, m, ha] (type ta), b f32 [k, ntok, hb]; returns token 0's outputs for every b head
static std::vector<float> run(ggml_backend_t be, ggml_type ta, int64_t k, int64_t m, int64_t ha, int64_t hb, int64_t ntok) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, ta, k, m, ha);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, ntok, hb);
    ggml_tensor * o = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    const std::vector<float> af = rnd((size_t)(k * m * ha), 11, 0.05f);   // same weights every run
    std::vector<uint8_t> aq(ggml_nbytes(a));
    if (ta == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(af.data(), (ggml_fp16_t *)aq.data(), (int64_t)af.size());
    } else {
        ggml_quantize_chunk(ta, af.data(), aq.data(), 0, m * ha, k, nullptr);
    }
    ggml_backend_tensor_set(a, aq.data(), 0, aq.size());
    // token 0 of every head is the same in every run; the other tokens differ
    std::vector<float> bf((size_t)(k * ntok * hb));
    for (int64_t h = 0; h < hb; ++h) {
        for (int64_t t = 0; t < ntok; ++t) {
            const std::vector<float> col = rnd((size_t)k, (uint32_t)(1000 + h * 16 + (t == 0 ? 0 : t + 100 * ntok)), 1.0f);
            memcpy(bf.data() + (h * ntok + t) * k, col.data(), k * sizeof(float));
        }
    }
    ggml_backend_tensor_set(b, bf.data(), 0, bf.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);

    std::vector<float> out((size_t)(m * ntok * hb));
    ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    std::vector<float> tok0((size_t)(m * hb));
    for (int64_t h = 0; h < hb; ++h) memcpy(tok0.data() + h * m, out.data() + (h * ntok) * m, m * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return tok0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_t be = nullptr;
    if (argc > 1) {
        be = ggml_backend_init_by_name(argv[1], nullptr);
    } else {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!be) { printf("BATCHINV_FAIL no backend\n"); return 2; }
    printf("backend %s\n", ggml_backend_name(be));

    struct cs { std::string name; ggml_type ta; int64_t k, m, ha, hb; };
    std::vector<cs> cases;
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS,
                         GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_F16 }) {
        cases.push_back({ std::string("matvec ") + ggml_type_name(t) + " k=5120", t, 5120, 512, 1, 1 });
    }
    cases.push_back({ "matvec q4_0 k=17408", GGML_TYPE_Q4_0, 17408, 256, 1, 1 });
    // attention, Qwen3.8-27B: 24 query heads on 4 KV heads, head dim 256
    cases.push_back({ "attn K x Q  (short k=256, n_kv=2048)", GGML_TYPE_F16, 256, 2048, 4, 24 });
    cases.push_back({ "attn V x P  (long k=n_kv=4096)",       GGML_TYPE_F16, 4096, 256, 4, 24 });
    cases.push_back({ "attn V x P  (long k=n_kv=32768)",      GGML_TYPE_F16, 32768, 256, 4, 24 });

    int ndiff = 0;
    for (const auto & c : cases) {
        const std::vector<float> ref = run(be, c.ta, c.k, c.m, c.ha, c.hb, 1);
        std::string line = c.name + ":";
        for (int64_t n = 2; n <= 4; ++n) {
            const std::vector<float> x = run(be, c.ta, c.k, c.m, c.ha, c.hb, n);
            double md = 0; size_t nd = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                if (memcmp(&ref[i], &x[i], sizeof(float)) != 0) { ++nd; md = std::max(md, (double)fabs(ref[i] - x[i])); }
            }
            char tmp[96];
            if (nd == 0) snprintf(tmp, sizeof(tmp), "  n=%lld same", (long long)n);
            else { snprintf(tmp, sizeof(tmp), "  n=%lld DIFF %zu/%zu max %.3g", (long long)n, nd, ref.size(), md); ++ndiff; }
            line += tmp;
        }
        printf("%s\n", line.c_str());
    }
    printf("%s\n", ndiff == 0 ? "BATCHINV_OK" : "BATCHINV_DIFF");
    ggml_backend_free(be);
    return ndiff == 0 ? 0 : 1;
}
