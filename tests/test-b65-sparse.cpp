// b65: sparse Vulkan buffer check - commit pages in VRAM, write/read tensors, move to system RAM and back, release.
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>

typedef ggml_backend_buffer_type_t (*buft_fn)(size_t);
typedef int (*commit_fn)(ggml_backend_buffer_t, const size_t *, const size_t *, int, int);
typedef bool (*move_fn)(ggml_backend_buffer_t, int, int);
typedef void (*release_fn)(ggml_backend_buffer_t, int);
typedef size_t (*bytes_fn)(ggml_backend_buffer_t, int);

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg) { printf("SPARSE_FAIL no Vulkan backend\n"); return 1; }
    auto get_buft = (buft_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_sparse_buffer_type");
    auto commit   = (commit_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_sparse_commit");
    auto move     = (move_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_sparse_move");
    auto release  = (release_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_sparse_release");
    auto bytes    = (bytes_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_sparse_bytes");
    if (!get_buft || !commit || !move || !release || !bytes) { printf("SPARSE_FAIL missing entry points\n"); return 1; }

    // a 4 GiB address range with one 256 MiB f32 tensor at 1 GiB and one at 3 GiB; only those get memory
    const size_t total = (size_t)4 << 30, tsz = (size_t)256 << 20;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(get_buft(0), total);
    if (!buf) { printf("SPARSE_FAIL alloc\n"); return 1; }
    ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tsz / 4);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tsz / 4);
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    ggml_backend_tensor_alloc(buf, a, base + ((size_t)1 << 30));
    ggml_backend_tensor_alloc(buf, b, base + ((size_t)3 << 30));
    size_t offs[2] = { (size_t)1 << 30, (size_t)3 << 30 }, sizes[2] = { tsz, tsz };
    int ha = commit(buf, &offs[0], &sizes[0], 1, 0);
    int hb = commit(buf, &offs[1], &sizes[1], 1, 0);
    printf("commit handles %d %d, VRAM %zu MiB, RAM %zu MiB\n", ha, hb, bytes(buf, 0) >> 20, bytes(buf, 1) >> 20);
    if (ha < 0 || hb < 0) { printf("SPARSE_FAIL commit\n"); return 1; }

    std::vector<float> va(tsz / 4), vb(tsz / 4), r(tsz / 4);
    for (size_t i = 0; i < va.size(); ++i) { va[i] = (float) i; vb[i] = -(float) i * 0.5f; }
    ggml_backend_tensor_set(a, va.data(), 0, tsz);
    ggml_backend_tensor_set(b, vb.data(), 0, tsz);
    int bad = 0;
    ggml_backend_tensor_get(a, r.data(), 0, tsz); bad += memcmp(r.data(), va.data(), tsz) != 0;
    ggml_backend_tensor_get(b, r.data(), 0, tsz); bad += memcmp(r.data(), vb.data(), tsz) != 0;
    printf("VRAM round trip: %s\n", bad ? "MISMATCH" : "ok");

    const bool m1 = move(buf, ha, 1);
    printf("move a to RAM: %s, VRAM %zu MiB, RAM %zu MiB\n", m1 ? "ok" : "refused", bytes(buf, 0) >> 20, bytes(buf, 1) >> 20);
    ggml_backend_tensor_get(a, r.data(), 0, tsz); const int bad_ram = memcmp(r.data(), va.data(), tsz) != 0;
    printf("after move to RAM: %s\n", bad_ram ? "MISMATCH" : "ok");
    const bool m2 = move(buf, ha, 0);
    ggml_backend_tensor_get(a, r.data(), 0, tsz); const int bad_back = memcmp(r.data(), va.data(), tsz) != 0;
    printf("move back to VRAM: %s, data %s\n", m2 ? "ok" : "refused", bad_back ? "MISMATCH" : "ok");
    release(buf, hb);
    printf("released b: VRAM %zu MiB\n", bytes(buf, 0) >> 20);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    const bool ok = !bad && (!m1 || (!bad_ram && m2 && !bad_back));
    printf("%s (system-RAM sparse binding %s)\n", ok ? "SPARSE_OK" : "SPARSE_FAIL", m1 ? "supported" : "not offered by the driver");
    return ok ? 0 : 1;
}
