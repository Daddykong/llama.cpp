#include "types.glsl"

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPEV4)
layout (binding = 0) readonly buffer AV4 {A_TYPEV4 data_a_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif
#if defined(A_TYPE_PACKED64)
layout (binding = 0) readonly buffer A_PACKED64 {A_TYPE_PACKED64 data_a_packed64[];};
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K) || defined(DATA_A_Q6_K)
// 16-byte view of the K-quant blocks (all are a multiple of 16 bytes) for the wide matvec path
layout (binding = 0) readonly buffer A_U128 {uvec4 data_a_u128[];};
#endif

#if defined(DATA_A_Q4_0R)
// Row-reordered Q4_0: each row holds all 16-byte quant blocks, then all fp16 scales
layout (binding = 0) readonly buffer A_R64 {uvec2 data_a_r64[];};
layout (binding = 0) readonly buffer A_R128 {uvec4 data_a_r128[];};
layout (binding = 0) readonly buffer A_R16 {float16_t data_a_r16[];};
#endif

#if defined(DATA_A_Q5_KR) || defined(DATA_A_Q8_0R) || defined(DATA_A_IQ4_XSR)
// Row-reordered Q5_K / Q8_0 (GGML_VK_Q5_K_REPACK / GGML_VK_Q8_0_REPACK): see reorder_q5_k.comp / reorder_q8_0.comp
layout (binding = 0) readonly buffer A_R128 {uvec4 data_a_r128[];};
layout (binding = 0) readonly buffer A_R32 {uint data_a_r32[];};
layout (binding = 0) readonly buffer A_R16 {float16_t data_a_r16[];};
#endif

#if (defined(DATA_A_Q4_0R) && defined(Q4_0R_K32)) || defined(DATA_A_Q5_KR) || defined(DATA_A_Q8_0R) || defined(DATA_A_IQ4_XSR)
#define A_REORDERED_K32 1
// q8_1_x4 blocks are 144 bytes (9 x 16): ds[4] then qs[32]
layout (binding = 1) readonly buffer B_R128 {ivec4 data_b_r128[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPEV2
layout (binding = 1) readonly buffer BV2 {B_TYPEV2 data_b_v2[];};
#endif
#ifdef B_TYPEV4
layout (binding = 1) readonly buffer BV4 {B_TYPEV4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};

#ifdef MUL_MAT_ID
layout (binding = 5) readonly buffer IDS {int data_ids[];};
#endif

