#if !defined(GGML_TYPE_IDS_COMP)
#define GGML_TYPE_IDS_COMP

// ggml_type enum values — must match ggml.h
#define GGML_TYPE_F32      0u
#define GGML_TYPE_F16      1u
#define GGML_TYPE_Q4_0     2u
#define GGML_TYPE_Q4_1     3u
// Vulkan-internal: row-reordered Q4_0 (GGML_VK_Q4_0_REPACK), reuses the removed Q4_2 slot
#define GGML_TYPE_Q4_0R    4u
// Vulkan-internal: row-reordered Q5_K / Q8_0 (GGML_VK_Q5_K_REPACK / GGML_VK_Q8_0_REPACK), unused type slots
#define GGML_TYPE_Q5_KR    5u
#define GGML_TYPE_Q8_0R    31u
// Vulkan-internal: row-reordered IQ4_XS (GGML_VK_IQ4_XS_REPACK), unused type slot
#define GGML_TYPE_IQ4_XSR  32u
// Vulkan-internal: row-reordered IQ4_NL (GGML_VK_IQ4_NL_REPACK), same layout as Q4_0R
#define GGML_TYPE_IQ4_NLR  33u
// Vulkan-internal: row-reordered Q6_K (GGML_VK_Q6_K_REPACK), unused type slot
#define GGML_TYPE_Q6_KR    36u
// Vulkan-internal: row-reordered Q4_1 (GGML_VK_Q4_1_REPACK), unused type slot
#define GGML_TYPE_Q4_1R    37u
#define GGML_TYPE_Q5_0     6u
#define GGML_TYPE_Q5_1     7u
#define GGML_TYPE_Q8_0     8u
#define GGML_TYPE_Q2_K    10u
#define GGML_TYPE_Q3_K    11u
#define GGML_TYPE_Q4_K    12u
#define GGML_TYPE_Q5_K    13u
#define GGML_TYPE_Q6_K    14u
#define GGML_TYPE_IQ2_XXS 16u
#define GGML_TYPE_IQ2_XS  17u
#define GGML_TYPE_IQ3_XXS 18u
#define GGML_TYPE_IQ1_S   19u
#define GGML_TYPE_IQ4_NL  20u
#define GGML_TYPE_IQ3_S   21u
#define GGML_TYPE_IQ2_S   22u
#define GGML_TYPE_IQ4_XS  23u
#define GGML_TYPE_IQ1_M   29u
#define GGML_TYPE_BF16    30u
#define GGML_TYPE_TQ1_0   34u
#define GGML_TYPE_TQ2_0   35u
#define GGML_TYPE_MXFP4   39u
#define GGML_TYPE_NVFP4   40u
#define GGML_TYPE_Q1_0    41u
#define GGML_TYPE_Q2_0    42u

#endif // !defined(GGML_TYPE_IDS_COMP)
