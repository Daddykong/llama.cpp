#pragma once
// b65: movable weight placement (LLAMA_W_SPARSE=1) shared by the model (owner), the KV cache (asks for VRAM) and the
// context (provides a GPU sync point). One process-wide instance: the experiment runs one model per process.
#include "ggml-backend.h"

#include <cstddef>
#include <functional>
#include <vector>

struct b65_place_entry {
    int il;
    ggml_backend_buffer_t buf;
    int handle;
    size_t bytes;
    int where;   // 0 VRAM, 1 system RAM
    int prio;    // position in the keep order: higher = moved out first
};

struct b65_place {
    std::vector<b65_place_entry> ffn;
    size_t vram_fixed = 0;
    double budget_mb = -1.0;
    std::function<void()> sync;
    bool active() const { return !ffn.empty() && budget_mb > 0; }
    size_t vram_weights() const;
    size_t evict(size_t need);   // FFN layers to system RAM until `need` bytes are freed; returns bytes freed
};

b65_place & b65_place_get();
