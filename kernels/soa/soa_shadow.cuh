// soa_shadow.cuh — Shadow buffer registry for split SoA (quants + scales arrays)
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>
#include <mutex>

struct soa_shadow_entry {
    void *quants;  // M*K bytes
    void *scales;  // M*bpr*2 bytes
};

inline std::unordered_map<const void*, soa_shadow_entry> soa_shadow_map;
inline std::mutex soa_shadow_mutex;

static inline void soa_shadow_register(const void* aos_data, void* quants, void* scales) {
    std::lock_guard<std::mutex> lock(soa_shadow_mutex);
    soa_shadow_map[aos_data] = {quants, scales};
}

static inline soa_shadow_entry soa_shadow_lookup(const void* aos_data) {
    std::lock_guard<std::mutex> lock(soa_shadow_mutex);
    auto it = soa_shadow_map.find(aos_data);
    return (it != soa_shadow_map.end()) ? it->second : soa_shadow_entry{nullptr, nullptr};
}

// Track which output buffers already have SiLU applied (from fused gate matmul)
#include <set>
static std::set<const void*> soa_silu_fused_set;
static std::mutex soa_silu_mutex;

static inline void soa_mark_silu_fused(const void *dst_data) {
    std::lock_guard<std::mutex> lock(soa_silu_mutex);
    soa_silu_fused_set.insert(dst_data);
}

static inline bool soa_check_silu_fused(const void *src_data) {
    std::lock_guard<std::mutex> lock(soa_silu_mutex);
    auto it = soa_silu_fused_set.find(src_data);
    if (it != soa_silu_fused_set.end()) {
        soa_silu_fused_set.erase(it);  // consume the flag (one-shot)
        return true;
    }
    return false;
}

// Redirect map: matmul_dst->data → silu_dst->data
// First pass: SiLU registers the redirect
// Second pass: matmul writes directly to silu_dst, SiLU becomes no-op
static std::unordered_map<const void*, void*> soa_silu_redirect;

static inline void soa_register_silu_redirect(const void *matmul_dst, void *silu_dst) {
    std::lock_guard<std::mutex> lock(soa_silu_mutex);
    soa_silu_redirect[matmul_dst] = silu_dst;
}

static inline void* soa_get_silu_redirect(const void *matmul_dst) {
    std::lock_guard<std::mutex> lock(soa_silu_mutex);
    auto it = soa_silu_redirect.find(matmul_dst);
    return (it != soa_silu_redirect.end()) ? it->second : nullptr;
}

// Registry for FFN UP matmul results (for SwiGLU mul fusion)
// Key: layer number extracted from tensor name
static std::unordered_map<int, void*> soa_ffn_up_results;
static std::mutex soa_ffn_up_mutex;

// Extract layer number from tensor name like 'blk.15.ffn_gate.weight'
static inline int soa_extract_layer(const char *name) {
    if (!name) return -1;
    const char *p = strstr(name, "blk.");
    if (!p) return -1;
    return atoi(p + 4);
}

static inline void soa_ffn_register_up(const char *name, void *data) {
    {static int rc=0; rc++; if(rc<=5){FILE*fp=fopen("/tmp/up_reg.log","a");if(fp){fprintf(fp,"REG %d: name=%s layer=%d ptr=%p\n",rc,name?name:"null",soa_extract_layer(name),data);fclose(fp);}}}
    int layer = soa_extract_layer(name);
    if (layer >= 0) {
        std::lock_guard<std::mutex> lock(soa_ffn_up_mutex);
        soa_ffn_up_results[layer] = data;
    }
}

static inline void* soa_ffn_lookup_up(const char *name) {
    {static int lc=0; lc++; if(lc<=5){int ly=soa_extract_layer(name);std::lock_guard<std::mutex> lk(soa_ffn_up_mutex);auto it=soa_ffn_up_results.find(ly);FILE*fp=fopen("/tmp/up_reg.log","a");if(fp){fprintf(fp,"LOOK %d: name=%s layer=%d found=%d\n",lc,name?name:"null",ly,it!=soa_ffn_up_results.end());fclose(fp);}}}
    int layer = soa_extract_layer(name);
    if (layer >= 0) {
        std::lock_guard<std::mutex> lock(soa_ffn_up_mutex);
        auto it = soa_ffn_up_results.find(layer);
        if (it != soa_ffn_up_results.end()) return it->second;
    }
    return nullptr;
}
