// soa_repack_kernel.cuh — GPU repack AoS → split arrays (quants + scales)
//
// Output: two separate contiguous arrays
//   quants_out: M * K bytes (all quants, row-major)
#include <cstdio>
//   scales_out: M * bpr * 2 bytes (all scales, row-major)
//
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define SOA_RP_QK 32
#define SOA_RP_BLOCK_AOS 34

// One thread per Q8_0 block
static __global__ void repack_q8_0_aos_to_soa(
    const uint8_t * __restrict__ aos_data,
    uint8_t * __restrict__ quants_out,    // M*K bytes
    uint8_t * __restrict__ scales_out,    // M*bpr*2 bytes
    int blocks_per_row, int ncols, int total_blocks)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= total_blocks) return;
    
    int row = bid / blocks_per_row;
    int col = bid % blocks_per_row;
    
    // AoS source
    const uint8_t *src = aos_data + (size_t)row * blocks_per_row * SOA_RP_BLOCK_AOS + col * SOA_RP_BLOCK_AOS;
    
    // Quants dest: row * K + col * 32
    uint8_t *dst_q = quants_out + (size_t)row * ncols + col * SOA_RP_QK;
    
    // Scales dest: row * bpr * 2 + col * 2
    uint8_t *dst_s = scales_out + (size_t)row * blocks_per_row * 2 + col * 2;
    
    // Copy scale (2 bytes)
    dst_s[0] = src[0];
    dst_s[1] = src[1];
    // Assert: scale must be non-negative (fp16 bit 15 = 0)
    if (src[1] & 0x80) {
        printf("REPACK NEG SCALE: row=%d col=%d src_byte1=0x%02x\n", row, col, src[1]);
        *(volatile int *)0 = 0xDEAD0003;
    }
    
    // Copy quants (32 bytes) via uint16 reads (src+2 is 2-byte aligned)
    const uint16_t *sq16 = (const uint16_t *)(src + 2);
    uint32_t *dq32 = (uint32_t *)dst_q;
    for (int i = 0; i < 8; i++) {
        dq32[i] = sq16[2*i] | ((uint32_t)sq16[2*i + 1] << 16);
    }
}
