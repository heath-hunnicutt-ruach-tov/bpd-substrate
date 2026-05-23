/* bpd_sw_gpu_batch.cu — Batched Smith-Waterman: N alignments simultaneously
 *
 * Each alignment gets its own block. 20 SMs on P4 = 20 alignments
 * running in parallel. For 1000 reads, that's 50 waves of 20.
 *
 * This is the #1 optimization CUDASW++4.0 uses: batch thousands of
 * short alignments to fill the GPU. Each warp handles one alignment.
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_sw_batch.so bench/bpd_sw_gpu_batch.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* Each block processes one complete alignment using shared memory.
 * Block ID = alignment index. Threads within block cooperate on
 * anti-diagonals of that single alignment. */
__global__ void k_sw_batch(
    const char* __restrict__ all_queries,  /* concatenated queries */
    const int* __restrict__ query_offsets, /* start offset per query */
    const int* __restrict__ query_lengths, /* length per query */
    const char* __restrict__ ref,          /* single reference */
    int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int max_qlen,                          /* for shared memory sizing */
    int* __restrict__ scores)              /* output: one score per alignment */
{
    int align_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int qlen = query_lengths[align_idx];
    int q_offset = query_offsets[align_idx];
    const char* query = all_queries + q_offset;

    /* Shared memory: H_prev_diag, H_prev_col, E_prev, F_prev */
    extern __shared__ int smem[];
    int* H_prev_diag = smem;
    int* H_prev_col = smem + (max_qlen + 1);
    int* E_prev = smem + 2 * (max_qlen + 1);
    int* F_prev = smem + 3 * (max_qlen + 1);

    /* Initialize shared memory */
    for (int i = tid; i <= qlen; i += block_size) {
        H_prev_diag[i] = 0;
        H_prev_col[i] = 0;
        E_prev[i] = 0;
        F_prev[i] = 0;
    }
    __syncthreads();

    int local_max = 0;
    int total_diags = qlen + rlen - 1;

    for (int d = 0; d < total_diags; d++) {
        int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
        int i_max = (d + 1 < qlen) ? d + 1 : qlen;
        int num_cells = i_max - i_min + 1;

        for (int c = tid; c < num_cells; c += block_size) {
            int i = i_min + c;
            int j = d + 2 - i;
            if (i < 1 || i > qlen || j < 1 || j > rlen) continue;

            int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;
            int h_diag = H_prev_diag[i - 1];
            int h_left = H_prev_col[i];
            int h_up = H_prev_col[i - 1];

            int e_from_h = h_left - gap_open;
            int e_from_e = E_prev[i] - gap_extend;
            int e_val = (e_from_h > e_from_e) ? e_from_h : e_from_e;

            int f1 = h_up - gap_open;
            int f2 = F_prev[i - 1] - gap_extend;
            int f_val = (f1 > f2) ? f1 : f2;

            int h = h_diag + s;
            if (e_val > h) h = e_val;
            if (f_val > h) h = f_val;
            if (h < 0) h = 0;

            if (h > local_max) local_max = h;

            F_prev[i] = f_val;
            E_prev[i] = e_val;
            H_prev_diag[i] = H_prev_col[i];
            H_prev_col[i] = h;
        }
        __syncthreads();
    }

    /* Reduce local_max across threads in this block */
    __shared__ int block_max;
    if (tid == 0) block_max = 0;
    __syncthreads();
    atomicMax(&block_max, local_max);
    __syncthreads();

    /* Thread 0 writes the result */
    if (tid == 0) {
        scores[align_idx] = block_max;
    }
}

extern "C" {

/* Batch align: N queries against one reference.
 * Returns N scores in the output array. */
void bpd_sw_batch_gpu(
    const char** queries, const int* query_lengths, int n_queries,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int block_size,
    int* scores)
{
    /* Concatenate queries */
    int total_qlen = 0;
    int max_qlen = 0;
    for (int i = 0; i < n_queries; i++) {
        total_qlen += query_lengths[i];
        if (query_lengths[i] > max_qlen) max_qlen = query_lengths[i];
    }

    char* concat = (char*)malloc(total_qlen);
    int* offsets = (int*)malloc(n_queries * sizeof(int));
    int off = 0;
    for (int i = 0; i < n_queries; i++) {
        offsets[i] = off;
        memcpy(concat + off, queries[i], query_lengths[i]);
        off += query_lengths[i];
    }

    /* Device memory */
    char *d_queries, *d_ref;
    int *d_offsets, *d_lengths, *d_scores;

    cudaMalloc(&d_queries, total_qlen);
    cudaMalloc(&d_ref, rlen);
    cudaMalloc(&d_offsets, n_queries * sizeof(int));
    cudaMalloc(&d_lengths, n_queries * sizeof(int));
    cudaMalloc(&d_scores, n_queries * sizeof(int));

    cudaMemcpy(d_queries, concat, total_qlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref, rlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, n_queries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, query_lengths, n_queries * sizeof(int), cudaMemcpyHostToDevice);

    /* Shared memory: 4 arrays of (max_qlen+1) ints */
    int smem_size = 4 * (max_qlen + 1) * sizeof(int);

    /* Launch: one block per alignment */
    k_sw_batch<<<n_queries, block_size, smem_size>>>(
        d_queries, d_offsets, d_lengths,
        d_ref, rlen,
        match_score, mismatch_score, gap_open, gap_extend,
        max_qlen, d_scores);
    cudaDeviceSynchronize();

    cudaMemcpy(scores, d_scores, n_queries * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_queries); cudaFree(d_ref);
    cudaFree(d_offsets); cudaFree(d_lengths); cudaFree(d_scores);
    free(concat); free(offsets);
}

} /* extern "C" */
