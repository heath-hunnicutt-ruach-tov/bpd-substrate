        // soa_dispatch_fused.inc — fused SoA dispatch (SWIGLU + optional x_bias residual)
        // Preconditions (checked at call site): gate_soa.quants && soa.quants && glu_op==SWIGLU
        //                                       gate_bias == nullptr; ids == nullptr.
        // ncols_dst is 1 for the typical fused FFN dispatch on Pascal, but we loop to handle 2+ too.
        {
            ggml_cuda_pool_alloc<float> scratch_alloc(ctx.pool(), (size_t)nrows);
            float * scratch_d = scratch_alloc.get();
            for (int j = 0; j < (int)ncols_dst; j++) {
                const char *act_col = (const char *)src1_q8_1.get() + (size_t)j * stride_col_y * 36;
                float *dst_col      = dst_d + (size_t)j * stride_col_dst;
                const float *xbias_col = nullptr;
                if (fusion_local.x_bias != nullptr) {
                    // x_bias is laid out per-column with stride_col_dst (matches stock's
                    // x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
                    // then load at j*stride_col_dst + threadIdx.x).
                    xbias_col = ((const float *)fusion_local.x_bias) + (size_t)j * stride_col_dst;
                }
                // Step 1: gate matmul + SiLU -> scratch
                gemv_soa_q8_0_q8_1<128, 1><<<nrows, 128, 0, stream>>>(
                    (const char *)gate_soa.quants, (const char *)gate_soa.scales,
                    act_col, scratch_d, nrows, ncols, bpr, nullptr, nullptr);
                // Step 2: up matmul, multiply by scratch (silu(gate_dot)), optional + xbias
                gemv_soa_q8_0_q8_1<128, 0><<<nrows, 128, 0, stream>>>(
                    (const char *)soa.quants, (const char *)soa.scales,
                    act_col, dst_col, nrows, ncols, bpr, scratch_d, xbias_col);
            }
        }
