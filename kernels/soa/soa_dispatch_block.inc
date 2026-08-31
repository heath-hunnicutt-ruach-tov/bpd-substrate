        for (int j = 0; j < (int)ncols_dst; j++) {
            const char *act_col = (const char *)src1_q8_1.get() + (size_t)j * stride_col_y * 36;
            float *dst_col = dst_d + (size_t)j * stride_col_dst;
            gemv_soa_q8_0_q8_1<128, 0><<<nrows, 128, 0, stream>>>(
                (const char *)soa.quants, (const char *)soa.scales,
                act_col, dst_col, nrows, ncols, bpr, nullptr, nullptr);
        }
