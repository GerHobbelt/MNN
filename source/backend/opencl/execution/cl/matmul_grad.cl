#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// if (!transA && !transB) ==> A' = C' * BT, B' = AT * C'
__kernel void matmul_grad(  GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input_a,
                            __read_only image2d_t input_b,
                            __read_only image2d_t input_c_diff,
                            __write_only image2d_t output_a_diff,
                            __write_only image2d_t output_b_diff,
                            __private const int channels,
                            __private const int channel_blocks
                            ) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);

    { // A' = C' * BT ==> matmul_transB
        FLOAT4 c_diff;
        FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        FLOAT result00 = 0;
        FLOAT result01 = 0;
        FLOAT result02 = 0;
        FLOAT result03 = 0;

        for (short pos = 0; pos < channel_blocks; pos += 1) {
            c_diff = RI_F(input_c_diff, SAMPLER, (int2)(pos, height_idx));

            short remain = (pos + 1) * 4 - channels;

            b0 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4));
            b1 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 1));
            b2 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 2));
            b3 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 3));

            if (remain == 3) {
                c_diff.y = 0;
                c_diff.z = 0;
                c_diff.w = 0;
            } else if (remain == 2) {
                c_diff.z = 0;
                c_diff.w = 0;
            } else if (remain == 1) {
                c_diff.w = 0;
            }

            result00 += dot(c_diff, b0);
            result01 += dot(c_diff, b1);
            result02 += dot(c_diff, b2);
            result03 += dot(c_diff, b3);
        }
    }

    { // B' = AT * C' ==> matmul_transA
        FLOAT4 v_zero = (FLOAT4)((FLOAT)0.0);

        FLOAT4 result10 = 0;
        FLOAT4 result11 = 0;
        FLOAT4 result12 = 0;
        FLOAT4 result13 = 0;

        for (short pos = 0; pos < channel_blocks; pos += 1) {
            FLOAT4 a0 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos));
            FLOAT4 a1 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+1));
            FLOAT4 a2 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+2));
            FLOAT4 a3 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, 4*pos+3));

            FLOAT4 c0 = RI_F(input_c_diff, SAMPLER, (int2)(width_blocks_idx, 4*pos));
            FLOAT4 c1 = RI_F(input_c_diff, SAMPLER, (int2)(width_blocks_idx, 4*pos+1));
            FLOAT4 c2 = RI_F(input_c_diff, SAMPLER, (int2)(width_blocks_idx, 4*pos+2));
            FLOAT4 c3 = RI_F(input_c_diff, SAMPLER, (int2)(width_blocks_idx, 4*pos+3));

            short remain = (pos + 1) * 4 - channels;

            a3 = ((remain >= 1) ? v_zero : a3);
            a2 = ((remain >= 2) ? v_zero : a2);
            a1 = ((remain >= 3) ? v_zero : a1);

            FLOAT4 a0_trans = (FLOAT4)(a0.x, a1.x, a2.x, a3.x);
            FLOAT4 a1_trans = (FLOAT4)(a0.y, a1.y, a2.y, a3.y);
            FLOAT4 a2_trans = (FLOAT4)(a0.z, a1.z, a2.z, a3.z);
            FLOAT4 a3_trans = (FLOAT4)(a0.w, a1.w, a2.w, a3.w);

            FLOAT4 c0_trans = (FLOAT4)(c0.x, c1.x, c2.x, c3.x);
            FLOAT4 c1_trans = (FLOAT4)(c0.y, c1.y, c2.y, c3.y);
            FLOAT4 c2_trans = (FLOAT4)(c0.z, c1.z, c2.z, c3.z);
            FLOAT4 c3_trans = (FLOAT4)(c0.w, c1.w, c2.w, c3.w);

            //matmul
            result10.x += dot(a0_trans, c0_trans);
            result10.y += dot(a0_trans, c1_trans);
            result10.z += dot(a0_trans, c2_trans);
            result10.w += dot(a0_trans, c3_trans);

            result11.x += dot(a1_trans, c0_trans);
            result11.y += dot(a1_trans, c1_trans);
            result11.z += dot(a1_trans, c2_trans);
            result11.w += dot(a1_trans, c3_trans);

            result12.x += dot(a2_trans, c0_trans);
            result12.y += dot(a2_trans, c1_trans);
            result12.z += dot(a2_trans, c2_trans);
            result12.w += dot(a2_trans, c3_trans);

            result13.x += dot(a3_trans, c0_trans);
            result13.y += dot(a3_trans, c1_trans);
            result13.z += dot(a3_trans, c2_trans);
            result13.w += dot(a3_trans, c3_trans);
        }
    }

    WI_F(output_a_diff, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result00, result01, result02, result03));

    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx), result10);
    if(4*height_blocks_idx+1 >= height) return;
    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx+1), result11);
    if(4*height_blocks_idx+2 >= height) return;
    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx+2), result12);
    if(4*height_blocks_idx+3 >= height) return;
    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx+3), result13);
}

// if (transA && !transB) ==> AT' = C' * BT ==> A' = B * CT', B' = ATT * C' = A * C'
__kernel void matmul_grad_transA (  GLOBAL_SIZE_2_DIMS
                                    __read_only image2d_t input_a,
                                    __read_only image2d_t input_b,
                                    __read_only image2d_t input_c_diff,
                                    __write_only image2d_t output_a_diff,
                                    __write_only image2d_t output_b_diff,
                                    __private const int channels,
                                    __private const int channel_blocks
                                    ) {

    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);

    { // A' = B * CT' ==> matmul_transB

        FLOAT4 b;
        FLOAT4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

        FLOAT result00 = 0;
        FLOAT result01 = 0;
        FLOAT result02 = 0;
        FLOAT result03 = 0;

        for (short pos= = 0; pos < channel_blocks; pos+=1){
            b = RI_F(input_b, SAMPLER, (int2)(pos, height_idx));

            short remain = (pos + 1) * 4 - channels;

            c0 = RI_F(input_c_diff, SAMPLER, (int2)(pos, width_blocks_idx * 4));
            c1 = RI_F(input_c_diff, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 1));
            c2 = RI_F(input_c_diff, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 2));
            c3 = RI_F(input_c_diff, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 3));

            if (remain == 3) {
                b.y = 0;
                b.z = 0;
                b.w = 0;
            } else if (remain == 2) {
                b.z = 0;
                b.w = 0;
            } else if (remain == 1) {
                b.w = 0;
            }

            result00 += dot(b, c0);
            result01 += dot(b, c1);
            result02 += dot(b, c2);
            result03 += dot(b, c3);
        }

    }
    { // B' = A * C' ==> matmul
        FLOAT4 a;
        FLOAT4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

        FLOAT result10 = 0;
        FLOAT result11 = 0;
        FLOAT result12 = 0;
        FLOAT result13 = 0;

        for (short pos = 0; pos < channel_blocks; pos += 1){
            a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

            short remain = (pos + 1) * 4 - channels;

            c0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4));
            c1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 1));
            c2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 2));
            c3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 3));

            if (remain == 3) {
                c1 = 0;
                c2 = 0;
                c3 = 0;
            } else if (remain == 2) {
                c2 = 0;
                c3 = 0;
            } else if (remain == 1) {
                c3 = 0;
            }

            FLOAT4 btmp0 = (FLOAT4)(c0.s0, c1.s0, c2.s0, c3.s0);
            FLOAT4 btmp1 = (FLOAT4)(c0.s1, c1.s1, c2.s1, c3.s1);
            FLOAT4 btmp2 = (FLOAT4)(c0.s2, c1.s2, c2.s2, c3.s2);
            FLOAT4 btmp3 = (FLOAT4)(c0.s3, c1.s3, c2.s3, c3.s3);

            result10 += dot(a, btmp0);
            result11 += dot(a, btmp1);
            result12 += dot(a, btmp2);
            result13 += dot(a, btmp3);
        }
    }

    WI_F(output_a_diff, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result00, result01, result02, result03));
    WI_F(output_b_diff, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result10, result11, result12, result13));
}

// if (!transA && transB) ==> A' = C' * BTT = C' * B, BT' = AT * C' ==> B' = CT' * A
__kernel void matmul_grad_transB (  GLOBAL_SIZE_2_DIMS
                                    __read_only image2d_t input_a,
                                    __read_only image2d_t input_b,
                                    __read_only image2d_t input_c_diff,
                                    __write_only image2d_t output_a_diff,
                                    __write_only image2d_t output_b_diff,
                                    __private const int channels,
                                    __private const int channel_blocks
) {

    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);

    { // A' = C' * B ==> matmul

        FLOAT4 c;
        FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        FLOAT result00 = 0;
        FLOAT result01 = 0;
        FLOAT result02 = 0;
        FLOAT result03 = 0;

        for(short pos = 0; pos < channel_blocks; pos += 1){
            c = RI_F(input_c_diff, SAMPLER, (int2)(pos, height_idx));

            short remain = (pos + 1) * 4 - channels;

            b0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4));
            b1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 1));
            b2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 2));
            b3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 3));

            if (remain == 3) {
                b1 = 0;
                b2 = 0;
                b3 = 0;
            } else if (remain == 2) {
                b2 = 0;
                b3 = 0;
            } else if (remain == 1) {
                b3 = 0;
            }

            FLOAT4 btmp0 = (FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
            FLOAT4 btmp1 = (FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
            FLOAT4 btmp2 = (FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
            FLOAT4 btmp3 = (FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

            result00 += dot(c, btmp0);
            result01 += dot(c, btmp1);
            result02 += dot(c, btmp2);
            result03 += dot(c, btmp3);
        }

    }
    { // B' = CT' * A ==> matmul_transA

        FLOAT4 v_zero = (FLOAT4)((FLOAT)0.0);
        FLOAT4 result10 = 0;
        FLOAT4 result11 = 0;
        FLOAT4 result12 = 0;
        FLOAT4 result13 = 0;

        for (short pos = 0; pos < channel_blocks; pos += 1) {
            FLOAT4 c0 = RI_F(input_c_diff, SAMPLER, (int2)(height_blocks_idx, 4*pos));
            FLOAT4 c1 = RI_F(input_c_diff, SAMPLER, (int2)(height_blocks_idx, 4*pos+1));
            FLOAT4 c2 = RI_F(input_c_diff, SAMPLER, (int2)(height_blocks_idx, 4*pos+2));
            FLOAT4 c3 = RI_F(input_c_diff, SAMPLER, (int2)(height_blocks_idx, 4*pos+3));

            FLOAT4 a0 = RI_F(input_a, SAMPLER, (int2)(width_blocks_idx, 4*pos));
            FLOAT4 a1 = RI_F(input_a, SAMPLER, (int2)(width_blocks_idx, 4*pos+1));
            FLOAT4 a2 = RI_F(input_a, SAMPLER, (int2)(width_blocks_idx, 4*pos+2));
            FLOAT4 a3 = RI_F(input_a, SAMPLER, (int2)(width_blocks_idx, 4*pos+3));

            short remain = (pos + 1) * 4 - channels;
            c3 = ((remain >= 1) ? v_zero : c3);
            c2 = ((remain >= 2) ? v_zero : c2);
            c1 = ((remain >= 3) ? v_zero : c1);

            FLOAT4 c0_trans = (FLOAT4)(c0.x, c1.x, c2.x, c3.x);
            FLOAT4 c1_trans = (FLOAT4)(c0.y, c1.y, c2.y, c3.y);
            FLOAT4 c2_trans = (FLOAT4)(c0.z, c1.z, c2.z, c3.z);
            FLOAT4 c3_trans = (FLOAT4)(c0.w, c1.w, c2.w, c3.w);

            FLOAT4 a0_trans = (FLOAT4)(a0.x, a1.x, a2.x, a3.x);
            FLOAT4 a1_trans = (FLOAT4)(a0.y, a1.y, a2.y, a3.y);
            FLOAT4 a2_trans = (FLOAT4)(a0.z, a1.z, a2.z, a3.z);
            FLOAT4 a3_trans = (FLOAT4)(a0.w, a1.w, a2.w, a3.w);

            //matmul
            result10.x += dot(c0_trans, a0_trans);
            result10.y += dot(c0_trans, a1_trans);
            result10.z += dot(c0_trans, a2_trans);
            result10.w += dot(c0_trans, a3_trans);

            result11.x += dot(c1_trans, a0_trans);
            result11.y += dot(c1_trans, a1_trans);
            result11.z += dot(c1_trans, a2_trans);
            result11.w += dot(c1_trans, a3_trans);

            result12.x += dot(c2_trans, a0_trans);
            result12.y += dot(c2_trans, a1_trans);
            result12.z += dot(c2_trans, a2_trans);
            result12.w += dot(c2_trans, a3_trans);

            result13.x += dot(c3_trans, a0_trans);
            result13.y += dot(c3_trans, a1_trans);
            result13.z += dot(c3_trans, a2_trans);
            result13.w += dot(c3_trans, a3_trans);
        }
    }

    WI_F(output_a_diff, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result00, result01, result02, result03));

    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx), result10);
    if(4*height_blocks_idx+1 >= height) return;
    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx+1), result11);
    if(4*height_blocks_idx+2 >= height) return;
    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx+2), result12);
    if(4*height_blocks_idx+3 >= height) return;
    WI_F(output_b_diff, (int2)(width_blocks_idx, 4*height_blocks_idx+3), result13);
}

// if (transA && transB) ==> AT' = C' * BTT  ==> A' = BT * CT', BT' = ATT * C'  ==>  B' = CT' * AT
__kernel void matmul_grad_transA_transB (  GLOBAL_SIZE_2_DIMS
                                    __read_only image2d_t input_a,
                                    __read_only image2d_t input_b,
                                    __read_only image2d_t input_c_diff,
                                    __write_only image2d_t output_a_diff,
                                    __write_only image2d_t output_b_diff,
                                    __private const int channels,
                                    __private const int channel_blocks
) {

    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);

    { // A' = BT * CT' ==> matmul_transA_transB

    }
    { // B' = CT' * AT ==> matmul_transA_transB

    }

}