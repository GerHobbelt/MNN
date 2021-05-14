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

#define MATMUL_V2
#ifdef MATMUL_V2

#ifndef VECTOR_WIDTH
#error VECTOR_WIDTH must be defined
#endif

inline FLOAT dotProd(FLOATX A, FLOATX B){
    FLOAT res = 0;
#if VECTOR_WIDTH >= 4
    res += dot(A.s0123, B.s0123);
#endif
#if VECTOR_WIDTH >= 8
    res += dot(A.s4567, B.s4567);
#endif
#if VECTOR_WIDTH >= 16
    res += dot(A.s89ab, B.s89ab);
    res += dot(A.scdef, B.scdef);
#endif
    return res;
}

inline void transpose(FLOATX *i, FLOATX *o){
#if VECTOR_WIDTH == 4
    o[0] = (FLOATX)(i[0].s0, i[1].s0, i[2].s0, i[3].s0);
    o[1] = (FLOATX)(i[0].s1, i[1].s1, i[2].s1, i[3].s1);
    o[2] = (FLOATX)(i[0].s2, i[1].s2, i[2].s2, i[3].s2);
    o[3] = (FLOATX)(i[0].s3, i[1].s3, i[2].s3, i[3].s3);
#elif VECTOR_WIDTH == 8
    o[0]  = (FLOATX)(i[0].s0, i[1].s0, i[2].s0, i[3].s0, i[4].s0, i[5].s0, i[6].s0, i[7].s0);
    o[1]  = (FLOATX)(i[0].s1, i[1].s1, i[2].s1, i[3].s1, i[4].s1, i[5].s1, i[6].s1, i[7].s1);
    o[2]  = (FLOATX)(i[0].s2, i[1].s2, i[2].s2, i[3].s2, i[4].s2, i[5].s2, i[6].s2, i[7].s2);
    o[3]  = (FLOATX)(i[0].s3, i[1].s3, i[2].s3, i[3].s3, i[4].s3, i[5].s3, i[6].s3, i[7].s3);
    o[4]  = (FLOATX)(i[0].s4, i[1].s4, i[2].s4, i[3].s4, i[4].s4, i[5].s4, i[6].s4, i[7].s4);
    o[5]  = (FLOATX)(i[0].s5, i[1].s5, i[2].s5, i[3].s5, i[4].s5, i[5].s5, i[6].s5, i[7].s5);
    o[6]  = (FLOATX)(i[0].s6, i[1].s6, i[2].s6, i[3].s6, i[4].s6, i[5].s6, i[6].s6, i[7].s6);
    o[7]  = (FLOATX)(i[0].s7, i[1].s7, i[2].s7, i[3].s7, i[4].s7, i[5].s7, i[6].s7, i[7].s7);
#elif VECTOR_WIDTH == 16
    o[0]  = (FLOATX)(i[0].s0, i[1].s0, i[2].s0, i[3].s0, i[4].s0, i[5].s0, i[6].s0, i[7].s0, i[8].s0, i[9].s0, i[10].s0, i[11].s0, i[12].s0, i[13].s0, i[14].s0, i[15].s0);
    o[1]  = (FLOATX)(i[0].s1, i[1].s1, i[2].s1, i[3].s1, i[4].s1, i[5].s1, i[6].s1, i[7].s1, i[8].s1, i[9].s1, i[10].s1, i[11].s1, i[12].s1, i[13].s1, i[14].s1, i[15].s1);
    o[2]  = (FLOATX)(i[0].s2, i[1].s2, i[2].s2, i[3].s2, i[4].s2, i[5].s2, i[6].s2, i[7].s2, i[8].s2, i[9].s2, i[10].s2, i[11].s2, i[12].s2, i[13].s2, i[14].s2, i[15].s2);
    o[3]  = (FLOATX)(i[0].s3, i[1].s3, i[2].s3, i[3].s3, i[4].s3, i[5].s3, i[6].s3, i[7].s3, i[8].s3, i[9].s3, i[10].s3, i[11].s3, i[12].s3, i[13].s3, i[14].s3, i[15].s3);
    o[4]  = (FLOATX)(i[0].s4, i[1].s4, i[2].s4, i[3].s4, i[4].s4, i[5].s4, i[6].s4, i[7].s4, i[8].s4, i[9].s4, i[10].s4, i[11].s4, i[12].s4, i[13].s4, i[14].s4, i[15].s4);
    o[5]  = (FLOATX)(i[0].s5, i[1].s5, i[2].s5, i[3].s5, i[4].s5, i[5].s5, i[6].s5, i[7].s5, i[8].s5, i[9].s5, i[10].s5, i[11].s5, i[12].s5, i[13].s5, i[14].s5, i[15].s5);
    o[6]  = (FLOATX)(i[0].s6, i[1].s6, i[2].s6, i[3].s6, i[4].s6, i[5].s6, i[6].s6, i[7].s6, i[8].s6, i[9].s6, i[10].s6, i[11].s6, i[12].s6, i[13].s6, i[14].s6, i[15].s6);
    o[7]  = (FLOATX)(i[0].s7, i[1].s7, i[2].s7, i[3].s7, i[4].s7, i[5].s7, i[6].s7, i[7].s7, i[8].s7, i[9].s7, i[10].s7, i[11].s7, i[12].s7, i[13].s7, i[14].s7, i[15].s7);
    o[8]  = (FLOATX)(i[0].s8, i[1].s8, i[2].s8, i[3].s8, i[4].s8, i[5].s8, i[6].s8, i[7].s8, i[8].s8, i[9].s8, i[10].s8, i[11].s8, i[12].s8, i[13].s8, i[14].s8, i[15].s8);
    o[9]  = (FLOATX)(i[0].s9, i[1].s9, i[2].s9, i[3].s9, i[4].s9, i[5].s9, i[6].s9, i[7].s9, i[8].s9, i[9].s9, i[10].s9, i[11].s9, i[12].s9, i[13].s9, i[14].s9, i[15].s9);
    o[10] = (FLOATX)(i[0].sa, i[1].sa, i[2].sa, i[3].sa, i[4].sa, i[5].sa, i[6].sa, i[7].sa, i[8].sa, i[9].sa, i[10].sa, i[11].sa, i[12].sa, i[13].sa, i[14].sa, i[15].sa);
    o[11] = (FLOATX)(i[0].sb, i[1].sb, i[2].sb, i[3].sb, i[4].sb, i[5].sb, i[6].sb, i[7].sb, i[8].sb, i[9].sb, i[10].sb, i[11].sb, i[12].sb, i[13].sb, i[14].sb, i[15].sb);
    o[12] = (FLOATX)(i[0].sc, i[1].sc, i[2].sc, i[3].sc, i[4].sc, i[5].sc, i[6].sc, i[7].sc, i[8].sc, i[9].sc, i[10].sc, i[11].sc, i[12].sc, i[13].sc, i[14].sc, i[15].sc);
    o[13] = (FLOATX)(i[0].sd, i[1].sd, i[2].sd, i[3].sd, i[4].sd, i[5].sd, i[6].sd, i[7].sd, i[8].sd, i[9].sd, i[10].sd, i[11].sd, i[12].sd, i[13].sd, i[14].sd, i[15].sd);
    o[14] = (FLOATX)(i[0].se, i[1].se, i[2].se, i[3].se, i[4].se, i[5].se, i[6].se, i[7].se, i[8].se, i[9].se, i[10].se, i[11].se, i[12].se, i[13].se, i[14].se, i[15].se);
    o[15] = (FLOATX)(i[0].sf, i[1].sf, i[2].sf, i[3].sf, i[4].sf, i[5].sf, i[6].sf, i[7].sf, i[8].sf, i[9].sf, i[10].sf, i[11].sf, i[12].sf, i[13].sf, i[14].sf, i[15].sf);
#endif
}

inline void dot1D(FLOATX *A, FLOATX *B, FLOATX *C){
#if VECTOR_WIDTH >= 4
    C->s0 += dotProd(*A, B[0]);
    C->s1 += dotProd(*A, B[1]);
    C->s2 += dotProd(*A, B[2]);
    C->s3 += dotProd(*A, B[3]);
#endif
#if VECTOR_WIDTH >= 8
    C->s4 += dotProd(*A, B[4]);
    C->s5 += dotProd(*A, B[5]);
    C->s6 += dotProd(*A, B[6]);
    C->s7 += dotProd(*A, B[7]);
#endif
#if VECTOR_WIDTH >= 16
    C->s8 += dotProd(*A, B[8]);
    C->s9 += dotProd(*A, B[9]);
    C->sa += dotProd(*A, B[10]);
    C->sb += dotProd(*A, B[11]);
    C->sc += dotProd(*A, B[12]);
    C->sd += dotProd(*A, B[13]);
    C->se += dotProd(*A, B[14]);
    C->sf += dotProd(*A, B[15]);
#endif
}

inline void setRemainingToZero(FLOATX *A, short remain){
#if VECTOR_WIDTH == 4
    A->s3 = ((remain >= 1) ? (FLOAT)(0) : A->s3);
    A->s2 = ((remain >= 2) ? (FLOAT)(0) : A->s2);
    A->s1 = ((remain >= 3) ? (FLOAT)(0) : A->s1);
#endif
#if VECTOR_WIDTH == 8
    A->s7 = ((remain >= 1) ? (FLOAT)(0) : A->s7);
    A->s6 = ((remain >= 2) ? (FLOAT)(0) : A->s6);
    A->s5 = ((remain >= 3) ? (FLOAT)(0) : A->s5);
    A->s4 = ((remain >= 4) ? (FLOAT)(0) : A->s4);
    A->s3 = ((remain >= 5) ? (FLOAT)(0) : A->s3);
    A->s2 = ((remain >= 6) ? (FLOAT)(0) : A->s2);
    A->s1 = ((remain >= 7) ? (FLOAT)(0) : A->s1);
#endif
#if VECTOR_WIDTH == 16
    A->sf =  ((remain >= 1) ? (FLOAT)(0) : A->sf);
    A->se =  ((remain >= 2) ? (FLOAT)(0) : A->se);
    A->sd =  ((remain >= 3) ? (FLOAT)(0) : A->sd);
    A->sc =  ((remain >= 4) ? (FLOAT)(0) : A->sc);
    A->sb =  ((remain >= 5) ? (FLOAT)(0) : A->sb);
    A->sa =  ((remain >= 6) ? (FLOAT)(0) : A->sa);
    A->s9 =  ((remain >= 7) ? (FLOAT)(0) : A->s9);
    A->s8 =  ((remain >= 8) ? (FLOAT)(0) : A->s8);
    A->s7 =  ((remain >= 9) ? (FLOAT)(0) : A->s7);
    A->s6 = ((remain >= 10) ? (FLOAT)(0) : A->s6);
    A->s5 = ((remain >= 11) ? (FLOAT)(0) : A->s5);
    A->s4 = ((remain >= 12) ? (FLOAT)(0) : A->s4);
    A->s3 = ((remain >= 13) ? (FLOAT)(0) : A->s3);
    A->s2 = ((remain >= 14) ? (FLOAT)(0) : A->s2);
    A->s1 = ((remain >= 15) ? (FLOAT)(0) : A->s1);
#endif
}


__kernel void matmul(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
#ifdef BIAS
        __read_only image2d_t input_c,
#endif
                     __write_only image2d_t output_c, __private const int K,
                     __private const int kBlocks) {
    // C := A * B + bias
    // A has dims (M, K)
    // B has dims (K, N)
    // C has dims (M, N)
    // TS (tile size) = VECTOR_WIDTH
    // num nBlocks = UP_DIV(N, TS)
    // num mBlocks = UP_DIV(M, 1)
    // num kBlocks = UP_DIV(K, TS)

    const int nBlock_idx = get_global_id(0); // Block index in direction of N
    const int mBlock_idx = get_global_id(1); // Block index in direction of M

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);
    FLOATX a;
    FLOATX b_arr[VECTOR_WIDTH];
    for (short i = 0; i < VECTOR_WIDTH; i++){
        b_arr[i] = 0;
    }

#ifdef BIAS
    FLOATX results = RI_F(input_c, SAMPLER, (int2)(nBlock_idx, 0));
#else
    FLOATX results = (FLOATX)(0);
#endif

    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(kBlock_idx, mBlock_idx));
        short remain = (kBlock_idx + 1) * VECTOR_WIDTH - K;
        for (short i = 0; i < VECTOR_WIDTH; i++){
            b_arr[i] = RI_F(input_b, SAMPLER, (int2)(nBlock_idx, kBlock_idx * VECTOR_WIDTH + i));
        }
        for (short i = 0; i < remain; i++){
            b_arr[VECTOR_WIDTH - 1 - i] = 0;
        }
        // There is no way to avoid typing out this part for each vector size
        FLOATX btmp_arr[VECTOR_WIDTH];

        transpose(b_arr, btmp_arr);

        dot1D(&a, btmp_arr, &results);
    }

    WI_F(output_c, (int2)(nBlock_idx, mBlock_idx), results);
}
#else
__kernel void matmul(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c, __private const int channels,
                     __private const int channel_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOATX a;
    FLOATX b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    #ifdef BIAS
    FLOATX temp = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOAT result0 = temp.x;
    FLOAT result1 = temp.y;
    FLOAT result2 = temp.z;
    FLOAT result3 = temp.w;
    #else
    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

        short remain = (pos + 1) * VECTOR_WIDTH - channels;

        b0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * VECTOR_WIDTH));
        b1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * VECTOR_WIDTH + 1));
        b2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * VECTOR_WIDTH + 2));
        b3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * VECTOR_WIDTH + 3));

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

        FLOATX btmp0 = (FLOATX)(b0.s0, b1.s0, b2.s0, b3.s0);
        FLOATX btmp1 = (FLOATX)(b0.s1, b1.s1, b2.s1, b3.s1);
        FLOATX btmp2 = (FLOATX)(b0.s2, b1.s2, b2.s2, b3.s2);
        FLOATX btmp3 = (FLOATX)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }
    WI_F(output_c, (int2)(width_blocks_idx, height_idx), (FLOATX)(result0, result1, result2, result3));
}
#endif

#ifdef MATMUL_V2
__kernel void matmul_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                            __read_only image2d_t input_b,
#ifdef BIAS
        __read_only image2d_t input_c,
#endif
                            __write_only image2d_t output_c, __private const int K,
                            __private const int kBlocks) {
    const int nBlock_idx    = get_global_id(0);
    const int mBlock_idx    = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);
    FLOATX a;
    FLOATX b_arr[VECTOR_WIDTH];
    for (short i = 0; i < VECTOR_WIDTH; i++){
        b_arr[i] = 0;
    }

#ifdef BIAS
    FLOATX results = RI_F(input_c, SAMPLER, (int2)(nBlock_idx, 0));
#else
    FLOATX results = (FLOATX)(0);
#endif

    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(kBlock_idx, mBlock_idx));

        for (short i = 0; i < VECTOR_WIDTH; i++){
            b_arr[i] = RI_F(input_b, SAMPLER, (int2)(kBlock_idx, nBlock_idx * VECTOR_WIDTH + i));
        }

        short remain = (kBlock_idx + 1) * VECTOR_WIDTH - K;
        setRemainingToZero(&a, remain);

        dot1D(&a, b_arr, &results);
    }
    WI_F(output_c, (int2)(nBlock_idx, mBlock_idx), results);
}
#else
__kernel void matmul_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c, __private const int channels,
                     __private const int channel_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOATX a;
    FLOATX b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    #ifdef BIAS
    FLOATX temp = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOAT result0 = temp.x;
    FLOAT result1 = temp.y;
    FLOAT result2 = temp.z;
    FLOAT result3 = temp.w;
    #else
    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

        short remain = (pos + 1) * VECTOR_WIDTH - channels;

        b0 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * VECTOR_WIDTH));
        b1 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * VECTOR_WIDTH + 1));
        b2 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * VECTOR_WIDTH + 2));
        b3 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * VECTOR_WIDTH + 3));
        if (remain == 3) {
            a.y = 0;
            a.z = 0;
            a.w = 0;
        } else if (remain == 2) {
            a.z = 0;
            a.w = 0;
        } else if (remain == 1) {
            a.w = 0;
        }

        result0 += dot(a, b0);
        result1 += dot(a, b1);
        result2 += dot(a, b2);
        result3 += dot(a, b3);
    }
    WI_F(output_c, (int2)(width_blocks_idx, height_idx), (FLOATX)(result0, result1, result2, result3));
}
#endif

#ifdef MATMUL_V2
__kernel void matmul_transA(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                            __read_only image2d_t input_b,
#ifdef BIAS
        __read_only image2d_t input_c,
#endif
                            __write_only image2d_t output_c,
                            __private const int K,
                            __private const int kBlocks,
                            __private const int M) {
    const int nBlock_idx = get_global_id(0);
    const int mBlock_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);

    FLOATX v_zero = (FLOATX)((FLOAT)0.0);
    FLOATX result_arr[VECTOR_WIDTH];
#ifdef BIAS
    FLOATX result_arr[0] = RI_F(input_c, SAMPLER, (int2)(nBlock_idx, 0));
    for (short i = 1; i < VECTOR_WIDTH; i++){
        result_arr[i] = result_arr[0];
    }
#else
    for (short i = 0; i < VECTOR_WIDTH; i++){
        result_arr[i] = 0;
    }
#endif

    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx += 1) {

        FLOATX a_arr[VECTOR_WIDTH];
        FLOATX b_arr[VECTOR_WIDTH];
        for(short i = 0; i < VECTOR_WIDTH; i++){
            a_arr[i] = RI_F(input_a, SAMPLER, (int2)(mBlock_idx, VECTOR_WIDTH*kBlock_idx + i));
            b_arr[i] = RI_F(input_b, SAMPLER, (int2)(nBlock_idx, VECTOR_WIDTH*kBlock_idx + i));
        }

        short remain = (kBlock_idx + 1) * VECTOR_WIDTH - K;
        for (short i = 0; i < remain; i++){
            a_arr[VECTOR_WIDTH - 1 - i] = v_zero;
        }

        FLOATX aTrans_arr[VECTOR_WIDTH];
        transpose(a_arr, aTrans_arr);

        FLOATX bTrans_arr[VECTOR_WIDTH];
        transpose(b_arr, bTrans_arr);

        //matmul
        for (short i = 0; i < VECTOR_WIDTH; i++){
            dot1D(&(aTrans_arr[i]), bTrans_arr, &(result_arr[i]));
        }
    }

    for (short i = 0; i < VECTOR_WIDTH && !(VECTOR_WIDTH*mBlock_idx + i >= M); i++){
        WI_F(output_c, (int2)(nBlock_idx, VECTOR_WIDTH*mBlock_idx + i), result_arr[i]);
    }
}
#else
__kernel void matmul_transA(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                 __read_only image2d_t input_b,
                #ifdef BIAS
                 __read_only image2d_t input_c,
                #endif
                 __write_only image2d_t output_c,
                 __private const int channels,
                 __private const int channel_blocks,
                  __private const int height) {
const int width_blocks_idx = get_global_id(0);
const int height_blocks_idx = get_global_id(1);

DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

FLOATX v_zero = (FLOATX)((FLOAT)0.0);
#ifdef BIAS
FLOATX result0 = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
FLOATX result1 = result0;
FLOATX result2 = result0;
FLOATX result3 = result0;
#else
FLOATX result0 = 0;
FLOATX result1 = 0;
FLOATX result2 = 0;
FLOATX result3 = 0;
#endif

for (short pos = 0; pos < channel_blocks; pos += 1) {
    FLOATX a0 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos));
    FLOATX a1 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos+1));
    FLOATX a2 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos+2));
    FLOATX a3 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos+3));

    FLOATX b0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, VECTOR_WIDTH*pos));
    FLOATX b1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, VECTOR_WIDTH*pos+1));
    FLOATX b2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, VECTOR_WIDTH*pos+2));
    FLOATX b3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, VECTOR_WIDTH*pos+3));

    short remain = (pos + 1) * VECTOR_WIDTH - channels;
    a3 = ((remain >= 1) ? v_zero : a3);
    a2 = ((remain >= 2) ? v_zero : a2);
    a1 = ((remain >= 3) ? v_zero : a1);

    FLOATX a0_trans = (FLOATX)(a0.x, a1.x, a2.x, a3.x);
    FLOATX a1_trans = (FLOATX)(a0.y, a1.y, a2.y, a3.y);
    FLOATX a2_trans = (FLOATX)(a0.z, a1.z, a2.z, a3.z);
    FLOATX a3_trans = (FLOATX)(a0.w, a1.w, a2.w, a3.w);

    FLOATX b0_trans = (FLOATX)(b0.x, b1.x, b2.x, b3.x);
    FLOATX b1_trans = (FLOATX)(b0.y, b1.y, b2.y, b3.y);
    FLOATX b2_trans = (FLOATX)(b0.z, b1.z, b2.z, b3.z);
    FLOATX b3_trans = (FLOATX)(b0.w, b1.w, b2.w, b3.w);

    //matmul
    result0.x += dot(a0_trans, b0_trans);
    result0.y += dot(a0_trans, b1_trans);
    result0.z += dot(a0_trans, b2_trans);
    result0.w += dot(a0_trans, b3_trans);

    result1.x += dot(a1_trans, b0_trans);
    result1.y += dot(a1_trans, b1_trans);
    result1.z += dot(a1_trans, b2_trans);
    result1.w += dot(a1_trans, b3_trans);

    result2.x += dot(a2_trans, b0_trans);
    result2.y += dot(a2_trans, b1_trans);
    result2.z += dot(a2_trans, b2_trans);
    result2.w += dot(a2_trans, b3_trans);

    result3.x += dot(a3_trans, b0_trans);
    result3.y += dot(a3_trans, b1_trans);
    result3.z += dot(a3_trans, b2_trans);
    result3.w += dot(a3_trans, b3_trans);
}
WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx), result0);
if(VECTOR_WIDTH*height_blocks_idx+1 >= height) return;
WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx+1), result1);
if(VECTOR_WIDTH*height_blocks_idx+2 >= height) return;
WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx+2), result2);
if(VECTOR_WIDTH*height_blocks_idx+3 >= height) return;
WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx+3), result3);

}
#endif

#ifdef MATMUL_V2
__kernel void matmul_transA_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                                   __read_only image2d_t input_b,
#ifdef BIAS
        __read_only image2d_t input_c,
#endif
                                   __write_only image2d_t output_c,
                                   __private const int K,
                                   __private const int kBlocks,
                                   __private const int M) {
    const int nBlock_idx = get_global_id(0);
    const int mBlock_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);

    FLOATX v_zero = (FLOATX)((FLOAT)0.0);
    FLOATX result_arr[VECTOR_WIDTH];
#ifdef BIAS
    result_arr[0] = RI_F(input_c, SAMPLER, (int2)(nBlock_idx, 0));
    for (short i = 1; i < VECTOR_WIDTH; i++){
        result_arr[i] = result_arr[0];
    }
#else
    for (short i = 0; i < VECTOR_WIDTH; i++){
        result_arr[i] = 0;
    }
#endif

    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx += 1) {

        FLOATX a_arr[VECTOR_WIDTH];
        FLOATX b_arr[VECTOR_WIDTH];

        for (short i = 0; i < VECTOR_WIDTH; i++){
            a_arr[i] = RI_F(input_a, SAMPLER, (int2)(mBlock_idx, VECTOR_WIDTH*kBlock_idx + i));
            b_arr[i] = RI_F(input_b, SAMPLER, (int2)(kBlock_idx, VECTOR_WIDTH*nBlock_idx + i));
        }

        short remain = (kBlock_idx + 1) * VECTOR_WIDTH - K;
        for (short i = 0; i < remain; i++){
            a_arr[VECTOR_WIDTH - 1 - i] = v_zero;
        }

        FLOATX aTrans_arr[VECTOR_WIDTH];
        transpose(a_arr, aTrans_arr);

        //matmul
        for (short i = 0; i < VECTOR_WIDTH; i++){
            dot1D(&(aTrans_arr[i]), b_arr, &(result_arr[i]));
        }
    }

    for (short i = 0; i < VECTOR_WIDTH && !(VECTOR_WIDTH*mBlock_idx+i >= M); i++){
        WI_F(output_c, (int2)(nBlock_idx, VECTOR_WIDTH*mBlock_idx+i), result_arr[i]);
    }
}
#else
__kernel void matmul_transA_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a,
                     __read_only image2d_t input_b,
                    #ifdef BIAS
                     __read_only image2d_t input_c,
                    #endif
                     __write_only image2d_t output_c,
                     __private const int channels,
                     __private const int channel_blocks,
                      __private const int height) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    FLOATX v_zero = (FLOATX)((FLOAT)0.0);
    #ifdef BIAS
    FLOATX result0 = RI_F(input_c, SAMPLER, (int2)(width_blocks_idx, 0));
    FLOATX result1 = result0;
    FLOATX result2 = result0;
    FLOATX result3 = result0;
    #else
    FLOATX result0 = 0;
    FLOATX result1 = 0;
    FLOATX result2 = 0;
    FLOATX result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        FLOATX a0 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos));
        FLOATX a1 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos+1));
        FLOATX a2 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos+2));
        FLOATX a3 = RI_F(input_a, SAMPLER, (int2)(height_blocks_idx, VECTOR_WIDTH*pos+3));

        FLOATX b0 = RI_F(input_b, SAMPLER, (int2)(pos, VECTOR_WIDTH*width_blocks_idx));
        FLOATX b1 = RI_F(input_b, SAMPLER, (int2)(pos, VECTOR_WIDTH*width_blocks_idx+1));
        FLOATX b2 = RI_F(input_b, SAMPLER, (int2)(pos, VECTOR_WIDTH*width_blocks_idx+2));
        FLOATX b3 = RI_F(input_b, SAMPLER, (int2)(pos, VECTOR_WIDTH*width_blocks_idx+3));
        
        short remain = (pos + 1) * VECTOR_WIDTH - channels;
        a3 = ((remain >= 1) ? v_zero : a3);
        a2 = ((remain >= 2) ? v_zero : a2);
        a1 = ((remain >= 3) ? v_zero : a1);

        FLOATX a0_trans = (FLOATX)(a0.x, a1.x, a2.x, a3.x);
        FLOATX a1_trans = (FLOATX)(a0.y, a1.y, a2.y, a3.y);
        FLOATX a2_trans = (FLOATX)(a0.z, a1.z, a2.z, a3.z);
        FLOATX a3_trans = (FLOATX)(a0.w, a1.w, a2.w, a3.w);

        //matmul
        result0.x += dot(a0_trans, b0);
        result0.y += dot(a0_trans, b1);
        result0.z += dot(a0_trans, b2);
        result0.w += dot(a0_trans, b3);
        
        result1.x += dot(a1_trans, b0);
        result1.y += dot(a1_trans, b1);
        result1.z += dot(a1_trans, b2);
        result1.w += dot(a1_trans, b3);
        
        result2.x += dot(a2_trans, b0);
        result2.y += dot(a2_trans, b1);
        result2.z += dot(a2_trans, b2);
        result2.w += dot(a2_trans, b3);
        
        result3.x += dot(a3_trans, b0);
        result3.y += dot(a3_trans, b1);
        result3.z += dot(a3_trans, b2);
        result3.w += dot(a3_trans, b3);
    }

    WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx), result0);
    if(VECTOR_WIDTH*height_blocks_idx+1 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx+1), result1);
    if(VECTOR_WIDTH*height_blocks_idx+2 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx+2), result2);
    if(VECTOR_WIDTH*height_blocks_idx+3 >= height) return;
    WI_F(output_c, (int2)(width_blocks_idx, VECTOR_WIDTH*height_blocks_idx+3), result3);
}
#endif