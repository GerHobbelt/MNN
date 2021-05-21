#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}
#ifndef MATMUL_V2
#define MATMUL_V2
#endif

#ifdef MATMUL_V2

#ifndef VECTOR_WIDTH
#error VECTOR_WIDTH must be defined
#endif

//#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
//#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define MUST_PRINT() (get_global_id(0) == 0 && get_global_id(1) == 0)

inline FLOATX load(const __global FLOAT* p, const short row, const short col, const short width, const short num_cols){
// P is pointer to buffer memory
// ROW IS ROW_IDX
// COL IS COLUMN_IDX WHERE COLUMN_WIDTH=4 IE ITERATOR VALUE
// WIDTH IS THE MEMORY WIDTH IE ROUND_UP(K, 4) OR ROUND_UP(N, 4)
// NUM_COLS IS THE NUMBER OF COLUMNS WITH WIDTH=4 IE UP_DIV(K, 4) OR UP_DIV(N, 4)
#if VECTOR_WIDTH == 4
    return vload4(row * num_cols + col, p);
#elif VECTOR_WIDTH == 8
// if the width is perfectly divisible by 8 (2, 4-block widths), always read full 8 element reads
    // Can possibly become compile-time directive
    if (width%8==0){
        return vload8((row * num_cols + col)/2, p);
    }

    // if not in the last column, read full 8 elements
    // must be two vload4 because of indexing multipliers embedded in vloadn
    // which prevents reading from the middle of n element blocks
    if (col < num_cols - 1){
        return (FLOAT8)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p));
    }

    // else the read will be a half-block
    return (FLOAT8)(vload4(row * num_cols + col, p), (FLOAT4)(0));
#elif VECTOR_WIDTH == 16
// if the width is perfectly divisible by 16 (4, 4-block widths),
    // always read full 16 element reads
    // Can possibly become compile-time directive
    if (width%16==0){
        return vload16((row * num_cols + col)/4, p);
    }

    // if not in the last 3 columns IE not 12 or less elems remaining, read 16 elems
    if (col < num_cols - 3){
        return (FLOAT16)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p), vload4(row * num_cols + col + 2, p), vload4(row * num_cols + col + 3, p));
    }

    // if the width is perfectly divisible by 8 but not 16,
    // ie width = (floor(num_cols/4))*16 + 8;
    // therefore read 8 elements w padding
    // Can possibly become compile-time directive
    if (width%8==0){
        return (FLOAT16)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p), (FLOAT8)(0));
    }

    // num_cols/4 is integer division -> floor(num_cols/4.0)
    // can possibly be compile time directive
    short remain = width - (num_cols/4)*16;

    if (remain == 4){
        return (FLOAT16)(vload4(row * num_cols + col, p), (FLOAT4)(0), (FLOAT8)(0));
    }
    if (remain == 12) {
        return (FLOAT16)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p), vload4(row * num_cols + col + 2, p), (FLOAT4)(0));
    }
#endif
}

inline FLOATX load_with_movement(const __global FLOAT* p, const short row, const short col, const short width, const short num_cols, short *num_vec4_moved){
    // P is pointer to buffer memory
    // ROW IS ROW_IDX
    // COL IS COLUMN_IDX WHERE COLUMN_WIDTH=4 IE ITERATOR VALUE
    // WIDTH IS THE MEMORY WIDTH IE ROUND_UP(K, 4) OR ROUND_UP(N, 4)
    // NUM_COLS IS THE NUMBER OF COLUMNS WITH WIDTH=4 IE UP_DIV(K, 4) OR UP_DIV(N, 4)
#if VECTOR_WIDTH == 4
    *num_vec4_moved = 1;
    return vload4(row * num_cols + col, p);
#elif VECTOR_WIDTH == 8
    // if the width is perfectly divisible by 8 (2, 4-block widths), always read full 8 element reads
    // Can possibly become compile-time directive
    if (width%8==0){
        *num_vec4_moved = 2;
        return vload8((row * num_cols + col)/2, p);
    }

    // if not in the last column, read full 8 elements
    // must be two vload4 because of indexing multipliers embedded in vloadn
    // which prevents reading from the middle of n element blocks
    if (col < num_cols - 1){
        *num_vec4_moved = 2;
        return (FLOAT8)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p));
    }

    // else the read will be a half-block
    *num_vec4_moved = 1;
    return (FLOAT8)(vload4(row * num_cols + col, p), (FLOAT4)(0));
#elif VECTOR_WIDTH == 16
    // if the width is perfectly divisible by 16 (4, 4-block widths),
    // always read full 16 element reads
    // Can possibly become compile-time directive
    if (width%16==0){
        *num_vec4_moved = 4;
        return vload16((row * num_cols + col)/4, p);
    }

    // if not in the last 3 columns IE not 12 or less elems remaining, read 16 elems
    if (col < num_cols - 3){
        *num_vec4_moved = 4;
        return (FLOAT16)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p), vload4(row * num_cols + col + 2, p), vload4(row * num_cols + col + 3, p));
    }

    // if the width is perfectly divisible by 8 but not 16,
    // ie width = (floor(num_cols/4))*16 + 8;
    // therefore read 8 elements w padding
    // Can possibly become compile-time directive
    if (width%8==0){
        *num_vec4_moved = 2;
        return (FLOAT16)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p), (FLOAT8)(0));
    }

    // num_cols/4 is integer division -> floor(num_cols/4.0)
    // can possibly be compile time directive
    short remain = width - (num_cols/4)*16;

    if (remain == 4){
        *num_vec4_moved = 1;
        return (FLOAT16)(vload4(row * num_cols + col, p), (FLOAT4)(0), (FLOAT8)(0));
    }
    if (remain == 12) {
        *num_vec4_moved = 3;
        return (FLOAT16)(vload4(row * num_cols + col, p), vload4(row * num_cols + col + 1, p), vload4(row * num_cols + col + 2, p), (FLOAT4)(0));
    }
#endif
}

inline void write(FLOATX *r, __global FLOAT* p, const short row, const short col, const short width, const short num_cols){
    // r is pointer to results FLOATX
    // P is pointer to buffer memory
    // ROW IS ROW_IDX
    // COL IS COLUMN_IDX WHERE COLUMN_WIDTH=4 IE ITERATOR VALUE
    // WIDTH IS THE MEMORY WIDTH IE ROUND_UP(K, 4) OR ROUND_UP(N, 4)
    // NUM_COLS IS THE NUMBER OF COLUMNS WITH WIDTH=4 IE UP_DIV(K, 4) OR UP_DIV(N, 4)
#if VECTOR_WIDTH==4
    vstore4(*r, row*num_cols+col, p);
#elif VECTOR_WIDTH==8
    // if the width is perfectly divisible by 8 (2, 4-block widths), always write full 8 elements
    // Can possibly become compile-time directive
    if (width%8==0){
        vstore8(*r, (row * num_cols + col)/2, p);
        return;
    }

    // if not in the last column, write full 8 elements
    // must be two vstore4 because of indexing multipliers embedded in vstoren
    // which prevents writing to the middle of n element blocks
    if (col < num_cols - 1){
        vstore4(r->s0123, row * num_cols + col, p);
        vstore4(r->s4567, row * num_cols + col + 1, p);
        return;
    }

    // else the write will be a half block
    vstore4(r->s0123, row * num_cols + col, p);
#elif VECTOR_WIDTH==16
    // if the width is perfectly divisible by 16 (4, 4-block widths), always write full 16 elements
    // Can possibly become compile-time directive
    if (width%16==0){
        vstore16(*r, (row*num_cols+col)/4, p);
        return;
    }

    // if not in last 3 columns IE not 12 or less elems remaining, read 16 elems
    if (col < num_cols - 3){
        vstore4(r->s0123, row * num_cols + col, p);
        vstore4(r->s4567, row * num_cols + col + 1, p);
        vstore4(r->s89ab, row * num_cols + col + 2, p);
        vstore4(r->scdef, row * num_cols + col + 3, p);
        return;
    }

    // if the width is perfectly divisible by 8 but not 16,
    // ie width = (floor(num_cols/4))*16 + 8;
    // therefore write 8 elements w padding
    // Can possibly become compile-time directive
    if (width%8 == 0){
        vstore4(r->s0123, row * num_cols + col, p);
        vstore4(r->s4567, row * num_cols + col + 1, p);
        return;
    }

    // num_cols/4 is integer division -> floor(num_cols/4.0)
    // can possibly be compile time directive
    short remain = width - (num_cols/4)*16;

    if(remain == 4){
        vstore4(r->s0123, row * num_cols + col, p);
        return;
    }
    if (remain == 12) {
        vstore4(r->s0123, row * num_cols + col, p);
        vstore4(r->s4567, row * num_cols + col + 1, p);
        vstore4(r->s89ab, row * num_cols + col + 2, p);
    }

#endif
}

inline void printFloatX(const FLOATX *f){
#if VECTOR_WIDTH==4
    printf("[%f, %f, %f, %f]\n", f->s0, f->s1, f->s2, f->s3);
#elif VECTOR_WIDTH==8
    printf("[%f, %f, %f, %f, %f, %f, %f, %f]\n", f->s0, f->s1, f->s2, f->s3, f->s4, f->s5, f->s6, f->s7);
#elif VECTOR_WIDTH==16
    printf("[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n", f->s0, f->s1, f->s2, f->s3, f->s4, f->s5, f->s6, f->s7, f->s8, f->s9, f->sa, f->sb, f->sc, f->sd, f->se, f->sf);
#endif
}

inline void printFloatXX(const FLOATX *f){
    for (short i = 0; i < VECTOR_WIDTH; i++){
        printFloatX(&(f[i]));
    }
}

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

inline void transpose(FLOATX *i, FLOATX *o) {
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

#define NUM_VEC4_PER_VECTOR VECTOR_WIDTH/4
#ifndef NBLOCKS
#error NBLOCKS must be defined
#endif

__kernel void matmul_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
        __global const FLOAT* input_b,
#ifdef BIAS
        __global const FLOAT* input_c,
#endif
        __global FLOAT* output_c,
        __private const int K,
        __private const int kBlocks,
        __private const int N,
        __private const int num_vec4_in_N,
        __private const int num_elems_in_N,
        __private const int num_vec4_in_K,
        __private const int num_elems_in_K) {

    const int nBlock_idx = get_global_id(0);// output W
    const int mBlock_idx = get_global_id(1);// output H

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);
    FLOATX a;
    __local FLOATX b_arr[NBLOCKS][VECTOR_WIDTH];
    __local int isBLoaded[NBLOCKS];

#ifdef BIAS
#error BIAS NOT IMPLEMENTED
#else
    FLOATX results = (FLOATX)(0);
#endif
    int offset_iter_K = 0;
    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx++) {
        isBLoaded[nBlock_idx] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        short offset_movement = 0;
        a = load_with_movement(input_a, mBlock_idx, offset_iter_K, num_elems_in_K, num_vec4_in_K, &offset_movement);

        if(!atomic_cmpxchg(&(isBLoaded[nBlock_idx]), 0, 1)){
            short remain = max((kBlock_idx + 1) * VECTOR_WIDTH - K, 0);
            for (short i = 0; i < VECTOR_WIDTH - remain; i++){
                b_arr[nBlock_idx][i] = load(input_b, kBlock_idx*VECTOR_WIDTH + i, nBlock_idx*NUM_VEC4_PER_VECTOR, num_elems_in_N, num_vec4_in_N);
            }
            for (short i = 0; i < remain; i++){
                b_arr[nBlock_idx][VECTOR_WIDTH - 1 - i] = 0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        FLOATX btmp_arr[VECTOR_WIDTH];
        transpose(&(b_arr[nBlock_idx]), btmp_arr);
        dot1D(&a, btmp_arr, &results);

        offset_iter_K += offset_movement;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    write(&results, output_c, mBlock_idx, nBlock_idx *  NUM_VEC4_PER_VECTOR, num_elems_in_N, num_vec4_in_N);
    barrier(CLK_LOCAL_MEM_FENCE);
}
#else
__kernel void matmul_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                     #ifdef BIAS
                     __global const FLOAT* input_c,
                     #endif
                     __global FLOAT* output_c, 
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int width_blocks) {
    const int width_blocks_idx = get_global_id(0);// output W
    const int height_idx       = get_global_id(1);// output H

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    #ifdef BIAS
    FLOAT4 temp = vload4(width_blocks_idx, input_c);

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
        const int inpa_offset = height_idx * channel_blocks + pos;
        a = vload4(inpa_offset, input_a);

        short remain = (pos + 1) * 4 - channels;
        const int inpb_offset = (pos*4) * width_blocks + width_blocks_idx;

        b0 = vload4(inpb_offset, input_b);
        b1 = vload4(inpb_offset + width_blocks, input_b);
        b2 = vload4(inpb_offset + width_blocks*2, input_b);
        b3 = vload4(inpb_offset + width_blocks*3, input_b);
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

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }

    const int out_offset = height_idx * width_blocks + width_blocks_idx;
    vstore4((FLOAT4)(result0, result1, result2, result3), out_offset, output_c);
}
#endif

#ifdef MATMUL_V2
__kernel void matmul_transB_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
        __global const FLOAT* input_b,
#ifdef BIAS
        __global const FLOAT* input_c,
#endif
        __global FLOAT* output_c,
        __private const int K,
        __private const int kBlocks,
        __private const int N,
        __private const int num_vec4_in_N,
        __private const int num_elems_in_N,
        __private const int num_vec4_in_K,
        __private const int num_elems_in_K) {

    const int nBlock_idx = get_global_id(0);
    const int mBlock_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);
    FLOATX a;
    FLOATX b_arr[VECTOR_WIDTH];

    #pragma unroll VECTOR_WIDTH
    for (short i = 0; i < VECTOR_WIDTH; i++){
        b_arr[i] = 0;
    }

#ifdef BIAS
#error BIAS NOT IMPLEMENTED
#else
    FLOATX results = (FLOATX)(0);
#endif

    int offset_iter_K = 0;
    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx += 1) {
        short offset_movement = 0;
        a = load_with_movement(input_a, mBlock_idx, offset_iter_K, num_elems_in_K, num_vec4_in_K, &offset_movement);

        #pragma unroll VECTOR_WIDTH
        for (short i = 0; i < VECTOR_WIDTH; i++){
            b_arr[i] = load(input_b, nBlock_idx * VECTOR_WIDTH + i, offset_iter_K, num_elems_in_K, num_vec4_in_K);
        }

        short remain = (kBlock_idx + 1) * VECTOR_WIDTH - K;
        setRemainingToZero(&a, remain);

        dot1D(&a, b_arr, &results);
        offset_iter_K += offset_movement;
    }

    write(&results, output_c, mBlock_idx, nBlock_idx *  NUM_VEC4_PER_VECTOR, num_elems_in_N, num_vec4_in_N);
}
#else
__kernel void matmul_transB_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                    #ifdef BIAS
                     __global const FLOAT* input_c,
                    #endif
                     __global FLOAT* output_c, 
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int width_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    #ifdef BIAS
    FLOAT4 temp = vload4(width_blocks_idx, input_c);
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
        const int inpa_offset = height_idx * channel_blocks + pos;
        a = vload4(inpa_offset, input_a);

        short remain = (pos + 1) * 4 - channels;
        const int inpb_offset = (width_blocks_idx*4) * channel_blocks + pos;

        b0 = vload4(inpb_offset, input_b);
        b1 = vload4(inpb_offset + channel_blocks, input_b);
        b2 = vload4(inpb_offset + channel_blocks*2, input_b);
        b3 = vload4(inpb_offset + channel_blocks*3, input_b);

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
    const int out_offset = height_idx * width_blocks + width_blocks_idx;
    vstore4((FLOAT4)(result0, result1, result2, result3), out_offset, output_c);
}
#endif
#ifdef MATMUL_V2
__kernel void matmul_transA_buf(GLOBAL_SIZE_2_DIMS
        __global const FLOAT* input_a,
        __global const FLOAT* input_b,
#ifdef BIAS
        __global const FLOAT* input_c,
#endif
        __global FLOAT* output_c,
        __private const int K,
        __private const int kBlocks,
        __private const int M,
        __private const int mBlocks,
        __private const int N,
        __private const int num_vec4_in_M,
        __private const int num_elems_in_M,
        __private const int num_vec4_in_N,
        __private const int num_elems_in_N) {

    const int nBlock_idx = get_global_id(0);
    const int mBlock_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);

    FLOATX v_zero = (FLOATX)((FLOAT)0.0);
    FLOATX result_arr[VECTOR_WIDTH];
    #ifdef BIAS
#error BIAS NOT IMPLEMENTED
    result_arr[0] = vloadX(nBlock_idx, input_c);
    for (short i = 1; i < VECTOR_WIDTH; i++){
        result_arr[i] = result_arr[0];
    }
    #else
    #pragma unroll VECTOR_WIDTH
    for (short i = 0; i < VECTOR_WIDTH; i++){
        result_arr[i] = (FLOATX)(0);
    }
    #endif

    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx++) {
        FLOATX a_arr[VECTOR_WIDTH];
        FLOATX b_arr[VECTOR_WIDTH];
#pragma unroll VECTOR_WIDTH
        for (short i = 0; i < VECTOR_WIDTH; i++){
            a_arr[i] = load(input_a, kBlock_idx*VECTOR_WIDTH + i, mBlock_idx * NUM_VEC4_PER_VECTOR, num_elems_in_M, num_vec4_in_M);
            b_arr[i] = load(input_b, kBlock_idx*VECTOR_WIDTH + i, nBlock_idx * NUM_VEC4_PER_VECTOR, num_elems_in_N, num_vec4_in_N);
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
#pragma unroll VECTOR_WIDTH
        for (short i = 0; i < VECTOR_WIDTH; i++){
            dot1D(&(aTrans_arr[i]), bTrans_arr, &(result_arr[i]));
        }
    }

    for (short i = 0; i < VECTOR_WIDTH && !(VECTOR_WIDTH*mBlock_idx+i >= M); i++){
        write(&(result_arr[i]), output_c, VECTOR_WIDTH*mBlock_idx + i, nBlock_idx *  NUM_VEC4_PER_VECTOR, num_elems_in_N, num_vec4_in_N);
    }
}
#else
__kernel void matmul_transA_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                 __global const FLOAT* input_b,
                #ifdef BIAS
                 __global const FLOAT* input_c,
                #endif
                 __global FLOAT* output_c,
                 __private const int channels,
                 __private const int channel_blocks,
                 __private const int height,
                 __private const int height_blocks,
                 __private const int width_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    FLOAT4 v_zero = (FLOAT4)((FLOAT)0.0);
    #ifdef BIAS
    FLOAT4 result0 = vload4(width_blocks_idx, input_c);
    FLOAT4 result1 = result0;
    FLOAT4 result2 = result0;
    FLOAT4 result3 = result0;
    #else
    FLOAT4 result0 = 0;
    FLOAT4 result1 = 0;
    FLOAT4 result2 = 0;
    FLOAT4 result3 = 0;
    #endif
    
    for (short pos = 0; pos < channel_blocks; pos += 1) {

        const int inpa_offset = (4*pos) * height_blocks + height_blocks_idx;
        FLOAT4 a0 = vload4(inpa_offset, input_a);
        FLOAT4 a1 = vload4(inpa_offset + height_blocks, input_a);
        FLOAT4 a2 = vload4(inpa_offset + height_blocks*2, input_a);
        FLOAT4 a3 = vload4(inpa_offset + height_blocks*3, input_a);

        const int inpb_offset = (4*pos) * width_blocks + width_blocks_idx;
        FLOAT4 b0 = vload4(inpb_offset, input_b);
        FLOAT4 b1 = vload4(inpb_offset + width_blocks, input_b);
        FLOAT4 b2 = vload4(inpb_offset + width_blocks*2, input_b);
        FLOAT4 b3 = vload4(inpb_offset + width_blocks*3, input_b);

        short remain = (pos + 1) * 4 - channels;
        a3 = ((remain >= 1) ? v_zero : a3);
        a2 = ((remain >= 2) ? v_zero : a2);
        a1 = ((remain >= 3) ? v_zero : a1);

        FLOAT4 a0_trans = (FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        FLOAT4 a1_trans = (FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        FLOAT4 a2_trans = (FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        FLOAT4 a3_trans = (FLOAT4)(a0.w, a1.w, a2.w, a3.w);

        FLOAT4 b0_trans = (FLOAT4)(b0.x, b1.x, b2.x, b3.x);
        FLOAT4 b1_trans = (FLOAT4)(b0.y, b1.y, b2.y, b3.y);
        FLOAT4 b2_trans = (FLOAT4)(b0.z, b1.z, b2.z, b3.z);
        FLOAT4 b3_trans = (FLOAT4)(b0.w, b1.w, b2.w, b3.w);

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
    const int out_offset = (4*height_blocks_idx) * width_blocks + width_blocks_idx;

    vstore4(result0, out_offset, output_c);
    if(4*height_blocks_idx+1 >= height) return;
    vstore4(result1, out_offset + width_blocks, output_c);
    if(4*height_blocks_idx+2 >= height) return;
    vstore4(result2, out_offset + width_blocks*2, output_c);
    if(4*height_blocks_idx+3 >= height) return;
    vstore4(result3, out_offset + width_blocks*3, output_c);
}
#endif

#ifdef MATMUL_V2
__kernel void matmul_transA_transB_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
            __global const FLOAT* input_b,
            #ifdef BIAS
            __global const FLOAT* input_c,
            #endif
            __global FLOAT* output_c,
            __private const int K,
            __private const int kBlocks,
            __private const int M,
            __private const int mBlocks,
            __private const int N,
            __private const int num_vec4_in_M,
            __private const int num_elems_in_M,
            __private const int num_vec4_in_N,
            __private const int num_elems_in_N,
            __private const int num_vec4_in_K,
            __private const int num_elems_in_K) {

    const int nBlock_idx = get_global_id(0);
    const int mBlock_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(nBlock_idx, mBlock_idx);

    FLOATX v_zero = (FLOATX)((FLOAT)0.0);
    FLOATX result_arr[VECTOR_WIDTH];
    #ifdef BIAS
#error BIAS NOT IMPLEMENTED
    result_arr[0] = vloadX(nBlock_idx, input_c);
    #pragma unroll
    for (short i = 1; i < VECTOR_WIDTH; i++){
        result_arr[i] = result_arr[0];
    }
    #else
    #pragma unroll VECTOR_WIDTH
    for (short i = 0; i < VECTOR_WIDTH; i++){
        result_arr[i] = (FLOATX)(0);
    }
    #endif

    int offset_iter_K = 0;

    for (short kBlock_idx = 0; kBlock_idx < kBlocks; kBlock_idx++) {
        short offset_movement = 0;
        FLOATX a_arr[VECTOR_WIDTH];
        FLOATX b_arr[VECTOR_WIDTH];

#pragma unroll VECTOR_WIDTH
        for (short i = 0; i < VECTOR_WIDTH; i++){
            a_arr[i] = load(input_a, kBlock_idx*VECTOR_WIDTH + i, mBlock_idx * NUM_VEC4_PER_VECTOR, num_elems_in_M, num_vec4_in_M);
            b_arr[i] = load_with_movement(input_b, nBlock_idx * VECTOR_WIDTH + i, offset_iter_K, num_elems_in_K, num_vec4_in_K, &offset_movement);
        }

        short remain = (kBlock_idx + 1) * VECTOR_WIDTH - K;
        for (short i = 0; i < remain; i++){
            a_arr[VECTOR_WIDTH - 1 - i] = v_zero;
        }

        FLOATX aTrans_arr[VECTOR_WIDTH];
        transpose(a_arr, aTrans_arr);

        //matmul
#pragma unroll VECTOR_WIDTH
        for (short i = 0; i < VECTOR_WIDTH; i++){
            dot1D(&(aTrans_arr[i]), b_arr, &(result_arr[i]));
        }

        offset_iter_K += offset_movement;
    }

    for (short i = 0; i < VECTOR_WIDTH && !(VECTOR_WIDTH*mBlock_idx+i >= M); i++){
        write(&(result_arr[i]), output_c, VECTOR_WIDTH*mBlock_idx + i, nBlock_idx *  NUM_VEC4_PER_VECTOR, num_elems_in_N, num_vec4_in_N);
    }
}
#else
__kernel void matmul_transA_transB_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                    #ifdef BIAS
                     __global const FLOAT* input_c,
                    #endif
                     __global FLOAT* output_c,
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int height,
                     __private const int height_blocks,
                     __private const int width_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_blocks_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_blocks_idx);

    FLOAT4 v_zero = (FLOAT4)((FLOAT)0.0);
    #ifdef BIAS
    FLOAT4 result0 = vload4(width_blocks_idx, input_c);
    FLOAT4 result1 = result0;
    FLOAT4 result2 = result0;
    FLOAT4 result3 = result0;
    #else
    FLOAT4 result0 = 0;
    FLOAT4 result1 = 0;
    FLOAT4 result2 = 0;
    FLOAT4 result3 = 0;
    #endif

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        const int inpa_offset = (4*pos) * height_blocks + height_blocks_idx;
        FLOAT4 a0 = vload4(inpa_offset, input_a);
        FLOAT4 a1 = vload4(inpa_offset + height_blocks, input_a);
        FLOAT4 a2 = vload4(inpa_offset + height_blocks*2, input_a);
        FLOAT4 a3 = vload4(inpa_offset + height_blocks*3, input_a);

        const int inpb_offset = (4*width_blocks_idx) * channel_blocks + pos;
        FLOAT4 b0 = vload4(inpb_offset, input_b);
        FLOAT4 b1 = vload4(inpb_offset + channel_blocks, input_b);
        FLOAT4 b2 = vload4(inpb_offset + channel_blocks*2, input_b);
        FLOAT4 b3 = vload4(inpb_offset + channel_blocks*3, input_b);

        short remain = (pos + 1) * 4 - channels;
        a3 = ((remain >= 1) ? v_zero : a3);
        a2 = ((remain >= 2) ? v_zero : a2);
        a1 = ((remain >= 3) ? v_zero : a1);

        FLOAT4 a0_trans = (FLOAT4)(a0.x, a1.x, a2.x, a3.x);
        FLOAT4 a1_trans = (FLOAT4)(a0.y, a1.y, a2.y, a3.y);
        FLOAT4 a2_trans = (FLOAT4)(a0.z, a1.z, a2.z, a3.z);
        FLOAT4 a3_trans = (FLOAT4)(a0.w, a1.w, a2.w, a3.w);

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

    const int out_offset = (4*height_blocks_idx) * width_blocks + width_blocks_idx;

    vstore4(result0, out_offset, output_c);
    if(4*height_blocks_idx+1 >= height) return;
    vstore4(result1, out_offset + width_blocks, output_c);
    if(4*height_blocks_idx+2 >= height) return;
    vstore4(result2, out_offset + width_blocks*2, output_c);
    if(4*height_blocks_idx+3 >= height) return;
    vstore4(result3, out_offset + width_blocks*3, output_c);
}
#endif