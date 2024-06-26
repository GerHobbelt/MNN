//
//  MNNGemmHybridInt4_sdot.S
//  MNN
//
//  Created by MNN on 2023/11/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro Int32ToFloat z0, z1, z2, z3
    scvtf \z0\().4s, \z0\().4s
    scvtf \z1\().4s, \z1\().4s
    scvtf \z2\().4s, \z2\().4s
    scvtf \z3\().4s, \z3\().4s
.endm

.macro MulScale d0, d1, d2, d3, s
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm

.macro Dequant c0, a0, z0, b0, s0, idx
    fmul \c0\().4s, \c0\().4s, \a0\().4s
    fmla \c0\().4s, \z0\().4s, \s0\().s[\idx]
    fadd \c0\().4s, \c0\().4s, \b0\().4s
.endm

asm_function MNNGemmHybridInt4FP32_sdot

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmHybridInt4FP32_sdot(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, float** param); 


// Auto: x0: C*, x1: A*, x2:B*, x3: src_depth_quad, x4: dst_step, x5: dst_depth_quad, x6: realSize, x7: param
// load from param: x7: alpha*, x8: zero*, x9: bias*, x10: sums*, x11: scales*
stp d14, d15, [sp, #(-16 * 9)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x27, x28, [sp, #(16 * 8)]

ldr x8, [x7, #0]
ldr x9, [x7, #8]
ldr x10, [x7, #16]
ldr x11, [x7, #24]
ldr x12, [x7, #32]

Start:
lsl x13, x3, #3 // x13 = src_depth_quad * UNIT * UNIT_SRC / 2(int8) = src_depth_quad * 8  = src_depth_quad << 3

TILE_4:
    cmp x6, #4
    blt TILE_1
    mov x14, x4       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_4:
    // dequant info for batch
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    dup v17.4s, wzr
    dup v18.4s, wzr
    dup v19.4s, wzr
    // mask
    movi v14.16b, #15
    // offset
    movi v15.16b, #8
LoopSz_TILE_4:
    // src    : 4(batch) x [1 x 4] : v4
    // weight : 4(oc) x [1 x 4] : v0
    // dst    : 4 x 4 x [1] : v16-v19
    ld1 {v0.8b}, [x25], #8    // weight
    ld1 {v4.16b}, [x24], x15   // src
    // int4->int8
    ushr v8.16b, v0.16b, #4
    and v9.16b, v0.16b, v14.16b
    sub v8.16b, v8.16b, v15.16b
    sub v9.16b, v9.16b, v15.16b
    zip1 v0.16b, v8.16b, v9.16b
    .inst 0x4f84e010 // sdot v16.4s, v0.16b, v4.4b[0] // batch0
    .inst 0x4fa4e011 // sdot v17.4s, v0.16b, v4.4b[1] // batch1
    .inst 0x4f84e812 // sdot v18.4s, v0.16b, v4.4b[2] // batch2
    .inst 0x4fa4e813 // sdot v19.4s, v0.16b, v4.4b[3] // batch3
    subs x26, x26, #1
    bne LoopSz_TILE_4

LoopSzEnd_TILE_4:
    add x7, x7, x13
    sub x27, x27, #1
    Int32ToFloat v16, v17, v18, v19
    // Int32ToFloat v20, v21, v22, v23
    // using float scale dequant for precison
    ld1 {v5.4s}, [x23]  // scales, 4 batch,so 4 scale

    MulScale v16, v17, v18, v19, v5

Tile4Dequant:
    ld1 {v0.4s}, [x19], #16  // alpha
    ld1 {v1.4s}, [x20], #16  // zero
    ld1 {v2.4s}, [x21], #16  // bias
    ld1 {v3.4s}, [x22]  // sums
    // alpha * sum + (zero * sums) + bias
    Dequant v16, v0, v1, v2, v3, 0
    Dequant v17, v0, v1, v2, v3, 1
    Dequant v18, v0, v1, v2, v3, 2
    Dequant v19, v0, v1, v2, v3, 3
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_4
Tile4End:
    sub x6, x6, #4      // bach -= 4
    add x0, x0, #64     // dst += 4 * 4 * sizeof(float32_t)
    add x1, x1, #16     // src += 4 * 4 * sizeof(int8_t)
    add x11, x11, #16    // sum += 4 * sizeof(float32_t)
    add x12, x12, #16    // scale += 4 * sizeof(float32_t)
    b TILE_4

TILE_1:
    cmp x6, #1
    blt End
    mov x14, x4       // dst_step
    lsr x15, x4, #2   // src_step = dst_step / 4, sizeof(float32_t)/4=sizeof(int8_t)
    mov x27, x5 // dst_depth_quad
    mov x28, x0 // dst
    mov x7, x2 // weight
    // dequant info
    mov x19, x8 // alpha
    mov x20, x9 // zero
    mov x21, x10 // bias
LoopDz_TILE_1:
    mov x22, x11 // sums
    mov x23, x12 // scales
    mov x24, x1  // src
    mov x25, x7 // weight
    mov x26, x3  // src_depth_quad
    // init
    dup v16.4s, wzr
    // mask
    movi v14.16b, #15
    // offset
    movi v15.16b, #8
LoopSz_TILE_1:
    // src    : 1(batch) x [1 x 4] : v4
    // weight : 4(oc) x [1 x 4] : v0
    // dst    : 1 x 4 x [1] : v16
    ld1 {v0.8b}, [x25], #8    // weight pack*pack*0.5
    ld1 {v4.s}[0], [x24], x15   // src
    // int4->int8
    ushr v8.16b, v0.16b, #4
    and v9.16b, v0.16b, v14.16b
    sub v8.16b, v8.16b, v15.16b
    sub v9.16b, v9.16b, v15.16b
    zip1 v0.16b, v8.16b, v9.16b
    
    .inst 0x4f84e010 // sdot v16.4s, v0.16b, v4.4b[0]

    subs x26, x26, #1
    bne LoopSz_TILE_1

LoopSzEnd_TILE_1:
    add x7, x7, x13
    sub x27, x27, #1
    scvtf v16.4s, v16.4s
    // using float scale dequant for precison
    ld1 {v4.s}[0], [x23]  // scales
    fmul v16.4s, v16.4s, v4.s[0]
Tile1Dequant:
    ld1 {v0.4s}, [x19], #16  // alpha
    ld1 {v1.4s}, [x20], #16  // zero
    ld1 {v2.4s}, [x21], #16  // bias
    ld1 {v3.s}[0], [x22]  // sums
    // alpha * sum + (zero * sumx) + bias
    fmla v2.4s, v0.4s, v16.4s
    fmla v2.4s, v1.4s, v3.s[0]
    st1 {v2.4s}, [x28], x14
    cmp x27, #1
    bge LoopDz_TILE_1
Tile1End:
    subs x6, x6, #1      // batch -= 1
    add x0, x0, #16     // dst += 1 * 4 * sizeof(float32_t)
    add x1, x1, #4      // src += 1 * 4 * sizeof(int8_t)
    add x11, x11, #4   // sum += 1 * sizeof(float32_t)
    add x12, x12, #4   // scale += 1 * sizeof(float32_t)
    bne TILE_1

End:
ldp x27, x28, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 9)
ret

#endif