//
//  MatmulExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/MatmulExecution.hpp"
//#include <string>

namespace MNN {
namespace OpenCL {

MatMulExecution::MatMulExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend,
                                 bool transposeA, bool transposeB) : Execution(backend)
                                 , mTransposeA(transposeA), mTransposeB(transposeB){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mAreadySetArg  = false;
}

ErrorCode MatMulExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    Tensor *input0 = inputs[0];
    Tensor *input1 = inputs[1];
    Tensor *output = outputs[0];

    std::vector<int> input0Shape = tensorShapeFormat(input0);
    std::vector<int> input1Shape = tensorShapeFormat(input1);
    std::vector<int> outputShape = tensorShapeFormat(output);

    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        if(mTransposeA) {
            mKernelName = mTransposeB ? "matmul_transA_transB":"matmul_transA";
        } else {
            mKernelName = mTransposeB ? "matmul_transB":"matmul";
        }

        if(inputs.size() > 2) {
            buildOptions.emplace("-DBIAS");
        }

#define MATMUL_V2
#ifdef MATMUL_V2

#ifndef VECTOR_WIDTH
#define VECTOR_WIDTH 16
#endif
        if (runtime->isSupportedFP16()){
            buildOptions.emplace("-DFLOATX=half" + std::to_string(VECTOR_WIDTH));
        } else {
            buildOptions.emplace("-DFLOATX=float" + std::to_string(VECTOR_WIDTH));
        }
        buildOptions.emplace("-DVECTOR_WIDTH=" + std::to_string(VECTOR_WIDTH));

        mKernel           = runtime->buildKernel("matmul", mKernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

    //处理二维矩阵相乘，N C相当于H W
    //二维矩阵相乘

    // C:=A*B
    // A := (M, K)
    // B := (K, N)
    // C := (M, N)

    const int M = mTransposeA ? input0Shape.at(3) : input0Shape.at(0); // height
    const int N = mTransposeB ? input1Shape.at(0) : input1Shape.at(3); // width
    const int K = mTransposeA ? input0Shape.at(0) : input0Shape.at(3); // outputChannel
    const int kBlocks = UP_DIV(K, VECTOR_WIDTH); // outputChannelBlocks
    const int mBlocks = mTransposeA ? UP_DIV(M, VECTOR_WIDTH) : UP_DIV(M, 1); // heightblocks
    const int nBlocks = UP_DIV(N, VECTOR_WIDTH); // widthblocks

    mGlobalWorkSize = {static_cast<uint32_t>(nBlocks), static_cast<uint32_t>(mBlocks)};

    int idx            = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, openCLImage(input0));
    mKernel.setArg(idx++, openCLImage(input1));
    if(inputs.size() > 2) {
        mKernel.setArg(idx++, openCLImage(inputs[2]));
    }
    mKernel.setArg(idx++, openCLImage(output));
    mKernel.setArg(idx++, static_cast<int>(K));
    mKernel.setArg(idx++, static_cast<int>(kBlocks));
    if(mTransposeA) {
        mKernel.setArg(idx++, static_cast<int>(M));
    }
    mLocalWorkSize = {mMaxWorkGroupSize / 64, 64, 0};

//        MNN_PRINT("Kernel: %s\tM: %i\tK: %i\tN: %i\tkBlocks: %i\tnBlocks: %i\tmBlocks: %i"
//                  "\tmGlobalWorkSize[0]: %i\tmGlobalWorkSize[1]: %i\tmMaxWorkGroupSize: %i\tmLocalWorkSize[0]: %i\n",
//                  mKernelName.c_str(), M, K, N, kBlocks, nBlocks, mBlocks, mGlobalWorkSize[0], mGlobalWorkSize[1],
//                  mMaxWorkGroupSize, mLocalWorkSize[0]);
#else
        mKernel           = runtime->buildKernel("matmul", mKernelName, buildOptions);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }

    //处理二维矩阵相乘，N C相当于H W
    //二维矩阵相乘
    if(mTransposeA) {
        const int height        = input0Shape.at(3);
        const int outputChannel = input0Shape.at(0);
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);
        const int heightblocks        = UP_DIV(height, 4);

        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(heightblocks)};
            int idx            = 0;
            mKernel.setArg(idx++, mGlobalWorkSize[0]);
            mKernel.setArg(idx++, mGlobalWorkSize[1]);
            mKernel.setArg(idx++, openCLImage(input0));
            mKernel.setArg(idx++, openCLImage(input1));
            if(inputs.size() > 2) {
                mKernel.setArg(idx++, openCLImage(inputs[2]));
            }
            mKernel.setArg(idx++, openCLImage(output));
            mKernel.setArg(idx++, static_cast<int>(outputChannel));
            mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
            mKernel.setArg(idx++, static_cast<int>(height));
            mLocalWorkSize = {mMaxWorkGroupSize / 64, 64, 0};
    }
    else {
        const int height        = input0Shape.at(0);
        const int outputChannel = input0Shape.at(3);
        const int width         = mTransposeB ? input1Shape.at(0): input1Shape.at(3);
        const int outputChannelBlocks = UP_DIV(outputChannel, 4);
        const int widthblocks         = UP_DIV(width, 4);

        mGlobalWorkSize = {static_cast<uint32_t>(widthblocks), static_cast<uint32_t>(height)};
        int idx            = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, openCLImage(input0));
        mKernel.setArg(idx++, openCLImage(input1));
        if(inputs.size() > 2) {
            mKernel.setArg(idx++, openCLImage(inputs[2]));
        }
        mKernel.setArg(idx++, openCLImage(output));
        mKernel.setArg(idx++, static_cast<int>(outputChannel));
        mKernel.setArg(idx++, static_cast<int>(outputChannelBlocks));
        mLocalWorkSize = {mMaxWorkGroupSize / 64, 64, 0};
    }
#endif

    return NO_ERROR;
}

ErrorCode MatMulExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

#ifdef LOG_VERBOSE
    MNN_PRINT("Start MatMulExecution onExecute... \n");
#endif

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, &event);
        
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us Matmul %s\n", costTime, mKernelName.c_str());
    #else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, runtime, nullptr);
    #endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("End MatMulExecution onExecute... \n");
#endif
    return NO_ERROR;
}

class MatMulCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_MatMul();
        return new MatMulExecution(inputs, op, backend, param->transposeA(), param->transposeB());
    }
};

OpenCLCreatorRegister<MatMulCreator> __matmul_op(OpType_MatMul, IMAGE);

} // namespace OpenCL
} // namespace MNN
