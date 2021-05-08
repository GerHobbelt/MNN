//
//  SGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SGD.hpp"
#include "OpGrad.hpp"
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;

namespace MNN {
namespace Train {
SGD::SGD(std::shared_ptr<Module> module) : ParameterOptimizer(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void SGD::setLearningRate(float rate) {
    mLearningRate = rate;
}

void SGD::setMomentum(float momentum) {
    mMomentum = momentum;
}

void SGD::setWeightDecay(float decay) {
    mWeightDecay = decay;
}

void SGD::setRegularizationMethod(RegularizationMethod method) {
    mRegularizationMethod = method;
}

float SGD::currentLearningRate() {
    return mLearningRate;
}

float SGD::getMomentum() {
    return mMomentum;
}

float SGD::getWeightDecay() {
    return mWeightDecay;
}

SGD::RegularizationMethod SGD::getRegularizationMethod() {
    return mRegularizationMethod;
}

Express::VARP SGD::regularizeParameters(Express::VARP param, Express::VARP grad) {
    VARP addWeightDecayGrad;
    if (mRegularizationMethod == L1) {
        auto temp          = _Sign(param);
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * temp + grad;
    } else if (mRegularizationMethod == L2) {
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * param + grad;
    } else if (mRegularizationMethod == L1L2) {
        auto temp          = _Sign(param);
        auto L1 = _Const(mWeightDecay, {}, NCHW) * temp;
        auto L2 = _Const(mWeightDecay, {}, NCHW) * param;
        addWeightDecayGrad = L1 + L2 + grad;
    }

    return addWeightDecayGrad;
}

Express::VARP SGD::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    auto lr         = _Const(mLearningRate, {}, NCHW);
    mHistory[param] = lr * grad + _Const(mMomentum, {}, NCHW) * mHistory[param];
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> SGD::onGetNextParameter(Express::VARP loss) {
    MNN::Timer _IntervalTimer;
    MNN::Timer _TotalTimer;
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);

    auto gradTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("OpGrad::grad() finished");

    auto parameters = module()->parameters();

    auto parametersTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("module()->parameters() finished");

    std::vector<VARP> prepareCompute;
    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            prepareCompute.emplace_back(iter);
        }
    }
    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }

    auto createPrepareComputeTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("createPrepareCompute");

    Variable::prepareCompute(prepareCompute);
    auto prepareComputeTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("prepareCompute() finished");


    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i=0; i<prepareCompute.size(); ++i) {
        auto info = prepareCompute[i]->getInfo();
        auto ptr = prepareCompute[i]->readMap<void>();
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return {};
        }
        auto newVar = _Const(ptr, info->dim, info->order, info->type);
        replaceOp[i]= newVar;
    }

    auto createReplaceTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("createReplace");

    for (int i=0; i<prepareCompute.size(); ++i) {
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }

    auto replaceTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("Variable::replace()");

    for (auto& iter : grad) {
        // apply regularization
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second       = newParameter;
    }

    auto applyTime = (float)_IntervalTimer.durationInUs();
    _IntervalTimer.reset();
//    MNN_PRINT("Apply");
    auto duration = (float)_TotalTimer.durationInUs();

//    MNN_PRINT("onGetNextParameter complete in %f ms.\n"
//              "\tOpGrad::grad() complete in %f ms (%f %%)\n"
//              "\tmodule()->parameters() complete in %f ms (%f %%)\n"
//              "\tcreatePrepareCompute complete in %f ms (%f %%)\n"
//              "\tprepareCompute() complete in %f ms (%f %%)\n"
//              "\tcreateReplace complete in %f ms (%f %%)\n"
//              "\tVariable::replace() complete in %f ms (%f %%)\n"
//              "\tApply complete in %f ms (%f %%)\n",
//              duration/1000.0f,
//            gradTime / 1000.0f, gradTime / duration * 100,
//            parametersTime / 1000.0f, parametersTime / duration * 100,
//            createPrepareComputeTime / 1000.0f, createPrepareComputeTime / duration * 100,
//            prepareComputeTime / 1000.0f, prepareComputeTime / duration * 100,
//            createReplaceTime / 1000.0f, createReplaceTime / duration * 100,
//            replaceTime / 1000.0f, replaceTime / duration * 100,
//            applyTime / 1000.0f, applyTime / duration * 100
//            );

    return grad;
}

} // namespace Train
} // namespace MNN
