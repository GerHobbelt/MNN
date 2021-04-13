//
//  MnistUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MnistUtils_hpp
#define MnistUtils_hpp
#include <MNN/expr/Module.hpp>
#include <MNN/MNNForwardType.h>
class MnistUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, std::string root, MNNForwardType forward = MNN_FORWARD_CUDA, uint epochs = 50);
};
#endif
