#ifndef KEYWORD_SPOT_CPP
#define KEYWORD_SPOT_CPP

#include "KeywordSpotter.hpp"
#include <MNN/expr/NN.hpp>

using namespace MNN::Express;

namespace MNN {
    namespace Train {
        namespace Model {

            KeywordSpotter::KeywordSpotter(){
                NN::ConvOption convOption;
                convOption.fusedActivationFunction = NN::Relu;
                convOption.kernelSize = {3, 3};
                convOption.channel = {1, 8};
                convOption.padMode = SAME;
                conv1.reset(NN::Conv(convOption));

                convOption.channel = {8, 64};
                convOption.padMode = VALID;
                conv2.reset(NN::Conv(convOption));

                fc1.reset(NN::Linear(25344, 4));
                dropout.reset(NN::Dropout(0.1));
                registerModel({conv1, conv2, fc1, dropout});
            }

            std::vector<VARP> KeywordSpotter::onForward(const std::vector<VARP>& inputs){
                VARP x = inputs[0];
                x = conv1->forward(x);
                x = _MaxPool(x, {2, 2}, {2, 2});
                x = conv2->forward(x);
                x = _MaxPool(x, {2, 2}, {2, 2});
                x = _Convert(x, NCHW);
                x = _Reshape(x, {0, -1});
                x = dropout->forward(x);
                x = fc1->forward(x);
                x = _Softmax(x, 1);
                return {x};
            }


        } // namespace Model
    } // namespace Train
} // namespace MNN

#endif // KEYWORD_SPOT_CPP