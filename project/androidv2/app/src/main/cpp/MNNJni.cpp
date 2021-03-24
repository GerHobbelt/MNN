#include <jni.h>
#include <string>
#include "train/source/demo/MnistUtils.hpp"
#include "train/source/demo/mnistTrain.cpp"
#include "train/source/models/Lenet.hpp"
#include "RandomGenerator.hpp"
#include <MNN/MNNForwardType.h>

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

extern "C" JNIEXPORT jstring JNICALL
Java_com_nkastanos_mnn_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nkastanos_mnn_MainActivity_runMnistDemo(JNIEnv *env, jobject thiz) {
    std::string root = "/data/local/tmp/mnist";
    RandomGenerator::generator(17);
    MNNForwardType forward = forwardType("OpenCL");

    // cpu 2292.893066 ms / 10 iter

    std::shared_ptr<Module> model(new Lenet);
//    std::shared_ptr<Module> model(new MnistV2); // Does not work for some reason. "Error for binary op: input0's type != input1's type" after at the end of epoch
    MnistUtils::train(model, root, forward);
    return 0;
}