#!/bin/bash

set -e

OPENMP="ON"
VULKAN="ON"
OPENCL="ON"
OPENGL="OFF"
OPENCV="OFF"
USE_THREAD_POOL="OFF"
RUN_LOOP=10

CLEAN=0
PUSH_MODEL=""
ABI="armeabi-v7a"
BITS=32

ANDROID_DIR=/data/local/tmp
WORK_DIR=`pwd`
BENCHMARK_MODEL_DIR=$WORK_DIR/benchmark/models

for arg in "$@"; do
    case $arg in
        -h|--help)
            echo "$package - Build MNN for 32-bit Android OpenCL GPU"
            echo " "
            echo "$package [options]"
            echo " "
            echo "options:"
            echo "-h, --help                  show brief help"
            echo "-c, --clean                 clean build directory before rebuild"
            exit 0
            ;;
        -c|--clean)
            CLEAN=1
            shift
            ;;
        -64|--64-bit)
            ABI="arm64-v8a"
            BITS=64
            shift
            ;;
        -p)
            shift
            PUSH_MODEL="-p"
            ;;
        *)
            shift # Remove generic argument from processing
            ;;
    esac
done

mkdir -p project/android/build_${BITS} && cd project/android/build_${BITS}

if [ ${CLEAN} -eq 1 ]; then
    make clean
fi

cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI=$ABI \
-DANDROID_STL=c++_static \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NATIVE_API_LEVEL=android-29  \
-DANDROID_TOOLCHAIN=clang \
-DMNN_USE_LOGCAT=true \
-DMNN_USE_SYSTEM_LIB=ON \
-DMNN_VULKAN=$VULKAN \
-DMNN_OPENGL=$OPENGL \
-DMNN_OPENCL=$OPENCL \
-DMNN_USE_OPENCV=$OPENCV \
-DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
-DMNN_BUILD_TRAIN=ON \
-DMNN_BUILD_TRAIN_MINI=OFF \
-DMNN_GPU_TRACE=ON \
-DMNN_OPENMP=$OPENMP \
-DMNN_USE_THREAD_POOL=$USE_THREAD_POOL \
-DMNN_BUILD_BENCHMARK=ON \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.

make -j4 runTrainDemo.out train.out

find . -name "*.so" | while read solib; do
    adb push $solib  $ANDROID_DIR
done

adb push runTrainDemo.out $ANDROID_DIR
adb shell chmod 0777 $ANDROID_DIR/runTrainDemo.out

if [ "" != "$PUSH_MODEL" ]; then
    adb shell "rm -rf $ANDROID_DIR/benchmark_models"
    adb push $BENCHMARK_MODEL_DIR $ANDROID_DIR/benchmark_models
fi

adb shell "cat /proc/cpuinfo > $ANDROID_DIR/train_benchmark.txt"
adb shell "echo >> $ANDROID_DIR/train_benchmark.txt"
adb shell "echo Build Flags: ABI=$ABI  OpenMP=$OPENMP Vulkan=$VULKAN OpenCL=$OPENCL OpenGL=$OPENGL >> $ANDROID_DIR/train_benchmark.txt"

# adb shell "mkdir $ANDROID_DIR/mnist"
# adb push ../MNN_Additional_Files/mnist/* $ANDROID_DIR/mnist/
# adb shell "gunzip $ANDROID_DIR/mnist/*"

# CPU
# adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out MnistTrainCustom $ANDROID_DIR/mnist CPU >> $ANDROID_DIR/train_benchmark.txt"
# Vulkan
# adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out MnistTrainCustom $ANDROID_DIR/mnist Vulkan >> $ANDROID_DIR/train_benchmark.txt"
# OpenCL
adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out MnistTrainCustom $ANDROID_DIR/mnist OpenCL >> $ANDROID_DIR/train_benchmark.txt"

adb pull $ANDROID_DIR/train_benchmark.txt .

cd ../../../