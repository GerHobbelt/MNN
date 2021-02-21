#!/bin/bash

set -e

CLEAN=0
ABI="armeabi-v7a"
BITS=32

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
        *)
            shift # Remove generic argument from processing
            ;;
    esac
done

mkdir -p project/android/build_${BITS} && cd project/android/build_${BITS}

if [ ${CLEAN} -eq 1 ]; then
    make clean
fi

if [ ${BITS} -eq 32 ]; then
    ./../build_32_opencl.sh \
        -DMNN_BUILD_TRAIN=ON \
        -DMNN_BUILD_TRAIN_MINI=OFF \
        -DMNN_GPU_TRACE=ON \
        -DMNN_OPENMP=OFF \
        -DMNN_USE_THREAD_POOL=ON 
fi
if [ ${BITS} -eq 64 ]; then
    ./../build_64_opencl.sh \
        -DMNN_BUILD_TRAIN=ON \
        -DMNN_BUILD_TRAIN_MINI=OFF \
        -DMNN_GPU_TRACE=ON \
        -DMNN_OPENMP=OFF \
        -DMNN_USE_THREAD_POOL=ON 
fi

# cmake ../../../ \
# -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
# -DCMAKE_BUILD_TYPE=Release \
# -DANDROID_ABI=$ABI \
# -DANDROID_STL=c++_static \
# -DCMAKE_BUILD_TYPE=Release \
# -DANDROID_NATIVE_API_LEVEL=android-29  \
# -DANDROID_TOOLCHAIN=clang \
# -DMNN_USE_SYSTEM_LIB=ON \
# -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
# -DMNN_GPU_TRACE=ON \
# -DMNN_OPENCL=ON \
# -DMNN_USE_OPENCV=OFF \
# -DMNN_BUILD_TRAIN=ON \
# -DMNN_BUILD_TRAIN_MINI=OFF \
# -DMNN_OPENMP=OFF \
# -DMNN_USE_THREAD_POOL=ON \
# -DNATIVE_LIBRARY_OUTPUT=. \
# -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

# make -j4

cd ../../../