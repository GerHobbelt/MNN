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

cd ../../../