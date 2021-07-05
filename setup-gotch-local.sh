#!/bin/bash


GOTCH_PATH=/home/jierui/go/src/github.com/ZonghaoWang/gotch
# Setup gotch for CUDA or non-CUDA device:
#=========================================
DUMMY_CUDA_FILE="$GOTCH_PATH/libtch/dummy_cuda_dependency.cpp"
# Check and delete old file if existing
if [ -f $DUMMY_CUDA_FILE ]
then
  echo "$DUMMY_CUDA_FILE existing. Deleting..."
  sudo rm $DUMMY_CUDA_FILE
fi

GOTCH_LIB_FILE="$GOTCH_PATH/libtch/lib.go"
if [ -f $GOTCH_LIB_FILE ]
then
  echo "$GOTCH_LIB_FILE existing. Deleting..."
  sudo rm $GOTCH_LIB_FILE
fi

# Create files for CUDA or non-CUDA device
if [ $CUDA_VERSION == "cpu" ]; then
  echo "creating $DUMMY_CUDA_FILE for CPU"
  sudo tee -a $DUMMY_CUDA_FILE > /dev/null <<EOT
extern "C" {
void dummy_cuda_dependency();
}

void dummy_cuda_dependency() {}
EOT

  echo "creating $GOTCH_LIB_FILE for CPU"
  sudo tee -a $GOTCH_LIB_FILE > /dev/null <<EOT
package libtch

// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I/usr/local/include
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu -L/lib64
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -I${SRCDIR}/libtorch/lib -I${SRCDIR}/libtorch/include -I${SRCDIR}/libtorch/include/torch/csrc/api/include -I${SRCDIR}/libtorch/include/torch/csrc
// #cgo LDFLAGS: -L${SRCDIR}/libtorch/lib
// #cgo CXXFLAGS: -I${SRCDIR}/libtorch/lib -I${SRCDIR}/libtorch/include -I${SRCDIR}/libtorch/include/torch/csrc/api/include -I${SRCDIR}/libtorch/include/torch/csrc
import "C"
EOT
else
  echo "creating $DUMMY_CUDA_FILE for GPU"
  sudo tee -a $DUMMY_CUDA_FILE > /dev/null <<EOT
extern "C" {
    void dummy_cuda_dependency();
}

namespace at {
    namespace cuda {
        int warp_size();
    }
}
void dummy_cuda_dependency() {
  at::cuda::warp_size();
}
EOT

  echo "creating $GOTCH_LIB_FILE for GPU"
  sudo tee -a $GOTCH_LIB_FILE > /dev/null <<EOT
package libtch

// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -lc10_cuda -ltorch_cuda
// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
import "C"
EOT
fi

sudo ldconfig
