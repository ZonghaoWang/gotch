package libtch

// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
// #cgo CFLAGS: -I${SRCDIR} -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I/usr/local/include -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -ltorch_cpu
// #cgo LDFLAGS: -Llibtorch/lib -L/lib64
// #cgo CXXFLAGS: -isystem libtorch/lib
// #cgo CXXFLAGS: -isystem libtorch/include
// #cgo CXXFLAGS: -isystem libtorch/include/torch/csrc/api/include
// #cgo CXXFLAGS: -isystem libtorch/include/torch/csrc
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #cgo linux,amd64,!nogpu CFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -Llibtorch/lib -lc10_cuda -ltorch_cuda
import "C"
