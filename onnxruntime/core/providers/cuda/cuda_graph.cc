#include "core/providers/cuda/cuda_graph.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace onnxruntime {

CUDAGraph::CUDAGraph() {
  cudaStreamCreate(&capture_stream_);
}

void CUDAGraph::CaptureBegin() {
  printf("++++++ CUDAGraph CaptureBegin ++++++ \n");
  cudaDeviceSynchronize();
  cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal);
}

void CUDAGraph::CaptureEnd() {
  printf("++++++ CUDAGraph CaptureEnd ++++++ \n");
  cudaStreamEndCapture(capture_stream_, &graph_);
  if (graph_ == NULL) {
    ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11200
  cudaGraphDebugDotPrint(graph_, "/datadrive/fhu/github/onnxruntime/build/Linux/Debug/cuda_graph_debug.log", 1);
#endif
// cudaGraphDebugDotPrint(graph_, "/datadrive/fhu/github/onnxruntime/build/Linux/Debug/cuda_graph_debug.log", 1);

  has_graph_ = true;
  cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0);
  has_graph_exec_ = true;
  cudaGraphDestroy(graph_);
  has_graph_ = false;
}

void CUDAGraph::Replay() {
  cudaDeviceSynchronize();
  // printf("++++++ cudaDeviceSynchronize CUDAGraph Replay Begin ++++++ \n");
  cudaGraphLaunch(graph_exec_, capture_stream_);
  cudaDeviceSynchronize();
  // printf("++++++ cudaDeviceSynchronize CUDAGraph Replay Finish ++++++ \n");
}

void CUDAGraph::Reset() {
  printf("++++++ CUDAGraph Reset ++++++ \n");
  if (has_graph_) {
    cudaGraphDestroy(graph_);
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    cudaGraphExecDestroy(graph_exec_);
    has_graph_exec_ = false;
  }
  is_capturing_ = false;
}

bool CUDAGraph::IsCapturing() const {
  return is_capturing_;
}

bool CUDAGraph::HasGraphExec() const {
  return has_graph_exec_;
}

void CUDAGraph::TurnOnCapture() {
  is_capturing_ = true;
}

void CUDAGraph::TurnOffCapture() {
  is_capturing_ = false;
}

CUDAGraph::~CUDAGraph() {
  Reset();
}

} // namespace onnxruntime
