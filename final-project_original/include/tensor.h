#pragma once

#include <vector>
#include <cstdio>
#include <cassert>

using std::vector;

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[4];
  float *buf = nullptr;
  int *buf_int = nullptr; //new
  size_t total_elements; //added for optimization [shpc]
  bool is_gpu;
  int device_id;

  Tensor(const vector<size_t> &shape_, int device_id_);
  Tensor(const vector<size_t> &shape_, float *host_buf_, int device_id_);
  Tensor(const vector<size_t> &shape_, int *host_buf_, int device_id_);
  ~Tensor();

  size_t num_elem();
};

typedef Tensor Parameter;
typedef Tensor Activation;