#pragma once

#include "tensor.h"

/* Operations (layers) */
void Batched_Embedding_CUDA(Tensor *in, Tensor *w, Tensor *out, size_t batch_size, cudaStream_t stream, int rank);
void Batched_Permute_CUDA(Tensor *in, Tensor *out, size_t batch_size, cudaStream_t stream, int rank);
void Batched_Conv1D_ReLU_GetMax_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *max_out, size_t batch_size, cudaStream_t stream, int rank);
void Batched_Concat_CUDA(const float* d_all_conv_results, float* d_concatenated_conv_results, size_t NUM_WORKERS, size_t batch_size, size_t N_FILTERS, cudaStream_t stream);
void Batched_Linear_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, cudaStream_t stream);
void Batched_Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, cudaStream_t stream);

