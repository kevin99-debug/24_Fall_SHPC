#include "matmul_single.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64
#define NEPT_X 4 //number of element per thread X
#define NEPT_Y 4 //number of element per thread Y

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  
  //shared memory
  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y * blockDim.y + ty) * NEPT_Y; //row. (.y가 row!!)
  int col = (blockIdx.x * blockDim.x + tx) * NEPT_X; //col. (.x가 col!!)

  float sum[4][4] = { {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f} };

  for(int t=0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) { //iterate to fully load shared mem.

    //load A, B to shared memory
    for(int i=0; i < NEPT_X; i++){
      for(int j=0; j <NEPT_Y; j++){
        int A_row = row + i;
        int A_col = t * BLOCK_SIZE + tx * NEPT_X + j;
        int B_row = t * BLOCK_SIZE + ty * NEPT_Y + i;
        int B_col = col + j;
        //for A
        if(A_row < M && A_col < K){
          A_block[ty * NEPT_Y + i][tx * NEPT_X + j] = A[A_row * K + A_col];
        } 
        else {
          A_block[ty * NEPT_Y + i][tx * NEPT_X + j] = 0.0f; //OOB
        }
        //for B
        if(B_row < K && B_col < N){
          B_block[ty * NEPT_Y + i][tx * NEPT_X + j] = B[B_row * N + B_col];
        } 
        else {
          B_block[ty * NEPT_Y + i][tx * NEPT_X + j] = 0.0f; //OOB
        }
      }
    }
    __syncthreads();

    //calculate
    for(int k=0; k < BLOCK_SIZE; k++) {
      for(int i=0; i < NEPT_X; i++) {
        for(int j=0; j < NEPT_Y; j++) {
          sum[i][j] += A_block[ty * NEPT_Y + i][k] * B_block[k][tx * NEPT_X + j];
        }
      }
    }
    __syncthreads();
  }

  //return results to C
  for(int i=0; i < NEPT_X; i++) {
    for(int j=0; j < NEPT_Y; j++) {
      if (row + i < M && col + j < N) {
        C[(row + i) * N + col + j] = sum[i][j];
      }
    }
  }
}

// Array of device (GPU) pointers
static float *a_d;
static float *b_d;
static float *c_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU
  CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  //kernel
  dim3 blockDim(BLOCK_SIZE / NEPT_X, BLOCK_SIZE / NEPT_Y); 
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  CUDA_CALL(cudaDeviceSynchronize());

  // Download C matrix from GPUs
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_initialize(int M, int N, int K) {
  
  int num_devices;
  // Only root process do something
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Allocate device memory 
  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));
}

void matmul_finalize() {

  // Free GPU memory
  CUDA_CALL(cudaFree(a_d));
  CUDA_CALL(cudaFree(b_d));
  CUDA_CALL(cudaFree(c_d));
}
