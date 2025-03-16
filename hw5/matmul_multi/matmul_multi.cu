#include "matmul_multi.h"
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

#define MAX_NUM_GPU 4
int num_devices = 0;

//original
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {

  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockDim.y * blockIdx.y + ty) * NEPT_Y; //row  
  int col = (blockDim.x * blockIdx.x + tx) * NEPT_X; //col

  float sum[NEPT_Y][NEPT_X] = {{0.0f}};
  //float sum[4][4] = { {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f} };

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
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

static cudaStream_t stream[MAX_NUM_GPU]; //create a stream to use multi-gpu

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  //register host as pinned memory
  CUDA_CALL(cudaHostRegister((void*)A, M * K * sizeof(float), cudaHostRegisterDefault));
  CUDA_CALL(cudaHostRegister((void*)B, K * N * sizeof(float), cudaHostRegisterDefault));
  CUDA_CALL(cudaHostRegister((void*)C, M * N * sizeof(float), cudaHostRegisterDefault));

  //upload A and B matrix to every GPU asynchronously!!!
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(a_d[i], A + Mbegin[i] * K,
                         (Mend[i] - Mbegin[i]) * K * sizeof(float),
                         cudaMemcpyHostToDevice, stream[i]));
    CUDA_CALL(
        cudaMemcpyAsync(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    dim3 blockDim(BLOCK_SIZE / NEPT_X, BLOCK_SIZE / NEPT_Y);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, ((Mend[i] - Mbegin[i]) + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<gridDim, blockDim, 0, stream[i]>>>(a_d[i], b_d[i], c_d[i], Mend[i]-Mbegin[i], N, K); 
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(stream[i]));
  }

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(C + Mbegin[i] * N, c_d[i],
                         (Mend[i] - Mbegin[i]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost, stream[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(stream[i]));
  }

  //unregister host memory
  CUDA_CALL(cudaHostUnregister((void*)A));
  CUDA_CALL(cudaHostUnregister((void*)B));
  CUDA_CALL(cudaHostUnregister((void*)C));
}

void matmul_initialize(int M, int N, int K) {

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d devices\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // Try printing more detailed information here
    printf("GPU %d: %s\n", i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Setup problem size for each GPU
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (M / num_devices) * i;
    Mend[i] = (M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = M;

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    CUDA_CALL(cudaStreamCreate(&stream[i]));
  }
}

void matmul_finalize() {

  // Free all GPU memory
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
    CUDA_CALL(cudaStreamDestroy(stream[i])); //destroy stream
  }
}
