#include "matmul.h"
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
#define NUM_STREAMS 2 //number of streams per GPU for overlapping

int num_devices = 0;

__global__ void matmul_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
  //float4 to store shared memory tiles (4 floats)
  __shared__ float4 A_block[(BLOCK_SIZE * BLOCK_SIZE) / 4]; //BLOCK_SIZE * BLOCK_SIZE / 4
  __shared__ float4 B_block[(BLOCK_SIZE * BLOCK_SIZE) / 4];

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  //each thread -> 4x4 block(NEPT_X * NEPT_Y) -> 4 float4
  //code before
  //int row = (blockDim.y * blockIdx.y + ty) * NEPT_Y; // Row
  //int col = (blockDim.x * blockIdx.x + tx) * NEPT_X; // Column
  int row_base = (blockIdx.y * BLOCK_SIZE) + ty * NEPT_Y; //blockDim.y = BLOCK_SIZE / NEPT_Y
  int col_base = (blockIdx.x * BLOCK_SIZE) + tx * NEPT_X; //blockDim.x = BLOCK_SIZE / NEPT_X

  float sum[NEPT_Y][NEPT_X] = {{0.0f}};
  
  //each thread -> 4x4 block(NEPT_X * NEPT_Y) -> 4 float4 -> load using vectorization
  #pragma unroll
  for (int t=0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) { //iterate to fully load shared mem.
    int A_block_col_start = t * BLOCK_SIZE; //row[row_base ~ row_base + 3] : col [A_block_col_start]
    int B_block_row_start = t * BLOCK_SIZE; //col[col_base ~ col_base + 3] : row [B_block_row_start]

    //load A, B to shared memory(vectorized)
    #pragma unroll
    for (int i = 0; i < NEPT_X; i++) {
      int A_row = row_base + i;
      int A_col = A_block_col_start + tx * NEPT_X; //starting col for A 
      int B_row = B_block_row_start + ty * NEPT_Y + i;
      int B_col = col_base; //starting col for B 

      const float4 *A_ptr = reinterpret_cast<const float4*>(&A[A_row * K + A_col]);
      const float4 *B_ptr = reinterpret_cast<const float4*>(&B[B_row * N + B_col]);

      float4 a_val = A_ptr[0]; //load one float4
      float4 b_val = B_ptr[0]; //load one float4

      //shared index memory (modified from hw5 because float4 is being used now)
      int A_SM_row = (ty * NEPT_Y + i);
      int A_SM_col = (tx * NEPT_X);
      int B_SM_row = (ty * NEPT_Y + i);
      int B_SM_col = (tx * NEPT_X);

      int A_SM_idx = (A_SM_row * BLOCK_SIZE + A_SM_col) / 4; 
      int B_SM_idx = (B_SM_row * BLOCK_SIZE + B_SM_col) / 4;

      A_block[A_SM_idx] = a_val;
      B_block[B_SM_idx] = b_val;
    }
    __syncthreads();

    //calculate
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      //4 floats from A_block & B_block at each iteration from shared mem
      //each row -> 64 floats -> 16 float4.
      #pragma unroll
      for (int i = 0; i < NEPT_X; i++) {
        float Avals_4[NEPT_X]; //4 floats from A(row) 
        int A_idx = (ty * NEPT_Y + i) * BLOCK_SIZE + k; //A_block : [(ty*NEPT_Y + i),k]
        float *A_float_ptr = reinterpret_cast<float*>(A_block);
        Avals_4[0] = A_float_ptr[A_idx];
        
        #pragma unroll
        for (int j = 0; j < NEPT_Y; j++) {
          int B_idx = k * BLOCK_SIZE + (tx * NEPT_X + j); //B_block : [k,(tx*NEPT_X + j)]
          float *B_float_ptr = reinterpret_cast<float*>(B_block);
          float Bval = B_float_ptr[B_idx];
          sum[i][j] += Avals_4[0] * Bval;
        }
      }
    }
    __syncthreads();
  }

  //return results to C
  #pragma unroll
  for (int i = 0; i < NEPT_X; i++) {
    #pragma unroll
    for (int j = 0; j < NEPT_Y; j++) {
      C[(row_base + i) * N + (col_base + j)] = sum[i][j];
    }
  }
}

// Array of device (GPU) pointers (modified to use multiple streams)
static float *a_d[MAX_NUM_GPU][NUM_STREAMS];
static float *b_d[MAX_NUM_GPU][NUM_STREAMS];
static float *c_d[MAX_NUM_GPU][NUM_STREAMS];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

static cudaStream_t streams[MAX_NUM_GPU][NUM_STREAMS]; //multiple streams per GPU

//host pointers
static const float *A_host = nullptr;
static const float *B_host = nullptr;
static float *C_host = nullptr;

void matmul_initialize(int M, int N, int K) {
  
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d device(s)\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

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

  //allocate device memory & create streams for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) { //overlap data transfer and kernel execution
      int total_rows = Mend[i] - Mbegin[i];
      int base = total_rows / NUM_STREAMS;
      int rem = total_rows % NUM_STREAMS;
      int rows_in_stream = base + (s < rem ? 1 : 0); //rows per stream

      CUDA_CALL(cudaMalloc(&a_d[i][s], rows_in_stream * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&b_d[i][s], K * N * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i][s], rows_in_stream * N * sizeof(float)));
      CUDA_CALL(cudaStreamCreate(&streams[i][s]));
    }
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K) { //overlapping H2D, kernel, D2H

  A_host = A;
  B_host = B;
  C_host = C;

  int chunks_per_gpu = NUM_STREAMS;
  int rows_per_chunk[MAX_NUM_GPU][NUM_STREAMS];
  int M_remaining[MAX_NUM_GPU];

  //compute chunk size per stream(local M)
  for(int i = 0; i < num_devices; i++) {
    M_remaining[i] = Mend[i] - Mbegin[i];
    int base = M_remaining[i] / chunks_per_gpu;
    int rem = M_remaining[i] % chunks_per_gpu;
    for(int s = 0; s < NUM_STREAMS; s++) {
      rows_per_chunk[i][s] = base + (s < rem ? 1 : 0);
    }
  }

  //Launch operations(data transfer, kernel) in a overlapping manner
  for(int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      int offset = 0;
      for(int prev = 0; prev < s; prev++) {
        offset += rows_per_chunk[i][prev];
      }
      int current_M = rows_per_chunk[i][s];
      if(current_M <= 0){ //no row assigned
        continue;
      } 

      //upload A and B matrix to every GPU asynchronously!!!
      CUDA_CALL(cudaMemcpyAsync(a_d[i][s], A_host + (Mbegin[i] + offset) * K, current_M * K * sizeof(float), cudaMemcpyHostToDevice, streams[i][s]));
      CUDA_CALL(cudaMemcpyAsync(b_d[i][s], B_host, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i][s]));

      dim3 blockDim(BLOCK_SIZE / NEPT_X, BLOCK_SIZE / NEPT_Y);
      dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (current_M + BLOCK_SIZE - 1) / BLOCK_SIZE);
      // Launch the kernel 
      matmul_kernel<<<gridDim, blockDim, 0, streams[i][s]>>>(a_d[i][s], b_d[i][s], c_d[i][s], current_M, N, K);

      //asynchronously copy the result back to host from GPU!!!
      CUDA_CALL(cudaMemcpyAsync(C_host + (Mbegin[i] + offset) * N, c_d[i][s], current_M * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i][s]));
    }
  }

  //synchronize(wait for all streams)
  for(int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      CUDA_CALL(cudaStreamSynchronize(streams[i][s]));
    }
  }
}

void matmul_finalize() {
  // Free all GPU memory(+ destroy streams)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      CUDA_CALL(cudaFree(a_d[i][s]));
      CUDA_CALL(cudaFree(b_d[i][s]));
      CUDA_CALL(cudaFree(c_d[i][s]));
      CUDA_CALL(cudaStreamDestroy(streams[i][s])); //destroy each stream
    }
  }
}