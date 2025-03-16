#include "matmul.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64
#define NEPT_X 4 // Number of elements per thread X
#define NEPT_Y 4 // Number of elements per thread Y

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
#define NUM_STREAMS 2 // Number of streams per GPU for overlapping

int num_devices = 0;

__global__ void matmul_kernel(const float *__restrict__ A, 
                              const float *__restrict__ B, 
                              float *__restrict__ C, 
                              int M, int N, int K) {
  // Redefine shared memory as float arrays for better access patterns
  __shared__ float A_block[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE * BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // Each thread processes a 4x4 tile (NEPT_X by NEPT_Y)
  // blockDim.x = BLOCK_SIZE / NEPT_X
  // blockDim.y = BLOCK_SIZE / NEPT_Y

  int row_base = (blockIdx.y * BLOCK_SIZE) + ty * NEPT_Y;
  int col_base = (blockIdx.x * BLOCK_SIZE) + tx * NEPT_X;

  float sum[NEPT_Y][NEPT_X];
  #pragma unroll
  for (int i = 0; i < NEPT_X; i++) {
    #pragma unroll
    for (int j = 0; j < NEPT_Y; j++) {
      sum[i][j] = 0.0f;
    }
  }

  // Load A and B tiles into shared memory using float4 for coalesced access
  #pragma unroll
  for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
    int A_tile_col_start = t * BLOCK_SIZE;
    int B_tile_row_start = t * BLOCK_SIZE;

    #pragma unroll
    for (int i = 0; i < NEPT_X; i++) {
      // Compute global A and B pointers
      int A_row = row_base + i;
      int A_col = A_tile_col_start + tx * NEPT_X; // Starting col for A load
      int B_row = B_tile_row_start + ty * NEPT_Y + i;
      int B_col = col_base; // Starting col for B load

      const float4 *A_ptr = reinterpret_cast<const float4*>(&A[A_row * K + A_col]);
      const float4 *B_ptr = reinterpret_cast<const float4*>(&B[B_row * N + B_col]);

      float4 a_val = A_ptr[0];
      float4 b_val = B_ptr[0];

      // Compute linear index in shared memory for float arrays
      int A_shm_idx = (ty * NEPT_Y + i) * BLOCK_SIZE + (tx * NEPT_X);
      int B_shm_idx = (ty * NEPT_Y + i) * BLOCK_SIZE + (tx * NEPT_X);

      // Store the loaded float4 into shared memory as individual floats
      A_block[A_shm_idx + 0] = a_val.x;
      A_block[A_shm_idx + 1] = a_val.y;
      A_block[A_shm_idx + 2] = a_val.z;
      A_block[A_shm_idx + 3] = a_val.w;

      B_block[B_shm_idx + 0] = b_val.x;
      B_block[B_shm_idx + 1] = b_val.y;
      B_block[B_shm_idx + 2] = b_val.z;
      B_block[B_shm_idx + 3] = b_val.w;
    }

    __syncthreads();

    // Perform the multiplication
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      #pragma unroll
      for (int i = 0; i < NEPT_X; i++) {
        float Aval = A_block[(ty * NEPT_Y + i) * BLOCK_SIZE + k];
        #pragma unroll
        for (int j = 0; j < NEPT_Y; j++) {
          float Bval = B_block[k * BLOCK_SIZE + (tx * NEPT_X + j)];
          sum[i][j] += Aval * Bval;
        }
      }
    }

    __syncthreads();
  }

  // Write results back to C
  #pragma unroll
  for (int i = 0; i < NEPT_X; i++) {
    #pragma unroll
    for (int j = 0; j < NEPT_Y; j++) {
      int Cr = row_base + i;
      int Cc = col_base + j;
      if (Cr < M && Cc < N) { // Ensure within bounds
        C[Cr * N + Cc] = sum[i][j];
      }
    }
  }
}

// Global device pointers to handle multiple streams per GPU
static float *a_d[MAX_NUM_GPU][NUM_STREAMS];
static float *b_d[MAX_NUM_GPU][NUM_STREAMS];
static float *c_d[MAX_NUM_GPU][NUM_STREAMS];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

static cudaStream_t streams[MAX_NUM_GPU][NUM_STREAMS]; // Multiple streams per GPU

// Host pointers (to be used directly without registration)
static const float *A_host = nullptr;
static const float *B_host = nullptr;
static float *C_host = nullptr;

// Function to initialize matrix multiplication
void matmul_initialize(int M, int N, int K) {
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices > MAX_NUM_GPU) {
    printf("Number of GPUs requested (%d) exceeds MAX_NUM_GPU (%d). Using %d GPUs.\n", 
           num_devices, MAX_NUM_GPU, MAX_NUM_GPU);
    num_devices = MAX_NUM_GPU;
  }

  if (num_devices == 0) {
    fprintf(stderr, "No CUDA devices found. Aborting.\n");
    exit(EXIT_FAILURE);
  }

  printf("Using %d device(s)\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    printf("GPU %d: %s\n", i, prop.name);
  }

  // Setup problem size for each GPU
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (M / num_devices) * i;
    Mend[i] = (M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = M; // Ensure the last GPU handles any remaining rows

  // Allocate device memory and create streams for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      // Calculate rows per stream
      int total_rows = Mend[i] - Mbegin[i];
      int base = total_rows / NUM_STREAMS;
      int rem = total_rows % NUM_STREAMS;
      int rows_in_stream = base + (s < rem ? 1 : 0);

      // Allocate memory for A chunk
      CUDA_CALL(cudaMalloc(&a_d[i][s], rows_in_stream * K * sizeof(float)));
      // Allocate memory for B (same across all streams since B is constant)
      CUDA_CALL(cudaMalloc(&b_d[i][s], K * N * sizeof(float)));
      // Allocate memory for C chunk
      CUDA_CALL(cudaMalloc(&c_d[i][s], rows_in_stream * N * sizeof(float)));
      // Create stream
      CUDA_CALL(cudaStreamCreate(&streams[i][s]));
    }
  }
}

// Function to perform matrix multiplication with overlapping H2D, kernel, D2H
void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // Assign host pointers
  A_host = A;
  B_host = B;
  C_host = C;

  // Determine chunk size per stream
  int chunks_per_gpu = NUM_STREAMS;
  int rows_per_chunk[MAX_NUM_GPU][NUM_STREAMS];
  int M_remaining[MAX_NUM_GPU];

  for(int i = 0; i < num_devices; i++) {
    M_remaining[i] = Mend[i] - Mbegin[i];
    int base = M_remaining[i] / chunks_per_gpu;
    int rem = M_remaining[i] % chunks_per_gpu;
    for(int s = 0; s < NUM_STREAMS; s++) {
      rows_per_chunk[i][s] = base + (s < rem ? 1 : 0);
    }
  }

  // Launch operations in a pipelined manner
  for(int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      int offset = 0;
      for(int prev = 0; prev < s; prev++) {
        offset += rows_per_chunk[i][prev];
      }
      int current_M = rows_per_chunk[i][s];
      if(current_M <= 0) continue; // Skip if no rows assigned

      // Asynchronously copy A chunk to device
      CUDA_CALL(cudaMemcpyAsync(a_d[i][s], 
                                A_host + (Mbegin[i] + offset) * K, 
                                current_M * K * sizeof(float), 
                                cudaMemcpyHostToDevice, 
                                streams[i][s]));

      // Asynchronously copy B to device (same for all streams since B is constant)
      CUDA_CALL(cudaMemcpyAsync(b_d[i][s], 
                                B_host, 
                                K * N * sizeof(float), 
                                cudaMemcpyHostToDevice, 
                                streams[i][s]));

      // Define grid and block dimensions
      dim3 blockDim(BLOCK_SIZE / NEPT_X, BLOCK_SIZE / NEPT_Y);
      dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (current_M + BLOCK_SIZE - 1) / BLOCK_SIZE);

      // Launch the kernel asynchronously
      matmul_kernel<<<gridDim, blockDim, 0, streams[i][s]>>>(
          a_d[i][s], b_d[i][s], c_d[i][s], current_M, N, K
      );

      // Check for kernel launch errors
      CUDA_CALL(cudaGetLastError());

      // Asynchronously copy the result back to host
      CUDA_CALL(cudaMemcpyAsync(C_host + (Mbegin[i] + offset) * N, 
                                c_d[i][s], 
                                current_M * N * sizeof(float), 
                                cudaMemcpyDeviceToHost, 
                                streams[i][s]));
    }
  }

  // Wait for all streams on all GPUs to finish
  for(int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      CUDA_CALL(cudaStreamSynchronize(streams[i][s]));
    }
  }
}

// Function to finalize and clean up resources
void matmul_finalize() {
  // Free all GPU memory and destroy streams
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for(int s = 0; s < NUM_STREAMS; s++) {
      CUDA_CALL(cudaFree(a_d[i][s]));
      CUDA_CALL(cudaFree(b_d[i][s]));
      CUDA_CALL(cudaFree(c_d[i][s]));
      CUDA_CALL(cudaStreamDestroy(streams[i][s])); // Destroy each stream
    }
  }
}