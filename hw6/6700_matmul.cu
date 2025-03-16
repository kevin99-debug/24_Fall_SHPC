#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

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

//kernel from hw5
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {

  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockDim.y * blockIdx.y + ty) * NEPT_Y; //row  
  int col = (blockDim.x * blockIdx.x + tx) * NEPT_X; //col

  float sum[NEPT_Y][NEPT_X] = {{0.0f}};

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

//host buffers for A,B,C
static float *A_local = NULL; //local
static float *C_local = NULL; //local
static float *B_buffer = NULL; //temp

//compute the local M for each process
static void compute_local_M(int M, int mpi_world_size, int mpi_rank, int *local_M, int *start_M) {
    int base = M / mpi_world_size;
    int rem = M % mpi_world_size;
    if (mpi_rank < rem) {
        *local_M = base + 1;
        *start_M = mpi_rank * (*local_M);
    } else {
        *local_M = base;
        *start_M = mpi_rank * (*local_M) + rem;
    }
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  int local_M = 0;
  int start_M = 0;
  compute_local_M(M, mpi_world_size, mpi_rank, &local_M, &start_M);

  // On root, prepare counts and displacements for Scatterv
  MPI_Request scatter_req;
  if (mpi_rank == 0) {
    int *send_counts = (int*) malloc(mpi_world_size * sizeof(int));
    int *displs = (int*) malloc(mpi_world_size * sizeof(int));
    if (send_counts == NULL || displs == NULL) {
      fprintf(stderr, "Memory allocation failed on root for Scatterv.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < mpi_world_size; i++) {
      int proc_M = 0;
      int proc_start = 0;
      compute_local_M(M, mpi_world_size, i, &proc_M, &proc_start);
      send_counts[i] = proc_M * K;
      displs[i] = proc_start * K;
    }
    //MPI_Scatterv(A,send_counts,displs,MPI_FLOAT,A_local,local_M * K,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Iscatterv(A, send_counts, displs, MPI_FLOAT, A_local, local_M * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &scatter_req);
    free(send_counts);
    free(displs);
  } 
  else {
    //MPI_Scatterv(NULL,NULL,NULL,MPI_FLOAT,A_local,local_M * K,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Iscatterv(NULL, NULL, NULL, MPI_FLOAT, A_local, local_M * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &scatter_req);
  }

  // Properly initialize B_buffer on the root before broadcasting
  MPI_Request bcast_req;
  if (mpi_rank == 0) {
    memcpy(B_buffer, B, K * N * sizeof(float));
  }

  // Broadcast matrix B to all processes
  //MPI_Bcast(B_buffer, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Ibcast(B_buffer, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &bcast_req);

  MPI_Wait(&scatter_req, MPI_STATUS_IGNORE);
  MPI_Wait(&bcast_req, MPI_STATUS_IGNORE);

  // Upload A and B to GPUs asynchronously
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
    if (num_devices > MAX_NUM_GPU) {
        num_devices = MAX_NUM_GPU;
    }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(a_d[i],A_local + Mbegin[i] * K,(Mend[i] - Mbegin[i]) * K * sizeof(float),cudaMemcpyHostToDevice,stream[i]));
    CUDA_CALL(cudaMemcpyAsync(b_d[i],B_buffer,K * N * sizeof(float),cudaMemcpyHostToDevice,stream[i]));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    dim3 blockDim(BLOCK_SIZE / NEPT_X, BLOCK_SIZE / NEPT_Y);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, ((Mend[i] - Mbegin[i]) + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_kernel<<<gridDim, blockDim, 0, stream[i]>>>(a_d[i],b_d[i],c_d[i],Mend[i] - Mbegin[i],N,K);
  }

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(C_local + Mbegin[i] * N,c_d[i],(Mend[i] - Mbegin[i]) * N * sizeof(float),cudaMemcpyDeviceToHost,stream[i]));
  }

  // Synchronize all streams
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(stream[i]));
  }

  // On root, prepare counts and displacements for Gatherv
  MPI_Request gatherv_req;
    if (mpi_rank == 0) {
        int *recv_counts = (int*) malloc(mpi_world_size * sizeof(int));
        int *displs = (int*) malloc(mpi_world_size * sizeof(int));
        if (recv_counts == NULL || displs == NULL) {
            fprintf(stderr, "Memory allocation failed on root for Gatherv.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < mpi_world_size; i++) {
            int proc_M = 0;
            int proc_start = 0;
            compute_local_M(M, mpi_world_size, i, &proc_M, &proc_start);
            recv_counts[i] = proc_M * N;
            displs[i] = proc_start * N;
        }
        MPI_Igatherv(C_local, local_M * N, MPI_FLOAT, C, recv_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD, &gatherv_req);
        free(recv_counts);
        free(displs);
    } else {
        MPI_Igatherv(C_local, local_M * N, MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD, &gatherv_req);
    }

    // Wait for Gatherv to complete
    MPI_Wait(&gatherv_req, MPI_STATUS_IGNORE);


}

void matmul_initialize(int M, int N, int K) {
  // TODO: FILL_IN_HERE
  //initialize MPI (rank, size)
  //calculate portion of matrix each MPI will handle
  //set up multi-gpu (similar to hw5)

  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  // Compute local M and start index
  int local_M = 0;
  int start_M = 0;
  compute_local_M(M, mpi_world_size, mpi_rank, &local_M, &start_M);
  if (mpi_rank == 0) {
    printf("MPI initialized with %d processes.\n", mpi_world_size);
    printf("Total M: %d, N: %d, K: %d\n", M, N, K);
  }
  printf("Rank %d: local M = %d, start_M = %d\n", mpi_rank, local_M, start_M);

  // Allocate host buffers for local A and C
  A_local = (float*) malloc(local_M * K * sizeof(float));
  C_local = (float*) malloc(local_M * N * sizeof(float));
  if (A_local == NULL || C_local == NULL) {
    fprintf(stderr, "Host memory allocation failed on rank %d.\n", mpi_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Allocate temporary buffer for B
  B_buffer = (float*) malloc(K * N * sizeof(float));
  if (B_buffer == NULL) {
    fprintf(stderr, "Host buffer allocation for B failed on rank %d.\n", mpi_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Register host buffers as pinned memory
  CUDA_CALL(cudaHostRegister(A_local, local_M * K * sizeof(float), cudaHostRegisterDefault));
  CUDA_CALL(cudaHostRegister(C_local, local_M * N * sizeof(float), cudaHostRegisterDefault));
  CUDA_CALL(cudaHostRegister(B_buffer, K * N * sizeof(float), cudaHostRegisterDefault));


  // Initialize multi-GPU setup
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices > MAX_NUM_GPU) {
    num_devices = MAX_NUM_GPU;
  }

  printf("Rank %d using %d GPU(s).\n", mpi_rank, num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));
    printf("Rank %d GPU %d: %s\n", mpi_rank, i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found on rank %d. Aborting.\n", mpi_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Setup problem size for each GPU
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (local_M / num_devices) * i;
    Mend[i] = (local_M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = local_M; // Last GPU takes the remainder

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
  // TODO: FILL_IN_HERE
  // Free device memory and destroy streams
  int num_devices = 0;
    CUDA_CALL(cudaGetDeviceCount(&num_devices));
    if (num_devices > MAX_NUM_GPU) {
        num_devices = MAX_NUM_GPU;
    }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
    CUDA_CALL(cudaStreamDestroy(stream[i]));
  }

  // Unregister and free host buffers
    if (A_local) {
        CUDA_CALL(cudaHostUnregister(A_local));
    }
    if (C_local) {
        CUDA_CALL(cudaHostUnregister(C_local));
    }
    if (B_buffer) {
        CUDA_CALL(cudaHostUnregister(B_buffer));
    }

}
