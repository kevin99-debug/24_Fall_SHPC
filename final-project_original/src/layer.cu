#include "layer.h"

#include <float.h>
#include <omp.h>
#include <thread>

/* Embedding*/
__global__ void Batched_Embedding_Kernel(int *in, float *w, float *out, size_t batch_size, size_t s, size_t H) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * s * H;
    if (idx < total) {
        size_t b = idx / (s * H);
        size_t seq = (idx / H) % s;
        size_t h = idx % H;
        int word_idx = in[b * s + seq];

        out[b * s * H + seq * H + h] = w[word_idx * H + h];
    }
}

void Batched_Embedding_CUDA(Tensor *in, Tensor *w, Tensor *out, size_t batch_size, cudaStream_t stream, int rank){
    //in : [batch_size, s] -> on device
    //w : [vocab_size, H] -> on device
    //out : [batch_size, s, H] -> alloc on device 
    size_t s = out->shape[1];
    size_t H = out->shape[2];

    int *d_in = in->buf_int;
    float *d_w = w->buf;
    float *d_out = out->buf;

    dim3 blockDim(256, 1, 1); // Threads per block
    dim3 gridDim((batch_size * s * H + blockDim.x - 1) / blockDim.x, 1, 1);

    Batched_Embedding_Kernel<<<gridDim, blockDim, 0, stream>>>(d_in, d_w, d_out, batch_size, s, H);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, " CUDA Error after Batched_Embedding_Kernel launch: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Permute*/
__global__ void Batched_Permute_kernel(float *in, float *out, size_t batch_size, size_t s, size_t H) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * s * H;

    if (idx < total) {
        size_t b = idx / (s * H);
        size_t rem = idx % (s * H);
        size_t x = rem / H;
        size_t y = rem % H;

        out[b * s * H + y * s + x] = in[b * s * H + x * H + y];
    }
}

void Batched_Permute_CUDA(Tensor *in, Tensor *out, size_t batch_size, cudaStream_t stream, int rank) {
    size_t s = in->shape[1];  //seq len
    size_t H = in->shape[2];  //hidden size

    float *d_in = in->buf;
    float *d_out = out->buf;

    size_t blockSize = 256;
    size_t total_threads = batch_size * s * H;
    size_t gridSize = (total_threads + blockSize - 1) / blockSize;

    // Launch kernel
    Batched_Permute_kernel<<<gridSize, blockSize, 0, stream>>>(d_in, d_out, batch_size, s, H);

    CHECK_CUDA(cudaGetLastError());
}

/* Conv1D 
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
#define TILE 64
__global__ void Batched_Conv1D_ReLU_GetMax_kernel(float *in, float *w, float *b, int s, int C, int OC, int K, float *max_out, size_t batch_size) {
    //in : [batch_size, C, s]
    //w : [OC, C, K]
    //b : [OC]
    //max_out : [batch_size, OC]
    size_t oc = blockIdx.x; //output channel idx
    size_t batch = blockIdx.y; // batch idx
    int tid = threadIdx.x; //thread idx (in block)
    size_t total_positions = s - K + 1; //size of output

    if (oc >= OC || batch >= batch_size){ 
        return; 
    }

    __shared__ float shared_max[TILE / 32]; //32 is WARP size. shared memory for warp-level max
    float thread_max = -FLT_MAX; //thread max

    for(size_t pos = tid; pos < total_positions; pos += blockDim.x){
        float sum = b[oc]; //bias

        for(int c = 0; c < C; c++) {
            for(int k = 0; k < K; k++) {
                int in_index = batch * C * s + c * s + pos + k;
                int w_index = oc * C * K + c * K + k;
                sum += in[in_index] * w[w_index];
            }
        }
        sum = fmaxf(sum, 0.0f); //relu
        thread_max = fmaxf(thread_max, sum); //update thread-max
    }

    //warp-level reduction
    for(int offset = 32 / 2; offset > 0; offset /= 2){
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }  
    //warp max -> shared memory
    if (threadIdx.x % 32 == 0) {
        shared_max[threadIdx.x / 32] = thread_max;
    }
    __syncthreads();

    //final reduction (in block)
    if (threadIdx.x < (TILE / 32)) {
        thread_max = shared_max[threadIdx.x];
        for(int offset = 32 / 2; offset > 0; offset /= 2){
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
        if (threadIdx.x == 0) {
            max_out[batch * OC + oc] = thread_max;
        }
    }
}

#define NUM_GPU 4
void Batched_Conv1D_ReLU_GetMax_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *max_out, size_t batch_size, cudaStream_t stream, int rank) {
    
    size_t s = in->shape[2]; //seq len
    size_t C = in->shape[1]; //input channels
    size_t OC = w->shape[0]; //output channels
    size_t K = w->shape[2]; //kernel size

    size_t distribute_B = (batch_size + NUM_GPU - 1) / NUM_GPU;
    auto use_gpu = [&](int gpu_idx) {
        size_t start_idx = gpu_idx * distribute_B;
        if (start_idx >= batch_size){
            return; //end
        } 
        size_t currB = std::min(distribute_B, batch_size - start_idx); //get current distributed batch

        float *d_in, *d_w, *d_b, *d_max_out;
        CHECK_CUDA(cudaSetDevice(gpu_idx)); 

        // Create a CUDA stream for this GPU
        cudaStream_t gpu_stream;
        CHECK_CUDA(cudaStreamCreate(&gpu_stream));
        //allocate memory
        CHECK_CUDA(cudaMalloc((void**)&d_in, currB * C * s * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_w, OC * C * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_b, OC * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_max_out, currB * OC * sizeof(float)));
        //copy to device
        CHECK_CUDA(cudaMemcpyAsync(d_in, in->buf + start_idx * C * s, currB * C * s * sizeof(float), cudaMemcpyHostToDevice, gpu_stream));
        CHECK_CUDA(cudaMemcpyAsync(d_w, w->buf, OC * C * K * sizeof(float), cudaMemcpyHostToDevice, gpu_stream));
        CHECK_CUDA(cudaMemcpyAsync(d_b, b->buf, OC * sizeof(float), cudaMemcpyHostToDevice, gpu_stream));

        dim3 blockDim(TILE);
        dim3 gridDim(static_cast<int>(OC), currB);
        Batched_Conv1D_ReLU_GetMax_kernel<<<gridDim, blockDim, 0, gpu_stream>>>(d_in, d_w, d_b, static_cast<int>(s), static_cast<int>(C), static_cast<int>(OC), static_cast<int>(K), d_max_out, currB);

        CHECK_CUDA(cudaMemcpyAsync(max_out->buf + start_idx * OC, d_max_out, currB * OC * sizeof(float), cudaMemcpyDeviceToHost, gpu_stream));
        CHECK_CUDA(cudaStreamSynchronize(gpu_stream));

        //free device memory
        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_max_out));

        CHECK_CUDA(cudaStreamDestroy(gpu_stream)); //destroy stream
    };

    std::vector<std::thread> gpu_threads;
    for(int gpu_idx = 0; gpu_idx < NUM_GPU; gpu_idx++) {
        gpu_threads.emplace_back(use_gpu, gpu_idx);
    }
    for (auto &t : gpu_threads) t.join();
}

/* Concat */
__global__ void Batched_Concat_kernel(const float* all_conv_results, float* concatenated_conv_results, size_t NUM_WORKERS, size_t batch_size, size_t N_FILTERS) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = NUM_WORKERS * batch_size * N_FILTERS;

    if(idx < total_elements){
        size_t worker = idx / (batch_size * N_FILTERS);
        size_t rem = idx % (batch_size * N_FILTERS);
        size_t sample = rem / N_FILTERS;
        size_t filter = rem % N_FILTERS;

        size_t src_idx = worker * batch_size * N_FILTERS + sample * N_FILTERS + filter; //src
        size_t dest_idx = sample * (NUM_WORKERS * N_FILTERS) + worker * N_FILTERS + filter; //dst
        concatenated_conv_results[dest_idx] = all_conv_results[src_idx]; //data copy
    }
}

void Batched_Concat_CUDA(const float* d_all_conv_results, float* d_concatenated_conv_results, size_t NUM_WORKERS, size_t batch_size, size_t N_FILTERS, cudaStream_t stream) {
    size_t total_elements = NUM_WORKERS * batch_size * N_FILTERS;

    size_t threads_per_block = 128;
    dim3 blockDim(128);
    dim3 gridDim((total_elements + threads_per_block - 1) / threads_per_block);

    Batched_Concat_kernel<<<gridDim, blockDim, 0, stream>>>(d_all_conv_results, d_concatenated_conv_results, NUM_WORKERS, batch_size, N_FILTERS);

    CHECK_CUDA(cudaGetLastError());
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */

/* GetMax
 * @param [in]   in: [C, s]
 * @param [out] out: [C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'C' is the channel size
 * 's' is the sequence length
 */

/* Linear 
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
//////////////////For Linear layer 0,1,2///////////////////// 
__global__ void Batched_Linear_ReLU_kernel(float *in, float *w, float *b, float *out, size_t M, size_t N, size_t batch_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * batch_size;

    if (idx < total) {
        size_t m = idx % M;
        size_t batch_idx = idx / M; 

        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            sum += in[batch_idx * N + n] * w[m * N + n];
        }
        sum += b[m]; //bias
        out[batch_idx * M + m] = sum > 0.0f ? sum : 0.0f; //ReLU
    }
}

void Batched_Linear_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, cudaStream_t stream) {
    size_t N = in->shape[1];          
    size_t M = w->shape[0];           

    size_t distribute_B = (batch_size + NUM_GPU - 1) / NUM_GPU; //distribute batch
    auto use_gpu = [&](int gpu_idx) {
        size_t start_idx = gpu_idx * distribute_B;
        if (start_idx >= batch_size) {
            return;
        } 
        size_t currB = std::min(distribute_B, batch_size - start_idx);

        CHECK_CUDA(cudaSetDevice(gpu_idx));

        float *d_in, *d_w, *d_b, *d_out;
        CHECK_CUDA(cudaMalloc((void**)&d_in, currB * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_w, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_b, M * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_out, currB * M * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, in->buf + start_idx * N, currB * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w, w->buf, M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, b->buf, M * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());

        dim3 blockDim(256);
        dim3 gridDim((M * currB + blockDim.x - 1) / blockDim.x);
        Batched_Linear_ReLU_kernel<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, M, N, currB);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(out->buf + start_idx * M, d_out, currB * M * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());

        //free
        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_out));
    };

    std::vector<std::thread> gpu_threads;
    for(int gpu = 0; gpu < NUM_GPU; gpu++) {
        gpu_threads.emplace_back(use_gpu, gpu);
    }
    for(auto &t : gpu_threads) {
        t.join();
    }
}

//////////////////For Linear layer 3/////////////////////
__global__ void Batched_Linear_kernel(float *in, float *w, float *b, float *out, size_t M, size_t N, size_t batch_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * batch_size;

    if (idx < total) {
        size_t m = idx % M;
        size_t b_idx = idx / M;

        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            sum += in[b_idx * N + n] * w[m * N + n];
        }
        sum += b[m]; //bias
        out[b_idx * M + m] = sum; //no activation(ReLU). This is the last layer
    }
}

void Batched_Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t batch_size, cudaStream_t stream) {
    size_t N = in->shape[1];        
    size_t M = w->shape[0];          

    size_t distribute_B = (batch_size + NUM_GPU - 1) / NUM_GPU; //distribute batch
    auto use_gpu = [&](int gpu_idx) {
        size_t start_idx = gpu_idx * distribute_B;
        if (start_idx >= batch_size){
            return;
        } 
        size_t currB = std::min(distribute_B, batch_size - start_idx);

        CHECK_CUDA(cudaSetDevice(gpu_idx));

        float *d_in, *d_w, *d_b, *d_out;
        CHECK_CUDA(cudaMalloc((void**)&d_in, currB * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_w, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_b, M * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_out, currB * M * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, in->buf + start_idx * N, currB * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w, w->buf, M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, b->buf, M * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());

        dim3 blockDim(512);
        dim3 gridDim((M * currB + blockDim.x - 1) / blockDim.x);
        Batched_Linear_kernel<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, M, N, currB);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(out->buf + start_idx * M, d_out, currB * M * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());

        //free
        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_out));
    };

    std::vector<std::thread> gpu_threads;
    for(int gpu = 0; gpu < NUM_GPU; gpu++) {
        gpu_threads.emplace_back(use_gpu, gpu);
    }
    for(auto &t : gpu_threads) {
        t.join();
    }
}