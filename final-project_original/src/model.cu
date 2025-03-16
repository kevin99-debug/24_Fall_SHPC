#include <mpi.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>

#include "layer.h"
#include "model.h"

#define CHECK_MPI(call)                                                        \
    do {                                                                       \
        int err = call;                                                        \
        if (err != MPI_SUCCESS) {                                              \
            char err_str[MPI_MAX_ERROR_STRING];                                \
            int err_len;                                                        \
            MPI_Error_string(err, err_str, &err_len);                          \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    err_str);                                                  \
            MPI_Abort(MPI_COMM_WORLD, err);                                    \
        }                                                                      \
    } while(0)

//number of Nodes 
#define NUM_WORKERS 4 
#define BATCH_SIZE 2048

//global pointers for Parameters and Activations
Parameter *linear0_w = nullptr; Parameter *linear0_b = nullptr;
Parameter *linear1_w = nullptr; Parameter *linear1_b = nullptr;
Parameter *linear2_w = nullptr; Parameter *linear2_b = nullptr;
Parameter *linear3_w = nullptr; Parameter *linear3_b = nullptr;
Parameter *conv_w[NUM_WORKERS] = {nullptr};
Parameter *conv_b[NUM_WORKERS] = {nullptr};
Parameter *emb_w = nullptr; 

Activation *emb_a = nullptr;
Activation *permute_a = nullptr;
Activation *concat_a = nullptr;
Activation *linear0_a = nullptr;
Activation *linear1_a = nullptr;
Activation *linear2_a = nullptr;
Activation *linear3_a = nullptr;
Activation *conv_out_a_workers[NUM_WORKERS] = {nullptr};

//preallocate device memory pointers for concatenation 
float* d_all_conv_results = nullptr;
float* d_concatenated_conv_results = nullptr;
//pinned host memory
float* host_processed_data = nullptr; //embed & permute data
float* host_all_conv_results_pinned = nullptr; //all_conv_results

void alloc_and_set_parameters(float *param, size_t param_size) {
    int rank, size;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    assert(size == NUM_WORKERS && "Number of MPI ranks must be equal to NUM_WORKERS.");

    CHECK_CUDA(cudaSetDevice(rank)); 
    
    size_t pos = 0;

    //allocate embedding weights
    emb_w = new Parameter({21635, EMBEDDING_DIM}, param + pos, rank); //[21635, 4096] 
    pos += 21635 * EMBEDDING_DIM;

    //allocate conv parameters
    int kernel_sizes[NUM_WORKERS] = {3, 5, 7, 9}; //kernel size differs for each rank

    for(int r = 0; r < NUM_WORKERS; r++) {
        if(r == rank){ //current rank (weights + bias)
            conv_w[r] = new Parameter({N_FILTERS, EMBEDDING_DIM, static_cast<size_t>(kernel_sizes[r])}, param + pos, rank); //[1024, 4096, K]
            pos += N_FILTERS * EMBEDDING_DIM * kernel_sizes[r];
            conv_b[r] = new Parameter({N_FILTERS}, param + pos, rank); //[1024]
            pos += N_FILTERS;
        }
        else{ //conv parameters of other ranks ignored. But their position is accumulated
            pos += N_FILTERS * EMBEDDING_DIM * kernel_sizes[r]; //weight
            pos += N_FILTERS; //bias
        }
    }

    //allocate linear parameters (rank 0)
    if(rank == 0){
        //linear0 layer
        linear0_w = new Parameter({2048, 4096}, param + pos, rank); //[2048, 4096]
        pos += 2048 * 4096;
        linear0_b = new Parameter({2048}, param + pos, rank); //[2048]
        pos += 2048;
        //linear1 layer
        linear1_w = new Parameter({1024, 2048}, param + pos, rank); //[1024, 2048]
        pos += 1024 * 2048;
        linear1_b = new Parameter({1024}, param + pos, rank); //[1024]
        pos += 1024;
        //linear2 layer
        linear2_w = new Parameter({512, 1024}, param + pos, rank); //[512, 1024]
        pos += 512 * 1024;
        linear2_b = new Parameter({512}, param + pos, rank); //[512]
        pos += 512;
        //linear3 layer
        linear3_w = new Parameter({2, 512}, param + pos, rank); //[2, 512]
        pos += 2 * 512;
        linear3_b = new Parameter({2}, param + pos, rank); //[2]
        pos += 2;
    }
    else{ //no linear for other ranks
        pos += (2048 * 4096) + 2048; //linear0_w, linear0_b
        pos += (1024 * 2048) + 1024; //linear1_w, linear1_b
        pos += (512 * 1024) + 512; //linear2_w, linear2_b
        pos += (2 * 512) + 2; //linear3_w, linear3_b
    }
}

void free_parameters() {
    int rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_CUDA(cudaSetDevice(rank));

    if(rank == 0){ //only rank0 has linear 
        delete emb_w;
        delete linear0_w;
        delete linear0_b;
        delete linear1_w;
        delete linear1_b;
        delete linear2_w;
        delete linear2_b;
        delete linear3_w;
        delete linear3_b;
    }
    delete conv_w[rank];
    delete conv_b[rank];
}

//initialize and preallocate memory for concat convolution results
void initialize_cc(size_t conv_result_size){
    size_t concatenated_conv_bytes = NUM_WORKERS * conv_result_size * sizeof(float); //size of concat
    //allocate device memory once
    CHECK_CUDA(cudaMalloc(&d_all_conv_results, concatenated_conv_bytes));
    CHECK_CUDA(cudaMalloc(&d_concatenated_conv_results, concatenated_conv_bytes));
    //allocate pinned host memory for all_conv_results
    CHECK_CUDA(cudaMallocHost(&host_all_conv_results_pinned, concatenated_conv_bytes));
}

//free preallocated memory for concat convolution results
void finalize_cc(){
    //free device memory
    if(d_all_conv_results){
        CHECK_CUDA(cudaFree(d_all_conv_results));
        d_all_conv_results = nullptr;
    }
    if(d_concatenated_conv_results){
        CHECK_CUDA(cudaFree(d_concatenated_conv_results));
        d_concatenated_conv_results = nullptr;
    }
    //free pinned host memory
    if(host_all_conv_results_pinned){
        CHECK_CUDA(cudaFreeHost(host_all_conv_results_pinned));
        host_all_conv_results_pinned = nullptr;
    }
}

void alloc_activations() {
    int rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_CUDA(cudaSetDevice(rank));

    emb_a = new Activation({BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM}, (float*)nullptr, rank); //embedding
    permute_a = new Activation({BATCH_SIZE, EMBEDDING_DIM, SEQ_LEN}, (float*)nullptr, rank); //permute
    
    if(rank == 0){ //rank 0 handles all linear layers
        conv_out_a_workers[0] = new Activation({BATCH_SIZE, N_FILTERS}, (float*)nullptr, rank); //conv0
        concat_a = new Activation({BATCH_SIZE, 4096}, (float*)nullptr, rank); 
        linear0_a = new Activation({BATCH_SIZE, 2048}, (float*)nullptr, rank); 
        linear1_a = new Activation({BATCH_SIZE, 1024}, (float*)nullptr, rank); 
        linear2_a = new Activation({BATCH_SIZE, 512}, (float*)nullptr, rank); 
        linear3_a = new Activation({BATCH_SIZE, 2}, (float*)nullptr, rank); 
    }
    else{ //other ranks -> only conv
        conv_out_a_workers[rank] = new Activation({BATCH_SIZE, N_FILTERS}, (float*)nullptr, rank);
    }
    // Allocate pinned host memory buffers for embedding and permutation data
    CHECK_CUDA(cudaMallocHost(&host_processed_data, BATCH_SIZE * EMBEDDING_DIM * SEQ_LEN * sizeof(float))); //e&p pinned host memory buffer
    size_t conv_result_size = BATCH_SIZE * N_FILTERS; //conv result size
    initialize_cc(conv_result_size);
}

void free_activations() {
    int rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_CUDA(cudaSetDevice(rank));

    delete emb_a;
    delete permute_a;
    delete conv_out_a_workers[rank];

    if(rank == 0){
        delete concat_a;
        delete linear0_a;
        delete linear1_a;
        delete linear2_a;
        delete linear3_a;
    }

    if(host_processed_data){ //free e&p pinned memory
        CHECK_CUDA(cudaFreeHost(host_processed_data));
        host_processed_data = nullptr;
    }
    finalize_cc();
}

void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
    int rank, size;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    assert(size == NUM_WORKERS && "Number of MPI ranks must be 4 (all Workers).");
    
    CHECK_CUDA(cudaSetDevice(rank));
    size_t total_batches = n_samples / BATCH_SIZE; //# of batches
    size_t batch_size = BATCH_SIZE;

    for(size_t batch_idx = 0; batch_idx < total_batches; ++batch_idx){ //iterate over batches
        
        if(rank ==0){
            //step 1: embed & permute batch
            size_t global_batch_start = batch_idx * batch_size * SEQ_LEN;
            std::vector<int> current_batch_inputs(batch_size * SEQ_LEN);
            memcpy(current_batch_inputs.data(), inputs + global_batch_start, batch_size * SEQ_LEN * sizeof(int));

            Tensor input_tensor({batch_size, SEQ_LEN}, current_batch_inputs.data(), rank);

            Batched_Embedding_CUDA(&input_tensor, emb_w, emb_a, batch_size, 0, rank);
            Batched_Permute_CUDA(emb_a, permute_a, batch_size, 0, rank);

            CHECK_CUDA(cudaStreamSynchronize(0)); //sync

            //step 2: e&p data to host
            CHECK_CUDA(cudaMemcpy(host_processed_data, permute_a->buf, batch_size * EMBEDDING_DIM * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost));

            //step 3: broadcast e&p data to all ranks
            MPI_Bcast(host_processed_data, batch_size * EMBEDDING_DIM * SEQ_LEN, MPI_FLOAT, 0, MPI_COMM_WORLD);

            //step 4: copy received data back to device
            CHECK_CUDA(cudaMemcpy(permute_a->buf, host_processed_data, batch_size * EMBEDDING_DIM * SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice));
        }
        else{ //rank1, rank2, rank3
            MPI_Bcast(host_processed_data, batch_size * EMBEDDING_DIM * SEQ_LEN, MPI_FLOAT, 0, MPI_COMM_WORLD);
            CHECK_CUDA(cudaMemcpy(permute_a->buf, host_processed_data, batch_size * EMBEDDING_DIM * SEQ_LEN * sizeof(float), cudaMemcpyHostToDevice));
        }
        //all ranks have e&p data for convolution
        //step 5: all ranks perform convolution layers
        Batched_Conv1D_ReLU_GetMax_CUDA(permute_a, conv_w[rank], conv_b[rank], conv_out_a_workers[rank], batch_size, 0, rank);
        CHECK_CUDA(cudaStreamSynchronize(0)); //sync

        //convolution Results to Host
        size_t conv_result_size = batch_size * N_FILTERS;
        CHECK_CUDA(cudaMemcpyAsync(host_all_conv_results_pinned + rank * (batch_size * N_FILTERS), conv_out_a_workers[rank]->buf, batch_size * N_FILTERS * sizeof(float), cudaMemcpyDeviceToHost, 0));
        CHECK_CUDA(cudaStreamSynchronize(0)); //sync

        if(rank !=0){ //rank1, rank2, rank3 -> conv result to rank0
            CHECK_MPI(MPI_Send(host_all_conv_results_pinned + rank * (batch_size * N_FILTERS), conv_result_size, MPI_FLOAT, 0, 100 + batch_idx, MPI_COMM_WORLD));
        }
        else{ //rank0 gathers conv result from other ranks(also from itself)
            for(int src=1; src < NUM_WORKERS; src++){ //recieve from other ranks
                CHECK_MPI(MPI_Recv(host_all_conv_results_pinned + src * conv_result_size, conv_result_size, MPI_FLOAT, src, 100 + batch_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            //copy conv results to device -> concat
            CHECK_CUDA(cudaMemcpy(d_all_conv_results, host_all_conv_results_pinned, NUM_WORKERS * conv_result_size * sizeof(float), cudaMemcpyHostToDevice));
            Batched_Concat_CUDA(d_all_conv_results, d_concatenated_conv_results, NUM_WORKERS, batch_size, N_FILTERS, 0);
            CHECK_CUDA(cudaMemcpy(concat_a->buf, d_concatenated_conv_results, NUM_WORKERS * conv_result_size * sizeof(float), cudaMemcpyDeviceToDevice));

            //linear layers on rank 0
            Batched_Linear_ReLU_CUDA(concat_a, linear0_w, linear0_b, linear0_a, BATCH_SIZE, 0);
            Batched_Linear_ReLU_CUDA(linear0_a, linear1_w, linear1_b, linear1_a, BATCH_SIZE, 0);
            Batched_Linear_ReLU_CUDA(linear1_a, linear2_w, linear2_b, linear2_a, BATCH_SIZE, 0);
            Batched_Linear_CUDA(linear2_a, linear3_w, linear3_b, linear3_a, BATCH_SIZE, 0);

            //copy final output to host
            std::vector<float> final_output(batch_size * 2);
            CHECK_CUDA(cudaMemcpy(final_output.data(), linear3_a->buf, batch_size * 2 * sizeof(float), cudaMemcpyDeviceToHost));
            memcpy(outputs + batch_idx * batch_size * 2, final_output.data(), batch_size * 2 * sizeof(float)); 
        }
        CHECK_CUDA(cudaStreamSynchronize(0));
    }
}
