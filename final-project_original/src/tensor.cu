#include "model.h"

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */

Tensor::Tensor(const std::vector<size_t> &shape_, int device_id_)
    : ndim(shape_.size()), buf(nullptr), buf_int(nullptr), total_elements(1), is_gpu(true), device_id(device_id_) {
    assert(ndim <= 4 && "Tensor constructor supports up to 4 dimensions.");
    for(size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    for(size_t i = ndim; i < 4; i++) { shape[i] = 1; }
    for(size_t i = 0; i < ndim; i++) { total_elements *= shape[i]; }

    CHECK_CUDA(cudaSetDevice(device_id));
    CHECK_CUDA(cudaMallocHost(&buf, total_elements * sizeof(float))); //allocate device memory for float data
}

//constructor for float buffers with host data
Tensor::Tensor(const std::vector<size_t> &shape_, float *host_buf_, int device_id_)
    : ndim(shape_.size()), buf(nullptr), buf_int(nullptr), total_elements(1), is_gpu(true), device_id(device_id_) {
    assert(ndim <= 4 && "Tensor constructor supports up to 4 dimensions.");
    for(size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    for(size_t i = ndim; i < 4; i++) { shape[i] = 1; } 
    for(size_t i = 0; i < ndim; i++) { total_elements *= shape[i]; }

    CHECK_CUDA(cudaSetDevice(device_id));
    CHECK_CUDA(cudaMallocHost(&buf, total_elements * sizeof(float))); //allocate device memory for float data

    if (host_buf_ != nullptr) { //if host buffer exists, copy data to device
        CHECK_CUDA(cudaMemcpy(buf, host_buf_, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    }
}


//constructor for int buffers with host data
Tensor::Tensor(const std::vector<size_t> &shape_, int *host_buf_, int device_id_)
    : ndim(shape_.size()), buf(nullptr), buf_int(nullptr), total_elements(1), is_gpu(true), device_id(device_id_) {
    assert(ndim <= 4 && "Tensor constructor supports up to 4 dimensions.");
    for(size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    for(size_t i = ndim; i < 4; i++) { shape[i] = 1; } 
    for(size_t i = 0; i < ndim; i++) { total_elements *= shape[i]; }

    CHECK_CUDA(cudaSetDevice(device_id));
    CHECK_CUDA(cudaMallocHost(&buf_int, total_elements * sizeof(int))); //allocate device memory for int data

    if (host_buf_ != nullptr) { //if host buffer exists, copy data to device
        CHECK_CUDA(cudaMemcpy(buf_int, host_buf_, total_elements * sizeof(int), cudaMemcpyHostToDevice));
    }
}

Tensor::~Tensor() {
    if(buf){ //free float buffer
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUDA(cudaFreeHost(buf));
        buf = nullptr;
    }

    if(buf_int){ //free int buffer
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUDA(cudaFreeHost(buf_int));
        buf_int = nullptr;
    }
}

size_t Tensor::num_elem() {
    return total_elements;
}