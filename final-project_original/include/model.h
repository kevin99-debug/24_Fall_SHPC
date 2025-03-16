#pragma once

#include "tensor.h"

/* Model Configuration */
#define VOCAB_SIZE 21635
#define EMBEDDING_DIM 4096
#define N_FILTERS 1024
#define N_CLASSES 2
#define SEQ_LEN 16


void alloc_and_set_parameters(float *param, size_t param_size);
void alloc_activations();
void predict_sentiment(int *inputs, float *outputs, size_t n_samples);
void free_parameters();
void free_activations();

