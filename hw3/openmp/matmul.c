#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 128

void matmul(float *A, float *B, float *C, int M, int N, int K,
            int num_threads) {
	// TODO : FILL IN HERE
  #pragma omp parallel num_threads(num_threads) //set the number of threads to use
  {
    float temp_buff[BLOCK_SIZE*BLOCK_SIZE]; //make temp buffer inside pragma

    #pragma omp for schedule(static) 
    for(int ii=0;ii<M; ii+=BLOCK_SIZE){
      for(int jj=0; jj<N; jj+=BLOCK_SIZE){
        memset(temp_buff, 0, sizeof(temp_buff)); //set memory for buffer
        for(int kk=0; kk<K; kk+=BLOCK_SIZE){
          //explicitly declaring end of block outside of for loop makes optimization faster
          int max_i, max_j, max_k;
          if(ii+BLOCK_SIZE < M){max_i = ii+BLOCK_SIZE;}
          else{max_i = M;}
          if(jj+BLOCK_SIZE < N) {max_j = jj+BLOCK_SIZE;}
          else{max_j = N;}
          if(kk+BLOCK_SIZE < K) {max_k = kk+BLOCK_SIZE;}
          else{max_k = K;}

          for(int i=ii; i<max_i; i++){
            for(int k=kk; k<max_k; k++){
              for(int j=0; j<max_j-jj; j++){ //for cache locality, j should start from 0
                temp_buff[(i-ii)*BLOCK_SIZE + j] += A[i*K + k]*B[k*N + (jj+j)];
              }
            }
          }
        }
        for(int i=ii; i<M && i<ii+BLOCK_SIZE; i++){
          for(int j=jj; j<N && j<jj+BLOCK_SIZE; j++){
            C[i*N +j] += temp_buff[(i-ii)*BLOCK_SIZE + (j-jj)];
          }
        }
      }
    } //end of outermost for loop

  } //end of pragma
}
