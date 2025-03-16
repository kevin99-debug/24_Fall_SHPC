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
  omp_set_num_threads(num_threads); //set the number of threads to use

  #pragma omp parallel 
  {
    //use a buffer of [BLOCK_SIZE * BLOCK_SIZE] to write to C
    //instead of writing directly to C every time,
    //accumulate results for a block
    float temp_buff[BLOCK_SIZE*BLOCK_SIZE];

    //implement a blocking method
    //loops for blocks
    #pragma omp for schedule(static)
    for(int ii=0; ii<M; ii+=BLOCK_SIZE){
      for(int jj=0; jj<N; jj+=BLOCK_SIZE){
        memset(temp_buff, 0, sizeof(temp_buff)); //set memory for buffer
        for(int kk=0; kk<K; kk+=BLOCK_SIZE){
          //set the end row for the blocks. this code handles leftovers as well
          int max_i, max_j, max_k;
          if(ii+BLOCK_SIZE < M){max_i = ii+BLOCK_SIZE;}
          else{max_i = M;}
          if(jj+BLOCK_SIZE < N) {max_j = jj+BLOCK_SIZE;}
          else{max_j = N;}
          if(kk+BLOCK_SIZE < K) {max_k = kk+BLOCK_SIZE;}
          else{max_k = K;}

          //access in i->k->j order. 
          //much more efficient memory access
          
          for(int i=ii; i<max_i; i++){
            for(int k=kk; k<max_k; k++){
              for(int j=0; j< max_j-jj; j++){ //iterate over BLOCK_SIZE
                temp_buff[(i-ii)*BLOCK_SIZE + j] += A[i*K + k] * B[k*N + (jj + j)];
              }
            }
          }
        }
        //one block of multiplication finished. temp_buffer full.
        //write to C(output)
        for(int i=ii; i<M && i<ii+BLOCK_SIZE; i++){
          for(int j=jj; j<N && j<jj+BLOCK_SIZE; j++){
            C[i*N + j] += temp_buff[(i-ii)*BLOCK_SIZE + (j-jj)];
          }
        }
      }
    }
  }

}
