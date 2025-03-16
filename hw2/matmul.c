#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 128

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[256];
static pthread_t threads[256];

static void *matmul_kernel(void *arg) {
  struct thread_arg *input = (struct thread_arg *)arg;
  const float *A = (*input).A;
  const float *B = (*input).B;
  float *C = (*input).C;
  int M = (*input).M;
  int N = (*input).N;
  int K = (*input).K;
  int num_threads = (*input).num_threads;
  int rank = (*input).rank;

  //HW2 implementation from here. 
  int rows_per_thread = M / num_threads;
  int start_row_idx = rank * rows_per_thread;
  int end_row_idx;

  if(rank == num_threads-1){ //last thread. end_row is M.
    end_row_idx = M;
  }
  else { //not the last thread
    end_row_idx = (rank+1) * rows_per_thread; //set end row to start of next thread.
  }

  //use a buffer of [BLOCK_SIZE * BLOCK_SIZE] to write to C
  //instead of writing directly to C every time,
  //accumulate results for a block
  float temp_buff[BLOCK_SIZE*BLOCK_SIZE];

  //implement a blocking method
  //loops for blocks
  for(int ii=start_row_idx; ii<end_row_idx; ii+=BLOCK_SIZE){
    for(int jj=0; jj<N; jj+=BLOCK_SIZE){
      memset(temp_buff, 0, sizeof(temp_buff)); //set memory for buffer
      for(int kk=0; kk<K; kk+=BLOCK_SIZE){
        //set the end row for the blocks. this code handles leftovers as well
        int max_i, max_j, max_k;
        if(ii+BLOCK_SIZE < end_row_idx){max_i = ii+BLOCK_SIZE;}
        else{max_i = end_row_idx;}
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
      for(int i=ii; i<end_row_idx && i<ii+BLOCK_SIZE; i++){
        for(int j=jj; j<N && j<jj+BLOCK_SIZE; j++){
          C[i*N + j] += temp_buff[(i-ii)*BLOCK_SIZE + (j-jj)];
        }
      }
    }
  }

  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
    if (err) {
      printf("pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      printf("pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
}