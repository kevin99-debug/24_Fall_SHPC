#define BLOCK_SIZE 32
#define NUM_ELEM_PER_THREAD 8

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  
  __local float A_block[BLOCK_SIZE][BLOCK_SIZE+2];
  __local float B_block[BLOCK_SIZE][BLOCK_SIZE+2];

  //get global id
  int globalRow = get_global_id(0) * NUM_ELEM_PER_THREAD;
  int globalCol = get_global_id(1) * NUM_ELEM_PER_THREAD;

  //get local id
  int localRow = get_local_id(0) * NUM_ELEM_PER_THREAD;
  int localCol = get_local_id(1) * NUM_ELEM_PER_THREAD;

  //accumulation value for current work-item
  float acc[NUM_ELEM_PER_THREAD][NUM_ELEM_PER_THREAD] = {{0.0f}};

  //loop over all BLOCKS to compute C[globalRow, globalCol]
  for(int t=0; t < (K + BLOCK_SIZE - 1)/ BLOCK_SIZE; t++){
    //load tiles of A and B to local memory
    for(int i=0; i < NUM_ELEM_PER_THREAD; i++){
      for(int j=0; j< NUM_ELEM_PER_THREAD; j++){
        if(globalRow + i < M && ((t * BLOCK_SIZE) + localCol + j) < K){
          A_block[localRow + i][localCol + j] = A[((globalRow + i) * K) + (t * BLOCK_SIZE) + localCol + j];
        }
        else{
          A_block[localRow + i][localCol + j] = 0.0f;
        }

        if(globalCol + j < N && ((t * BLOCK_SIZE) + localRow + i) < K){
          B_block[localRow + i][localCol + j] = B[(t * BLOCK_SIZE + localRow + i) * N + globalCol + j];
        }
        else{
          B_block[localRow + i][localCol + j] = 0.0f;
        }
      }
    } //end of tile loop

    //synchronize tiles
    barrier(CLK_LOCAL_MEM_FENCE);

    //compute on loaded tile
    for (int k = 0; k < BLOCK_SIZE; k++) {
      for (int i = 0; i < NUM_ELEM_PER_THREAD; i++) {
        for (int j = 0; j < NUM_ELEM_PER_THREAD; j++) {
          acc[i][j] += A_block[localRow + i][k] * B_block[k][localCol + j];
        }
      }
    }
    //synchronize tiles
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  //write to global memory
  for (int i = 0; i < NUM_ELEM_PER_THREAD; i++) {
    for (int j = 0; j < NUM_ELEM_PER_THREAD; j++) {
      if (globalRow + i < M && globalCol + j < N) {
        C[(globalRow + i) * N + globalCol + j] = acc[i][j];
      }
    }
  }

}
