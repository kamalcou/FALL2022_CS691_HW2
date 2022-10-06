/*
Name: Md Kamal Hossain Chowdhury
Email:mhchowdhury@crimson.ua.edu
Fall 2022: CS 691 
Homework #: 2
Instructions to compile the program: (for example: nvcc -o hw2 hw2.cpp)
Instructions to run the program: (for example: ./hw2 10000)
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <mkl.h>
#include <cublas_v2.h>

// const int TILE_DIM = 32;
// const int BLOCK_rowS = 8;
// const int NUM_REPS = 100;
#define BLOCK_SIZE 16
#define TILE_WIDTH 16

/* function to measure time taken */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void printarray(double *A, int N) {
  int i, j;
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
        
        printf(" %.1lf ",A[i*N+j]);   
    }
    printf("\n");
  }
}




void naive_multi(double *A, double *B,double *C, int N)
{
    //double rslt[N*N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N+j] = 0;
 
            for (int k = 0; k < N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
}

void cblas_multi(double *A, double *B,double *C, int N){

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, 1.0, A, N, B, N, 0.0, C, N);
}

double rand_range(double min, double max){

    double random=((double) rand())/RAND_MAX;
    double range=(max-min)*random;
    double number=min+range;

    return  number;

}


int compare_array(double *A,double *B, int N) {
  int flag=1;
  for (int i = 0; i < N; i++)
    for (int j = 0; j< N; j++){

      double maxError = 0.001;
  
      if(maxError<fabs(A[i*N+j]-B[i*N+j])){
        printf("Failed naive C[%d][%d]=%lf produced C[%d][%d]=%lf\n",i,j,A[i*N+j],i,j,B[i*N+j]);
        flag= 0;
        return flag;
      }
  
    }

  return flag;
}

__global__ void sharedMatrixMulti(double *A, double *B, double *C, int Width)
{
  __shared__ double tiledA[TILE_WIDTH][TILE_WIDTH];
  __shared__ double tiledB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  // Identify the row and column of the P element to work on
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  double Pvalue = 0;
  
  for (int m = 0; m < (Width+TILE_WIDTH-1)/TILE_WIDTH; ++m) {
    
         if(row*Width + m*TILE_WIDTH+tx<Width*Width)
          {tiledA[ty][tx] = A[row*Width + m*TILE_WIDTH+tx];}
          else{
            tiledA[ty][tx] = 0;
          }
          if((m*TILE_WIDTH+ty)*Width+col<Width*Width){
              tiledB[ty][tx] = B[(m*TILE_WIDTH+ty)*Width+col];
          }
          else{

            tiledB[ty][tx] =0;
          }
          
          __syncthreads();
          for (int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += tiledA[ty][k] * tiledB[k][tx];
          
          __syncthreads();
          }
  }
   if(row<Width && col<Width){
        C[row*Width+col] = Pvalue;
        }
  
}



__global__ void naive_GPU(double *A, double *B, double *C,int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < N) {
		double sum = 0;
		for (int i = 0; i < N; i++)
			sum += A[row * N + i] * B[i * N + col];
		C[row * N + col] = sum;
	}
}

void cublas_multi(double *A, double *B, double *C, int N) {
	

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
  double alpha=1.0;
  double beta=0.0;


	
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, A, N, B, N, &beta, C, N);

	// Destroy the handle
	cublasDestroy(handle);
}

int main(int argc, char **argv) {
  int N;
  double *A,*B,*C;
  int i, j;
  double t1, t2;

  double *cblas_C;

  double *d_A,*d_B,*d_C,*d_cublasC,*d_sharedC,*h_C,*h_cublas_C,*h_sharedC;
  
  



  N = atoi(argv[1]);
  

  /* Allocate memory for the arrays */
  A = (double *) malloc(N*N*sizeof(double));
  B = (double *) malloc(N*N*sizeof(double));
  C = (double *) malloc(N*N*sizeof(double));
  h_C = (double *) malloc(N*N*sizeof(double));
  h_cublas_C = (double *) malloc(N*N*sizeof(double));
  h_sharedC = (double *) malloc(N*N*sizeof(double)); 
  cblas_C = (double *) malloc(N*N*sizeof(double));


  //allocate memory in the device 
  checkCuda(cudaMalloc(&d_A, N*N*sizeof(double)));
  checkCuda(cudaMalloc(&d_B, N*N*sizeof(double)));
  checkCuda(cudaMalloc(&d_C, N*N*sizeof(double)));
  checkCuda(cudaMalloc(&d_cublasC, N*N*sizeof(double)));
  checkCuda(cudaMalloc(&d_sharedC, N*N*sizeof(double)));


  
  
  srand(time(NULL));

  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
        A[i*N+j]= rand_range(0,10);  //random number generation for A
        // A[i*N+j]= 1;
    }
   
  }
  
  
   for(i=0;i<N;i++){
    for(j=0;j<N;j++){
        B[i*N+j]= rand_range(0,10);   //random number generation for B
        // B[i*N+j]= 1;    
    }
  }
  
  //Naïve CPU Version 
  t1 = gettime();
    naive_multi(A,B,C,N);  //CPU naive matrix multi
  t2 = gettime();
   
  
  printf("Naïve CPU Version Matrix multiplication for size = %d  time taken = %f ms\n",
          N, (t2-t1)*1000);
  
  //cblas_dgemm CPU Version
  t1 = gettime();
    cblas_multi(A,B,cblas_C,N);
  t2 = gettime();

  printf("cblas_dgemm CPU Version Matrix multiplication for size = %d  time taken = %f ms\n",
          N, (t2-t1)*1000);

  compare_array(C,cblas_C,N);  //correctness check


 //GPU programming
  
 //Simple GPU Version

  checkCuda(cudaMemcpy(d_A, A, N*N*sizeof(double),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_B, B, N*N*sizeof(double),cudaMemcpyHostToDevice));


  dim3 threadsPerBlock(((N+BLOCK_SIZE-1)/BLOCK_SIZE), ((N+BLOCK_SIZE-1)/BLOCK_SIZE), 1);
  dim3 blocksPerGrid(BLOCK_SIZE, BLOCK_SIZE, 1);
  
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;


  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaEventRecord(stopEvent, 0));
   

  naive_GPU<<<threadsPerBlock,blocksPerGrid>>>(d_A, d_B, d_C, N);


  
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  printf("Simple GPU Version time Matrix multiplication for size = %d time taken=%f ms\n",N,ms);

  checkCuda(cudaMemcpy(h_C, d_C, N*N*sizeof(double), cudaMemcpyDeviceToHost));
  
  compare_array(C,h_C,N);  //correctness check
 
  //Shared Memory GPU Version
  checkCuda(cudaMemcpy(d_A, A, N*N*sizeof(double),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_B, B, N*N*sizeof(double),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_sharedC, h_sharedC, N*N*sizeof(double),cudaMemcpyHostToDevice));


  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaEventRecord(stopEvent, 0));

  sharedMatrixMulti<<<threadsPerBlock,blocksPerGrid>>>(d_A,d_B,d_sharedC,N);
  

  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  printf("Shared Memory GPU Version for size = %d taken time=%f ms\n",N,ms);

  checkCuda(cudaMemcpy(h_sharedC, d_sharedC, N*N*sizeof(double), cudaMemcpyDeviceToHost));
  compare_array(C,h_sharedC,N);  //correctness check

  

  //cublasDgemm GPU Version 

  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaEventRecord(stopEvent, 0));
  cublas_multi(d_B, d_A, d_cublasC, N) ;  
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

  printf("cublasDgemm GPU Version for size = %d taken time=%f ms\n",N,ms);

  checkCuda(cudaMemcpy(h_cublas_C, d_cublasC, N*N*sizeof(double), cudaMemcpyDeviceToHost));
  compare_array(C,h_cublas_C,N);  //correctness check
  
   
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_sharedC);
  cudaFree(d_cublasC);

  free(A);
  free(B);
  free(C);
  free(h_C);
  free(h_cublas_C);
  free(h_sharedC);
  free(cblas_C);

  return 0;
}


