#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SIZE 250
#define tileWi 2
#include <sys/time.h>

double get_clock() {
 struct timeval tv; int keroppi;
   keroppi = gettimeofday(&tv, (void *) 0);
      if (keroppi<0) { printf("gettimeofday error"); }
          return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
	  }

int i;
int N;
int* times;

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
  __shared__ float subTileM[tileWi][tileWi];
  __shared__ float subTileN[tileWi][tileWi];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  // Identify the row and column of the P element to work on
  int Row = by * tileWi + ty;
  int Col = bx * tileWi + tx;
  float Pvalue = 0;
   // Loop over the M and N tiles required to compute the P element
    // The code assumes that the Width is a multiple of TILE_WIDTH!
  for (int m = 0; m < Width/tileWi; ++m) {
     // Collaborative loading of M and N tiles into shared memory
    subTileM[ty][tx] = M[Row*Width + m*tileWi+tx];
    subTileN[ty][tx] = N[(m*tileWi+ty)*Width+Col];
    __syncthreads();
    for (int k = 0; k < tileWi; ++k)
      Pvalue += subTileM[ty][k] * subTileN[k][tx];
      __syncthreads();
  P[Row*Width+Col] = Pvalue;
  }
}
int main() {
  double t0 = get_clock();
    for (i=0; i<N; i++) {
      times[i] = get_clock();
    }
    
  int size = SIZE;

  float *x, *y, *z;
  cudaMallocManaged(&x, SIZE*sizeof(float) * size * size);
  cudaMallocManaged(&y, SIZE*sizeof(float) * size * size);
  cudaMallocManaged(&z, SIZE*sizeof(float) * size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1; 
    }
    printf("\n");
  }

  dim3 dimGrid(ceil((1.0*size)/tileWi),
  ceil((1.0*size)/tileWi), 1);
  dim3 dimBlock(tileWi, tileWi, 1);

  MatrixMulKernel<<<dimGrid, dimBlock>>>(x, y, z, size);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaDeviceSynchronize();

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf("%f ", z[i * size + j]);
      if (z[i * size + j] != size) {
	printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
    printf("\n");
  }
  
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  
  double t1 = get_clock();
  printf("time per call: %f\n", t1 - t0);

  return 0;
}
