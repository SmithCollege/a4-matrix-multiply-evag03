#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SIZE 250
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

 __global__ void MatrixMulOnDevice(float* A, float* B, float* C, int Width) {
  for (int i = 0; i < Width; ++i) {
    for (int j = 0; j < Width; ++j) {
      float sum = 0;
      for (int k = 0; k < Width; ++k) {
	float a = A[i * Width + k];
	float b = B[k * Width + j];
	sum += a * b;
      }
      C[i * Width + j] = sum;
    }
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

  MatrixMulOnDevice<<<1, 128>>>(x, y, z, size);

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
