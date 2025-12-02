#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define BASE_TYPE float

__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, 
                          int Acols, int Bcols) {
    int i0 = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;
    BASE_TYPE sum = 0;

    for (int k = 0; k < Acols; k++) {
        sum += A[i0 + k] * B[k * Bcols + j0];
    }

    int ind = Bcols * (blockDim.y * blockIdx.y + threadIdx.y) + 
              blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = sum;
}

int toMultiple(int a, int b) {
    int mod = a % b;
    return (mod != 0) ? a + (b - mod) : a;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int Arows = 512, Acols = 512;
    int Brows = Acols, Bcols = 512;
    
    Arows = toMultiple(Arows, BLOCK_SIZE);
    Acols = toMultiple(Acols, BLOCK_SIZE);
    Brows = toMultiple(Brows, BLOCK_SIZE);
    Bcols = toMultiple(Bcols, BLOCK_SIZE);
    
    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);
    
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);
    
    BASE_TYPE *d_A = NULL, *d_B = NULL, *d_C = NULL;
    
    for (int i = 0; i < Arows * Acols; ++i) h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    for (int i = 0; i < Brows * Bcols; ++i) h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);
    
    cudaEventRecord(start);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Acols, Bcols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Matrix Multiplication Time: %.2f ms\n", time);
    printf("A: %dx%d, B: %dx%d, C: %dx%d\n", Arows, Acols, Brows, Bcols, Arows, Bcols);
    
    cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);
    
    // Проверка правильности (выборочно)
    int check_i = 10, check_j = 10;
    BASE_TYPE cpu_sum = 0;
    for (int k = 0; k < Acols; k++) {
        cpu_sum += h_A[check_i * Acols + k] * h_B[k * Bcols + check_j];
    }
    BASE_TYPE gpu_val = h_C[check_i * Bcols + check_j];
    printf("Verification: CPU[%d,%d] = %f, GPU[%d,%d] = %f, Difference = %f\n", 
           check_i, check_j, cpu_sum, check_i, check_j, gpu_val, fabs(cpu_sum - gpu_val));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
