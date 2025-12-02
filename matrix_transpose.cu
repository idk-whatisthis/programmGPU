#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define BASE_TYPE float

__global__ void matrixTranspose(const BASE_TYPE *A, BASE_TYPE *AT, int rows, int cols) {
    int iA = cols * (blockDim.y * blockIdx.y + threadIdx.y) + 
             blockDim.x * blockIdx.x + threadIdx.x;
    int iAT = rows * (blockDim.x * blockIdx.x + threadIdx.x) + 
              blockDim.y * blockIdx.y + threadIdx.y;
    AT[iAT] = A[iA];
}

int toMultiple(int a, int b) {
    int mod = a % b;
    return (mod != 0) ? a + (b - mod) : a;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int rows = 2048;
    int cols = 2048;
    
    rows = toMultiple(rows, BLOCK_SIZE);
    cols = toMultiple(cols, BLOCK_SIZE);
    
    size_t size = rows * cols * sizeof(BASE_TYPE);
    
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(size);
    BASE_TYPE *h_AT = (BASE_TYPE *)malloc(size);
    BASE_TYPE *d_A = NULL;
    BASE_TYPE *d_AT = NULL;
    
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_AT, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(cols / BLOCK_SIZE, rows / BLOCK_SIZE);
    
    cudaEventRecord(start);
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_AT, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Matrix Transpose Time: %.2f ms\n", time);
    printf("Matrix size: %dx%d, Block size: %dx%d\n", rows, cols, BLOCK_SIZE, BLOCK_SIZE);
    
    cudaMemcpy(h_AT, d_AT, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_AT);
    free(h_A);
    free(h_AT);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
