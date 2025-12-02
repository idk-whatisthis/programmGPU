
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void createMatrix(int *A, const int n) {
    A[threadIdx.y * n + threadIdx.x] = 10 * threadIdx.y + threadIdx.x;
}

int main() {
    const int n = 1024;
    size_t size = n * n * sizeof(int);
    
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *d_B = NULL;
    
    cudaMalloc((void **)&d_B, size);
    
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(n / 32, n / 32);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    createMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_B, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    
    printf("Create Matrix Time: %.2f ms\n", time);
    printf("Matrix size: %dx%d\n", n, n);
    
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

