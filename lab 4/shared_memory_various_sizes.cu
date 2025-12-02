#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define BASE_TYPE float

__global__ void matrixMultShared(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Arows, int Acols, int Bcols) {
    __shared__ BASE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    BASE_TYPE sum = 0;
    
    for (int tile = 0; tile < (Acols + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        if (row < Arows && (tile * BLOCK_SIZE + threadIdx.x) < Acols) {
            As[threadIdx.y][threadIdx.x] = A[row * Acols + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (col < Bcols && (tile * BLOCK_SIZE + threadIdx.y) < Acols) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * Bcols + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < Arows && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

void test_matrix_size(int size) {
    printf("\\n=== Testing Matrix Size: %dx%d ===\\n", size, size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int Arows = size, Acols = size;
    int Brows = size, Bcols = size;
    
    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);
    
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);
    
    BASE_TYPE *d_A, *d_B, *d_C;
    
    for (int i = 0; i < Arows * Acols; ++i) h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    for (int i = 0; i < Brows * Bcols; ++i) h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C, Csize);
    
    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((Bcols + BLOCK_SIZE - 1) / BLOCK_SIZE, (Arows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEventRecord(start);
    matrixMultShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Arows, Acols, Bcols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Time: %.2f ms\\n", time);
    printf("Performance: %.2f GFLOP/s\\n", 
           (2.0 * size * size * size) / (time * 1e6)); // 2*n^3 operations
    
    cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);
    
    // Быстрая проверка корректности
    BASE_TYPE check_sum = 0;
    for (int i = 0; i < 10; i++) {
        check_sum += h_C[i];
    }
    printf("Quick check sum: %f\\n", check_sum);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Shared Memory Performance Scaling Test ===\\n");
    
    // Тестируем разные размеры матриц
    test_matrix_size(256);
    test_matrix_size(512);
    test_matrix_size(1024);
    test_matrix_size(2048);
    
    printf("\\n=== Scaling Analysis ===\\n");
    printf("As matrix size doubles, computation time should increase ~8x (O(n³))\\n");
    printf("but GPU utilization may improve for larger sizes.\\n");
    
    return 0;
}
