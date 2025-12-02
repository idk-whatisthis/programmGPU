#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define BASE_TYPE float

// Базовая версия (глобальная память)
__global__ void matrixMultGlobal(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Acols, int Bcols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    BASE_TYPE sum = 0;
    for (int k = 0; k < Acols; k++) {
        sum += A[row * Acols + k] * B[k * Bcols + col];
    }
    
    if (row < Acols && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

// Версия с разделяемой памятью
__global__ void matrixMultShared(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Acols, int Bcols) {
    __shared__ BASE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    BASE_TYPE sum = 0;
    
    for (int tile = 0; tile < (Acols + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Загрузка тайлов в разделяемую память
        if (row < Acols && (tile * BLOCK_SIZE + threadIdx.x) < Acols) {
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
        
        // Вычисление произведения для тайла
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < Acols && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int Arows = 512, Acols = 512;
    int Brows = Acols, Bcols = 512;
    
    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);
    
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C_global = (BASE_TYPE *)malloc(Csize);
    BASE_TYPE *h_C_shared = (BASE_TYPE *)malloc(Csize);
    
    BASE_TYPE *d_A, *d_B, *d_C_global, *d_C_shared;
    
    // Инициализация матриц
    for (int i = 0; i < Arows * Acols; ++i) h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    for (int i = 0; i < Brows * Bcols; ++i) h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    
    cudaMalloc((void **)&d_A, Asize);
    cudaMalloc((void **)&d_B, Bsize);
    cudaMalloc((void **)&d_C_global, Csize);
    cudaMalloc((void **)&d_C_shared, Csize);
    
    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((Bcols + BLOCK_SIZE - 1) / BLOCK_SIZE, (Arows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Тест глобальной памяти
    printf("=== GLOBAL MEMORY VERSION ===\n");
    cudaEventRecord(start);
    matrixMultGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_global, Acols, Bcols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_global;
    cudaEventElapsedTime(&time_global, start, stop);
    printf("Time: %.2f ms\n", time_global);
    
    // Тест разделяемой памяти
    printf("=== SHARED MEMORY VERSION ===\n");
    cudaEventRecord(start);
    matrixMultShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_shared, Acols, Bcols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);
    printf("Time: %.2f ms\n", time_shared);
    
    printf("=== COMPARISON ===\n");
    printf("Global memory: %.2f ms\n", time_global);
    printf("Shared memory: %.2f ms\n", time_shared);
    printf("Speedup: %.2fx\n", time_global / time_shared);
    
    // Проверка корректности
    cudaMemcpy(h_C_global, d_C_global, Csize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_shared, d_C_shared, Csize, cudaMemcpyDeviceToHost);
    
    // Сравнение результатов
    int errors = 0;
    for (int i = 0; i < Arows * Bcols; i++) {
        if (fabs(h_C_global[i] - h_C_shared[i]) > 1e-3) {
            errors++;
            if (errors < 5) {
                printf("Mismatch at element %d: Global=%f, Shared=%f\n", i, h_C_global[i], h_C_shared[i]);
            }
        }
    }
    printf("Total errors: %d\n", errors);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    cudaFree(d_C_shared);
    free(h_A);
    free(h_B);
    free(h_C_global);
    free(h_C_shared);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
