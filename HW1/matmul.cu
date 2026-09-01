#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// GPU kernel - one thread per output element
__global__ void matmul_gpu(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

// CPU baseline
void matmul_cpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
          sum += A[i*N+k] * B[k*N+j];
        C[i*N+j] = sum;
      }
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    size_t bytes = N * N * sizeof(float);

    // Allocate and initialize host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)rand()/RAND_MAX;
        h_B[i] = (float)rand()/RAND_MAX;
    }

    // CPU timing
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = (t1.tv_sec - t0.tv_sec)*1e3
                  + (t1.tv_nsec - t0.tv_nsec)/1e6;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t start, stop, k_start, k_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_stop);

    // Host to Device transfer
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2d_ms;
    cudaEventElapsedTime(&h2d_ms, start, stop);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Kernel timing
    cudaEventRecord(k_start);
    matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(k_stop);
    cudaEventSynchronize(k_stop);
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, k_start, k_stop);

    // Device to Host transfer
    cudaEventRecord(start);
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2h_ms;
    cudaEventElapsedTime(&d2h_ms, start, stop);

    float transfer_ms = h2d_ms + d2h_ms;
    float gpu_total = transfer_ms + kernel_ms;
    float speedup = cpu_ms / gpu_total;

    printf("N=%d | CPU=%.2fms | Kernel=%.2fms | H2D+D2H=%.2fms | Speedup=%.2fx\n",
           N, cpu_ms, kernel_ms, transfer_ms, speedup);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
