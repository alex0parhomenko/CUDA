#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <utility>

using namespace std;

__global__ void jacobiKernel(float* A, float* b, int matrix_size, float* x_prev, float* x_now)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < matrix_size)
    {
        float sum = b[thread_idx];
        int A_idx = matrix_size * thread_idx;
        for (int i = 0; i < matrix_size; ++i) 
            sum -= thread_idx != i ?  A[A_idx + i] * x_prev[i] : 0;

        x_now[thread_idx] = sum / A[A_idx + thread_idx];
    }
}


int main(int argc, char *argv[])
{
    FILE* matrix_file = fopen(argv[1], "r");
    int matrix_size = 0, iter_count = 20000;
    fscanf(matrix_file, "%d", &matrix_size);
    float* A = new float[matrix_size * matrix_size];
    float* b = new float[matrix_size];
    float* x = new float[matrix_size];
    float* A_cuda, *b_cuda, *x_prev_cuda, *x_now_cuda;
    assert( cudaSuccess == cudaMalloc((void **) &A_cuda,  matrix_size * matrix_size * sizeof(float)));
    assert( cudaSuccess == cudaMalloc((void **) &b_cuda, matrix_size * sizeof(float)));
    assert( cudaSuccess == cudaMalloc((void **) &x_prev_cuda, matrix_size * sizeof(float)));
    assert( cudaSuccess == cudaMalloc((void **) &x_now_cuda, matrix_size * sizeof(float)));   

    for (int i = 0; i < matrix_size; i++) {
        x[i] = 0;
        for (int j = 0; j < matrix_size; j++) {
            fscanf(matrix_file, "%f", &A[i* matrix_size + j]); 
        }
        fscanf(matrix_file, "%f", &b[i]);
    }
    
    cudaMemcpy(A_cuda, A, sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(x_prev_cuda, x, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(x_now_cuda, x, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
    int blocks_count = 32;
    int threads_count = matrix_size / blocks_count + 1; 
 
    for (int i = 0; i < iter_count; i++) {
        jacobiKernel<<< blocks_count, threads_count >>>(A_cuda, b_cuda, matrix_size, x_prev_cuda, x_now_cuda);
        
        swap(x_prev_cuda, x_now_cuda);
    }
    cudaMemcpy(x, x_prev_cuda, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < matrix_size; i++) {
        printf("%f ", x[i]); 
    }
    cudaFree(A_cuda); 
    cudaFree(b_cuda); 
    cudaFree(x_prev_cuda);
    cudaFree(x_now_cuda);
    delete[] A;
    delete[] b;
    delete[] x; 
    return 0;
}
