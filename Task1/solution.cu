#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <assert.h>

__global__ void jacobiKernel(float* A, float* b, int N, float* x_now, float* x_next)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float sum = 0.0;
        int idx_Ai = N * idx;
        for (int j = 0; j < N; ++j)
            if (idx != j)
                sum += A[idx_Ai + j] * x_now[j];

        x_next[idx] = (b[idx] - sum) / A[idx_Ai + idx];
    }
}

void parse_argv(int argc, char *argv[], char **fname, int *iter, int *blockSize)
{
    static struct option long_options[] =
    {
        {"file",       required_argument, NULL, 'f'},
        {"iterations", optional_argument, NULL, 'i'},
        {"blockSize",  optional_argument, NULL, 'b'},
        {NULL, 0, NULL, 0}
    };

    int ch = 0;
    while ((ch = getopt_long(argc, argv, "f:i:b:", long_options, NULL)) != -1) {
        switch (ch) {
             case 'f' : *fname = optarg;
                 break;
             case 'i' : *iter = atoi(optarg);
                 break;
             case 'b' : *blockSize = atoi(optarg);
                 break;
             default:
                 abort();
        }
    }
}

int main(int argc, char *argv[])
{
    int N, i, iter = 10000, blockSize = 512;
    char *fname = NULL;

    parse_argv(argc, argv, &fname, &iter, &blockSize);

    FILE *file = fopen(fname, "r");
    if (file == NULL)
        exit(EXIT_FAILURE);

    fscanf(file, "%d", &N);
    printf("N = %d, iter = %d, blocksize = %d\n", N, iter, blockSize);

    float *A = (float *)calloc(N * N, sizeof(float));
    float *b = (float *)calloc(N, sizeof(float));
    float *x = (float *)calloc(N, sizeof(float));

    assert(A != NULL);
    assert(b != NULL);
    assert(x != NULL);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fscanf(file, "%f", &A[N * i + j]);
        }
        fscanf(file, "%f", &b[i]);
    }

    float *x0_d, *x1_d, *A_d, *b_d;
    assert(cudaSuccess == cudaMalloc((void **) &A_d,  N * N * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &b_d,      N * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &x0_d,     N * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &x1_d,     N * sizeof(float)));

    cudaMemcpy(A_d, A,  sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b,  sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x0_d, x, sizeof(float) * N, cudaMemcpyHostToDevice);

    int nBlocks = (N + blockSize - 1) / blockSize;

    printf("Running Jacobi method...\n");
    for (i = 0; i < iter; ++i)
    {
        float *xnext = (i % 2 ? x0_d : x1_d);
        float *xnow  = (i % 2 ? x1_d : x0_d);
        jacobiKernel <<< nBlocks, blockSize >>> (A_d, b_d, N, xnow, xnext);
    }
    cudaMemcpy(x, (iter % 2 ? x1_d : x0_d), sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(A_d); cudaFree(b_d); cudaFree(x0_d); cudaFree(x1_d);
    free(A); free(b);

    printf("\nResult after %d iterations:\n", iter);
    for (i = 0; i < N; i++)
        printf("x[%d] = %f\n", i, x[i]);

    return 0;
}
