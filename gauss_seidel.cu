#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

#include "mms.hpp"
#include "residual.hpp"
#include "residual.cuh"
#include "norm.hpp"
#include "norm.cuh"
#include "gauss_seidel.hpp"
#include "gauss_seidel.cuh"

#define USE_CPU 0
#define EPSILON 1e-12

int main(int argc, char **argv) {

        int n = 1 << 12;
        int max_iter = 10;
        size_t num_bytes = n * n * sizeof(double);
        double *f = (double*)malloc(num_bytes);
        double *u = (double*)malloc(num_bytes);
        double *res = (double*)malloc(num_bytes);
        double h = 1.0 / n;
        double err = 0.0;


        memset(f, 0, num_bytes);
        memset(u, 0, num_bytes);
        memset(res, 0, num_bytes);


        init_forcing(f, n, h);

        double r = l1norm(f, n, h);


        printf("Problem size: %d x %d \n", n, n);
        printf("|f|_1 = %g \n", r);
        printf("Threshold: r|f|_1 = %g, r = %g \n", r * EPSILON, EPSILON);

        printf("Iteration \t Residual \n");
        double res_norm = 0.0;
        int iter = 0;


        int stride = 1000;
#if USE_CPU
        do {
                gauss_seidel(u, f, n);
                residual(res, u, f, n);
                res_norm = l1norm(res, n, h);
                if (iter % stride == 0)
                printf("%-7d \t %-7.4g  \n", iter, res_norm);
                iter++;
        } while ( res_norm > r * EPSILON && iter < max_iter);
        printf("Residual: %g achieved after %d iterations. \n", res_norm, iter);
        printf("Checking solution...\n");
        exact_solution(res, n,  h);
        err = error(res, u, n, h);
        printf("Error: %g \n", err);
#endif


        double *d_u, *d_f, *d_res, *d_sum;

        cudaMalloc((void**)&d_u, num_bytes);
        cudaMalloc((void**)&d_f, num_bytes);
        cudaMalloc((void**)&d_res, num_bytes);
        cudaMalloc((void**)&d_sum, sizeof(double));

        cudaMemset(d_u, 0, num_bytes);
        cudaMemset(d_res, 0, num_bytes);
        cudaMemcpy(d_f, f, num_bytes, cudaMemcpyHostToDevice);

        iter = 0;
        do {
                gauss_seidel_H(d_u, d_f, n);
                residual_H(d_res, d_u, d_f, n);
                res_norm = l1norm_H(d_sum, d_res, n, h);
                if (iter % stride == 0)
                printf("%-7d \t %-7.4g  \n", iter, res_norm);
                iter++;
        } while ( res_norm > r * EPSILON && iter < max_iter);
        printf("Residual: %g achieved after %d iterations. \n", res_norm, iter);
        printf("Checking solution...\n");
        exact_solution(res, n,  h);
        cudaMemcpy(u, d_u, num_bytes, cudaMemcpyDeviceToHost);
        

        err = error(res, u, n, h);
        printf("Error: %g \n", err);

        free(f);
        free(u);
        free(res);
        cudaFree(d_f);
        cudaFree(d_u);
        cudaFree(d_res);

}
