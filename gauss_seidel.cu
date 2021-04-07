#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>
#include <chrono>

#include "definitions.hpp"
#include "mms.hpp"
#include "residual.hpp"
#include "norm.hpp"
#include "gauss_seidel.hpp"
#include "solver.hpp"

#if USE_GPU
#include "gauss_seidel.cuh"
#include "residual.cuh"
#include "norm.cuh"
#endif


int main(int argc, char **argv) {

        using Number=double;

        int stride = 100;
        int n = 1 << 10;
        int max_iter = 1000;
        double eps = 1e-12;
        size_t num_bytes = n * n * sizeof(double);
        double *f = (double*)malloc(num_bytes);
        double *u = (double*)malloc(num_bytes);
        double *r = (double*)malloc(num_bytes);
        double h = 1.0 / n;
        double err = 0.0;


        memset(f, 0, num_bytes);
        memset(u, 0, num_bytes);
        memset(r, 0, num_bytes);
        


        init_forcing(f, n, h);

        // Scale norm by 1 / h^2 since the forcing function contains the scaling by h^2 (not the
        // difference operators)
        double fnorm = l1norm(f, n, h) / h / h;


        printf("Problem size: %d x %d \n", n, n);
        printf("|f|_1 = %g \n", fnorm);
        printf("Threshold: r|f|_1 = %g, r = %g \n", fnorm * eps, eps);


#if USE_CPU
        GaussSeidel<Number> gs(u, f, n);
        Residual<Number> res(r, u, f, n);
        L1Norm<Number> norm(r, n, h);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        solve(gs, res, norm, fnorm, max_iter, eps, stride);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> elapsed = t2 - t1;
        printf("Solver took: %g ms\n", elapsed.count());

        printf("Checking solution...\n");
        exact_solution(r, n,  h);
        err = error(r, u, n, h);
        printf("Error: %g \n", err);
#endif



#if USE_GPU

        Number *d_u, *d_f, *d_r;

        cudaMalloc((void**)&d_u, num_bytes);
        cudaMalloc((void**)&d_f, num_bytes);
        cudaMalloc((void**)&d_r, num_bytes);

        cudaMemset(d_u, 0, num_bytes);
        cudaMemset(d_r, 0, num_bytes);
        cudaMemcpy(d_f, f, num_bytes, cudaMemcpyHostToDevice);

        GaussSeidelCU<Number> d_gs(d_u, d_f, n);
        ResidualCU<Number> d_res(d_r, d_u, d_f, n);
        L1NormCU<Number> d_norm(d_r, n, h);

        solve(d_gs, d_res, d_norm, fnorm, max_iter, eps, stride);

        cudaFree(d_f);
        cudaFree(d_u);
        cudaFree(d_r);

#endif

#if USE_CPU
        free(f);
        free(u);
        free(r);
#endif


}
