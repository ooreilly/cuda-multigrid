#pragma once

#include <grid.cuh>
#include <definitions.cuh>


template <typename T>
__global__ void cuda_gauss_seidel_red_black_kernel(
    T *u, const T *f, const int n, const T h, const int color, const int ix0 = 1,
    const int iy0 = 1, const int ixn = 1, const int iyn = 1) {
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;

        if (!inbounds(j, i, n, n, ix0, iy0, ixn, iyn)) return;

        if ((i + j) % 2 != color) return;

        u[j + i * n] = -0.25 * (h * h * f[j + i * n] - u[j + 1 + i * n] -
                                u[j - 1 + i * n] - u[j + (i + 1) * n] -
                                u[j + (i - 1) * n]);
}

template <typename T>
void cuda_gauss_seidel_red_black(T *u, const T *f, const int n, const T h) {

        dim3 threads (32, 4, 1);
        dim3 blocks ( (n - 1) / threads.x + 1, (n - 1) / threads.y  + 1, 1);
        cuda_gauss_seidel_red_black_kernel<T><<<blocks, threads>>>(u, f, n, h, 0);
        cuda_gauss_seidel_red_black_kernel<T><<<blocks, threads>>>(u, f, n, h, 1);
        CUDACHECK(cudaGetLastError());
}

template <typename T>
__global__ void cuda_poisson_residual_kernel(T *r, const T *u, const T *f,
                                             const int n, const T h,
                                             const int ix0 = 1, const int iy0 = 1,
                                             const int ixn = 1, const int jxn = 1) {
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        
        if (!inbounds(j, i, n, n, ix0, iy0, ixn, ixn)) return;

        T hi2 = 1.0 / (h * h);
        r[j + i * n] =
            f[j + i * n] -
            (u[j + 1 + i * n] + u[j - 1 + i * n] + -4.0 * u[j + i * n] +
             u[j + (i + 1) * n] + u[j + (i - 1) * n]) *
                hi2;
}

template <typename T>
__global__ void cuda_exact_solution_kernel(T *u, const int n, const T h, const T modes=1.0,
                const int bx=0, const int by=0, const int ex=0, const int ey=0) {
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        if (!inbounds(j, i, n, n, bx, by, ex, ey)) return;

        T s = 2.0 * M_PI * modes / (h * (n - 1));
        u[j + n * i] = sin(s * h * j) * sin(s * h * i);

}
template <typename T>
void cuda_exact_solution(T *u, const int n, const T h, const T modes=1.0) {
        dim3 threads (32, 4, 1);
        dim3 blocks ( (n - 1) / threads.x + 1, (n - 1) / threads.y  + 1, 1);
        cuda_exact_solution_kernel<T><<<blocks, threads>>>(u, n, h);
        CUDACHECK(cudaGetLastError());
}

template <typename T>
void cuda_poisson_residual(T *r, const T *u, const T *f, const int n, const T h) {
        dim3 threads (32, 4, 1);
        dim3 blocks ( (n - 1) / threads.x + 1, (n - 1) / threads.y  + 1, 1);
        cuda_poisson_residual_kernel<T><<<blocks, threads>>>(r, u, f, n, h);
        CUDACHECK(cudaGetLastError());
}

class CUDAGaussSeidelRedBlack {
        public:
                CUDAGaussSeidelRedBlack() { }
        template <typename P>
                CUDAGaussSeidelRedBlack(P& p) { }

        template <typename T>
        void operator()(T *u, const T *f, const int n, const T h) {
                cuda_gauss_seidel_red_black(u, f, n, h);
        }

        template <typename P>
        void operator()(P& p) {
                cuda_gauss_seidel_red_black(p.u, p.f, p.n, p.h);
        }
        const char *name() {
                return "CUDA Gauss-Seidel (red-black)";
        }

};

template <enum norm_type normt=L1NORM, typename T=double>
class CUDAPoisson {

        public:
                int n;
                int l;
                T h;
                T modes;
                T *u, *f, *r;
                size_t num_bytes;
                CUDANorm<normt> normfcn;

        CUDAPoisson(int l, T h, T modes) : l(l), h(h), modes(modes) {
                l = l;
                n = (1 << l) + 1;
                num_bytes = sizeof(T) * n * n;
                CUDACHECK(cudaMalloc((void**)&u, num_bytes));
                CUDACHECK(cudaMalloc((void**)&f, num_bytes));
                CUDACHECK(cudaMalloc((void**)&r, num_bytes));
                CUDACHECK(cudaMemset(u, 0, num_bytes));
                CUDACHECK(cudaMemset(f, 0, num_bytes));
                CUDACHECK(cudaMemset(r, 0, num_bytes));

                T *hf = (T*)malloc(num_bytes);
                forcing_function(hf, n, h, modes);
                CUDACHECK(cudaMemcpy(f, hf, num_bytes, cudaMemcpyHostToDevice));

                free(hf);
        }
        
        void residual(void) {
                cuda_poisson_residual(r, u, f, n, h);
        }

        T error(void) {
                T *v;
                CUDACHECK(cudaMalloc((void**)&v, num_bytes));
                CUDACHECK(cudaMemset(v, 0, num_bytes));
                cuda_exact_solution(v, n, h, modes);
                cuda_grid_subtract(r, u, v, n, n);
                T err = 0.0;
                err = normfcn(r, n, n, h, h);
                CUDACHECK(cudaFree(v));
                return err;
        }

        T norm(void) {
                T r_norm = normfcn(r, n, n, h, h);
                return r_norm;
        }

        ~CUDAPoisson() {
                CUDACHECK(cudaFree(u));
                CUDACHECK(cudaFree(f));
                CUDACHECK(cudaFree(r));
        }



};
