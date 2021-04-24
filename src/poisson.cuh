#pragma once

#include "definitions.cuh"

__device__ __inline__ bool inbounds(const int i, const int j, const int n,
                                    const int i0, const int j0, const int in,
                                    const int jn) {

        if (i < i0 || j < j0) return false; 
        if (i > n - 1 - in || j > n - 1 - jn) return false; 
        return true;
}

template <typename T>
__global__ void cuda_gauss_seidel_red_black_kernel(
    T *u, const T *f, const int n, const T h, const int color, const int i0 = 1,
    const int j0 = 1, const int in = 1, const int jn = 1) {
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;

        if (!inbounds(i, j, n, i0, j0, in, jn)) return;

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
}

template <typename T>
__global__ void cuda_poisson_residual_kernel(T *r, const T *u, const T *f,
                                             const int n, const T h,
                                             const int i0 = 1, const int j0 = 1,
                                             const int in = 1,
                                             const int jn = 1) {
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        
        if (!inbounds(i, j, n, i0, j0, in, jn)) return;

        T hi2 = 1.0 / (h * h);
        r[j + i * n] =
            f[j + i * n] -
            (u[j + 1 + i * n] + u[j - 1 + i * n] + -4.0 * u[j + i * n] +
             u[j + (i + 1) * n] + u[j + (i - 1) * n]) *
                hi2;
}

template <typename T>
void cuda_poisson_residual(T *r, const T *u, const T *f, const int n, const T h) {
        dim3 threads (32, 4, 1);
        dim3 blocks ( (n - 1) / threads.x + 1, (n - 1) / threads.y  + 1, 1);
        cuda_poisson_residual_kernel<T><<<blocks, threads>>>(r, u, f, n, h);
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

template <typename T>
class CUDAPoisson {

        public:
                int n;
                int l;
                T h;
                T modes;
                T *u, *f, *r;
                size_t num_bytes;

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
                T *v = (T*)malloc(num_bytes);
                T *hr = (T*)malloc(num_bytes);
                T *hu = (T*)malloc(num_bytes);
                memset(v, 0, num_bytes);
                exact_solution(v, n, h, modes);
                cudaMemcpy(hr, r, num_bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(hu, u, num_bytes, cudaMemcpyDeviceToHost);
                grid_subtract(hr, hu, v, n, n);
                T err = grid_l1norm(hr, n, n, h);
                free(v);
                return err;
        }

        T norm(void) {
                T *hr = (T*)malloc(num_bytes);
                cudaMemcpy(hr, r, num_bytes, cudaMemcpyDeviceToHost);
                T r_norm = grid_l1norm(hr, n, n, h);
                free(hr);

                return r_norm;
        }

        ~CUDAPoisson() {
                CUDACHECK(cudaFree(u));
                CUDACHECK(cudaFree(f));
                CUDACHECK(cudaFree(r));
        }



};
