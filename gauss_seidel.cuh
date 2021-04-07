#include "definitions.cuh"

__global__ void gauss_seidel_kernel(double *u, const double *f, const int n, const int color) {

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        if (idx == 0 || idy == 0 || idx >= n - 1 || idy >= n - 1) return;

        int j = idx;
        int i = idy;

        if ( (i % 2 != j % 2 && color == 0) || (i % 2 == j % 2 && color == 1)) {
                u[j + n * i] = 
                    0.25 * (u[j - 1 + n * i] + u[j + 1 + n * i] + u[j + n * (i - 1)] +
                            u[j + n * (i + 1)] + f[j + n * i]);
        }

}


void gauss_seidel_H(double *u, const double *f, const int n) {

        dim3 threads ( 32, 4);
        dim3 blocks ( (n - 1) / threads.x + 1, (n - 1) / threads.y + 1);

        gauss_seidel_kernel<<<blocks, threads>>>(u, f, n, 0);
        CUDACHECK(cudaGetLastError());
        gauss_seidel_kernel<<<blocks, threads>>>(u, f, n, 1);
        CUDACHECK(cudaGetLastError());

}


template <typename Number=double>
class GaussSeidelCU {
        private:
                Number *u;
                const Number *f;
                int n;

        public:
         GaussSeidelCU(Number *u, const Number *f, const int n)
             : u(u), f(f), n(n) {}

         void operator()(void) {
                gauss_seidel_H(u, f, n);
         }

         const char* name(void) {return "Gauss-Seidel (CUDA)";}
};
