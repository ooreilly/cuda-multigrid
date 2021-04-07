#include "definitions.cuh"

__global__ void residual_kernel(double *res, const double *u, const double *f, const int n) {

        int j = threadIdx.x + blockIdx.x * blockDim.x;
        int i = threadIdx.y + blockIdx.y * blockDim.y;

        if (j == 0 || i == 0 || j >= n - 1 || i >= n - 1) return;


        res[j + n * i] =
            (u[j - 1 + n * i] + u[j + 1 + n * i] + -4 * u[j + n * i] +
             u[j + n * (i - 1)] + u[j + n * (i + 1)]) +
            + f[j + n * i];
}


void residual_H(double *res, const double *u, const double *f, const int n) {

        dim3 threads ( 32, 4);
        dim3 blocks ( (n - 1) / threads.x + 1, (n - 1) / threads.y + 1, 1);

        residual_kernel<<<blocks, threads>>>(res, u, f, n);
        CUDACHECK(cudaGetLastError());

}

template <typename Number=double>
class ResidualCU {

        private:
                Number *res;
                const Number *u, *f;
                const int n;

        public:
                ResidualCU(Number *res, const Number *u, const Number *f, const int n)
             : res(res), u(u), f(f), n(n) {}

                void operator()(void){
                       residual_H(res, u, f, n); 
                }
};
