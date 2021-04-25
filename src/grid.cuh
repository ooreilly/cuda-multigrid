#pragma once
#include <definitions.cuh>

enum norm_type {L1NORM, L2NORM};

__host__ __device__ __inline__ bool inbounds(const int ix, const int iy, const int nx, const int ny, 
                                    const int ix0, const int iy0, const int ixn,
                                    const int iyn) {

        if (ix < ix0 || iy < iy0) return false; 
        if (ix > nx - 1 - ixn || iy > ny - 1 - iyn) return false; 
        return true;
}

template <typename T>
__global__ void cuda_grid_x_kernel(T *x, const int nx, const int ny, const T h) {
        int ix = threadIdx.x + blockDim.x * blockIdx.x;
        int iy = threadIdx.y + blockDim.y * blockIdx.y;

        if (ix >= nx || iy >= ny) return;

        x[ix + nx * iy] = ix * h;

}

template <typename T>
void cuda_grid_x(T *x, const int nx, const int ny, const T h) {
        dim3 threads (32, 4, 1);
        dim3 blocks ( (nx - 1) / threads.x + 1, (ny - 1) / threads.y  + 1, 1);
        cuda_grid_x_kernel<T><<<blocks, threads>>>(x, nx, ny, h);
        CUDACHECK(cudaGetLastError());
}

template <typename T>
__global__ void cuda_grid_y_kernel(T *x, const int nx, const int ny, const T h) {
        int ix = threadIdx.x + blockDim.x * blockIdx.x;
        int iy = threadIdx.y + blockDim.y * blockIdx.y;

        if (ix >= nx || iy >= ny) return;

        x[ix + nx * iy] = iy * h;

}

template <typename T>
void cuda_grid_y(T *x, const int nx, const int ny, const T h) {
        dim3 threads (32, 4, 1);
        dim3 blocks ( (nx - 1) / threads.x + 1, (ny - 1) / threads.y  + 1, 1);
        cuda_grid_y_kernel<T><<<blocks, threads>>>(x, nx, ny, h);
        CUDACHECK(cudaGetLastError());
}

template<typename T>
__global__ void cuda_grid_restrict_kernel(T *yc, const int nxc, const int nyc, const T *xf, const int nxf, const int nyf, const T a = 0.0, const T b=1.0,
                const int ix0=1, const int iy0=1, const int ixn=1, const int iyn=1) {
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        const T c0 = 0.25;
        const T c1 = 0.5;
        if (!inbounds(j, i, nxc, nyc, ix0, iy0, ixn, iyn)) return;
        yc[j + nxc * i] = a * yc[j + nxc * i] +
                          b * (c0 * c0 * xf[2 * j - 1 + nxf * (2 * i - 1)] +
                               c0 * c1 * xf[2 * j + nxf * (2 * i - 1)] +
                               c0 * c0 * xf[2 * j + 1 + nxf * (2 * i - 1)] +
                               +c1 * c0 * xf[2 * j - 1 + nxf * (2 * i + 0)] +
                               c1 * c1 * xf[2 * j + nxf * (2 * i + 0)] +
                               c1 * c0 * xf[2 * j + 1 + nxf * (2 * i + 0)] +
                               +c0 * c0 * xf[2 * j - 1 + nxf * (2 * i + 1)] +
                               c0 * c1 * xf[2 * j + nxf * (2 * i + 1)] +
                               c0 * c0 * xf[2 * j + 1 + nxf * (2 * i + 1)]);
}

template<typename T>
void cuda_grid_restrict(T *yc, const int nxc, const int nyc, const T *xf, const int nxf, const int nyf, const T a = 0.0, const T b=1.0,
                const int i0=1, const int j0=1, const int in=1, const int jn=1) {
        assert(nxf == 2 * (nxc - 1) + 1);
        dim3 threads (32, 4, 1);
        dim3 blocks ( (nxc - 1) / threads.x + 1, (nyc - 1) / threads.y  + 1, 1);
        cuda_grid_restrict_kernel<T><<<blocks, threads>>>(yc, nxc, nyc, xf, nxf, nyf, a, b, i0, j0, in, jn);
        CUDACHECK(cudaGetLastError());
}

template <typename T>
__global__ void cuda_grid_prolongate_kernel(T *yf, const int nxf, const int nyf,
                                 const T *xc, const int nxc, const int nyc,
                                 const T a = 0.0, const T b = 1.0,
                                 const int bx = 0, const int by = 0,
                                 const int ex = 0, const int ey = 0) {
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        if (!inbounds(j, i, nxc, nyc, bx, by, ex, ey)) return;

        yf[2 * j + nxf * 2 * i] =
            a * yf[2 * j + nxf * 2 * i] + b * xc[j + nxc * i];
        if (j < nxc - 1)
                yf[2 * j + 1 + nxf * 2 * i] =
                    a * yf[2 * j + 1 + nxf * 2 * i] +
                    0.5 * b *
                        (xc[j + nxc * i] + xc[j + 1 + nxc * i]);
        if (i < nxc - 1)
                yf[2 * j + nxf * (2 * i + 1)] =
                    a * yf[2 * j + nxf * (2 * i + 1)] +
                    0.5 * b *
                        (xc[j + nxc * i] +
                         xc[j + nxc * (i + 1)]);
        if (i < nxc - 1 && j < nxc - 1)
                yf[2 * j + 1 + nxf * (2 * i + 1)] =
                    + a * yf[2 * j + 1 + nxf * (2 * i + 1)] +
                    0.25 * b *
                        (xc[j + nxc * i] +
                         xc[j + nxc * (i + 1)] +
                         xc[j + 1 + nxc * i] +
                         xc[j + 1 + nxc * (i + 1)]);
}

template <typename T>
void cuda_grid_prolongate(T *yf, const int nxf, const int nyf, const T *xc,
                          const int nxc, const int nyc, const T a = 0.0,
                          const T b = 1.0, const int bx = 0, const int by = 0,
                          const int ex = 0, const int ey = 0) {
        assert(nxf == 2 * (nxc - 1) + 1);
        dim3 threads (32, 4, 1);
        dim3 blocks ( (nxc - 1) / threads.x + 1, (nyc - 1) / threads.y  + 1, 1);
        cuda_grid_prolongate_kernel<T><<<blocks, threads>>>(yf, nxf, nyf, xc, nxc, nyc, a, b, bx, by, ex, ey);
        CUDACHECK(cudaGetLastError());


}

template<typename T>
__global__ void cuda_grid_subtract_kernel(T *z, const T *x, const T *y, const int nx, const int ny, 
                const int ix0=0, const int iy0=0, const int ixn=0, const int iyn=0) {
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        if (!inbounds(j, i, nx, ny, ix0, iy0, ixn, iyn)) return;
        z[j + nx * i] = x[j + nx * i] - y[j + nx * i];
}

template <typename T>
void cuda_grid_subtract(T *z, const T *x, const T *y, const int nx,
                        const int ny, const int i0 = 0, const int j0 = 0,
                        const int in = 0, const int jn = 0) {
        dim3 threads (32, 4, 1);
        dim3 blocks ( (nx - 1) / threads.x + 1, (ny - 1) / threads.y  + 1, 1);
        cuda_grid_subtract_kernel<T><<<blocks, threads>>>(z, x, y, nx, ny);
}

template <enum norm_type norm, typename T=double>
__global__ void norm_kernel(T *temp, const T *u, const int N) {

                const int warpSize = 32;
                int numThreads = blockDim.x;
                int numValuesPerBlockPerThread = (N - 1) / gridDim.x / numThreads + 1;
                int numValuesPerBlock = numValuesPerBlockPerThread * numThreads;
                int idx = threadIdx.x + blockIdx.x * numValuesPerBlock;
                int warpID = threadIdx.x / warpSize;
                int lane = threadIdx.x % warpSize;
                int numWarps = numThreads / warpSize;

                double partialSum = 0.0;
                __shared__ T sPartialSum[1024];
                int end = idx + numValuesPerBlock;
                int iEnd = end > N ? N : end;

                for (int i = idx; i < iEnd; i += numThreads) {
                        switch (norm) {
                                case L1NORM:
                                        partialSum += fabs(u[i]);
                                        break;
                                case L2NORM:
                                        partialSum += u[i] * u[i];
                                        break;
                        }
                }

                T val = partialSum;
                for (int i = 16; i > 0; i /= 2)
                        val += __shfl_down_sync(0xffffffff, val, i);

                if (lane == 0) sPartialSum[warpID] = val;

                __syncthreads();

                if (lane == 0 && warpID == 0) {
                        double blockSum = 0.0;
                        for (int i = 0; i < numWarps; ++i) {
                                blockSum += sPartialSum[i];
                        }

                        atomicAdd(temp, blockSum);
                }
}


template <enum norm_type norm, typename T=double>
T norm_H(T *tmp, const T *u, const int n, const int device=0) {

        double out[1] = {0.0};

        cudaMemset(tmp, 0, sizeof(T));

        int numSM = 0;
        const int threads = 128;
        cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);
        norm_kernel<norm><<<numSM, threads>>>(tmp, u, n);
        CUDACHECK(cudaGetLastError());
        cudaMemcpy(out, tmp, sizeof(T), cudaMemcpyDeviceToHost);
        CUDACHECK(cudaGetLastError());

        return out[0];
}

template <enum norm_type norm, typename T=double>
class CUDANorm {
        private:
                T *sum;
        public:
                CUDANorm() {
                        CUDACHECK(cudaMalloc((void**)&sum, sizeof(T)));
                }

                //TODO: Add bounds options to norm
                T operator()(const T *u, const int nx, const int ny,
                             const T hx, const T hy, const int device=0) {
                        CUDACHECK(cudaMemset(sum, 0, sizeof(T)));
                        return norm_H<norm>(sum, u, nx * ny) * hx * hy;
                }

                ~CUDANorm() {
                        CUDACHECK(cudaFree(sum));
                }

};
