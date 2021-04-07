#include "definitions.cuh"

__global__ void l1norm_kernel(double *temp, const double *u, const double h, const int N) {

                const int warpSize = 32;
                int numThreads = blockDim.x;
                int numValuesPerBlockPerThread = (N - 1) / gridDim.x / numThreads + 1;
                int numValuesPerBlock = numValuesPerBlockPerThread * numThreads;
                int idx = threadIdx.x + blockIdx.x * numValuesPerBlock;
                int warpID = threadIdx.x / warpSize;
                int lane = threadIdx.x % warpSize;
                int numWarps = numThreads / warpSize;

                double partialSum = 0.0;
                __shared__ double sPartialSum[1024];
                int end = idx + numValuesPerBlock;
                int iEnd = end > N ? N : end;
                for (int i = idx; i < iEnd; i += numThreads) {
                        partialSum += fabs(u[i]) * h * h;
                }

                double val = partialSum;
                for (int i = 16; i > 0; i /= 2)
                        val += __shfl_down_sync(0xffffffff, val, i);

                if (lane == 0) sPartialSum[warpID] = val;

                __syncthreads();

                if (lane == 0 && warpID == 0) {
                        double blockSum = 0.0;
                        for (int i = 0; i < numWarps; ++i) {
                                blockSum += sPartialSum[i];
                        }

                        double val = atomicAdd(temp, blockSum);
                }
}


double l1norm_H(double *tmp, double *u, const int n, const double h, int device=0) {

        double out[1] = {0.0};

        cudaMemset(tmp, 0, sizeof(double));

        int numSM = 0;
        const int threads = 128;
        cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device);
        l1norm_kernel<<<numSM, threads>>>(tmp, u, h, n * n);
        CUDACHECK(cudaGetLastError());
        cudaMemcpy(out, tmp, sizeof(double), cudaMemcpyDeviceToHost);
        CUDACHECK(cudaGetLastError());

        return out[0];

}

