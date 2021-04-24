#pragma once
#define CUDACHECK(call) cudacheck_impl(call, __FILE__, __LINE__, __func__)


void cudacheck_impl(cudaError_t err, const char *file, const int line,
                    const char *func) {
        if (cudaSuccess != err) {
                fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n", file, line,
                        func, cudaGetErrorString(err));
                fflush(stderr);
                exit(EXIT_FAILURE);
        }
}

