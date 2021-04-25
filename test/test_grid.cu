#include <stdio.h>
#include <grid.hpp>
#include <grid.cuh>
#include <assertions.hpp>
#include <definitions.cuh>


template <typename T>
int test_gridpoints(const int nx, const int ny, const T h) {

        printf("Testing grid points generation with nx = %d ny = %d and h = %f \n", nx, ny, h);
        size_t num_bytes = nx * ny * sizeof(T);
        T *x = (T*)malloc(num_bytes);

        grid_x(x, nx, ny, h);

        approx(x[0], 0.0);
        approx(x[1], h);
        approx(x[nx-1], h * (nx - 1));
        approx(x[(ny - 1) * nx - 1], h * (nx - 1));
        approx(x[(ny - 1) * nx], 0.0);

        
        grid_y(x, nx, ny, h);

        approx(x[0], 0.0);
        approx(x[1], 0.0);
        approx(x[nx-1], 0.0);

        approx(x[nx], h);
        approx(x[nx * (ny - 1)], (ny - 1) * h);

        free(x);

        return test_report();
}

template <typename T>
int test_gridnorm(const int nx, const int ny) {
        T hx = 1.0 / (nx - 1);
        T hy = 1.0 / (ny - 1);
        printf(
            "Testing L1 and L2 norm on grid with nx = %d ny = %d and hx = %f hy = %f "
            "\n",
            nx, ny, hx, hy);
        size_t num_bytes = nx * ny * sizeof(T);
        T *x = (T*)malloc(num_bytes);

        for (int i = 0; i < nx * ny; ++i)
                x[i] = 1.0;

        T norm = grid_l1norm(x, nx, ny, hx, hy, 0, 0, 0, 0);
        approx(norm, nx * ny * hx * hy);
        
        norm = grid_l1norm(x, nx, ny, hx, hy, 1, 0, 0, 0);
        approx(norm, (nx - 1) * ny * hx * hy);
        
        norm = grid_l1norm(x, nx, ny, hx, hy, 0, 1, 0, 0);
        approx(norm, nx * (ny - 1) * hx * hy);

        norm = grid_l1norm(x, nx, ny, hx, hy, 0, 0, 1, 1);
        approx(norm, (nx - 1) * (ny - 1) * hx * hy);

        norm = grid_l2norm(x, nx, ny, hx, hy, 0, 0, 0, 0);
        approx(norm, nx * ny * hx * hy);
        
        norm = grid_l2norm(x, nx, ny, hx, hy, 1, 0, 0, 0);
        approx(norm, (nx - 1) * ny * hx * hy);
        
        norm = grid_l2norm(x, nx, ny, hx, hy, 0, 1, 0, 0);
        approx(norm, nx * (ny - 1) * hx * hy);

        // Extract the second last point in the corner
        grid_x(x, nx, ny, hx);
        norm = grid_l2norm(x, nx, ny, hx, hy, nx - 2, ny - 2, 1, 1);
        approx(norm, (1 - hx) * (1 - hx) * hx * hy);

        return test_report();
}

template <typename T>
int cuda_test_gridnorm(const int nx, const int ny) {
        T hx = 1.0 / (nx - 1);
        T hy = 1.0 / (ny - 1);
        printf(
            "CUDA Testing L1 and L2 norm on grid with nx = %d ny = %d and hx = %f hy = %f "
            "\n",
            nx, ny, hx, hy);
        size_t num_bytes = nx * ny * sizeof(T);
        T *x = (T*)malloc(num_bytes);

        for (int i = 0; i < nx * ny; ++i)
                x[i] = 1.0;

        T *d_x;
        CUDACHECK(cudaMalloc((void**)&d_x, num_bytes));
        CUDACHECK(cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice));

        CUDANorm<L1NORM, T> l1norm;
        T norm = l1norm(d_x, nx, ny, hx, hy);
        approx(norm, nx * ny * hx * hy);

        CUDANorm<L2NORM, T> l2norm;
        norm = l2norm(d_x, nx, ny, hx, hy);
        approx(norm, nx * ny * hx * hy);
        
        CUDACHECK(cudaFree(d_x));

        return test_report();
}



template <typename T>
int cuda_test_gridpoints(const int nx, const int ny, const T h) {

        printf("CUDA Testing grid points generation with nx = %d ny = %d and h = %f \n", nx, ny, h);
        size_t num_bytes = nx * ny * sizeof(T);
        T *d_x, *x;
        cudaMalloc((void**)&d_x, num_bytes);
        x = (T*)malloc(num_bytes);

        cuda_grid_x(d_x, nx, ny, h);

        CUDACHECK(cudaMemcpy(x, d_x, num_bytes, cudaMemcpyDeviceToHost));

        approx(x[0], 0.0);
        approx(x[1], h);
        approx(x[nx-1], h * (nx - 1));
        approx(x[(ny - 1) * nx - 1], h * (nx - 1));
        approx(x[(ny - 1) * nx], 0.0);

        cuda_grid_y(d_x, nx, ny, h);
        
        CUDACHECK(cudaMemcpy(x, d_x, num_bytes, cudaMemcpyDeviceToHost));

        approx(x[0], 0.0);
        approx(x[1], 0.0);
        approx(x[nx-1], 0.0);

        approx(x[nx], h);
        approx(x[nx * (ny - 1)], (ny - 1) * h);

        free(x);
        CUDACHECK(cudaFree(d_x));

        return test_report();
}

template <typename T>
void restriction(const char *axis, const T *xc, T *yc, T *zc, const int nxc,
                 const int nyc, const T hc, const T *xf, const int nxf, const int nyf, const T hf) {
        grid_restrict(yc, nxc, nyc, xf, nxf, nyf);
        grid_subtract(zc, xc, yc, nxc, nyc);


        T l1_err = grid_l1norm(zc, nxc, nyc, hc, hc, 1, 1, 1, 1);
        T l2_err = grid_l2norm(zc, nxc, nyc, hc, hc, 1, 1, 1, 1);

        approx(l1_err, 0.0);
        approx(l2_err, 0.0);

        printf("Restriction in %s l1-error: %g, l2-error: %g \n", axis, l1_err, l2_err);
}


template <typename T>
void prolongation(const char *axis, const T *xf, T *yf, T *zf, const int nxf,
                 const int nyf, const T hf, const T *xc, const int nxc, const int nyc, const T hc) {
        grid_prolongate(yf, nxf, nyf, xc, nxc, nyc);
        grid_subtract(zf, xf, yf, nxf, nyf);

        T l1_err = grid_l1norm(zf, nxf, nyf, hf, hf);
        T l2_err = grid_l2norm(zf, nxf, nyf, hf, hf);

        approx(l1_err, 0.0);
        approx(l2_err, 0.0);

        printf("Prolongation in %s l1-error: %g, l2-error: %g \n", axis, l1_err, l2_err);
}

template <typename T>
void restriction_prolongation_info(const int nxf, const int nyf, const T hf, const int nxc,
                                   const int nyc, const T hc) {
        printf(
            "Testing grid restriction and prolongation with fine grid "
            "[%d %d], hf=%g, and coarse grid [%d %d], hc=%g. \n",
            nxf, nyf, hf, nxc, nyc, hc);

}

template <typename T>
int test_restriction_prolongation(const int nxc, const int nyc, const T hc) {

        T hf = 0.5 * hc;
        int nxf = 2 * (nxc - 1) + 1;
        int nyf = 2 * (nyc - 1) + 1;
        restriction_prolongation_info(nxf, nyf, hf, nxc, nyc, hc);

        size_t num_bytesf = sizeof(T) * nxf * nyf;
        size_t num_bytesc = sizeof(T) * nxc * nyc;

        T *xf = (T*)malloc(num_bytesf);
        T *yf = (T*)malloc(num_bytesf);
        T *zf = (T*)malloc(num_bytesf);
        T *xc = (T*)malloc(num_bytesc);
        T *yc = (T*)malloc(num_bytesc);
        T *zc = (T*)malloc(num_bytesc);

        // Test restriction and prolongation in the x-direction
        grid_x(xf, nxf, nyf, hf);
        grid_x(xc, nxc, nyc, hc);
        restriction("x", xc, yc, zc, nxc, nyc, hc, xf, nxf, nyf, hf);
        prolongation("x", xf, yf, zf, nxf, nyf, hf, xc, nxc, nyc, hc);

        // Test restriction and prolongation in the y-direction
        grid_y(xf, nxf, nyf, hf);
        grid_y(xc, nxc, nyc, hc);
        restriction("y", xc, yc, zc, nxc, nyc, hc, xf, nxf, nyf, hf);
        prolongation("y", xf, yf, zf, nxf, nyf, hf, xc, nxc, nyc, hc);

        free(xf);
        free(yf);
        free(zf);

        free(xc);
        free(yc);
        free(zc);

        return test_report();
}

template <typename T>
void cuda_restriction(const char *axis, const T *xc, T *yc, T *zc, const int nxc,
                 const int nyc, const T hc, const T *xf, const int nxf, const int nyf, const T hf) {

        size_t num_bytes = sizeof(T) * nxc * nyc;
        cuda_grid_restrict(yc, nxc, nyc, xf, nxf, nyf);
        cuda_grid_subtract(zc, xc, yc, nxc, nyc);

        T *hzc = (T*)malloc(num_bytes);
        CUDACHECK(cudaMemcpy(hzc, zc, num_bytes, cudaMemcpyDeviceToHost));

        //TODO: Compute norms on device once there's support for bounds control
        T l1_err = grid_l1norm(hzc, nxc, nyc, hc, hc, 1, 1, 1, 1);
        T l2_err = grid_l2norm(hzc, nxc, nyc, hc, hc, 1, 1, 1, 1);

        approx(l1_err, 0.0);
        approx(l2_err, 0.0);

        printf("CUDA Restriction in %s l1-error: %g, l2-error: %g \n", axis, l1_err, l2_err);
}

template <typename T>
void cuda_prolongation(const char *axis, const T *xf, T *yf, T *zf, const int nxf,
                 const int nyf, const T hf, const T *xc, const int nxc, const int nyc, const T hc) {

        size_t num_bytes = sizeof(T) * nxf * nyf;
        cuda_grid_prolongate(yf, nxf, nyf, xc, nxc, nyc);
        cuda_grid_subtract(zf, xf, yf, nxf, nyf);

        T *hzf = (T*)malloc(num_bytes);
        CUDACHECK(cudaMemcpy(hzf, zf, num_bytes, cudaMemcpyDeviceToHost));

        //TODO: Compute norms on device once there's support for bounds control
        T l1_err = grid_l1norm(hzf, nxf, nyf, hf, hf, 1, 1, 1, 1);
        T l2_err = grid_l2norm(hzf, nxf, nyf, hf, hf, 1, 1, 1, 1);

        approx(l1_err, 0.0);
        approx(l2_err, 0.0);

        printf("CUDA Prolongation in %s l1-error: %g, l2-error: %g \n", axis, l1_err, l2_err);
}

template <typename T>
int cuda_test_restriction_prolongation(const int nxc, const int nyc, const T hc) {
        T hf = 0.5 * hc;
        int nxf = 2 * (nxc - 1) + 1;
        int nyf = 2 * (nyc - 1) + 1;
        restriction_prolongation_info(nxf, nyf, hf, nxc, nyc, hc);

        size_t num_bytesf = sizeof(T) * nxf * nyf;
        size_t num_bytesc = sizeof(T) * nxc * nyc;

        T *xf, *yf, *zf, *xc, *yc, *zc;
        cudaMalloc((void**)&xf, num_bytesf);
        cudaMalloc((void**)&yf, num_bytesf);
        cudaMalloc((void**)&zf, num_bytesf);
        cudaMalloc((void**)&xc, num_bytesc);
        cudaMalloc((void**)&yc, num_bytesc);
        cudaMalloc((void**)&zc, num_bytesc);

        cuda_grid_x(xf, nxf, nyf, hf);
        cuda_grid_x(xc, nxc, nyc, hc);
        cuda_restriction("x", xc, yc, zc, nxc, nyc, hc, xf, nxf, nyf, hf);
        cuda_prolongation("x", xf, yf, zf, nxf, nyf, hf, xc, nxc, nyc, hc);

        cuda_grid_y(yf, nxf, nyf, hf);
        cuda_grid_y(yc, nxc, nyc, hc);
        cuda_restriction("y", xc, yc, zc, nxc, nyc, hc, xf, nxf, nyf, hf);
        cuda_prolongation("y", xf, yf, zf, nxf, nyf, hf, xc, nxc, nyc, hc);

        CUDACHECK(cudaFree(xf));
        CUDACHECK(cudaFree(yf));
        CUDACHECK(cudaFree(zf));
        CUDACHECK(cudaFree(xc));
        CUDACHECK(cudaFree(yc));
        CUDACHECK(cudaFree(zc));

        return test_report();
}

int main(int argc, char **argv) {

        int err = 0;
        {
                int nx = 20;
                int ny = 20;
                double h = 1.0;
                err |= test_gridpoints(nx, ny, h);
                err |= cuda_test_gridpoints(nx, ny, h);
        }

        {
                int nx = 21;
                int ny = 20;
                double h = 0.5;
                err |= test_gridpoints(nx, ny, h);
        }

        {
                int nx = 21;
                int ny = 31;
                err |= test_gridnorm<double>(nx, ny);
                err |= cuda_test_gridnorm<double>(nx, ny);
        }

        {
                int nxc = 4;
                int nyc = 4;
                double hf = 0.3;
                err |= test_restriction_prolongation(nxc, nyc, hf);
                err |= cuda_test_restriction_prolongation(nxc, nyc, hf);
        }

        return err;

}
