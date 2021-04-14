#include <stdio.h>
#include <grid.hpp>
#include <assertions.hpp>


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
void restriction(const char *axis, const T *xc, T *yc, T *zc, const int nxc,
                 const int nyc, const T hc, const T *xf, const int nxf, const int nyf, const T hf) {
        grid_restrict(yc, nxc, nyc, xf, nxf, nyf);
        grid_subtract(zc, xc, yc, nxc, nyc);


        T l1_err = grid_l1norm(zc, nxc, nyc, hc, 1, nxc - 1, 1, nyc - 1);
        T l2_err = grid_l2norm(zc, nxc, nyc, hc, 1, nxc - 1, 1, nyc - 1);

        approx(l1_err, 0.0);
        approx(l2_err, 0.0);

        printf("Restriction in %s l1-error: %g, l2-error: %g \n", axis, l1_err, l2_err);
}


template <typename T>
void prolongation(const char *axis, const T *xf, T *yf, T *zf, const int nxf,
                 const int nyf, const T hf, const T *xc, const int nxc, const int nyc, const T hc) {
        grid_prolongate(yf, nxf, nyf, xc, nxc, nyc);
        grid_subtract(zf, xf, yf, nxf, nyf);

        T l1_err = grid_l1norm(zf, nxf, nyf, hf);
        T l2_err = grid_l2norm(zf, nxf, nyf, hf);

        approx(l1_err, 0.0);
        approx(l2_err, 0.0);

        printf("Prolongation in %s l1-error: %g, l2-error: %g \n", axis, l1_err, l2_err);
}

template <typename T>
int test_restriction_prolongation(const int nxc, const int nyc, const T hc) {

        T hf = 0.5 * hc;
        int nxf = 2 * (nxc - 1) + 1;
        int nyf = 2 * (nyc - 1) + 1;
        printf(
            "Testing grid restriction and prolongation with fine grid "
            "[%d %d], hf=%g, and coarse grid [%d %d], hc=%g. \n",
            nxf, nyf, hf, nxc, nyc, hc);

        size_t num_bytesf = sizeof(T) * nxf * nyf;
        size_t num_bytesc = sizeof(T) * nxc * nyc;

        T *xf = (T*)malloc(num_bytesf);
        T *yf = (T*)malloc(num_bytesf);
        T *zf = (T*)malloc(num_bytesf);
        T *xc = (T*)malloc(num_bytesc);
        T *yc = (T*)malloc(num_bytesc);
        T *zc = (T*)malloc(num_bytesc);

        // Test restriction  and prolongation in the x-direction
        grid_x(xf, nxf, nyf, hf);
        grid_x(xc, nxc, nyc, hc);
        restriction("x", xc, yc, zc, nxc, nyc, hc, xf, nxf, nyf, hf);
        prolongation("x", xf, yf, zf, nxf, nyf, hf, xc, nxc, nyc, hc);

        // Test restriction in the y-direction
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

int main(int argc, char **argv) {

        int err = 0;
        {
                int nx = 20;
                int ny = 20;
                double h = 1.0;
                err |= test_gridpoints(nx, ny, h);
        }

        {
                int nx = 21;
                int ny = 20;
                double h = 0.5;
                err |= test_gridpoints(nx, ny, h);
        }

        {
                int nxc = 4;
                int nyc = 4;
                double hf = 0.3;
                err |= test_restriction_prolongation(nxc, nyc, hf);
        }

        return err;

}
