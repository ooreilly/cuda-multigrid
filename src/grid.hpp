#pragma once
#include <assert.h>

template <typename T>
void grid_x(T *x, const int nx, const int ny, const T h) {
        for (int i = 0; i < ny; ++i) {
                for (int j = 0; j < nx; ++j) {
                        x[j + nx * i] = j * h;
                }
        }
}

template <typename T>
void grid_y(T *y, const int nx, const int ny, const T h) {
        for (int i = 0; i < ny; ++i) {
                for (int j = 0; j < nx; ++j) {
                        y[j + nx * i] = i * h;
                }
        }
}

template<typename T>
void grid_restrict(T *yc, const int nxc, const int nyc, const T *xf, const int nxf, const int nyf, const T a = 0.0, const T b=1.0) {
        assert(nxf == 2 * (nxc - 1) + 1);
        for (int i = 0; i < nyc; ++i) {
                for (int j = 0; j < nxc; ++j) {
                        yc[j + nxc * i] =
                            a * yc[j + nxc * i] + b * xf[2 * j + nxf * 2 * i];
                }
        }
}

template<typename T>
void grid_prolongate(T *yf, const int nxf, const int nyf, const T *xc, const int nxc, const int nyc, const T a=0.0, const T b=1.0) {
        assert(nxf == 2 * (nxc - 1) + 1);

        for (int i = 0; i < nyc; ++i) {
                for (int j = 0; j < nxc; ++j) {
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
        }
}

template<typename T>
void grid_subtract(T *z, const T *x, const T *y, const int nx, const int ny) {
        for (int i = 0; i < nx * ny; ++i)
                z[i] = x[i] - y[i];
}

template<typename T>
double grid_l1norm(const T *x, const int nx, const int ny, const T h) {
        double out = 0.0;
        for (int i = 0; i < nx * ny; ++i) 
                out += fabs(x[i]) * h * h;
        return out;
}

template<typename T>
double grid_l2norm(const T *x, const int nx, const int ny, const T h) {
        double out = 0.0;
        for (int i = 0; i < nx * ny; ++i) 
                out += x[i] * x[i] * h * h;
        return out;
}

template <typename T>
void grid_print(const T *x, const int nx, const int ny) {
        for (int i = 0; i < ny; ++i) {
                for (int j = 0; j < nx; ++j)
                        printf("%-5.3g ", fabs(x[j + i * nx]));
                printf("\n");
        }
}
