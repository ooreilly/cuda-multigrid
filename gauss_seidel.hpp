#pragma once
#include <algorithm>
#include <omp.h>

template <typename Number = double>
void gauss_seidel_impl(Number *u, const Number *f, const int n, const int i0,
                       const int j0, const int in, const int jn) {
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
        for (int i = i0; i < in; i += 2) {
                for (int j = j0; j < in; j += 2) {
                        u[j + n * i] =
                            0.25 *
                                (u[j - 1 + n * i] + u[j + 1 + n * i] +
                                 u[j + n * (i - 1)] + u[j + n * (i + 1)] +
                            f[j + n * i]);
                        }
        }

}


template <typename Number=double>
void gauss_seidel(Number *u, const Number *f, const int n) {
        gauss_seidel_impl(u, f, n, 1, 1, n - 1, n - 1);
        gauss_seidel_impl(u, f, n, 1, 2, n - 1, n - 1);
        gauss_seidel_impl(u, f, n, 2, 1, n - 1, n - 1);
        gauss_seidel_impl(u, f, n, 2, 2, n - 1, n - 1);
}

template <typename Number=double>
class GaussSeidel {
        private:
                Number *u;
                const Number *f;
                int n;

        public:
         GaussSeidel(Number *u, const Number *f, const int n)
             : u(u), f(f), n(n) {}

         void operator()(void) {
                gauss_seidel(this->u, this->f, this->n);
         }

         const char* name(void) {return "Gauss-Seidel";}
};

