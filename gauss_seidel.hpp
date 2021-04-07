#pragma once

template <typename Number=double>
void gauss_seidel(Number *u, const Number *f, const int n) {

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        if (i % 2 == j % 2) {
                        u[j + n * i] =
                            0.25 *
                                (u[j - 1 + n * i] + u[j + 1 + n * i] +
                                 u[j + n * (i - 1)] + u[j + n * (i + 1)] +
                            f[j + n * i]);
                        }
                }
        }

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        if (i % 2 != j % 2) {
                        u[j + n * i] =
                            0.25 *
                                (u[j - 1 + n * i] + u[j + 1 + n * i] +
                                 u[j + n * (i - 1)] + u[j + n * (i + 1)] +
                            f[j + n * i]);
                        }
                }
        }

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

