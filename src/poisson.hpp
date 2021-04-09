#pragma once
// Solves Poisson's equation: u_xx + u_yy = -f

template <typename T>
void gauss_seidel(T *u, const T *f, const int n, const T h) {

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        u[j + i * n] =
                            0.25 * (u[j + 1 + i * n] + u[j - 1 + i * n] +
                                    u[j + (i + 1) * n] + u[j + (i - 1) * n] +
                                    h * h * f[j + i * n]);
                }
        }

}

template <typename T>
void gauss_seidel_red_black(T *u, const T *f, const int n, const T h) {

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        if ((i + j) % 2 == 0) {
                        u[j + i * n] =
                            0.25 * (u[j + 1 + i * n] + u[j - 1 + i * n] +
                                    u[j + (i + 1) * n] + u[j + (i - 1) * n] +
                                    h * h * f[j + i * n]);
                        }
                }
        }

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        if ((i + j) % 2 == 1) {
                        u[j + i * n] =
                            0.25 * (u[j + 1 + i * n] + u[j - 1 + i * n] +
                                    u[j + (i + 1) * n] + u[j + (i - 1) * n] +
                                    h * h * f[j + i * n]);
                        }
                }
        }

}

template <typename T>
void residual(T *r, const T *u, const T *f, const int n, const T h) {

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        r[j + i * n] = 
                        u[j + 1 + i * n] + u[j - 1 + i * n] +
                                        - 4.0 * u[j + i * n] + u[j + (i + 1) * n] +
                                        u[j + (i - 1) * n] + h * h * f[j + i * n];
                }
        }

}

template <typename T>
void forcing_function(T *f, const int n, const T h, const T modes=1.0) {

        T s = 2.0 * M_PI * modes / (h * (n - 1));
        for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                        f[j + n * i] = 2 * s * s * sin(s * h * j) * sin(s * h * i);
                }
        }

}

template <typename T>
void exact_solution(T *u, const int n, const T h, const T modes=1.0) {
        T s = 2.0 * M_PI * modes / (h * (n - 1));
        for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                        u[j + n * i] = sin(s * h * j) * sin(s * h * i);
                }
        }
}

class GaussSeidel {
        public:
        template <typename T>
        void operator()(T *u, const T *f, const int n, const T h) {
                gauss_seidel(u, f, n, h);
        }

        const char *name() {
                return "Gauss-Seidel";
        }

};

class GaussSeidelRedBlack {
        public:
        template <typename T>
        void operator()(T *u, const T *f, const int n, const T h) {
                gauss_seidel_red_black(u, f, n, h);
        }
        const char *name() {
                return "Gauss-Seidel (red-black)";
        }

};

class SolverOptions {
        public:
        int verbose = 0;
        int max_iterations = 1000000;
        double eps = 1e-12;
        int info = 1.0;
        int mms = 0;
};

class SolverOutput {
        public:
                double residual;
                int iterations;
                double error;
};

