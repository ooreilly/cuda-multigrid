#pragma once
#include <grid.hpp>
// Solves Poisson's equation: Lu = f, Lu = -(u_xx + u_yy)

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

        T hi2 = 1.0 / (h * h);
        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        r[j + i * n] = 
                        -(u[j + 1 + i * n] + u[j - 1 + i * n] +
                                        - 4.0 * u[j + i * n] + u[j + (i + 1) * n] +
                                        u[j + (i - 1) * n]) * hi2 -  f[j + i * n];
                }
        }

}

template <typename T>
void forcing_function(T *f, const int n, const T h, const T modes=1.0) {

        T s = 2.0 * M_PI * modes / (h * (n - 1));
        memset(f, 0, n * n * sizeof(T));
        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
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

template <typename T>
__inline__ void base_case(T *u, const T *f, const T h) {
        u[1 + 3 * 1] = 0.25 * f[1 + 3 * 1] * h * h;
}


template <typename T, typename F>
void multigrid_v_cycle(const int l, F& solver, T *u, T *f, T *r, T *tmp, T **v, const T h) {

        int nu = (1 << l) + 1; 
        int nv = (1 << (l - 1)) + 1;
        //printf("nu = %d nv = %d \n", nu, nv);
        T *d = (T*)malloc(sizeof(T) * nv * nv);
        T *vl = (T*)malloc(sizeof(T) * nv * nv);
        memset(vl, 0, sizeof(T) * nv * nv);
        memset(d, 0, sizeof(T) * nv * nv);

        if (l == 1) {
                base_case(u, f, h);
                return;
        }
        
        //printf("f^%d \n", l);
        //grid_print(f, nu, nu);

        // Smooth out high frequency components
        solver(u, f, nu, h);

        //printf("u^%d \n", l);
        //grid_print(u, nu, nu);

        //printf("f^%d \n", l);
        //grid_print(f, nu, nu);

        memset(tmp, 0, sizeof(T) * nu * nu);
        residual(tmp, u, f, nu, h);

        //printf("r^%d \n", l);
        //grid_print(tmp, nu, nu);
        grid_restrict(d, nv, nv, tmp, nu, nu, 0.0, 1.0);
        //printf("r^%d \n", l - 1);
        //grid_print(d, nv, nv);

        multigrid_v_cycle(l - 1, solver, vl, d, r, tmp, v, 2 * h); 

        //printf("v^%d \n", l - 1);
        //grid_print(vl, nv, nv);

        //printf("u^%d \n", l);
        //grid_print(u, nu, nu);

        // Prolongate and subtract solution u^l = u^l -  Pv^(l-1)
        grid_prolongate(u, nu, nu, vl, nv, nv, 1.0, -1.0);
        
        //printf("P^%d \n", l);
        //grid_print(u, nu, nu);

        // Smooth out high frequency components
        solver(u, f, nu, h);

        //printf("u^%d \n", l);
        //grid_print(u, nu, nu);

        free(d);
        free(vl);


}

template <typename T>
class Multigrid {
        private:
                T **v, *r, *tmp;
                int l;
        public:
                Multigrid(const int l) : l(l) {
                        // v[0] : fine grid, v[n-1] : coarse grid
                        v = (T**)malloc(sizeof(T) * l);
                        for (int i = 0; i < l; ++i) {
                                int n = (1 << (l - 0)) + 1;
                                int num_bytes =  sizeof(T) * n * n;
                                v[i] = (T*)malloc(num_bytes);
                                memset(v[i], 0, num_bytes);
                        }

                        int n = (1 << l) + 1;
                        r = (T*)malloc(sizeof(T) * n * n);
                        tmp = (T*)malloc(sizeof(T) * n * n);

                }

                template <typename F>
                void operator()(F& solver, T *u, T *f, const T h) {
                        multigrid_v_cycle<T, F>(l, solver, u, f, r, tmp, v, h);
                }

                ~Multigrid(void) {
                        for (int i = 0; i < l; ++i) 
                                free(v[i]);
                        free(v);
                        free(r);
                        free(tmp);
                }

};

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

