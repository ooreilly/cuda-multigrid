#pragma once
#include <grid.hpp>
#include <algorithm>
// Solves Poisson's equation: Lu = f, Lu = u_xx + u_yy

template <typename T>
void gauss_seidel(T *u, const T *f, const int n, const T h) {

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        u[j + i * n] =
                            - 0.25 * (
                                    h * h * f[j + i * n]
                                    -
                                    u[j + 1 + i * n] - u[j - 1 + i * n]
                                    -
                                    u[j + (i + 1) * n] - u[j + (i - 1) * n]);
                }
        }

}

template <typename T>
void gauss_seidel_red_black(T *u, const T *f, const int n, const T h) {

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        if ( (i + j) % 2 == 0) {
                        u[j + i * n] =
                            - 0.25 * (
                                    h * h * f[j + i * n]
                                    -
                                    u[j + 1 + i * n] - u[j - 1 + i * n]
                                    -
                                    u[j + (i + 1) * n] - u[j + (i - 1) * n]);
                        }
                }
        }

        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        if ( (i + j) % 2 == 1) {
                        u[j + i * n] =
                            - 0.25 * (
                                    h * h * f[j + i * n]
                                    -
                                    u[j + 1 + i * n] - u[j - 1 + i * n]
                                    -
                                    u[j + (i + 1) * n] - u[j + (i - 1) * n]);
                        }
                }
        }
}

template <typename T>
void poisson_residual(T *r, const T *u, const T *f, const int n, const T h) {

        T hi2 = 1.0 / (h * h);
        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        r[j + i * n] = 
                        f[j + i * n] - (
                                        u[j + 1 + i * n] + u[j - 1 + i * n] +
                                        - 4.0 * u[j + i * n] + u[j + (i + 1) * n] +
                                        u[j + (i - 1) * n]) * hi2;
                }
        }

}

template <typename T>
void forcing_function(T *f, const int n, const T h, const T modes=1.0) {

        T s = 2.0 * M_PI * modes / (h * (n - 1));
        memset(f, 0, n * n * sizeof(T));
        for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                        f[j + n * i] = -2 * s * s * sin(s * h * i) * sin(s * h * j);
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
        u[1 + 3 * 1] = -0.5 * f[1 + 3 * 1] * h * h;
}

template <typename T, typename S>
void multigrid_v_cycle(const int l, S& smoother, T *u, T *f, T *r, T *tmp, T **v, const T h) {

        if (l == 1) {
                base_case(u, f, h);
                return;
        }

        int nu = (1 << l) + 1; 
        int nv = (1 << (l - 1)) + 1;
        T *d = (T*)malloc(sizeof(T) * nv * nv);
        T *vl = (T*)malloc(sizeof(T) * nv * nv);
        memset(vl, 0, sizeof(T) * nv * nv);
        memset(d, 0, sizeof(T) * nv * nv);

        smoother(u, f, nu, h);

        memset(tmp, 0, sizeof(T) * nu * nu);
        poisson_residual(tmp, u, f, nu, h);

        grid_restrict(d, nv, nv, tmp, nu, nu, 0.0, 1.0);

        multigrid_v_cycle(l - 1, smoother, vl, d, r, tmp, v, 2 * h); 

        // Prolongate and add correction u^l = u^l +  Pv^(l-1)
        grid_prolongate(u, nu, nu, vl, nv, nv, 1.0, 1.0);

        smoother(u, f, nu, h);

        free(d);
        free(vl);


}

template <typename F, typename P, typename T>
class Multigrid {
        private:
                T **v, *r, *tmp;
                int l;
                F smoother;
        public:

                Multigrid() { }
                Multigrid(P& p) : l(p.l) {
                        // v[0] : fine grid, v[n-1] : coarse grid
                        //v = (T**)malloc(sizeof(T) * p.l);
                        //for (int i = 0; i < l; ++i) {
                        //        int n = (1 << (p.l - 0)) + 1;
                        //        int num_bytes =  sizeof(T) * p.n * p.n;
                        //        v[i] = (T*)malloc(num_bytes);
                        //        memset(v[i], 0, num_bytes);
                        //}

                        int n = (1 << p.l) + 1;
                        r = (T*)malloc(sizeof(T) * n * n);
                        tmp = (T*)malloc(sizeof(T) * n * n);

                }

                void operator()(P& p) {
                        multigrid_v_cycle<T, F>(p.l, smoother, p.u, p.f, r, tmp, v, p.h);
                }

                ~Multigrid(void) {
                        //for (int i = 0; i < l; ++i) 
                        //        free(v[i]);
                        //        free(v);
                        free(r);
                        free(tmp);
                }

                const char *name() {
                        static char name[2048];
                        sprintf(name, "Multi-Grid<%s>", smoother.name());
                        return name;
                }

};

class GaussSeidel {
        public:
                GaussSeidel() { }
        template <typename P>
                GaussSeidel(P& p) { }
        template <typename P>
        void operator()(P& p) {
                gauss_seidel(p.u, p.f, p.n, p.h);
        }

        template <typename T>
        void operator()(T *u, T *f, const int n, const T h) {
                gauss_seidel(u, f, n, h);
        }

        const char *name() {
                return "Gauss-Seidel";
        }

};

class GaussSeidelRedBlack {
        public:
                GaussSeidelRedBlack() { }
        template <typename P>
                GaussSeidelRedBlack(P& p) { }
        template <typename T>
        void operator()(T *u, const T *f, const int n, const T h) {
                gauss_seidel_red_black(u, f, n, h);
        }

        template <typename P>
        void operator()(P& p) {
                gauss_seidel_red_black(p.u, p.f, p.n, p.h);
        }
        const char *name() {
                return "Gauss-Seidel (red-black)";
        }

};

template <typename T>
class Poisson {
        public:
                int n;
                int l;
                T h;
                T modes;
                T *u, *f, *r;
                size_t num_bytes;

        Poisson(int l, T h, T modes) : l(l), h(h), modes(modes) {
                n = (1 << l) + 1;
                num_bytes = sizeof(T) * n * n;
                u = (T*)malloc(num_bytes);
                f = (T*)malloc(num_bytes);
                r = (T*)malloc(num_bytes);
                memset(u, 0, num_bytes);
                memset(f, 0, num_bytes);
                memset(r, 0, num_bytes);
                forcing_function(f, n, h, modes);
        }

        T error() {
                T *v = (T*)malloc(num_bytes);
                memset(v, 0, num_bytes);
                exact_solution(v, n, h, modes);
                grid_subtract(r, u, v, n, n);
                T err = grid_l1norm(r, n, n, h);
                free(v);
                return err;
        }

        void residual(void) {
                poisson_residual(r, u, f, n, h);
        }

        T norm(void) {
                return grid_l1norm(r, n, n, h);
        }

        ~Poisson() {
                free(u);
                free(f);
                free(r);
        }
};


