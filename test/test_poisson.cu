#include <stdio.h>

#include <poisson.hpp>
#include <assertions.hpp>
#include <grid.hpp>

template <typename T, typename F>
SolverOutput test_gauss_seidel(const int n, const T h, const T modes, SolverOptions opts) {
        size_t num_bytes = sizeof(T) * n * n;
        T *u = (T*)malloc(num_bytes);
        T *f = (T*)malloc(num_bytes);
        T *r = (T*)malloc(num_bytes);
        memset(u, 0, num_bytes);
        memset(f, 0, num_bytes);
        memset(r, 0, num_bytes);

        forcing_function(f, n, h, modes);
        T fnorm = grid_l1norm(f, n, n, h); 

        F solver;

        if (opts.verbose) {
                printf("Solver: %s \n", solver.name());
                printf("|f|_1 = %g, eps = %g, eps|f|_1 = %g \n", fnorm, opts.eps, opts.eps * fnorm);
                printf("Iteration \t Residual\n");
        }
        
        T res = 0.0;
        int iter = 0;
        do {
                solver(u, f, n, h);
                residual(r, u, f, n, h);
                res = grid_l1norm(r, n, n, h);
                iter++;
                if (iter % opts.info == 0 && opts.verbose)
                printf("%-7d \t %-7.7g \n", iter, res);

        } while (res > opts.eps * fnorm && (iter < opts.max_iterations || opts.max_iterations < 0));

        SolverOutput out;
        out.iterations = iter;
        out.residual = res;

        if (opts.mms) {
                T *v = (T*)malloc(num_bytes);
                memset(v, 0, num_bytes);
                exact_solution(v, n, h, modes);
                grid_subtract(r, u, v, n, n);
                out.error = grid_l1norm(r, n, n, h);
                free(v);
        }

        free(u);
        free(f);
        free(r);

        return out;
}

template <typename T, typename F>
void convergence_test(const int num_grids) {
        int n = 16;
        T h = 1.0;
        T modes = 1.0;
        T rate = 0.0;
        T err1 = 0.0;
        F solver;
        printf("Solver: %s \n", solver.name());
        printf("Refinement \t Iterations \t Residual \t Error \t\t Rate \n");
        SolverOptions opts;
        opts.max_iterations = -1;
        opts.eps = 1e-8;
        opts.mms = 1;
        for (int i = 0; i < num_grids; ++i) {
                n = 2 * (n - 1) + 1;
                h = h / 2;
                SolverOutput out = test_gauss_seidel<T, F>(n, h, modes, opts);
                rate = log2(err1 / out.error);
                printf("%-7d \t %-7d \t %-5.5g \t %-5.5g \t %-5.5g \n", i,
                       out.iterations, out.residual, out.error, rate);
                err1 = out.error;
        }

}

int main(int argc, char **argv) {

        SolverOptions opts;
        test_gauss_seidel<double, GaussSeidel>(100, 1.0, 1.0, opts);
        test_gauss_seidel<double, GaussSeidelRedBlack>(100, 1.0, 1.0, opts);

        convergence_test<double, GaussSeidel>(3);
        convergence_test<double, GaussSeidelRedBlack>(3);

}
