#include <stdio.h>

#include <poisson.hpp>
#include <assertions.hpp>
#include <grid.hpp>
#include <solver.hpp>


template <typename S, typename P, typename T=double>
void convergence_test(const int num_grids, SolverOptions opts) {
        T rate = 0.0;
        T err1 = 0.0;
        T modes = 1.0;
        int l = 2;
        T h = 1.0;
        printf("MMS convergence test\n");
        {
                S tmp;
                printf("Solver: %s \n", tmp.name());
        }
        printf("Refinement \t Iterations \t Residual \t Error \t\t Rate \n");
        for (int i = 0; i < num_grids; ++i) {
                P problem(l, h, modes);
                S solver(problem);
                SolverOutput out = solve(solver, problem, opts);
                rate = log2(err1 / out.error);
                printf("%-7d \t %-7d \t %-5.5g \t %-5.5g \t %-5.5g \n", i,
                       out.iterations, out.residual, out.error, rate);
                err1 = out.error;
                l++;
                h /= 2;
        }
}


int main(int argc, char **argv) {

        using Number = double;
        SolverOptions opts;
        opts.verbose = 1;
        opts.info = 1;
        opts.max_iterations = 20;//e4;
        opts.eps = 1e-8;
        opts.mms = 1;
        int l = 4;
        int n = (1 << l) + 1;
        double h = 1.0 / (n - 1);
        double modes = 1.0;
        using Problem = Poisson<Number>;

        {
                Problem problem(l, h, modes);
                using Smoother=GaussSeidelRedBlack;
                using MG=Multigrid<Smoother, Problem, Number>;
                MG mg(problem);
                auto out = solve(mg, problem, opts);
                printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);

                opts.verbose = 0;
                convergence_test<MG, Problem>(10, opts);
        }

        //{
        //opts.verbose = 1;
        //Problem problem(7, h, modes);
        //GaussSeidel gs;
        //auto out = solve(gs, problem, opts);
        //printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);
        //}
        //convergence_test<GaussSeidel, Problem, Number>(problem, 3, opts);
        //convergence_test<double, GaussSeidelRedBlack>(5, opts);

}
