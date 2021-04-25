#include <stdio.h>

#include <poisson.hpp>
#include <poisson.cuh>
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
        printf("Grid Size \t Iterations \t Time (ms) \t Residual \t Error \t\t Rate \n");
        for (int i = 0; i < num_grids; ++i) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                P problem(l, h, modes);
                S solver(problem);

                cudaEventRecord(start);
                SolverOutput out = solve(solver, problem, opts);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float elapsed = 0;
                cudaEventElapsedTime(&elapsed, start, stop);

                rate = log2(err1 / out.error);
                int n = (1 << l) + 1;
                printf("%4d x %-4d \t %-7d \t %-5.5f \t %-5.5g \t %-5.5g \t %-5.5f \n", 
                       n, n,
                       out.iterations, elapsed, out.residual, out.error, rate);
                err1 = out.error;
                l++;
                h /= 2;
        }
}


int main(int argc, char **argv) {

        using Number = double;
        SolverOptions opts;
        opts.verbose = 1;
        opts.info = 10;
        opts.max_iterations = 1e4;
        opts.eps = 1e-8;
        opts.mms = 1;
        int l = 4;
        int n = (1 << l) + 1;
        double h = 1.0 / (n - 1);
        double modes = 1.0;
        using Problem = Poisson<Number>;

    
        {

                using Problem = Poisson<Number>;
                Problem problem(l, h, modes);
                using Smoother=GaussSeidelRedBlack;
                Smoother solver;
                auto out = solve(solver, problem, opts);
                printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);

        }
        
        {
                Problem problem(l, h, modes);
                using Smoother=GaussSeidelRedBlack;
                using MG=Multigrid<Smoother, Problem, Number>;
                MG mg(problem);
                auto out = solve(mg, problem, opts);
                printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);

        }      

        {
                using CUDAProblem = CUDAPoisson<L1NORM, Number>;
                CUDAProblem problem(l, h, modes);
                using CUDASmoother = CUDAGaussSeidelRedBlack;

                CUDASmoother solver;
                auto out = solve(solver, problem, opts);
                printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);
        }

        {
                using CUDAProblem = CUDAPoisson<L1NORM, Number>;
                using CUDASmoother = CUDAGaussSeidelRedBlack;
                using CUDAMG = CUDAMultigrid<CUDASmoother, CUDAProblem, Number>;
                
                CUDAProblem problem(l, h, modes);
                CUDAMG solver(problem);
                auto out = solve(solver, problem, opts);
                printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);
        }

        {
                using Smoother=GaussSeidelRedBlack;
                using MG=Multigrid<Smoother, Problem, Number>;
                opts.verbose = 0;
                
                int num_refinements = 12;
                convergence_test<MG, Problem>(num_refinements, opts);
        }
        {
                using CUDAProblem = CUDAPoisson<L1NORM, Number>;
                using CUDASmoother = CUDAGaussSeidelRedBlack;
                using CUDAMG = CUDAMultigrid<CUDASmoother, CUDAProblem, Number>;
                
                opts.verbose = 0;
                int num_refinements = 12;
                convergence_test<CUDAMG, CUDAProblem>(num_refinements, opts);
        }
}
