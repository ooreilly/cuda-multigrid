#pragma once

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


template <typename F, typename P, typename T=double>
SolverOutput solve(F& solver, P& problem, SolverOptions opts) {

        if (opts.verbose) {
                printf("Solver: %s \n", solver.name());
                printf("Iteration \t Residual\n");
        }
        
        T res = 0.0;
        int iter = 0;
        do {
                solver(problem);
                problem.residual();
                res = problem.norm();
                iter++;
                if (iter % opts.info == 0 && opts.verbose)
                printf("%-7d \t %-7.7g \n", iter, res);

        } while (res > opts.eps && (iter < opts.max_iterations || opts.max_iterations < 0));

        SolverOutput out;
        out.iterations = iter;
        out.residual = res;

        if (opts.mms) {
                out.error = problem.error();
        }

        return out;
}

