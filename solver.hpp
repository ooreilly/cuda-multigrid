template <typename Number>
class SolveInfo {
        int converged = 0;
        Number number;
};

template <typename Solver, typename Residual, typename Norm,
          typename Number = double>
double solve(Solver &solver, Residual &residual, Norm &norm,
             const Number r = 1.0f, const int max_iter = -1,
             const Number eps = 1e-12, const int stride = 1) {
        int iter = 0;
        double res_norm = 0;
        printf("Solver: %s \n", solver.name());
        printf("Iteration \t Residual \n");
        do {
                solver();
                residual();
                res_norm = norm();
                if (iter % stride == 0)
                        printf("%-7d \t %-7.4g  \n", iter + 1, res_norm);
                iter++;
        } while (res_norm > r * eps && (max_iter < 0 || iter < max_iter));
        printf("Residual: %g achieved after %d iterations. \n", res_norm, iter);
        return res_norm;
}
