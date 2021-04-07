void gauss_seidel(double *u, const double *f, const int n) {

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
