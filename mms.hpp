

double error(const double *u, const double *v, const int n, const double h) {
        double out = 0.0;
        for (int i = 0; i < n * n; ++i) {
                out += fabs(u[i] - v[i]) * h * h;
        }
        return out;
}

void init_forcing(double *f, const int n, const double h) { 
        double Lx = h * (n - 1);
        double Ly = h * (n - 1);
        for (int i = 1; i < n - 1; ++i) {
                        double yi = i * h;
                for (int j = 1; j < n - 1; ++j) {
                        double xj = j * h;
                        f[j + n * i] = 
                                 4.0 * M_PI * M_PI * h * h *
                                (
                                sin(2.0 * M_PI * xj / Lx) * sin(2.0 * M_PI * yi / Ly) / (Lx * Lx)
                                +
                                sin(2.0 * M_PI * xj / Lx) * sin(2.0 * M_PI * yi / Ly) / (Ly * Ly)
                                );
                }
        }
}

void exact_solution(double *v, const int n, const double h) { 
        double Lx = h * (n - 1);
        double Ly = h * (n - 1);
        for (int i = 0; i < n; ++i) {
                        double yi = i * h;
                for (int j = 0; j < n; ++j) {
                        double xj = j * h;
                        v[j + n * i] = sin(2.0 * M_PI * xj / Lx) * sin(2.0 * M_PI * yi / Ly);
                }
        }
}
