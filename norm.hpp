
double l1norm(const double *u, const int n, const double h) {
        double out = 0.0;
        for (int i = 0; i < n * n; ++i) {
                out += fabs(u[i]) * h * h;
        }
        return out;
}

