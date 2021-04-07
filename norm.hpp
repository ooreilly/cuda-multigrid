#pragma once

double l1norm(const double *u, const int n, const double h) {
        double out = 0.0;
        for (int i = 0; i < n * n; ++i) {
                out += fabs(u[i]) * h * h;
        }
        return out;
}


template <typename Number=double>
class L1Norm {
        private:
                const Number *u;
                const int n;
                const Number h;
        public:
                L1Norm(const Number *u, const int n, const Number h) : u(u), n(n), h(h) {}

                Number operator()(void) {
                        return l1norm(u, n, h);
                }

};
