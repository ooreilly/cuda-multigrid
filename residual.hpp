#pragma once

void residual(double *res,  const double *u, const double *f, const int n) {
        for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                        res[j + n * i] =
                                (u[j - 1 + n * i] + u[j + 1 + n * i] +
                                 - 4 * u[j + n * i] + 
                                 u[j + n * (i - 1)] + u[j + n * (i + 1)]) +
                             + f[j + n * i];
                }
        }

}


template <typename Number=double>
class Residual {

        private:
                Number *res;
                const Number *u, *f;
                const int n;

        public:
                Residual(Number *res, const Number *u, const Number *f, const int n)
             : res(res), u(u), f(f), n(n) {}

                void operator()(void){
                       residual(res, u, f, n); 
                }
};
