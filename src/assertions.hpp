#pragma once

#ifndef HALT_ON_ERROR
#define HALT_ON_ERROR 0
#endif 

#ifndef FTOL
#define FTOL 1e-6
#endif

#ifndef GTOL
#define GTOL 1e-12
#endif

int num_pass = 0;
int num_fail = 0;
int num_tests = 0;

#define equals(x, y) equals_impl((x), (y), #x, #y, __FILE__, __LINE__, __func__)
#define approx(x, y) approx_impl((x), (y), #x, #y, __FILE__, __LINE__, __func__)

template <typename T>
void equals_impl(T a, T b, const char *astr, const char *bstr, const char *file,
                 const int line, const char *func) {
        if (a != b) {
                printf("%s:%d %s() %s == %s : %d == %d  \n", file, line, func, astr, bstr, a, b);
                num_fail++;
                if (HALT_ON_ERROR) exit(-1);
        } else {
                num_pass++;
        }
}

template <typename T>
void approx_impl(T a, T b, const char *astr, const char *bstr, const char *file,
                 const int line, const char *func) {
        int err = 0;

        // Single precision
        if (sizeof(T) == 4 && (a - b > 0 && a - b > FTOL) || (b - a > 0 && b - a > FTOL)) {
                printf("%s:%d %s() %s == %s : %f == %f  \n", file, line, func, astr, bstr, a, b);
                err = 1;

        }

        // Double precision
        if (sizeof(T) == 8 && (a - b > 0 && a - b > GTOL) || (b - a > 0 && b - a > GTOL)) {
                printf("%s:%d %s() %s == %s : %g == %g  \n", file, line, func, astr, bstr, a, b);
                err = 1;

        }

        if (HALT_ON_ERROR && err) exit(-1);
        if (err) num_fail++;
        else num_pass++;
}

int test_report(void) {
        printf("Number of tests passed: %d failed: %d \n", num_pass, num_fail);
        int num_fail_out = num_fail;
        num_pass = num_fail = num_tests = 0;
        return num_fail_out;
}

