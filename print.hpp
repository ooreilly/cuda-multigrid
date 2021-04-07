#pragma once

void print_matrix(double *u, const int n) {
        for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                        printf("%-5.3f ", fabs(u[j + i * n]));
                }
                printf("\n");
        }
        printf("\n");

}


