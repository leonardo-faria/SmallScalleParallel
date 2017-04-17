#pragma once
#include <omp.h>
#include "matrix.h"

double omp_csr_matrixvector(csr_matrix matrix,double* x, double* y, int omp_threads);
double omp_ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y, int omp_threads);