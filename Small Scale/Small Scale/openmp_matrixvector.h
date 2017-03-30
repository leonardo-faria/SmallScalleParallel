#pragma once
#include <omp.h>
#include "matrix.h"

void omp_csr_matrixvector(csr_matrix matrix,double* x, double* y);
void omp_ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y);