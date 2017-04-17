#pragma once
#include "matrix.h"

double csr_matrixvector(csr_matrix matrix, double* x, double* y);
double ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y);