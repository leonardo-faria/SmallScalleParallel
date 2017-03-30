#pragma once
#include "matrix.h"

void csr_matrixvector(csr_matrix matrix, double* x, double* y);
void ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y);