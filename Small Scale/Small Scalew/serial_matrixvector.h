#pragma once
#include "matrix.h"

void csr_matrixvector(csr_matrix matrix, std::vector<double> x, std::vector<double> &y);
void ellpack_matrixvector(ellpack_matrix matrix, std::vector<double> x, std::vector<double> &y);