#include "serial_matrixvector.h"

void csr_matrixvector(csr_matrix matrix, std::vector<double> x, std::vector<double> &y) {
	for (int i = 0; i < matrix.rows; i++) {
		double t = 0;
		for (int j = matrix.irp[i]; j < matrix.irp[i + 1]; j++)
			t = t + matrix.as[j] * x[matrix.ja[j]];
		y[i] = t;
	}
}

void ellpack_matrixvector(ellpack_matrix matrix, std::vector<double> x, std::vector<double>& y) {
	for (int i = 0; i < matrix.rows; i++) {
		double t = 0;
		for (int j = 0; j < matrix.maxnzr; j++)
			t = t + matrix.as[i][j] * x[matrix.ja[i][j]];
		y[i] = t;
	}
}
