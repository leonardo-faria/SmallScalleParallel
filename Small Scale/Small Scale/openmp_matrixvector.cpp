#include "openmp_matrixvector.h"

double omp_csr_matrixvector(csr_matrix matrix, double* x, double* y,int omp_threads) {
	double time = omp_get_wtime();
#pragma omp parallel num_threads(omp_threads)
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < matrix.rows; i++) {
			double t = 0;
			for (int j = matrix.irp[i]; j < matrix.irp[i + 1]; j++)
				t = t + matrix.as[j] * x[matrix.ja[j]];
			y[i] = t;
		}
	}
	return (omp_get_wtime()-time)*1000000;
}

double omp_ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y, int omp_threads) {
	double time = omp_get_wtime();

#pragma omp parallel num_threads(omp_threads)
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < matrix.rows; i++) {
			double t = 0;
			 int I = i*matrix.maxnzr;
			for (int j = 0; j < matrix.maxnzr; j++)
				t = t + matrix.as[I+j] * x[matrix.ja[I + j]];
			y[i] = t;
		}
	}
	return (omp_get_wtime() - time) * 1000000;
}
