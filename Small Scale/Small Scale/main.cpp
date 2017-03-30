#include <iostream>
#include "openmp_matrixvector.h"
#include "serial_matrixvector.h"
#include "cuda_matrixvector.cuh"
#include "matrix.h"

#include <time.h>
#include <windows.h>

LONGLONG measure_time(void(*f)(csr_matrix matrix, double*, double*), csr_matrix m, double*x, double* y) {
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	(*f)(m, x, y);
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	return ElapsedMicroseconds.QuadPart;
}
LONGLONG measure_time(void(*f)(ellpack_matrix, double*, double*), ellpack_matrix m, double* x, double* y) {
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	(*f)(m, x, y);
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	return ElapsedMicroseconds.QuadPart;
}

bool dif(double n1, double n2) {
	return abs(n1 - n2) > abs(n1/100000);
}
int main(int argc, char* argv[]) {
	csr_matrix csr("../matrices/cage4.mtx");
	ellpack_matrix ell("../matrices/cage4.mtx");

	for (size_t asd = 0; asd < 10; asd++) {
		double *x;
		x = (double*)malloc(sizeof(double)*csr.collumns);
		double *y1 = (double*)malloc(sizeof(double)*csr.collumns);
		double *y2 = (double*)malloc(sizeof(double)*csr.collumns);
		double *y3 = (double*)malloc(sizeof(double)*csr.collumns);
		double *y4 = (double*)malloc(sizeof(double)*csr.collumns);
		double *y5 = (double*)malloc(sizeof(double)*csr.collumns);
		double *y6 = (double*)malloc(sizeof(double)*csr.collumns);
		for (size_t i = 0; i < csr.collumns; i++) {
			x[i]=i + 1;
			y1[i] = 0;
			y2[i] = 0;
			y3[i] = 0;
			y4[i] = 0;
			y5[i] = 0;
			y6[i] = 0;
		}

		std::cout << "CSR-SER time was: " << measure_time(csr_matrixvector, csr, x, y1) << std::endl;
		std::cout << "CSR-PAR time was: " << measure_time(omp_csr_matrixvector, csr, x, y2) << std::endl;
		std::cout << "CSR-CUDA time was: " << measure_time(cuda_csr_matrixvector, csr, x, y3) << std::endl;
		std::cout << "ELL-SER time was: " << measure_time(ellpack_matrixvector, ell, x, y4) << std::endl;
		std::cout << "ELL-PAR time was: " << measure_time(omp_ellpack_matrixvector, ell, x, y5) << std::endl;
		std::cout << "ELL-CUDA time was: " << measure_time(cuda_ellpack_matrixvector, ell, x, y6) << std::endl;
		std::cout << "\n\n";
		for (size_t i = 0; i < csr.collumns; i++) {
			if (dif(y1[i], y2[i]))
				std::cout << "fds2\n";
				if (dif(y1[i], y3[i]))
				std::cout << "fds3\n";
			if (dif(y1[i], y4[i]))
				std::cout << "fds4\n";
			if (dif(y1[i], y5[i]))
				std::cout << "fds5\n";
			if (dif(y1[i], y6[i]))
				std::cout << "fds6\n";
			//std::cout << y1[i] << "\t" << y2[i] << "\t" << y3[i] << "\t" << y4[i] << "\t" << y5[i] << "\t" << y6[i] << "\n";
		
		}
		delete[] y1;
		delete[] y2;
		delete[] y3;
		delete[] y4;
		delete[] y5;
		delete[] y6;
		delete[] x;
	}
	int n;
	std::cin >> n;

}