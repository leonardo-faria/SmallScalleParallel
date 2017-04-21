#include <iostream>
#include <fstream>
#include "openmp_matrixvector.h"
#include "serial_matrixvector.h"
#include "cuda_matrixvector.cuh"
#include "matrix.h"
#include <math.h>
#include <time.h>
#include <sstream> 


/*
LONGLONG measure_time(void(*f)(csr_matrix, double*, double*), csr_matrix m, double*x, double* y) {
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
*/

const int MAX_OMP_THREADS = omp_get_max_threads();

std::ofstream serial_results;
std::ofstream omp_results;
std::ofstream cuda_results;


void error(double* a1, double* a2, double size, double& max, double& total) {
	double temp_max = 0;
	double temp_total = 0;
	double error;
	for (size_t i = 0; i < size; i++) {
		error = fabs((a1[i] - a2[i]));
		if (temp_max < error)
			temp_max = error;
		temp_total += error;
	}
	max += temp_max;
	total += temp_total;
}

void perform_test(std::string filename, int maxtries) {

	serial_results << filename.substr(0, filename.size() - 4) << ",";
	omp_results << filename.substr(0, filename.size() - 4) << ",";
	cuda_results << filename.substr(0, filename.size() - 4) << ",";

	csr_matrix csr("../matrices/" + filename);
	ellpack_matrix ell("../matrices/" + filename);
	serial_results << csr.nonzeros << "," << csr.collumns << "," << csr.rows << ",";
	omp_results << MAX_OMP_THREADS << "," << csr.nonzeros << "," << csr.collumns << "," << csr.rows << ",";
	cuda_results << csr.nonzeros << "," << csr.collumns << "," << csr.rows << ",";

	double sercsr_time = 0;
	double serell_time = 0;
	double ser_total_error = 0;
	double ser_max_error = 0;

	std::vector<double> openmpcsr_time;
	std::vector<double> openmpell_time;
	std::vector<double> openmpcsr_avg_error;
	std::vector<double> openmpell_avg_error;
	std::vector<double> openmpcsr_max_error;
	std::vector<double> openmpell_max_error;
	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++) {
		openmpcsr_time.push_back(0);
		openmpell_time.push_back(0);
		openmpcsr_avg_error.push_back(0);
		openmpell_avg_error.push_back(0);
		openmpcsr_max_error.push_back(0);
		openmpell_max_error.push_back(0);
	}

	double cudacsr_time = 0;
	double cudaell_time = 0;
	double cudacsr_total_error = 0;
	double cudaell_total_error = 0;
	double cudacsr_max_error = 0;
	double cudaell_max_error = 0;
	for (size_t tries = 0; tries < maxtries; tries++) {
		double *x;
		x = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_ser_csr = (double*)malloc(sizeof(double)*csr.rows);
		double *y_ser_ell = (double*)malloc(sizeof(double)*csr.rows);
		double *y_omp_csr = (double*)malloc(sizeof(double)*csr.rows);
		double *y_omp_ell = (double*)malloc(sizeof(double)*csr.rows);
		double *y_cuda_csr = (double*)malloc(sizeof(double)*csr.rows);
		double *y_cuda_ell = (double*)malloc(sizeof(double)*csr.rows);
		for (size_t i = 0; i < csr.collumns; i++) {
			x[i] = 1;
		}	for (size_t i = 0; i < csr.rows; i++) {
			y_ser_csr[i] = 0;
			y_ser_ell[i] = 0;
			y_omp_csr[i] = 0;
			y_omp_ell[i] = 0;
			y_cuda_csr[i] = 0;
			y_cuda_ell[i] = 0;
		}

		sercsr_time += csr_matrixvector(csr, x, y_ser_csr);
		serell_time += ellpack_matrixvector(ell, x, y_ser_ell);
		error(y_ser_ell, y_ser_csr, csr.collumns, ser_max_error, ser_total_error);
		for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++) {
			openmpcsr_time[omp_threads] += omp_csr_matrixvector(csr, x, y_omp_csr, omp_threads + 1);
			openmpell_time[omp_threads] += omp_ellpack_matrixvector(ell, x, y_omp_ell, omp_threads + 1);

			error(y_omp_csr, y_ser_csr, csr.collumns, openmpcsr_max_error[omp_threads], openmpcsr_avg_error[omp_threads]);
			error(y_omp_ell, y_ser_ell, csr.collumns, openmpell_max_error[omp_threads], openmpell_avg_error[omp_threads]);
		}
		cudacsr_time += cuda_csr_matrixvector(csr, x, y_cuda_csr);
		cudaell_time += cuda_ellpack_matrixvector(ell, x, y_cuda_ell);
		error(y_cuda_csr, y_ser_csr, csr.collumns, cudacsr_max_error, cudacsr_total_error);
		error(y_cuda_ell, y_ser_ell, csr.collumns, cudaell_max_error, cudaell_total_error);
		

		delete[] y_ser_csr;
		delete[] y_ser_ell;
		delete[] y_omp_csr;
		delete[] y_omp_ell;
		delete[] y_cuda_csr;
		delete[] y_cuda_ell;
		delete[] x;
	}

	serial_results << ((csr.nonzeros*2.0) / (sercsr_time / ((double)maxtries))) << "," << ((csr.nonzeros*2.0) / (serell_time / ((double)maxtries))) << "," << (ser_max_error / ((double)maxtries)) << "," << (ser_total_error / ((double)maxtries)) << ",";

	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++)
		omp_results << ((csr.nonzeros*2.0) / (openmpcsr_time[omp_threads] / ((double)maxtries))) << ",";
	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++)
		omp_results << ((csr.nonzeros*2.0) / (openmpell_time[omp_threads] / ((double)maxtries))) << ",";
	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++)
		omp_results << (openmpcsr_max_error[omp_threads] / ((double)maxtries)) << "," << (openmpcsr_avg_error[omp_threads] / ((double)maxtries)) << ",";
	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++)
		omp_results << (openmpcsr_max_error[omp_threads] / ((double)maxtries)) << "," << (openmpell_avg_error[omp_threads] / ((double)maxtries)) << ",";
	cuda_results << ((csr.nonzeros*2.0) / (cudacsr_time / ((double)maxtries))) << ",";
	cuda_results << ((csr.nonzeros*2.0) / (cudaell_time / ((double)maxtries))) << ",";
	cuda_results << (cudacsr_max_error / ((double)maxtries)) << "," << (cudacsr_total_error / ((double)maxtries)) << ",";
	cuda_results << (cudaell_max_error / ((double)maxtries)) << "," << (cudaell_total_error / ((double)maxtries)) << ",";

	serial_results << "\n";
	omp_results << "\n";
	cuda_results << "\n";
}




int main(int argc, char* argv[]) {

	serial_results.open("serial_results.csv");
	omp_results.open("omp_results.csv");
	cuda_results.open("cuda_results.csv");

	serial_results << "Matrix,nonzeros,collumns,rows,csr,ell,Max error,Total error\n";

	omp_results << "Matrix,Max threads,nonzeros,collumns,rows,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << "(" << i << ") csr,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << "(" << i << ") ell,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << " (" << i << ") Max error csr, (" << i << ")Total error csr, ";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << " (" << i << ") Max error ell, (" << i << ")Total error ell, ";

	omp_results << "\n";

	cuda_results << "Matrix,nonzeros,collumns,rows,csr,ell,Max error csr,Total error csr,Max error ell,Total error ell,\n";

	perform_test("cage4.mtx", 1);

	serial_results.close();
	omp_results.close();
	cuda_results.close();
	int a;
	scanf("%d", &a);
}