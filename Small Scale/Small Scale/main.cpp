#include <iostream>
#include <fstream>
#include "openmp_matrixvector.h"
#include "serial_matrixvector.h"
#include "cuda_matrixvector.cuh"
#include "matrix.h"

#include <time.h>
#include <windows.h>   
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
bool dif(double n1, double n2) {
	return abs(n1 - n2) > abs(n1 / 100000);
}



void perform_test(std::string filename, int maxtries) {

	serial_results << filename.substr(0, filename.size() - 4) << ",";
	omp_results << filename.substr(0, filename.size() - 4) << ",";
	cuda_results << filename.substr(0, filename.size() - 4) << ",";

	csr_matrix csr("../matrices/" + filename);
	ellpack_matrix ell("../matrices/" + filename);
	serial_results << csr.nonzeros << "," << csr.collumns << "," << csr.rows << ",";
	omp_results<< MAX_OMP_THREADS << "," << csr.nonzeros << "," << csr.collumns << "," << csr.rows << ",";
	cuda_results << csr.nonzeros << "," << csr.collumns << "," << csr.rows << ",";

	double sercsr_time = 0;
	double serell_time = 0;
	std::vector<double> openmpcsr_time;
	std::vector<double>  openmpell_time;
	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++) {
		openmpcsr_time.push_back(0);
		openmpell_time.push_back(0);
	}
	double cudacsr_time = 0;
	double cudaell_time = 0;

	for (size_t tries = 0; tries < maxtries; tries++) {
		double *x;
		x = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_ser_csr = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_ser_ell = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_omp_csr = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_omp_ell = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_cuda_csr = (double*)malloc(sizeof(double)*csr.collumns);
		double *y_cuda_ell = (double*)malloc(sizeof(double)*csr.collumns);
		for (size_t i = 0; i < csr.collumns; i++) {
			x[i] = i + 1;
			y_ser_csr[i] = 0;
			y_ser_ell[i] = 0;
			y_omp_csr[i] = 0;
			y_omp_ell[i] = 0;
			y_cuda_csr[i] = 0;
			y_cuda_ell[i] = 0;
		}
		
		sercsr_time += csr_matrixvector(csr, x, y_ser_csr);
		serell_time += ellpack_matrixvector(ell, x, y_ser_ell);

		for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++) {
			openmpcsr_time[omp_threads] += omp_csr_matrixvector(csr, x, y_omp_csr,omp_threads+1);
			openmpell_time[omp_threads] += omp_ellpack_matrixvector(ell, x, y_omp_ell, omp_threads+1);
		}

		cudacsr_time += cuda_csr_matrixvector(csr, x, y_cuda_csr);
		cudaell_time += cuda_ellpack_matrixvector(ell, x, y_cuda_ell);



		for (size_t i = 0; i < csr.collumns; i++) {
			if (dif(y_ser_csr[i], y_ser_ell[i]))
				std::cout << "fds2\n";
			if (dif(y_ser_csr[i], y_omp_csr[i]))
				std::cout << "fds3\n";
			if (dif(y_ser_csr[i], y_omp_ell[i]))
				std::cout << "fds4\n";
			if (dif(y_ser_csr[i], y_cuda_csr[i]))
				std::cout << "fds5\n";
			if (dif(y_ser_csr[i], y_cuda_ell[i]))
				std::cout << "fds6\n";

		}
		delete[] y_ser_csr;
		delete[] y_ser_ell;
		delete[] y_omp_csr;
		delete[] y_omp_ell;
		delete[] y_cuda_csr;
		delete[] y_cuda_ell;
		delete[] x;
	}

	serial_results << ((csr.nonzeros*2.0) / (sercsr_time / ((double)maxtries))) << "," << ((csr.nonzeros*2.0) / (serell_time / ((double)maxtries))) << ",";


	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++)
		omp_results << ((csr.nonzeros*2.0) / (openmpcsr_time[omp_threads] / ((double)maxtries))) << ",";	
	for (int omp_threads = 0; omp_threads < MAX_OMP_THREADS; omp_threads++)
		omp_results << ((csr.nonzeros*2.0) / (openmpell_time[omp_threads] / ((double)maxtries))) << ",";
	
	cuda_results << ((csr.nonzeros*2.0) / (cudacsr_time / ((double)maxtries))) << "," << ((csr.nonzeros*2.0) / (cudaell_time / ((double)maxtries))) << ",";

	serial_results << "\n";
	omp_results << "\n";
	cuda_results << "\n";
}

bool has_mtx_extension(char const *name) {
	size_t len = strlen(name);
	return len > 4 && strcmp(name + len - 4, ".mtx") == 0;
}

void test_folder(std::string folder) {
	serial_results.open(folder+"serial_results.csv");
	omp_results.open(folder + "omp_results.csv");
	cuda_results.open(folder + "cuda_results.csv");

	serial_results << "Matrix,nonzeros,collumns,rows,csr,ell,\n";

	omp_results << "Matrix,Max threads,nonzeros,collumns,rows,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << "(" << i << ") csr,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << "(" << i << ") ell,";
	omp_results << "\n";

	cuda_results << "Matrix,nonzeros,collumns,rows,csr,ell,\n";



	serial_results.close();
	omp_results.close();
	cuda_results.close();
}

int main(int argc, char* argv[]) {

	serial_results.open("serial_results.csv");
	omp_results.open("omp_results.csv");
	cuda_results.open("cuda_results.csv");
	
	serial_results << "Matrix,nonzeros,collumns,rows,csr,ell,\n";
	
	omp_results << "Matrix,Max threads,nonzeros,collumns,rows,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << "(" << i << ") csr,";
	for (int i = 1; i <= MAX_OMP_THREADS; i++)
		omp_results << "(" << i << ") ell,";
	omp_results << "\n";
	
	cuda_results << "Matrix,nonzeros,collumns,rows,csr,ell,\n";

	perform_test("olm1000.mtx", 10);

	serial_results.close();
	omp_results.close();
	cuda_results.close();
	int a;
	std::cin >> a;
}