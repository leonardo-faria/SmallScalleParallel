#include "matrix.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
extern "C" {
#include "mmio.h"
}

ellpack_matrix::ellpack_matrix(std::string filename) : matrix(filename) {
	maxnzr = 0;
	std::vector<std::vector<int>> work_ja = std::vector<std::vector<int>>();
	std::vector<std::vector<double>> work_as = std::vector<std::vector<double>>();
	int col = 0;
	for (size_t i = 0; i < coo_irp.size(); i++) {
		col = coo_irp[i];
		work_ja.push_back(std::vector<int>());
		work_as.push_back(std::vector<double>());
		size_t j;
		for (j = i; j < coo_irp.size() && coo_irp[j] == col; j++) {
			work_ja[work_ja.size() - 1].push_back(coo_ja[j]);
			work_as[work_as.size() - 1].push_back(coo_as[j]);
		}
		if (maxnzr < work_ja[work_ja.size() - 1].size())
			maxnzr = work_ja[work_ja.size() - 1].size();
		i = j - 1;
	}
	for (size_t i = 0; i < work_ja.size(); i++) {
		while (work_ja[i].size() < maxnzr) {
			work_ja[i].push_back(work_ja[i][work_ja[i].size() - 1]);
			work_as[i].push_back(0);
		}
	}

	ja = (int*)malloc(sizeof(int)*work_ja.size()*maxnzr);
	as = (double*)malloc(sizeof(double)*work_as.size()*maxnzr);
	for (size_t i = 0; i < work_ja.size(); i++) {
		for (size_t j = 0; j < maxnzr; j++) {
			ja[i*maxnzr +j] = work_ja[i][j];
			as[i*maxnzr + j] = work_as[i][j];
		}
		work_ja[i].clear();
		work_as[i].clear();
		std::vector<int>().swap(work_ja[i]);
		std::vector<double>().swap(work_as[i]);

	}
	work_ja.clear();
	work_as.clear();
	std::vector<std::vector<int>>().swap(work_ja);
	std::vector<std::vector<double>>().swap(work_as);

	coo_irp.clear();
	coo_as.clear();
	coo_ja.clear();
	std::vector<int>().swap(coo_irp);
	std::vector<int>().swap(coo_ja);
	std::vector<double>().swap(coo_as);
}

csr_matrix::csr_matrix(std::string filename) : matrix(filename) {

	work_irp = std::vector<int>();
	work_irp.push_back(0);
	for (size_t i = 0; i < coo_irp.size(); i++) {
		int col = coo_irp[i];
		size_t j;
		for (j = i; j < coo_irp.size() && coo_irp[j] == col; j++);
		if (j < coo_irp.size())
			work_irp.push_back(j);
		i = j - 1;
	}
	work_irp.push_back(coo_irp.size());

	irp_size = work_irp.size();
	irp = (int*)malloc(sizeof(int)*work_irp.size());
	ja = (int*)malloc(sizeof(int)*coo_ja.size());
	as = (double*)malloc(sizeof(double)*coo_as.size());
	std::copy(work_irp.begin(), work_irp.end(), irp);
	std::copy(coo_ja.begin(), coo_ja.end(), ja);
	std::copy(coo_as.begin(), coo_as.end(), as);

	coo_irp.clear();
	coo_as.clear();
	coo_ja.clear();
	std::vector<int>().swap(coo_irp);
	std::vector<int>().swap(coo_ja);
	std::vector<double>().swap(coo_as);
}

matrix::matrix(std::string filename) {
	int ret_code;
	MM_typecode matcode;
	FILE *f;
	if ((f = fopen(filename.c_str(), "r")) == NULL)
		throw std::exception(((std::string) ("Unable to open file " + filename)).c_str());
	if (mm_read_banner(f, &matcode) != 0) {
		throw std::exception("Could not process Matrix Market banner.\n");
	}
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
		mm_is_sparse(matcode)) {
		throw std::exception(((std::string) ("Sorry, this application does not support Market Market type: [" + (std::string) mm_typecode_to_str(matcode) + "%s]\n")).c_str());
		exit(1);
	}
	if ((ret_code = mm_read_mtx_crd_size(f, &rows, &collumns, &nonzeros)) != 0)
		throw std::exception("Unable to read matrix crd size");
	//irp = std::vector<int>((int *)malloc(nonzeros * sizeof(int)), nonzeros);
	coo_irp = std::vector<int>();
	coo_ja = std::vector<int>();
	coo_as = std::vector<double>();
	int temp_irp, temp_ja;
	double temp_as;
	for (int i = 0; i < nonzeros; i++) {
		fscanf(f, "%d %d %lg\n", &temp_irp, &temp_ja, &temp_as);

		coo_irp.push_back(temp_irp);
		coo_ja.push_back(temp_ja);
		coo_as.push_back(temp_as);
		coo_irp[coo_irp.size() - 1]--;
		coo_ja[coo_ja.size() - 1]--;

		if ((mm_is_hermitian(matcode) || mm_is_symmetric(matcode) || mm_is_skew(matcode)) && temp_irp != temp_ja) {
			coo_irp.push_back(temp_ja);
			coo_ja.push_back(temp_irp);
			coo_as.push_back(temp_as);
			coo_irp[coo_irp.size() - 1]--;
			coo_ja[coo_ja.size() - 1]--;

		}
	}
	if (mm_is_hermitian(matcode) || mm_is_symmetric(matcode) || mm_is_skew(matcode))
		nonzeros = coo_ja.size();
	work_irp = std::vector<int>(coo_irp.size());
	work_ja = std::vector<int>(coo_ja.size());
	work_as = std::vector<double>(coo_as.size());

	BottomUpMergeSort(coo_irp.size());
	work_irp.clear();
	work_as.clear();
	work_ja.clear();
	std::vector<int>().swap(work_irp);
	std::vector<int>().swap(work_ja);
	std::vector<double>().swap(work_as);
}

void matrix::BottomUpMergeSort(int n) {
	// Each 1-element run in A is already "sorted".
	// Make successively longer sorted runs of length 2, 4, 8, 16... until whole array is sorted.
	for (int width = 1; width < n; width = 2 * width) {
		// Array A is full of runs of length width.
		for (int i = 0; i < n; i = i + 2 * width) {
			// Merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
			// or copy A[i:n-1] to B[] ( if(i+width >= n) )
			BottomUpMerge(i, std::min(i + width, n), std::min(i + 2 * width, n));
		}
		// Now work array B is full of runs of length 2*width.
		// Copy array B to array A for next iteration.
		// A more efficient implementation would swap the roles of A and B.
		CopyArray(n);
		// Now array A is full of runs of length 2*width.
	}
}
void matrix::BottomUpMerge(int iLeft, int iRight, int iEnd) {
	int i = iLeft, j = iRight;
	// While there are elements in the left or right runs...
	for (int k = iLeft; k < iEnd; k++) {
		// If left run head exists and is <= existing right run head.
		if (i < iRight && (j >= iEnd || coo_irp[i] < coo_irp[j] || (coo_irp[i] == coo_irp[j] && coo_ja[i] <= coo_ja[j]))) {
			work_as[k] = coo_as[i];
			work_irp[k] = coo_irp[i];
			work_ja[k] = coo_ja[i];
			i = i + 1;
		} else {
			work_as[k] = coo_as[j];
			work_irp[k] = coo_irp[j];
			work_ja[k] = coo_ja[j];
			j = j + 1;
		}
	}
}
void  matrix::CopyArray(int n) {
	for (int i = 0; i < n; i++) {
		coo_as[i] = work_as[i];
		coo_irp[i] = work_irp[i];
		coo_ja[i] = work_ja[i];
	}
}
