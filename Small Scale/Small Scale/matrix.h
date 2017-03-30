#pragma once
#include <string>
#include <vector>
class matrix {
public:
	matrix(std::string filename);
	virtual ~matrix() = 0 {};
	int rows;
	int collumns;
	int nonzeros;
protected:
	std::vector<int> work_irp;
	std::vector<int> work_ja;
	std::vector<double> work_as;
	std::vector<int> coo_irp;
	std::vector<int> coo_ja;
	std::vector<double> coo_as;
	void BottomUpMergeSort(int n);
	void BottomUpMerge(int iLeft, int iRight, int iEnd);
	void CopyArray(int n);
	
};

class csr_matrix :public matrix {
public:
	csr_matrix(std::string filename);
	int irp_size;
	int* irp;
	int* ja;
	double* as;
};

class ellpack_matrix :public matrix {
public:
	ellpack_matrix(std::string filename);
	int maxnzr;
	int* ja;
	double* as;
};