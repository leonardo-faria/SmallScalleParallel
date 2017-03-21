#pragma once
#include <vector>
class matrix {
public:
	matrix(int rows,int collumns);
	virtual ~matrix() = 0 {};
	const int rows;
	const int collumns;
};

class csr_matrix :public matrix {
public:
	csr_matrix();
	std::vector<int> irp;
	std::vector<int> ja;
	std::vector<double> as;
};

class ellpack_matrix :public matrix {
public:
	ellpack_matrix();
	const int maxnzr;
	std::vector<std::vector<int>> ja;
	std::vector<std::vector<double>> as;
};