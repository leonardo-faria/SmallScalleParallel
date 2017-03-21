#include <iostream>
#include "cuda_matrixvector.h"
#include "openmp_matrixvector.h"


/*
smallscale ( -dir <dir> ) | ( -f <filename> )  -o <output>
*/

void readParameters(int n, char* argv[]) {
	if (strcmp(argv[n], "-dir") == 0) {

	} else if (strcmp(argv[n], "-f") == 0) {

	} else if (strcmp(argv[n], "-o") == 0) {

	}
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		std::cout << "Incorrect number arguments.\nsmallscale ( -dir <dir> ) | ( -f <filename> )  -o <output>" << std::endl;
	}
	readParameters(1, argv);
	readParameters(3, argv);




}