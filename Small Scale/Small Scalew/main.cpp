#include <iostream>

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
	if (argc != 4) {
		std::cout << "Incorrect number arguments.\nsmallscale ( -dir <dir> ) | ( -f <filename> )  -o <output>" << std::endl;
	}
	readParameters(0, argv);
	readParameters(2, argv);




}