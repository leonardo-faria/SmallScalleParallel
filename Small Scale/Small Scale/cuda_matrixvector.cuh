#pragma once

#include "matrix.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>  
#include <iostream>


__host__ float cuda_csr_matrixvector(csr_matrix matrix, double* x, double* y);
__host__ float cuda_ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y);