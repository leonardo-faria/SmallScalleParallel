#include "cuda_matrixvector.cuh"

#define BLOCK_DIM 64
#define GRID_DIM 64
template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void device_cuda_csr_matrixvector(int num_rows, int* irp, int* ja, double* as, double* x, double* y) {
	__shared__ double sdata[blockSize];
	unsigned int tid = threadIdx.x;
	for (int i = blockIdx.x; i < num_rows; i += gridDim.x) {
		sdata[tid] = 0;
		for (int j = irp[i] + tid; j < irp[i + 1]; j += blockDim.x) {
			sdata[tid] += as[j] * x[ja[j]];
		}
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize>(sdata, tid);
		if (tid == 0) y[i] = sdata[0];

	}

}

template <unsigned int blockSize>
__global__ void device_cuda_ellpack_matrixvector(int num_rows,int  maxnzr, int* ja, double* as, double* x, double* y) {
	
	__shared__ double sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i_delta = maxnzr*gridDim.x;
	unsigned int i_end = num_rows*maxnzr;
	unsigned int I = blockIdx.x;
	for (int i = maxnzr*blockIdx.x; i < i_end; i += i_delta) {
		sdata[tid] = 0;
		unsigned int j_end = i + maxnzr;
		for (int j = tid+i; j < j_end; j+=blockDim.x) {
		sdata[tid] = as[j] * x[ja[j]];
		}
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize>(sdata, tid);
		if (tid == 0) y[I] = sdata[0];
		I += gridDim.x;
	}

}

__host__ float cuda_csr_matrixvector(csr_matrix matrix, double* x, double* y) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int *d_ja, *d_irp;
	double *d_as, *d_x, *d_y;

	cudaMalloc((void**)&d_irp, matrix.irp_size * sizeof(int));
	cudaMalloc((void**)&d_ja, matrix.nonzeros * sizeof(int));
	cudaMalloc((void**)&d_as, matrix.nonzeros * sizeof(double));

	cudaMalloc((void**)&d_y, matrix.collumns * sizeof(double));
	cudaMalloc((void**)&d_x, matrix.collumns * sizeof(double));

	cudaMemcpy(d_irp, matrix.irp, matrix.irp_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ja, matrix.ja, matrix.nonzeros * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_as, matrix.as, matrix.nonzeros * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, matrix.collumns * sizeof(double), cudaMemcpyHostToDevice);


	cudaEventRecord(start);
	device_cuda_csr_matrixvector<BLOCK_DIM> << <GRID_DIM, BLOCK_DIM >> > (matrix.rows, d_irp, d_ja, d_as, d_x, d_y); cudaEventRecord(stop, 0);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	cudaMemcpy(y, d_y, matrix.collumns * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_ja);
	cudaFree(d_irp);
	cudaFree(d_as);
	cudaFree(d_y);
	cudaFree(d_x);
	return time*1000;
}

__host__ float cuda_ellpack_matrixvector(ellpack_matrix matrix, double* x, double* y) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int *d_ja;
	double *d_as, *d_x, *d_y;

	cudaMalloc((void**)&d_ja, matrix.maxnzr*matrix.rows * sizeof(int));
	cudaMalloc((void**)&d_as, matrix.maxnzr*matrix.rows * sizeof(double));

	cudaMalloc((void**)&d_y, matrix.collumns * sizeof(double));
	cudaMalloc((void**)&d_x, matrix.collumns * sizeof(double));

	cudaMemcpy(d_ja, matrix.ja, matrix.maxnzr*matrix.rows * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_as, matrix.as, matrix.maxnzr*matrix.rows * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, matrix.collumns * sizeof(double), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);
	device_cuda_ellpack_matrixvector<BLOCK_DIM> << <GRID_DIM, BLOCK_DIM >> > (matrix.rows, matrix.maxnzr, d_ja, d_as, d_x, d_y); cudaEventRecord(stop, 0);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	cudaMemcpy(y, d_y, matrix.collumns * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_ja);
	cudaFree(d_as);
	cudaFree(d_y);
	cudaFree(d_x);
	return time*1000;
}

/*
int maine(csr_matrix matrix, std::vector<double> x, std::vector<double>& y) {
	const int bsx = 16;
	const int bsy = 64;
	const dim3 BLOCK_DIM(bsx, bsy);

	int *d_ja, *d_irp;
	double *d_as, *d_x, *d_y;

	cudaMalloc((void**)&d_irp, matrix.irp.size() * sizeof(int));
	cudaMalloc((void**)&d_ja, matrix.ja.size() * sizeof(int));
	cudaMalloc((void**)&d_as, matrix.as.size() * sizeof(double));
	cudaMalloc((void**)&d_x, x.size() * sizeof(double));
	cudaMalloc((void**)&d_y, y.size() * sizeof(double));
	cudaMemcpy(d_ja, matrix.ja., ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, COLS * sizeof(float), cudaMemcpyHostToDevice);




	const dim3 GRID_DIM(ROWS / BLOCK_DIM.x, COLS / BLOCK_DIM.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	gpuMatrixVector << <GRID_DIM, BLOCK_DIM >> > (COLS, d_A, d_x);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time1;
	cudaEventElapsedTime(&time1, start, stop);
	std::cout << "MULT time:" << time1 << "ms\n"; // Very accurate
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//reduce << <ROWS, 1 >> > (COLS,d_A, d_y );
	reduce6<512> << <ROWS, 512, 512 * sizeof(float) >> > (d_A, d_y, COLS);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "REDUCE time:" << time << "ms\n"; // Very accurate
	cudaEventDestroy(start);
	cudaEventDestroy(stop);




	cudaMemcpy(h_y_d, d_y, ROWS * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_A, d_A, ROWS*COLS * sizeof(int), cudaMemcpyDeviceToHost);



	float diff = 0.0f;
	for (int row = 0; row < ROWS; ++row) {
		diff = std::max(diff, std::abs((h_y[row] - h_y_d[row]) / h_y[row]));
	}
	std::cout << "Max diff = " << diff << std::endl;

	std::cout << "giga flops=" << (2 * ROWS*COLS) / (1000 * (time1 + time));
	cudaFree(d_A);
	cudaFree(d_y);
	cudaFree(d_x);

	delete[] h_A;
	delete[] h_y;
	delete[] h_y_d;
	delete[] h_x;

	int n;
	std::cin >> n;
	return 0;
}*/