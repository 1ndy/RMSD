#include "rmsd.cuh"
#include <algorithm>
#include <stdio.h>

double compute_rmsd_gpu(std::vector<point*> s1, std::vector<point*> s2) {
	//first convert the vectors into arrays
	int size1 = s1.size();
	int size2 = s2.size();

	int blocksPerGrid;
	int threadsPerBlock;
	int min = std::min(size1, size2);
	int size = min + (128 - min % 128); //make it an even multiple of 128 for cuda
	threadsPerBlock = std::min(size, 128);
	blocksPerGrid = size / threadsPerBlock;
	printf("Running CUDA with %d threads and %d blocks on %d points\n", threadsPerBlock, blocksPerGrid, min);

	int* h_s1Ax = (int*)calloc(size, sizeof(point));
	int* h_s1Ay = (int*)calloc(size, sizeof(point));
	int* h_s1Az = (int*)calloc(size, sizeof(point));

	int* h_s2Ax = (int*)calloc(size, sizeof(point));
	int* h_s2Ay = (int*)calloc(size, sizeof(point));
	int* h_s2Az = (int*)calloc(size, sizeof(point));


	//allocate the host while copying arrays
	int i;
	point* p;
	for (i = 0; i < std::min(size,size1); i++) {
		p = s1.at(i);
		h_s1Ax[i] = p->x;
		h_s1Ay[i] = p->y;
		h_s1Az[i] = p->z;
	}
	for (i = 0; i < std::min(size,size2); i++) {
		p = s2.at(i);
		h_s2Ax[i] = p->x;
		h_s2Ay[i] = p->y;
		h_s2Az[i] = p->z;
	}

	//allocate space on the device
	int* d_s1Ax = 0;
	int* d_s1Ay = 0;
	int* d_s1Az = 0;
	
	int* d_s2Ax = 0;
	int* d_s2Ay = 0;
	int* d_s2Az = 0;
	
	cudaMalloc(&d_s1Ax, sizeof(int) * size);
	cudaMalloc(&d_s1Ay, sizeof(int) * size);
	cudaMalloc(&d_s1Az, sizeof(int) * size);
	
	cudaMalloc(&d_s2Ax, sizeof(int) * size);
	cudaMalloc(&d_s2Ay, sizeof(int) * size);
	cudaMalloc(&d_s2Az, sizeof(int) * size);

	//allocate a results array
	int* d_sR = 0;
	cudaMalloc(&d_sR, sizeof(int) * size);

	//copy memory
	cudaMemcpy(d_s1Ax, h_s1Ax, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s1Ay, h_s1Ay, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s1Az, h_s1Az, sizeof(int) * size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_s2Ax, h_s2Ax, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s2Ay, h_s2Ay, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s2Az, h_s2Az, sizeof(int) * size, cudaMemcpyHostToDevice);

	//now we can start the computation
	double n_inverse = 1.0 / std::min(size1, size2);
	int sum = 0;
	sumPointDistancesGPU<<<blocksPerGrid, threadsPerBlock>>>(d_s1Ax, d_s1Ay, d_s1Az, d_s2Ax, d_s2Ay, d_s2Az, d_sR);
	cudaDeviceSynchronize();

	int* results = (int*)malloc(sizeof(int) * std::min(size1, size2));
	cudaMemcpy(results, d_sR, std::min(size1, size2) * sizeof(int), cudaMemcpyDeviceToHost);

	#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < std::min(size1, size2); i++) {
		sum += results[i];
	}

	double radicand = n_inverse * double(sum);
	return sqrt(radicand);
}

__global__
void sumPointDistancesGPU(int* s1x, int* s1y, int* s1z, int* s2x, int* s2y, int* s2z, int *r) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int x1 = s1x[i];
	int y1 = s1y[i];
	int z1 = s1z[i];
	int x2 = s2x[i];
	int y2 = s2y[i];
	int z2 = s2z[i];

	int distance = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
	//printf("distance between (%d, %d, %d) and (%d, %d, %d): %d\n", x1, y1, z1, x2, y2, z2, distance);
	r[i] = distance;
}