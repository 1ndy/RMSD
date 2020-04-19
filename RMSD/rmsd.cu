#include "rmsd.cuh"
#include <algorithm>
#include <stdio.h>

float compute_rmsd_gpu(std::vector<point*> s1, std::vector<point*> s2) {
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

	float* h_s1Ax = (float*)calloc(size, sizeof(point));
	float* h_s1Ay = (float*)calloc(size, sizeof(point));
	float* h_s1Az = (float*)calloc(size, sizeof(point));

	float* h_s2Ax = (float*)calloc(size, sizeof(point));
	float* h_s2Ay = (float*)calloc(size, sizeof(point));
	float* h_s2Az = (float*)calloc(size, sizeof(point));


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
	float* d_s1Ax = 0;
	float* d_s1Ay = 0;
	float* d_s1Az = 0;
	
	float* d_s2Ax = 0;
	float* d_s2Ay = 0;
	float* d_s2Az = 0;
	
	cudaMalloc(&d_s1Ax, sizeof(float) * size);
	cudaMalloc(&d_s1Ay, sizeof(float) * size);
	cudaMalloc(&d_s1Az, sizeof(float) * size);
	
	cudaMalloc(&d_s2Ax, sizeof(float) * size);
	cudaMalloc(&d_s2Ay, sizeof(float) * size);
	cudaMalloc(&d_s2Az, sizeof(float) * size);

	//allocate a results array
	float* d_sR = 0;
	cudaMalloc(&d_sR, sizeof(float) * size);

	//copy memory
	cudaMemcpy(d_s1Ax, h_s1Ax, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s1Ay, h_s1Ay, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s1Az, h_s1Az, sizeof(float) * size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_s2Ax, h_s2Ax, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s2Ay, h_s2Ay, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s2Az, h_s2Az, sizeof(float) * size, cudaMemcpyHostToDevice);

	//now we can start the computation
	float n_inverse = 1.0 / std::min(size1, size2);
	float sum = 0;
	sumPointDistancesGPU<<<blocksPerGrid, threadsPerBlock>>>(d_s1Ax, d_s1Ay, d_s1Az, d_s2Ax, d_s2Ay, d_s2Az, d_sR);
	cudaDeviceSynchronize();

	float* results = (float*)malloc(sizeof(float) * std::min(size1, size2));
	cudaMemcpy(results, d_sR, std::min(size1, size2) * sizeof(float), cudaMemcpyDeviceToHost);

	#pragma omp parallel for reduction(+:sum)
	for (i = 0; i < std::min(size1, size2); i++) {
		sum += results[i];
	}

	float radicand = n_inverse * float(sum);
	return sqrt(radicand);
}

__global__
void sumPointDistancesGPU(float* s1x, float* s1y, float* s1z, float* s2x, float* s2y, float* s2z, float* r) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float x1 = s1x[i];
	float y1 = s1y[i];
	float z1 = s1z[i];
	float x2 = s2x[i];
	float y2 = s2y[i];
	float z2 = s2z[i];

	float distance = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
	//printf("distance between (%d, %d, %d) and (%d, %d, %d): %d\n", x1, y1, z1, x2, y2, z2, distance);
	r[i] = distance;
}