#include <vector>
#include <cuda_runtime.h>

#include "point.h"


float compute_rmsd_gpu(std::vector<point*> s1, std::vector<point*> s2);

__global__ void sumPointDistancesGPU(float* s1x, float* s1y, float* s1z, float* s2x, float* s2y, float* s2z, float* r);
