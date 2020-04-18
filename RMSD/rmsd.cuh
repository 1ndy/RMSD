#include <vector>
#include <cuda_runtime.h>

#include "point.h"


double compute_rmsd_gpu(std::vector<point*> s1, std::vector<point*> s2);

__global__ void sumPointDistancesGPU(int* s1x, int* s1y, int* s1z, int* s2x, int* s2y, int* s2z, int* r);
