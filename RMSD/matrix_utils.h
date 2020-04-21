#pragma once
#include <vector>
#include "point.h"

struct matrix_t {
	int m;
	int n;
	float** grid;
};

void createMatrix(int x, int y);

matrix_t convertSTLVectorToArray(std::vector<point*> s1, std::vector<point*> s2);

matrix_t transposeMatrix(matrix_t a);

matrix_t multiplyMatrices(matrix_t a, matrix_t b);

matrix_t invertMatrix(matrix_t a);

matrix_t exponentiate_matrix(matrix_t a);

void normalizeWithCentroid(matrix_t a);

float calculateCentroid(matrix_t a);
