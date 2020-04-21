#include "matrix_utils.h"

matrix_t createMatrix(int x, int y)
{
	matrix_t r;
	r.m = x;
	r.n = y;
	r.grid = (float**)calloc(x, sizeof(float*));
	for (int i = 0; i < x; i++) {
		r.grid[i] = (float*)calloc(3, sizeof(float));
	}
	return r;
}

matrix_t convertSTLVectorToArray(std::vector<point*> s1)
{
	int size = s1.size();
	matrix_t a = createMatrix(size, 3);
	for (int i = size - 1; i >= 0; i--) {
		point* p = s1.at(i);
		a.grid[i][0] = p->x;
		a.grid[i][1] = p->y;
		a.grid[i][2] = p->z;
	}
	return a;
}

matrix_t transposeMatrix(matrix_t a)
{
	matrix_t ta = createMatrix(a.n, a.m);
	for (int i = 0; i < a.m; i++) {
		for (int j = 0; j < a.n; j++) {
			ta.grid[j][i] = a.grid[i][j];
		}
	}
	return ta;
}

matrix_t multiplyMatrices(matrix_t a, matrix_t b)
{
	return matrix_t();
}

matrix_t invertMatrix(matrix_t a)
{
	return matrix_t();
}

matrix_t exponentiate_matrix(matrix_t a)
{
	return matrix_t();
}

void normalizeWithCentroid(matrix_t a)
{
}

float calculateCentroid(matrix_t a)
{
	return 0.0f;
}
