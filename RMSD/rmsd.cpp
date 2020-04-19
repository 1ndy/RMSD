#include "rmsd.h"

float compute_rmsd_cpu(std::vector<point*> s1, std::vector<point*> s2) {
	int len1 = int(s1.size());
	int len2 = int(s2.size());
	int sum = sumPointDistances(s1, s2);
	float n_inverse = 1.0 / float(std::min(len1, len2));
	float radicand = n_inverse * float(sum);
	float result = sqrt(radicand);
	return result;
}

float sumPointDistances(std::vector<point*> s1, std::vector<point*> s2) {
	int len1 = int(s1.size());
	int len2 = int(s2.size());
	int i;
	int runs = std::min(len1, len2);
	float total = 0;
	float distance = 0;
	//#pragma omp parallel for reduction(+:total)
	for (i = 0; i < runs; i++) {
		distance = distanceBetweenTwoPoints(s1.at(i), s2.at(i));
		total += distance;
	}
	return total;
}

float distanceBetweenTwoPoints(point* p1, point* p2) {
	return distanceBetweenTwoCoordsSquared(p1->x, p2->x) + distanceBetweenTwoCoordsSquared(p1->y, p2->y) + distanceBetweenTwoCoordsSquared(p1->z, p2->z);
}

float distanceBetweenTwoCoordsSquared(float c1, float c2) {
	return (c1 - c2) * (c1 - c2);
}