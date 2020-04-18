#include "rmsd.h"

double compute_rmsd_cpu(std::vector<point*> s1, std::vector<point*> s2) {
	int len1 = int(s1.size());
	int len2 = int(s1.size());
	int sum = sumPointDistances(s1, s2);
	double n_inverse = 1.0 / double(std::min(len1, len2));
	double radicand = n_inverse * double(sum);
	double result = sqrt(radicand);
	//printf("%f * %d\n", n_inverse, sum);
	return result;
}

int sumPointDistances(std::vector<point*> s1, std::vector<point*> s2) {
	int len1 = int(s1.size());
	int len2 = int(s2.size());
	int i;
	int runs = std::min(len1, len2);
	int total = 0;
	int distance = 0;
	//#pragma omp parallel for reduction(+:total)
	for (i = 0; i < runs; i++) {
		distance = distanceBetweenTwoPoints(s1.at(i), s2.at(i));
		//printf("distance between (%d, %d, %d) and (%d, %d, %d): %d\n", s1->at(i)->x, s1->at(i)->y, s1->at(i)->z, s2->at(i)->x, s2->at(i)->y, s2->at(i)->z, distance);
		total += distance;
	}
	return total;
}

int distanceBetweenTwoPoints(point* p1, point* p2) {
	return distanceBetweenTwoCoordsSquared(p1->x, p2->x) + distanceBetweenTwoCoordsSquared(p1->y, p2->y) + distanceBetweenTwoCoordsSquared(p1->z, p2->z);
}

int distanceBetweenTwoCoordsSquared(int c1, int c2) {
	return (c1 - c2) * (c1 - c2);
}