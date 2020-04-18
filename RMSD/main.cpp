#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "point.h"
#include "PointReader.h"
#include "rmsd.cuh"

double compute_rmsd_cpu(std::vector<point*> s1, std::vector<point*> s2);
int sumPointDistances(std::vector<point*> s1, std::vector<point*> s2);
int distanceBetweenTwoPoints(point* p1, point* p2);
int distanceBetweenTwoCoordsSquared(int c1, int c2);

int main() {
	std::string fname1 = "structure1.txt";
	std::string fname2 = "structure2.txt";

	PointReader pr1(fname1);
	PointReader pr2(fname2);
	
	std::vector<point*> points1 = pr1.getPoints();
	std::vector<point*> points2 = pr2.getPoints();


	auto cpu_start_time = std::chrono::high_resolution_clock::now();
	double rmsd = compute_rmsd_cpu(points1, points2);
	auto cpu_stop_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop_time - cpu_start_time);

	printf("RMSD-CPU: %f\n", rmsd);
	std::cout << "Computed in " << duration.count() << " microseconds" << std::endl;
	
	
	cpu_start_time = std::chrono::high_resolution_clock::now();
	rmsd = compute_rmsd_gpu(points1, points2);
	cpu_stop_time = std::chrono::high_resolution_clock::now();
	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop_time - cpu_start_time);

	printf("RMSD-GPU: %f\n", rmsd);
	std::cout << "Computed in " << duration2.count() << " microseconds" << std::endl;
	
	
	
	
	return 0;
}

double compute_rmsd_cpu(std::vector<point*> s1, std::vector<point*> s2) {
	int len1 = int(s1.size());
	int len2 = int(s1.size());
	int sum = sumPointDistances(s1, s2);
	double n_inverse = 1.0 / double(std::min(len1, len2));
	double radicand = n_inverse * double(sum);
	double result = std::sqrt(radicand);
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