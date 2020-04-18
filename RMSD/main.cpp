#include <cstdio>
#include <iostream>
#include <chrono>
#include <string>

#include "PointReader.h"
#include "rmsd.cuh"
#include "rmsd.h"

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
