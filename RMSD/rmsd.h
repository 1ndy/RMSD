#pragma once
#include <vector>
#include <algorithm>

#include "point.h"

double compute_rmsd_cpu(std::vector<point*> s1, std::vector<point*> s2);

int sumPointDistances(std::vector<point*> s1, std::vector<point*> s2);

int distanceBetweenTwoPoints(point* p1, point* p2);

int distanceBetweenTwoCoordsSquared(int c1, int c2);