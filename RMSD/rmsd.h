#pragma once
#include <vector>
#include <algorithm>

#include "point.h"

float compute_rmsd_cpu(std::vector<point*> s1, std::vector<point*> s2);

float sumPointDistances(std::vector<point*> s1, std::vector<point*> s2);

float distanceBetweenTwoPoints(point* p1, point* p2);

float distanceBetweenTwoCoordsSquared(float c1, float c2);