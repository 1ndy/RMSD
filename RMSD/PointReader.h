#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

#include "point.h"

#define LINE_SIZE 255

class PointReader
{
public:
	PointReader(std::string fname);
	std::vector<point*> getPoints();
private:
	std::vector<point*> points;
};
