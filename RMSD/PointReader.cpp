#include "PointReader.h"

//takes a filename to instantiate
//opens the file and reads three ints off every line
//stores those points in a list
PointReader::PointReader(std::string fname) {
	std::ifstream file(fname);
	points = std::vector<point*>();

	if (!file.is_open()) {
		fprintf(stderr, "Could not open file '%s'\n", fname.c_str());
	}
	else {
		int line_number = 1;
		int read_len = 0;
		char input[LINE_SIZE];
		while (!file.eof()) {
			file.getline(input, LINE_SIZE - 1);
			point* p = new point;
			read_len = sscanf(input, "%f %f %f", &(p->x), &(p->y), &(p->z));
			if (read_len != 3) {
				fprintf(stderr, "Error in file on line %d\n", line_number);
			}
			else {
				points.push_back(p);
			}
			line_number++;
		}
		file.close();
	}
}

std::vector<point*> PointReader::getPoints() {
	return points;
}