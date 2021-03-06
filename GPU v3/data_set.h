#ifndef DATA_SET_H
#define DATA_SET_H

#include <string>
#include <vector>

class data_set
{
public:
	data_set(std::string file_name, int features_size, int tests_size, bool is_class_first);
	std::vector<float> features;
	std::vector<float> answers;
	int features_size;
	int tests_size;
};

#endif // DATA_SET_H