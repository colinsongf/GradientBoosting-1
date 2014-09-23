#ifndef DATA_SET_H
#define DATA_SET_H

#include <string>
#include <thrust\host_vector.h>

class data_set
{
public:
	data_set(std::string file_name, int features_size, int tests_size, bool is_class_first);
	thrust::host_vector<float> features;
	thrust::host_vector<float> answers;
	int features_size;
	int tests_size;
};

#endif // DATA_SET_H