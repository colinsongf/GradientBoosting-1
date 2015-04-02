#ifndef DATA_SET_H
#define DATA_SET_H

#include <string>
#include <vector>
#include "test.h"

class data_set
{
public:
	data_set(std::string file_name, int features_size, int tests_size, bool is_class_first);
	data_set(std::string file_name, int features_size, int tests_size, bool is_class_first, int extra_features);
	data_set(const data_set& other);
	std::vector<float> features;
	std::vector<float> answers;
	std::vector<test> tests;
	int features_size;
	int tests_size;
	void make_train_test_csv(std::string file_name);
	void make_my_data(std::string file_name);
};

#endif // DATA_SET_H
