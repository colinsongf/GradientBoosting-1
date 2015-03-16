#include <sstream>
#include <fstream>
#include <iostream>
#include "data_set.h"

data_set::data_set(std::string file_name, int features_size, int tests_size, bool is_class_first)
	: features_size(features_size), tests_size(tests_size)
{
	answers.resize(tests_size);
	features.resize(features_size * tests_size);
	std::ifstream data_stream(file_name.c_str());
	std::string line;
	for (int i = 0; i < tests_size; i++)
	{
		getline(data_stream, line);
		std::istringstream iss(line);
		float ans;
		float feature;
		std::vector<float> temp;
		if (is_class_first)
		{
			iss >> ans;
			answers[i] = ans;
			for (int j = 0; j < features_size; j++)
			{
				iss >> feature;
				features[j * tests_size + i] = feature;
				temp.push_back(feature);
			}
			tests.push_back(test(temp, ans));
		}
		else
		{
			for (int j = 0; j < features_size; j++)
			{
				iss >> feature;
				features[j * tests_size + i] = feature;
				temp.push_back(feature);
			}
			iss >> ans;
			answers[i] = ans;
			tests.push_back(test(temp, ans));
		}
	}
}
