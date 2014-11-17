#include <sstream>
#include <fstream>
#include "data_set.h"

data_set::data_set(std::string file_name, int features_size, int tests_size, bool is_class_first)
	: features_size(features_size), tests_size(tests_size)
{
	answers.resize(tests_size);
	features.resize(features_size * tests_size);
	std::ifstream data_stream(file_name);
	std::string line;
	for (int i = 0; i < tests_size; i++)
	{
		getline(data_stream, line);
		std::istringstream iss(line);
		if (is_class_first)
		{
			iss >> answers[i];
			for (int j = 0; j < features_size; j++)
			{
				iss >> features[j * tests_size + i];
			}
		}
		else
		{
			for (int j = 0; j < features_size; j++)
			{
				iss >> features[j * tests_size + i];
			}
			iss >> answers[i];
		}
	}
}