#include <sstream>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cstdlib>
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

data_set::data_set(std::string file_name, int _features_size, int tests_size, bool is_class_first, int extra_features)
	: tests_size(tests_size)
{
	int old_features = _features_size;
	features_size = old_features + extra_features;
	
	answers.resize(tests_size);
	features.resize(features_size * tests_size);
	std::ifstream data_stream(file_name.c_str());
	std::string line;
	srand(time(NULL));
	float min_w = 0.1;
	float max_w = 50.0;
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
			for (int j = 0; j < old_features; j++)
			{
				iss >> feature;
				features[j * tests_size + i] = feature;
				temp.push_back(feature);
			}
			for (int j = old_features; j < features_size; j++)
			{
				feature = ((max_w - min_w) * ((float)rand() / (float)RAND_MAX) + min_w);
				features[j * tests_size + i] = feature;
				temp.push_back(feature);
			}
			tests.push_back(test(temp, ans));
		}
		else
		{
			for (int j = 0; j < old_features; j++)
			{
				iss >> feature;
				features[j * tests_size + i] = feature;
				temp.push_back(feature);
			}
			for (int j = old_features; j < features_size; j++)
			{
				feature = ((max_w - min_w) * ((float)rand() / (float)RAND_MAX) + min_w);
				features[j * tests_size + i] = feature;
				temp.push_back(feature);
			}
			iss >> ans;
			answers[i] = ans;
			tests.push_back(test(temp, ans));
		}
	}
}

data_set::data_set(const data_set& other) : features(other.features), answers(other.answers), tests(other.tests),
	features_size(other.features_size), tests_size(other.tests_size) {} 