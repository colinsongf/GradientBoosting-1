#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
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

void data_set::make_train_test_csv(std::string file_name)
{
	std::ofstream train_features(file_name + "_train_features.csv");
	std::ofstream test_features(file_name + "_test_features.csv");
	std::ofstream train_targets(file_name + "_train_targets.csv");
	std::ofstream test_targets(file_name + "_test_targets.csv");
	int train_size = (int)floor(tests_size * 0.7);
	std::random_shuffle(tests.begin(), tests.end());
	train_features.precision(5);
	test_features.precision(5);
	train_targets.precision(5);
	test_targets.precision(5);
	for (int i = 0; i < train_size; i++)
	{
		train_targets << std::fixed << tests[i].answer << std::endl;
		for (int j = 0; j < features_size; j++)
		{
			train_features << std::fixed << tests[i].features[j];
			if (j < features_size - 1)
			{
				train_features << ",";
			}
		}
		train_features << std::endl;
	}
	for (int i = train_size; i < tests_size; i++)
	{
		test_targets << std::fixed << tests[i].answer << std::endl;
		for (int j = 0; j < features_size; j++)
		{
			test_features << std::fixed << tests[i].features[j];
			if (j < features_size - 1)
			{
				test_features << ",";
			}
		}
		test_features << std::endl;
	}
}


void data_set::make_my_data(std::string file_name)
{
	std::ofstream train_set(file_name + "_train_set.csv");
	std::ofstream test_set(file_name + "_test_set.csv");
	int train_size = (int)floor(tests_size * 0.7);
	train_set.precision(5);
	test_set.precision(5);
	for (int i = 0; i < train_size; i++)
	{
		for (int j = 0; j < features_size; j++)
		{
			train_set << std::fixed << tests[i].features[j] << " ";
		}
		train_set << std::fixed << tests[i].answer << std::endl;
	}
	for (int i = train_size; i < tests_size; i++)
	{
		for (int j = 0; j < features_size; j++)
		{
			test_set << std::fixed << tests[i].features[j] << " ";
		}
		test_set << std::fixed << tests[i].answer << std::endl;
	}
}

data_set::data_set(const data_set& other) : features(other.features), answers(other.answers), tests(other.tests),
	features_size(other.features_size), tests_size(other.tests_size) {} 
