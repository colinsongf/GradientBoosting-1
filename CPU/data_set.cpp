#include <vector>
#include <sstream>
#include <fstream>
#include "data_set.h"

data_set::data_set(std::string file_name, int features_size, bool is_class_first)
{
	std::ifstream data_stream(file_name);
	std::string line;
	while (getline(data_stream, line))
	{
		std::vector<float> features;
		float value;
		float ans;
		std::istringstream iss(line);
		while (iss >> value)
		{
			features.push_back(value);
		}
		if (is_class_first)
		{
			ans = features.front();
			features.erase(features.begin());
		}
		else
		{
			ans = features.back();
			features.pop_back();
		}
		features.erase(features.begin() + features_size, features.end());
		tests.push_back(test(features, ans));
	}
}

test& data_set::operator[](size_t index)
{
	return tests[index];
}

data_set::iterator data_set::begin()
{
	return tests.begin();
}

data_set::iterator data_set::end()
{
	return tests.end();
}

size_t data_set::size()
{
	return tests.size();
}