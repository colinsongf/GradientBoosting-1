#include <vector>
#include <sstream>
#include <fstream>
#include "data_set.h"

data_set::data_set(std::string file_name)
{
	std::ifstream data_stream(file_name);
	std::string line;
	while (getline(data_stream, line))
	{
		std::vector<double> features;
		double value;
		double ans;
		std::istringstream iss(line);
		iss >> ans;
		while (iss >> value)
		{
			features.push_back(value);
		}
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