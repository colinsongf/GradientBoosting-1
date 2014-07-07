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
		data.push_back(make_pair(ans, features));
	}
}

std::pair<double, std::vector<double> >& data_set::operator[](int index)
{
	return data[index];
}

data_set::iterator data_set::begin()
{
	return data.begin();
}
data_set::iterator data_set::end()
{
	return data.end();
}