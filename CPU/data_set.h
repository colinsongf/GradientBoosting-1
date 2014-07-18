#ifndef DATA_SET_H
#define DATA_SET_H

#include <string>
#include <vector>
#include "test.h"

class data_set
{
public:
	typedef std::vector<test>::iterator iterator;
	data_set(std::string file_name, bool is_class_first = true); //format: ans feature_1 feature_2 ... feature_n || feature_1 feature_2 ... feature_n ans
	test& operator[](size_t index);
	iterator begin();
	iterator end();
	size_t size();
private:
	std::vector<test> tests;
};

#endif // DATA_SET_H