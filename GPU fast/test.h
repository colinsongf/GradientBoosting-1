#ifndef TEST_H
#define TEST_H

#include <vector>

struct test
{
	test(std::vector<double> const& features, double anwser = 0.0);
	std::vector<double> features;
	double anwser;
};
#endif // TEST_H