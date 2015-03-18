#ifndef TEST_H
#define TEST_H

#include <vector>

struct test
{
	test(std::vector<float> const& features, float answer = 0.0);
	test(const test& other);
	std::vector<float> features;
	float answer;
};
#endif // TEST_H