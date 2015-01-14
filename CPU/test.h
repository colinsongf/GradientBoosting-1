#ifndef TEST_H
#define TEST_H

#include <vector>

struct test
{
	test(std::vector<float> const& features, float anwser = 0.0);
	std::vector<float> features;
	float anwser;
};
#endif // TEST_H