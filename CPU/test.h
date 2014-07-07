#ifndef TEST_H
#define TEST_H

#include <vector>

class test
{
public:
	test(std::vector<double> const& features);
	double operator[](int index) const;
private:
	std::vector<double> features;
};
#endif // TEST_H