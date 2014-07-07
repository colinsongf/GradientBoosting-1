#include <vector>
#include "test.h"

test::test(std::vector<double> const& features) : features(features) {}

double test::operator[](int index) const
{
	return features[index];
}