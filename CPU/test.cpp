#include <vector>
#include "test.h"

test::test(std::vector<double> const& features) : features(features) {}
double test::get_feature(int index) const
{
	return features[index];
}
