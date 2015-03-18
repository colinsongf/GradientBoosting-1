#include <vector>
#include "test.h"

test::test(std::vector<float> const& features, float answer) : features(features), answer(answer) {}

test::test(const test& other) : features(other.features), answer(other.answer) {}
