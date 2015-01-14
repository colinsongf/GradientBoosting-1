#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include <vector>
#include "data_set.h"
#include "tree.h"

class gradient_boosting
{
public:
	gradient_boosting(data_set& train_set, int iterations, int max_leafs, int max_depth);
	float calculate_anwser(test& _test);
	float calculate_error(data_set& test_set);
private:
	data_set get_pseudo_residuals_set();
	float calculate_coefficient();
	float calculate_loss_function(float arg);
	size_t size();
	data_set train_set;
	std::vector<tree> trees;
	std::vector<float> coefficients;
};
#endif // GRADIENT_BOOSTING_H