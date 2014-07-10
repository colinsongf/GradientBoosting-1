#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include <vector>
#include "data_set.h"
#include "tree.h"

class gradient_boosting
{
public:
	gradient_boosting(data_set& train_set, int iterations, int max_leafs);
	double calculate_anwser(test& _test);
	double calculate_error(data_set& test_set);
private:
	data_set get_pseudo_residuals_set();
	double calculate_coefficient();
	double calculate_loss_function(double arg);
	size_t size();
	data_set train_set;
	std::vector<tree> trees;
	std::vector<double> coefficients;
};
#endif // GRADIENT_BOOSTING_H