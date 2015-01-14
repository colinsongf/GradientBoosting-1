#include <cmath>
#include "gradient_boosting.h"
#include <cstdio>

gradient_boosting::gradient_boosting(data_set& train_set, int iterations, int max_leafs, int max_depth) : train_set(train_set)
{
	trees.push_back(tree(train_set, 1, max_depth));
	coefficients.push_back(1.0);
	for (int i = 0; i < iterations; i++)
	{
		printf("iter #%d\n", i);
		data_set pseudo_residuals = get_pseudo_residuals_set();
		trees.push_back(tree(pseudo_residuals, max_leafs, max_depth));
		coefficients.push_back(calculate_coefficient());
	}
}

float gradient_boosting::calculate_anwser(test& _test)
{
	float ans = 0;
	for (size_t i = 0; i < size(); i++)
	{
		ans += coefficients[i] * trees[i].calculate_anwser(_test);
	}
	return ans;
}

float gradient_boosting::calculate_error(data_set& test_set)
{
	float error = 0;
	for (data_set::iterator cur_test = test_set.begin(); cur_test != test_set.end(); cur_test++)
	{
		float ans = calculate_anwser(*cur_test);
		error += ((ans - cur_test->anwser) * (ans - cur_test->anwser));
	}
	error /= (1.0 * test_set.size());
	return error;
}

data_set gradient_boosting::get_pseudo_residuals_set()
{
	data_set pseudo_residuals(train_set);
	for (data_set::iterator cur_test = pseudo_residuals.begin(); cur_test != pseudo_residuals.end(); cur_test++)
	{
		cur_test->anwser -= calculate_anwser(*cur_test);
	}
	return pseudo_residuals;
}

float gradient_boosting::calculate_coefficient() // golden section search
{
	float left = 0;
	float right = 1;
	float phi = (1.0 + sqrt(5.0)) / 2.0;
	float eps = 1e-2;
	float x1 = right - (right - left) / phi;
	float x2 = left + (right - left) / phi;
	float y1 = calculate_loss_function(x1);
	float y2 = calculate_loss_function(x2);
	while (abs(right - left) > eps)
	{
		if (y1 >= y2)
		{
			left = x1;
			x1 = x2;
			y1 = y2;
			x2 = left + (right - left) / phi; 
			y2 = calculate_loss_function(x2);
		}
		else
		{
			right = x2;
			x2 = x1;
			y2 = y1;
			x1 = right - (right - left) / phi;
			y1 = calculate_loss_function(x1);
		}
	}
	return (left + right) / 2.0;
}

float gradient_boosting::calculate_loss_function(float arg)
{
	float loss = 0;
	for (data_set::iterator cur_test = train_set.begin(); cur_test != train_set.end(); cur_test++)
	{
		float ans = calculate_anwser(*cur_test) + arg * trees.back().calculate_anwser(*cur_test);
		loss += ((ans - cur_test->anwser) * (ans - cur_test->anwser));
	}
	return 0.5 * loss;
}

size_t gradient_boosting::size()
{
	return coefficients.size();
}