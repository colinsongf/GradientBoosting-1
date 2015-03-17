#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include "gradient_boosting.h"

gradient_boosting::gradient_boosting(data_set& train_set, int iterations, int max_leafs, int max_depth) : train_set(train_set)
{
	trees.push_back(tree(train_set, 1, max_depth));
	coefficients.push_back(1.0);
	for (int i = 0; i < iterations; i++)
	{
		//printf("iter #%d\n", i);
		data_set pseudo_residuals = get_pseudo_residuals_set();
		trees.push_back(tree(pseudo_residuals, max_leafs, max_depth));
		//clock_t time = clock();
		coefficients.push_back(calculate_coefficient());
		//time = clock() - time;
		//printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	}
}

float gradient_boosting::calculate_answer(test& _test)
{
	float ans = 0;
	for (size_t i = 0; i < size(); i++)
	{
		ans += coefficients[i] * trees[i].calculate_answer(_test);
	}
	return ans;
}

float gradient_boosting::calculate_error(data_set& test_set)
{
	float error = 0;
	for (int i = 0; i < test_set.tests_size; i++)
	{
		float ans = calculate_answer(test_set.tests[i]);
		error += ((ans - test_set.answers[i]) * (ans - test_set.answers[i]));
	}
	error /= (1.0 * test_set.tests_size);
	return error;
}

data_set gradient_boosting::get_pseudo_residuals_set()
{
	data_set pseudo_residuals(train_set);
	for (int i = 0; i < pseudo_residuals.tests_size; i++)
	{
		float ans = calculate_answer(pseudo_residuals.tests[i]);
		pseudo_residuals.tests[i].answer -= ans;
		pseudo_residuals.answers[i] -= ans;
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
	for (int i = 0; i < train_set.tests_size; i++)
	{
		float ans = calculate_answer(train_set.tests[i]) + arg * trees.back().calculate_answer(train_set.tests[i]);
		loss += ((ans - train_set.answers[i]) * (ans - train_set.answers[i]));
	}
	return 0.5 * loss;
}

size_t gradient_boosting::size()
{
	return coefficients.size();
}
