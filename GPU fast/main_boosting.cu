#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <iostream>
#include "gradient_boosting.h"

int main()
{
	data_set train_set("housing-train.txt", false);
	data_set test_set("housing-test.txt", false);
	gradient_boosting grad_boost(train_set, 4, 10);
	double error = grad_boost.calculate_error(test_set);
	std::cout << " test error: " << error << std::endl;
	/*double best_error = 1e5;
	int best_i = 0;
	int best_j = 0;
	for (int i = 0; i < 20; i++)
	{
		for (int j = 1; j < 30; j++)
		{
			gradient_boosting grad_boost(train_set, i, j);
			double error = grad_boost.calculate_error(test_set);
			if (error < best_error)
			{
				best_error = error;
				best_i = i;
				best_j = j;
			}
			std::cout << "i: " << i << " j: " << j << " test error: " << error << std::endl;
		}
	}
	std::cout << best_error << " i: " << best_i << " j: " << best_j << std::endl;*/
	return 0;
}