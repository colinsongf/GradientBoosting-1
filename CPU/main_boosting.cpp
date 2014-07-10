#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <iostream>
#include "gradient_boosting.h"

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	data_set train_set("wine-train.txt");
	data_set test_set("wine-test.txt");
	gradient_boosting grad_boost(train_set, 4, 10);
	double error = grad_boost.calculate_error(test_set);
	std::cout << " test error: " << error << std::endl;
	/*double best_error = 1e5;
	int best_i = 0;
	int best_j = 0;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 2; j < 17; j+=2)
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