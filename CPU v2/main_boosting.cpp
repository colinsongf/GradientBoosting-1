#include <iostream>
#include <cstdio>
#include <ctime>

#include "gradient_boosting.h"

int main()
{
	int features_size = 21;
	int tests_size = 1500;
	data_set train_set("Prototask.train", features_size, tests_size, false);
	data_set test_set("Prototask.test", features_size, 1193, false);
	freopen("out.txt", "w", stdout);
	/*
	clock_t time = clock();
	gradient_boosting grad_boost(train_set, 100, 1000000, 4);
	time = clock() - time;
	printf("total boost time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	double error = grad_boost.calculate_error(test_set);
	std::cout << " test error: " << error << std::endl;
	*/
	clock_t sum = 0;
	for (int i = -1; i < 1; i++)
	{
		clock_t time = clock();
		gradient_boosting grad_boost(train_set, 100, 1000000, 4);
		time = clock() - time;
		if (i >= 0)
		{
			sum += time;
		}
		printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	}
	sum /= 1.0;
	printf("avg boost time: %f\n\n", (float)sum / CLOCKS_PER_SEC);
	fclose(stdout);


	//double error = grad_boost.calculate_error(test_set);
	//std::cout << " test error: " << error << std::endl;
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
	std::cout << best_error << " i: " << best_i << " j: " << best_j << std::endl;
	*/
	return 0;
}
