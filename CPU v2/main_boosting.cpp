#include <iostream>
#include <cstdio>
#include <ctime>

#include "gradient_boosting.h"

int main()
{
	int features_size = 100;
	int tests_size = 165249;
	data_set train_set("data2.txt", features_size, tests_size, true);
	

	//int features_size = 21;
	//int tests_size = 2693;
	//data_set train_set("proto_all.txt", features_size, tests_size, false);

	//data_set train_set("prototask_train_set.csv", features_size, 1885, false);
	//data_set test_set("prototask_test_set.csv", features_size, 808, false);


	//data_set train_set("big_data_train_set.csv", features_size, 115674, false);
	//data_set test_set("big_data_test_set.csv", features_size, 49575, false);


	//data_set test_set("Prototask.test", features_size, 1193, false);
	//freopen("out.txt", "w", stdout);
	/*
	clock_t time = clock();
	gradient_boosting grad_boost(train_set, 100, 1000000, 4);
	time = clock() - time;
	printf("total boost time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	float error = grad_boost.calculate_error(test_set);
	std::cout << " test error: " << error << std::endl;
	*/
	clock_t sum = 0;
	for (int i = -1; i < 0; i++)
	{
		clock_t time = clock();
		gradient_boosting grad_boost(train_set, 4, 1000000, 4);
		time = clock() - time;
		/*float err = grad_boost.calculate_error(train_set);
		std::cout << "train err: " << err << std::endl;
		err = grad_boost.calculate_error(test_set);
		std::cout << "test err: " << err << std::endl;
		*/
		if (i >= 0)
		{
			sum += time;
		}
		printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	}
	sum /= 1.0;
	printf("avg boost time: %f\n\n", (float)sum / CLOCKS_PER_SEC);
	//fclose(stdout);


	//float error = grad_boost.calculate_error(test_set);
	//std::cout << " test error: " << error << std::endl;
	/*float best_error = 1e5;
	int best_i = 0;
	int best_j = 0;
	for (int i = 0; i < 20; i++)
	{
		for (int j = 1; j < 30; j++)
		{
			gradient_boosting grad_boost(train_set, i, j);
			float error = grad_boost.calculate_error(test_set);
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
