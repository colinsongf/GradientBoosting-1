#include <cstdlib>
#include <iostream>
#include <ctime>
#include "tree.cuh"
#include "thrust\device_vector.h"

int main()
{
	int features_size = 21;
	int tests_size = 1500;
	clock_t time = clock();
	data_set train_set("Prototask.train", features_size, tests_size, false);
	tree t(train_set, 1000000);
	/*data_set test_set("Prototask.test", false);
	float error = t.calculate_error(test_set);
	std::cout << "test error: " << error << std::endl;
	t.print();*/
	time = clock() - time;
	printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	cudaDeviceReset();
	return 0;
}