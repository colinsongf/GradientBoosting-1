#include <cstdlib>
#include <iostream>
#include <ctime>
#include "tree.cuh"
#include "thrust\device_vector.h"

int main()
{
	//thrust::device_vector<thrust::device_vector<int>> gg;
	clock_t time = clock();
	data_set train_set("Prototask.train5", 5, false);
	tree t(train_set, 1000000);
	/*data_set test_set("Prototask.test", false);
	double error = t.calculate_error(test_set);
	std::cout << "test error: " << error << std::endl;
	t.print();*/
	time = clock() - time;
	printf("time: %f\n\n", (double)time / CLOCKS_PER_SEC);
	return 0;
}