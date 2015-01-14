#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>
#include <ctime>
#include <iostream>
#include "tree.h"

int main()
{
	clock_t time1 = clock();
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	data_set train_set("Prototask.train", 5, false);
	data_set test_set("Prototask.test", 5, false);
	//data_set train_set("proto.small", 21, false);
	//data_set test_set("proto.small", 21, false);
	clock_t time2 = clock();
	tree t(train_set, 1000000, 4);
	float error = t.calculate_error(train_set);
	std::cout << "test error: " << error << std::endl;
	clock_t time3 = clock();

	time1 = time3 - time1;
	time2 = time3 - time2;
	printf("calc time: %f\n\n", (float)time2 / CLOCKS_PER_SEC);
	printf("time: %f\n\n", (float)time1 / CLOCKS_PER_SEC);

	/*clock_t sum = 0;
	for (int i = -1; i < 10; i++)
	{
		int features_size = 21;
		int tests_size = 1500;
		clock_t time = clock();
		data_set train_set("Prototask.train", 21, false);
		tree t(train_set, 1000000, 6);
		time = clock() - time;
		if (i >= 0)
		{
			sum += time;
		}
		printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	}
	sum /= 10.0;
	printf("avg time: %f\n\n", (float)sum / CLOCKS_PER_SEC);
	*/
	return 0;
}