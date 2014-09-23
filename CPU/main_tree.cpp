#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>
#include <ctime>
#include <iostream>
#include <chrono>
#include "tree.h"

int main()
{
	clock_t time = clock();
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	data_set train_set("Prototask.train3", 21, false);
	auto start = std::chrono::high_resolution_clock::now();
	tree t(train_set, 1000000);
	/*data_set test_set("Prototask.test", false);
	double error = t.calculate_error(test_set);
	std::cout << "test error: " << error << std::endl;
	t.print();*/
	auto end = std::chrono::high_resolution_clock::now();
	time = clock() - time;
	printf("time: %f\n\n", (double)time / CLOCKS_PER_SEC);
	auto elapsed = end - start;
	std::cout << "chrono time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << '\n';  // clock ticks (seconds)
	return 0;
}