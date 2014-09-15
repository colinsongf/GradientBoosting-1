#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>
#include <ctime>
#include <iostream>
#include "tree.h"

int main()
{
	clock_t time = clock();
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	data_set train_set("Prototask.train4", 21, false);
	tree t(train_set, 1000000);
	/*data_set test_set("Prototask.test", false);
	double error = t.calculate_error(test_set);
	std::cout << "test error: " << error << std::endl;
	t.print();*/
	time = clock() - time;
	printf("time: %f\n\n", (double)time / CLOCKS_PER_SEC);
	return 0;
}