#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <iostream>
#include "tree.h"

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	data_set train_set("wine-train.txt");
	tree t(train_set, 1024);
	data_set test_set("wine-test.txt");
	double error = t.calculate_error(test_set);
	std::cout << "test error: " << error << std::endl;
	return 0;
}