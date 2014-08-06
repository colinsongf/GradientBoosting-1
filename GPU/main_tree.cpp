#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <iostream>
#include "tree.h"

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	data_set train_set("housing-train.txt", false);
	tree t(train_set, 1024);
	data_set test_set("housing-test.txt", false);
	double error = t.calculate_error(test_set);
	std::cout << "test error: " << error << std::endl;
	t.print();
	return 0;
}