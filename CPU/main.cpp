#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "tree.h"

using namespace std;

int main()
{
	data_set train_set("wine-train.txt");
	tree t(train_set, 1024);
	data_set test_set("wine-test.txt");
	double error = t.calculate_error(test_set);
	cout << "test error: " << error << endl;
	return 0;
}
