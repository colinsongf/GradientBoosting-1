#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "tree.h"

using namespace std;

int main()
{
	vector<pair<double, vector<double> > > train_set;
	ifstream train_stream("wine-train.txt");
	string line;
	while (getline(train_stream, line))
	{
		vector<double> test;
		double value;
		double ans;
		istringstream iss(line);
		iss >> ans;
		while (iss >> value)
		{
			test.push_back(value);
		}
		train_set.push_back(make_pair(ans, test));
	}
	tree t(train_set, 1024);
	vector<pair<double, vector<double> > > test_set;
	ifstream test_stream("wine-test.txt");
	while (getline(test_stream, line))
	{
		vector<double> test;
		double value;
		double ans;
		istringstream iss(line);
		iss >> ans;
		while (iss >> value)
		{
			test.push_back(value);
		}
		test_set.push_back(make_pair(ans, test));
	}
	double error = 0;
	for (size_t i = 0; i < test_set.size(); i++)
	{
		double ans = t.calculate(test_set[i]);
		error += ((ans - test_set[i].first) * (ans - test_set[i].first));
	}
	error /= (1.0 * test_set.size());
	cout << "test error: " << error << endl;
	return 0;
}
