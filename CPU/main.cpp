#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

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
	
	return 0;
}
