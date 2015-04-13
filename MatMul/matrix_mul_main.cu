#include "matrix_mul.cuh"
#include <string>
#include <cstdio>

using namespace std;

int main()
{
	string a_file_name = "a.txt";
	string b_file_name = "b.txt";
	string output_file_name = "out.txt";
	int a_size = 3;
	int b_size = 4;
	int features_size = 3;
	int block_size = 3;
	int n = 2;

	matrix_mul m(a_file_name, b_file_name, a_size, b_size, features_size);
	m.calculate(output_file_name, n, block_size);

	return 0;
}
