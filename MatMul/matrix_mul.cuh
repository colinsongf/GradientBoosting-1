#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

#include <string>

class matrix_mul
{
public:
	matrix_mul(std::string a_file_name, std::string b_file_name, int a_size,
			int b_size, int features_size);
	void calculate(std::string output_file_name, int n, int block_size);
private:
	void print_c_matrix(float* c_device, int a_actual_size);
	std::string a_file_name;
	std::string b_file_name;
	int a_size;
	int b_size;
	int features_size;
};

#endif // MATRIX_MUL_CUH
