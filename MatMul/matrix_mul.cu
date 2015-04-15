#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <cublas_v2.h>
#include "matrix_mul.cuh"

matrix_mul::matrix_mul(std::string a_file_name, std::string b_file_name, int a_size,
		int b_size, int features_size)
		: a_file_name(a_file_name),
		  b_file_name(b_file_name),
		  a_size(a_size),
		  b_size(b_size),
		  features_size(features_size) {}

struct comparator
{
	float* c_device;
	int i;
	int b_size;
	__host__ __device__ comparator(float* c_device, int& i, int& b_size) : c_device(c_device), i(i), b_size(b_size) {}
    bool __host__ __device__ operator()(const int& id1, const int& id2) const
    {
    	return c_device[i * b_size + id1] > c_device[i * b_size + id2];
    }
};

void matrix_mul::calculate(std::string output_file_name, int n, int block_size)
{
	std::ifstream a_stream(a_file_name.c_str());
	std::ifstream b_stream(b_file_name.c_str());
	std::ofstream output_stream(output_file_name.c_str());
	std::vector<int> b_ids;
	std::vector<float> b(b_size * features_size);
	char const tab_delim = '\t';
	std::string line;
	thrust::device_vector<int> b_ids_device;
	for (int i = 0; i < b_size; i++) //read "b" matrix
	{
		if (i % 1000000 == 0)
		{
			printf("Reading B: %d of %d\n", i, b_size);
		}
		getline(b_stream, line);
		std::istringstream line_stream(line);
		std::string value;
		getline(line_stream, value, tab_delim);
		b_ids.push_back(atoi(value.c_str()));
		for (int j = 0; j < features_size; j++)
		{
			getline(line_stream, value, tab_delim);
			b[j * b_size + i] = atof(value.c_str());
		}
		b_ids_device.push_back(i);
	}
	thrust::device_vector<float> b_device(b);

	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;

	int parts = a_size / block_size + 1;

	for (int id = 0; id < parts; id++) //read "a" matrix by parts
	{
		printf("Part %d of %d\n", id, parts);
		cudaDeviceSyncronize();
		clock_t time = clock();
		int a_actual_size = (id == parts - 1) ? (a_size % block_size) : block_size;
		std::vector<int> a_ids;
		std::vector<float> a(a_actual_size * features_size);
		for (int i = 0; (i < block_size) && (id * block_size + i < a_size); i++)
		{
			getline(a_stream, line);
			std::istringstream line_stream(line);
			std::string value;
			getline(line_stream, value, tab_delim);
			a_ids.push_back(atoi(value.c_str()));
			for (int j = 0; j < features_size; j++)
			{
				getline(line_stream, value, tab_delim);
				a[i * features_size + j] = (float)atof(value.c_str());
			}
		}
		thrust::device_vector<float> a_device(a);
		thrust::device_vector<float> c_device(a_actual_size * b_size);
		
		cudaDeviceSyncronize();
		time = clock() - time;
		printf("read a: %f sec\n", (float)time / CLOCKS_PER_SEC);
		
		time = clock();

		// matrix multiplication
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b_size, a_actual_size, features_size, &alpha,
				thrust::raw_pointer_cast(&b_device[0]), b_size, thrust::raw_pointer_cast(&a_device[0]),
				features_size, &beta, thrust::raw_pointer_cast(&c_device[0]), b_size);
				
		cudaDeviceSyncronize();
		time = clock() - time;
		printf("matrix mul: %f sec\n", (float)time / CLOCKS_PER_SEC);
		
		time = clock();
		
		//print_c_matrix(thrust::raw_pointer_cast(&c_device[0]), a_actual_size);

		clock_t sort_time = 0;
		clock_t temp_time;
		
		for (int i = 0; i < a_actual_size; i++) //select "n" best ids
		{
			cudaDeviceSyncronize();
			temp_time = clock();
			thrust::sort(b_ids_device.begin(), b_ids_device.end(), comparator(thrust::raw_pointer_cast(&c_device[0]), i, b_size));
			cudaDeviceSyncronize();
			temp_time = clock() - temp_time;
			sort_time += temp_time;
			thrust::host_vector<int> sorted_ids(b_ids_device);
			output_stream << a_ids[i];
			for (int j = 0; j < n; j++)
			{
				output_stream << '\t' << b_ids[sorted_ids[j]];
			}
			output_stream << std::endl;
		}
		
		cudaDeviceSyncronize();
		time = clock() - time;
		printf("only sorting: %f sec\n", (float)sort_time / CLOCKS_PER_SEC);
		printf("sorting with output: %f sec\n\n", (float)time / CLOCKS_PER_SEC);
		
	}
	status = cublasDestroy(handle);
}

void matrix_mul::print_c_matrix(float* c_device, int a_actual_size)
{
	float* c = (float*)malloc(a_actual_size * b_size * sizeof(float));
	cudaMemcpy(c, c_device, a_actual_size * b_size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < a_actual_size; i++)
	{
		for (int j = 0; j < b_size; j++)
		{
			printf("%f ", c[i * b_size + j]);
		}
		printf("\n");
	}
	free(c);
}
