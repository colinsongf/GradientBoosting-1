#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include "matrix_mul.cuh"

#define BLOCK_SIZE 8

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

__global__ void select(float* c_device, int* ids_matrix, int n, int b_size, int a_actual_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < a_actual_size)
	{
		int offset = x * b_size;
		int l = 0;
		int r = b_size - 1;
		float temp_value;
		int temp_id;
		for(;;)
		{
			if (l != r)
			{
				temp_value = c_device[offset + (l + r) / 2];
				c_device[offset + (l + r) / 2] = c_device[offset + r];
				c_device[offset + r] = temp_value;
				
				temp_id = ids_matrix[offset + (l + r) / 2];
				ids_matrix[offset + (l + r) / 2] = ids_matrix[offset + r];
				ids_matrix[offset + r] = temp_id;
			}
			float mid = c_device[r];
			int i = l - 1;
			for (int j = l; j <= r; j++) 
			{
				if (c_device[j] >= mid)
				{
					i++;
					temp_value = c_device[offset + i];
					c_device[offset + i] = c_device[offset + j];
					c_device[offset + j] = temp_value;
					
					temp_id = ids_matrix[offset + i];
					ids_matrix[offset + i] = ids_matrix[offset + j];
					ids_matrix[offset + j] = temp_id;
				}
			}
			if (i < n)
			{
				l = i + 1;
			}
			else if (i > n)
			{
				r = i - 1;
			}
			else
			{
				return;
			}
		}
	}
} 

void matrix_mul::calculate(std::string output_file_name, int n, int block_size)
{
	std::ifstream a_stream(a_file_name.c_str());
	std::ifstream b_stream(b_file_name.c_str());
	//std::ofstream output_stream(output_file_name.c_str());
	FILE* fout = fopen(output_file_name.c_str(), "w");
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
		cudaDeviceSynchronize();
		clock_t time = clock();
		int a_actual_size = (id == parts - 1) ? (a_size % block_size) : block_size;
		std::vector<int> a_ids;
		std::vector<float> a(a_actual_size * features_size);
		//thrust::host_vector<int> ids_matrix_host(a_actual_size * b_size);
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
			//thrust::copy(b_ids_device.begin(), b_ids_device.end(), ids_matrix_host.begin() + i * b_size);
			
		}
		thrust::device_vector<float> a_device(a);
		thrust::device_vector<float> c_device(a_actual_size * b_size);
		//thrust::device_vector<int> ids_matrix(ids_matrix_host);
		
		
		cudaDeviceSynchronize();
		time = clock() - time;
		printf("read a: %f sec\n", (float)time / CLOCKS_PER_SEC);
		
		time = clock();

		// matrix multiplication
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b_size, a_actual_size, features_size, &alpha,
				thrust::raw_pointer_cast(&b_device[0]), b_size, thrust::raw_pointer_cast(&a_device[0]),
				features_size, &beta, thrust::raw_pointer_cast(&c_device[0]), b_size);
				
		cudaDeviceSynchronize();
		time = clock() - time;
		printf("matrix mul: %f sec\n", (float)time / CLOCKS_PER_SEC);
		
		time = clock();
		
		//print_c_matrix(thrust::raw_pointer_cast(&c_device[0]), a_actual_size);

		clock_t sort_time = 0;
		clock_t temp_time;
		
		/*
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(1 + a_actual_size / BLOCK_SIZE, 1);
		
		select<<<grid, block>>>(thrust::raw_pointer_cast(&c_device[0]), thrust::raw_pointer_cast(&ids_matrix[0]), n, b_size, a_actual_size);
		cudaDeviceSynchronize();
		*/
		
		thrust::device_vector<float> out_f(b_size);
		thrust::device_vector<int> out_i(b_size);
		
		
		for (int i = 0; i < a_actual_size; i++) //select "n" best ids
		{
			cudaDeviceSynchronize();
			size_t  temp_storage_bytes  = 0;
			void* d_temp_storage = NULL;
			temp_time = clock();
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(&c_device[0]) + i * b_size,
				thrust::raw_pointer_cast(&out_f[0]), thrust::raw_pointer_cast(&b_ids_device[0]),
				thrust::raw_pointer_cast(&out_i[0]), b_size);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(&c_device[0]) + i * b_size,
				thrust::raw_pointer_cast(&out_f[0]), thrust::raw_pointer_cast(&b_ids_device[0]),
				thrust::raw_pointer_cast(&out_i[0]), b_size);
			//thrust::sort_by_key(c_device.begin() + i * b_size, c_device.begin() + i * b_size + n, b_ids_device.begin());
			cudaDeviceSynchronize();
			cudaFree(d_temp_storage);
			
			cudaDeviceSynchronize();
			temp_time = clock() - temp_time;
			sort_time += temp_time;
			
			
			//thrust::host_vector<int> sorted_ids(out_i);
			//output_stream << a_ids[i];
			fprintf(fout, "%d", a_ids[i]);
			for (int j = 0; j < n; j++)
			{
				fprintf(fout, "\t%d", b_ids[(int)out_i[j]]);
				//output_stream << '\t' << b_ids[sorted_ids[j]];
			}
			fprintf(fout, "\n");
			//output_stream << std::endl;
		}
		
		cudaDeviceSynchronize();
		time = clock() - time;
		printf("only sorting: %f sec\n", (float)sort_time / CLOCKS_PER_SEC);
		printf("sorting with print: %f sec\n\n", (float)time / CLOCKS_PER_SEC);
		
	}
	fclose(fout);
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
