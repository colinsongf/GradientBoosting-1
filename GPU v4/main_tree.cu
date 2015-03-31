#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <cstdio>
#include "tree.cuh"
#include "thrust/device_vector.h"

__global__ void warmup(int* out)
{
	int ans = 0;
	for (int i = 0; i < 1000; i++)
	{
		ans += i * (i + 1);
	}
	*out = ans;
}

int main()
{
	dim3 block(32, 1);
	dim3 grid(3, 1);
	int* out;
	cudaMalloc(&out, 10 * sizeof(float));
	warmup<<<grid, block>>>(out);
	cudaDeviceSynchronize();
	cudaFree(out);

	clock_t sum = 0;
	int features_size = 100;
	int tests_size = 165249;
	data_set train_set("data2.txt", features_size, tests_size, true);
	//data_set test_set("Prototask.test", features_size, 1193, false);
	//data_set train_set("proto.small", features_size, tests_size, false);
	//freopen("out.txt", "w", stdout);


	for (int i = -1; i < 0; i++)
	{
		//cudaDeviceSynchronize();
		clock_t time = clock();
		tree t(train_set, 1000000, 4);
		//float err = t.calculate_answer(train_set.tests[0]);
		//std::cout << "test err: " << err << std::endl;
		cudaDeviceSynchronize();
		time = clock() - time;
		if (i >= 0)
		{
			sum += time;
		}
		printf("tree time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	}
	sum /= 1.0;
	printf("avg time: %f\n\n", (float)sum / CLOCKS_PER_SEC);
	
	//fclose(stdout);
	/*clock_t time = clock();
	tree t(train_set, 1000000, 15);
	time = clock() - time;
	float err = t.calculate_error(train_set);
	std::cout << "test err: " << err << std::endl;
	printf("time: %f\n\n", (float)time / CLOCKS_PER_SEC);
	*/
	//cudaDeviceReset();
	return 0;
}
