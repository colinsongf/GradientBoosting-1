#include "test_d.cuh"

test_d::test_d(test t)
{
	cudaMalloc(&anwser, sizeof(double));
	cudaMemcpy(anwser, &t.anwser, sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&features, t.features.size() * sizeof(double));
	cudaMemcpy(features, &t.features[0], t.features.size() * sizeof(double), cudaMemcpyHostToDevice);
}

test_d::test_d() {}


test_d::~test_d()
{
	/*cudaFree(anwser);
	cudaFree(features);*/
}
