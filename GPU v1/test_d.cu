#include "test_d.cuh"

test_d::test_d(test t) : anwser(t.anwser)
{
	cudaMalloc(&features, t.features.size() * sizeof(double));
	cudaMemcpy(features, &t.features[0], t.features.size() * sizeof(double), cudaMemcpyHostToDevice);
}

//test_d::~test_d()
//{
//	cudaFree(features);
//}
