#include <set>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <thrust\reduce.h>
#include <thrust\replace.h>
#include <thrust\execution_policy.h>
#include "device_launch_parameters.h"
#include <thrust\host_vector.h>
#include "tree.cuh"

#define INF 1e5
#define INF_INT 100000
#define EPS 1e-5
#define BLOCK_SIZE 16
#define MAX_TESTS 1500
#define SLOT_SIZE 32

float calc_root_mse(data_set& train_set,  float avg, float n)
{
	float ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (int i = 0; i < train_set.tests_size; i++)
	{
		ans += ((train_set.answers[i] - avg) * (train_set.answers[i] - avg));
	}
	ans /= n; 
	return ans;
}

__host__ __device__ node::node(const node& other) : depth(other.depth), is_leaf(other.is_leaf), is_exists(other.is_exists),
	node_mse(other.node_mse), output_value(other.output_value), size(other.size), split_value(other.split_value),
	subtree_mse(other.subtree_mse), sum(other.sum) {}

__host__ __device__ node::node(int depth) : depth(depth)
{
	is_leaf = false;
	is_exists = true;
	node_mse = 0;
	size = 0;
}

__host__ __device__ triple::triple(int test_id, float feature, float answer) : test_id(test_id), feature(feature), answer(answer) {};

bool __host__ __device__ operator<(const triple& lhs, const triple& rhs)
{
	return lhs.feature < rhs.feature;
}

/*tree::tree(const tree& other) : feature_id_at_depth(other.feature_id_at_depth), leafs(other.leafs), max_leafs(other.max_leafs)
{
	//TODO!
	root = new node(*other.root);
	layers.resize(feature_id_at_depth.size() + 1);
	if (!layers.empty())
	{
		fill_layers(root);
	}
}*/

__global__ void make_last_layer_gpu(node* nodes, int depth, int layer_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < layer_size)
	{
		nodes[x + layer_size - 1].is_leaf = true;
	}
}

//__constant__ float const_answers[1500];

tree::tree(data_set& train_set, int max_leafs) : max_leafs(max_leafs)
{
	features_size = train_set.features_size;
	tests_size = train_set.tests_size;
	cudaMalloc(&nodes, (pow(2, features_size + 1) - 1) * sizeof(node));
	cudaMalloc(&feature_id_at_depth, features_size * sizeof(int));
	cudaMalloc(&used_features, features_size * sizeof(bool));
	cudaMalloc(&features, features_size * tests_size * sizeof(float));
	cudaMalloc(&answers, tests_size * sizeof(float));
	cudaMemsetAsync(used_features, false, features_size * sizeof(bool));
	std::set<int> features_set;
	for (size_t i = 0; i < train_set.features_size; i++)
	{
		features_set.insert(i);
	}
	cudaMemcpyAsync(features, &train_set.features[0], sizeof(float) * tests_size * features_size, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(answers, &train_set.answers[0], sizeof(float) * tests_size, cudaMemcpyHostToDevice);
	std::vector<triple> sorted(tests_size * features_size);
	for (int i = 0; i < features_size; i++)
	{
		for (int j = 0; j < tests_size; j++)
		{
			sorted[i * tests_size + j].test_id = j;
			sorted[i * tests_size + j].feature = train_set.features[i * tests_size + j];
			sorted[i * tests_size + j].answer = train_set.answers[j];
		}
		std::sort(sorted.begin() + i * tests_size, sorted.begin() + (i + 1) * tests_size);
	}
	sorted_tests = thrust::device_vector<triple> (sorted);
	//cudaMemcpyToSymbol(const_answers, &train_set.answers[0], sizeof(float) * tests_size);
	auto start = std::chrono::high_resolution_clock::now();
	leafs = 1;
	depth = 0;
	node root(0);
	root.sum = 0;
	root.size = 0;
	for (int i = 0; i < tests_size; i++)
	{
		root.size++;
		root.sum += train_set.answers[i];
	}
	root.output_value = root.sum / root.size;
	root.node_mse = calc_root_mse(train_set, root.output_value, root.size); 
	cudaMemcpyAsync(nodes, &root, sizeof(node), cudaMemcpyHostToDevice);
	float new_error = root.node_mse;
	float old_error = new_error + EPS;
	while (/*new_error < old_error &&*/ leafs < max_leafs && !features_set.empty())
	{
		cudaDeviceSynchronize();
		make_layer(depth);
		std::pair<int, float> feature_and_error = fill_layer();
		features_set.erase(feature_and_error.first);
		depth++;
		old_error = new_error;
		new_error = feature_and_error.second;
		std::cout << "level " << depth << " created. training error: " << new_error << std::endl;
	}
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + pow(2, depth) / (1 + BLOCK_SIZE), 1);
	make_last_layer_gpu<<<grid, block>>>(nodes, depth, pow(2, depth));
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = end - start;
	std::cout << "leafs before pruning: " << leafs << std::endl;
	std::cout << "calculating time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
	//prune(0);                                         //*******************TODO!!
	//std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
}

tree::~tree()
{
	cudaFree(nodes);
	cudaFree(feature_id_at_depth);
	cudaFree(used_features);
	cudaFree(features);
	cudaFree(answers);
	//delete_node(root);
}

float tree::calculate_answer(test& _test)
{
	int cur_id = 0;
	node cur = nodes[cur_id];
	while (!cur.is_leaf)
	{
		if (_test.features[feature_id_at_depth[cur.depth]] < cur.split_value)
		{
			cur = nodes[cur_id * 2 + 1];
			cur_id = cur_id * 2 + 1;
		}
		else
		{
			cur = nodes[cur_id * 2 + 2];
			cur_id = cur_id * 2 + 2;
		}
	}
	return cur.output_value;
}

//float tree::calculate_error(data_set& test_set)
//{
//	float error = 0;
//	for (data_set::iterator cur_test = test_set.begin(); cur_test != test_set.end(); cur_test++)
//	{
//		float ans = calculate_answer(*cur_test);
//		error += ((ans - cur_test->answer) * (ans - cur_test->answer));
//	}
//	error /= (1.0 * test_set.size());
//	return error;
//}

/*void tree::print()
{
	std::cout << "************TREE (layers structure)**************" << std::endl;
	for (size_t i = 0; i < layers.size(); i++)
	{
		std::cout << "layer " << i << "; layer size: " << layers[i].size() << std::endl; 
	}
	std::cout << "************TREE (DFS pre-order)**************" << std::endl;
	print(root);
	std::cout << "**********************************************" << std::endl;
}*/

__device__ void delete_node(node* nodes, int node_id, int* new_leafs)
{
	if (!nodes[node_id].is_exists)
	{
		return;
	}
	delete_node(nodes, node_id * 2 + 1, new_leafs);
	delete_node(nodes, node_id * 2 + 2, new_leafs);
	nodes[node_id].is_exists = false;
	if (nodes[node_id].is_leaf)
	{
		new_leafs[node_id]--;
	}
}

__global__ void make_layer_gpu(node* nodes, int begin_id, int end_id, int* new_leafs)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (begin_id + x < end_id)
	{
		int i = begin_id + x;
		if (!nodes[i].is_leaf && nodes[i].is_exists)
		{
			nodes[2 * i + 1].is_exists = true;
			nodes[2 * i + 2].is_exists = true;
			nodes[2 * i + 1].depth = nodes[i].depth + 1;
			nodes[2 * i + 2].depth = nodes[i].depth + 1;
			new_leafs[x]++;
		}
		else
		{
			nodes[2 * i + 1].is_exists = false;
			nodes[2 * i + 2].is_exists = false;
		}
	}
}

void tree::make_layer(int depth)
{
	thrust::device_vector<int> new_leafs(pow(2, depth), 0);
	int begin_id = pow(2, depth) - 1;
	int end_id = begin_id + pow(2, depth);
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + pow(2, depth) / (1 + BLOCK_SIZE), 1);
	make_layer_gpu<<<grid, block>>>(nodes, begin_id, end_id, thrust::raw_pointer_cast(&new_leafs[0]));
	cudaDeviceSynchronize();
	leafs += thrust::reduce(new_leafs.begin(), new_leafs.end());
}

__device__ void calc_subtree_mse(node* nodes, int node_id, float* features, float* answers,
								 int tests_size, int* feature_id_at_depth)
{
	float error = 0;
	float ans = 0;
	for (int i = 0; i < tests_size; i++)
	{
		bool is_test_in_node = false;
		int cur_node_id = 0;
		while (!nodes[cur_node_id].is_leaf)
		{
			if (cur_node_id == node_id)
			{
				is_test_in_node = true;
			}
			cur_node_id = features[feature_id_at_depth[nodes[cur_node_id].depth] * tests_size + i] < nodes[cur_node_id].split_value ?
				cur_node_id * 2 + 1 : cur_node_id * 2 + 2;
		}
		if (is_test_in_node)
		{
			ans = nodes[cur_node_id].output_value;
			error += ((ans - answers[i]) * (ans - answers[i]));
		}
	}
	error /= (1.0 * nodes[node_id].size);
	nodes[node_id].subtree_mse = error;
}

__global__ void prune_gpu(node* nodes, int node_id, bool* need_go_deeper, float* features, float* answers, int tests_size,
						  int* feature_id_at_depth, int* new_leafs)
{
	*need_go_deeper = false;
	if (!nodes[node_id].is_leaf && nodes[node_id].is_exists)
	{
		calc_subtree_mse(nodes, 2 * node_id + 1, features, answers, tests_size, feature_id_at_depth);
		calc_subtree_mse(nodes, 2 * node_id + 2, features, answers, tests_size, feature_id_at_depth);
		if (nodes[node_id].node_mse <= nodes[2 * node_id + 1].subtree_mse + nodes[2 * node_id + 2].subtree_mse)
		{
			nodes[node_id].is_leaf = true;
			new_leafs[node_id]++;
			delete_node(nodes, 2 * node_id + 1, new_leafs);
			delete_node(nodes, 2 * node_id + 2, new_leafs);
		}
		else
		{
			*need_go_deeper = true;
		}
	}
}

void tree::prune(int node_id)
{
	thrust::device_vector<int> new_leafs((pow(2, features_size + 1) - 1));
	bool* need_go_deeper;
	bool need_go_deeper_h;
	cudaMalloc(&need_go_deeper, sizeof(bool));
	dim3 block(1, 1);
	dim3 grid(1, 1);
	prune_gpu<<<grid, block>>>(nodes, node_id, need_go_deeper, features, answers, tests_size, feature_id_at_depth,
		thrust::raw_pointer_cast(&new_leafs[0]));
	cudaDeviceSynchronize();
	cudaMemcpy(&need_go_deeper_h, need_go_deeper, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(need_go_deeper);
	leafs += thrust::reduce(new_leafs.begin(), new_leafs.end());
	if (need_go_deeper_h)
	{
		prune(2 * node_id + 1);
		prune(2 * node_id + 2);
	}
}

/*void tree::print(node* n)
{
	for (int i = 0; i < n->depth; i++)
	{
		std::cout << "-";
	}
	if (n->is_leaf)
	{
		std::cout << "leaf. output value: " << n->output_value << std::endl;
	}
	else
	{
		std::cout << "split feature: " << feature_id_at_depth[n->depth] << "; ";
		std::cout << "split value: " << n->split_value << std::endl;
		print(n->left);
		print(n->right);
	}
}

void tree::fill_layers(node* n)
{
	layers[n->depth].push_back(n);
	if (!n->is_leaf)
	{
		fill_layers(n->left);
		fill_layers(n->right);
	}
}*/

__global__ void fill_node_id_of_test(node* nodes, int* node_id_of_test, int* feature_id_at_depth,
									 float* features, int tests_size, int depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < tests_size)
	{
		int cur = 0;
		while (nodes[cur].depth < depth && !nodes[cur].is_leaf && nodes[cur].is_exists)
		{
			if (features[feature_id_at_depth[nodes[cur].depth] * tests_size + i] < nodes[cur].split_value)
			{
				cur = 2 * cur + 1;
			}
			else
			{
				cur = 2 * cur + 2;
			}
		}
		node_id_of_test[i] = cur;
	}
}

__global__ void fill_is_tests_in_node(int* node_id_of_test, unsigned int* is_tests_in_node,	int tests_size, int layer_size, int tests_per_slot)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < tests_per_slot)
	{
		for (int j = i * SLOT_SIZE; j < (i + 1) * SLOT_SIZE && j < tests_size; j++)
		{
			int id_shifted = node_id_of_test[j] - layer_size + 1;
			if (id_shifted >= 0)
			{
				is_tests_in_node[id_shifted * tests_per_slot + j / SLOT_SIZE] |= 1 << (j % SLOT_SIZE);
			}
		}
	}
}    


__global__ void calc_split_gpu2(node* nodes, int* node_id_of_test, float* errors, float* split_values,
									 float* features, float* answers, int tests_size, bool* used_features,
									 int features_size, int layer_size, unsigned int* is_tests_in_node, 
									 triple* sorted_tests, int tests_per_slot)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < tests_size && y < features_size && !used_features[y])
	{
		int node_id = node_id_of_test[x];
		if (node_id < layer_size - 1)
		{
			return;
		}
		if (nodes[node_id].is_leaf)
		{
			errors[y * tests_size + x] = nodes[node_id].node_mse;
			return;
		}
		float feature_value = features[y * tests_size + x];
		float split_value = feature_value;
		int split_id = tests_size - 1;
		for (;split_id >= 0; split_id--)
		{
			if (sorted_tests[y * tests_size + split_id].feature == feature_value)
			{
				break;
			}
		}
		int node_split_id_shifted = node_id - layer_size + 1;
		if (split_id < tests_size - 1)
		{
			for (int i = split_id + 1; i < tests_size; i++)
			{
				triple t = sorted_tests[y * tests_size + i];
				int test_id = t.test_id;
				if (is_tests_in_node[node_split_id_shifted * tests_per_slot + test_id / SLOT_SIZE] & (1 << (test_id % SLOT_SIZE)))
				{
					split_value = t.feature;
					break;
				}
			}
		}
		split_value = (feature_value + split_value) / 2.0 + EPS;
		float l_sum = 0;
		float r_sum = 0;
		int l_size = 0;
		int r_size = 0;
		float l_avg = 0;
		float r_avg = 0;
		float l_err = 0;
		float r_err = 0;
		for (int i = 0; i <= split_id; i++)
		{
			triple t = sorted_tests[y * tests_size + i];
			int test_id = t.test_id;
			int exists = (is_tests_in_node[node_split_id_shifted * tests_per_slot + test_id / SLOT_SIZE] >> (test_id % SLOT_SIZE)) & 1;
			l_sum += exists * t.answer;
			l_size += exists;
		}
		if (split_id < tests_size - 1)
		{
			for (int i = split_id + 1; i < tests_size; i++)
			{
				triple t = sorted_tests[y * tests_size + i];
				int test_id = t.test_id;
				int exists = (is_tests_in_node[node_split_id_shifted * tests_per_slot + test_id / SLOT_SIZE] >> (test_id % SLOT_SIZE)) & 1;
				r_sum += exists * t.answer;
				r_size += exists;
			}
		}
		l_avg = (l_size > 0) ? (l_sum / l_size) : 0; 
		r_avg = (r_size > 0) ? (r_sum / r_size) : 0; 
		for (int i = 0; i <= split_id; i++)
		{
			triple t = sorted_tests[y * tests_size + i];
			int test_id = t.test_id;
			int exists = (is_tests_in_node[node_split_id_shifted * tests_per_slot + test_id / SLOT_SIZE] >> (test_id % SLOT_SIZE)) & 1;
			l_err += exists * (t.answer - l_avg) * (t.answer - l_avg); 
		}
		if (split_id < tests_size - 1)
		{
			for (int i = split_id + 1; i < tests_size; i++)
			{
				triple t = sorted_tests[y * tests_size + i];
				int test_id = t.test_id;
				int exists = (is_tests_in_node[node_split_id_shifted * tests_per_slot + test_id / SLOT_SIZE] >> (test_id % SLOT_SIZE)) & 1;
				r_err += exists * (t.answer - r_avg) * (t.answer - r_avg);
			}
		}
		l_err = (l_size > 0) ? (l_err / l_size) : 0;
		r_err = (r_size > 0) ? (r_err / r_size) : 0;
		errors[y * tests_size + x] = l_err + r_err;
		split_values[y * tests_size + x] = split_value;
	}
}

__global__ void calc_min_error(int* node_id_of_test, float* pre_errors, float* pre_split_values,
							   float* errors, float* split_values, int tests_size, int features_size,
							   int layer_size, bool* used_features)
{
	__shared__ int node_id_of_test_shared[MAX_TESTS];
	int tests_per_thread = tests_size / (BLOCK_SIZE - 1);
	for (int i = 0; i < tests_per_thread; i++)
	{
		if (threadIdx.y * tests_per_thread + i >= tests_size)
		{
			break;
		}
		node_id_of_test_shared[threadIdx.y * tests_per_thread + i] =
			node_id_of_test[threadIdx.y * tests_per_thread + i]; 
	}
	__syncthreads();
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < features_size && !used_features[y])
	{
		for (int i = 0; i < tests_size; i++)
		{
			int node_id = node_id_of_test_shared[i];
			if (node_id < layer_size - 1)
			{
				continue;
			}
			if (pre_errors[y * tests_size + i] < errors[y * layer_size + node_id + 1 - layer_size])
			{
				errors[y * layer_size + node_id + 1 - layer_size] = pre_errors[y * tests_size + i];
				split_values[y * layer_size + node_id + 1 - layer_size] = pre_split_values[y * tests_size + i];
			}
		}
	}
}

__global__ void calc_best_feature(float* errors, bool* used_features, int* best_features, float* best_errors,
									 int features_size, int layer_size, int* feature_id_at_depth, int depth)
{
	float best_error = INF;
	int best_feature = -1;
	for (int i = 0; i < features_size; i++)
	{
		if (!used_features[i])
		{
			float cur_error = errors[i * layer_size];
			if (cur_error < best_error)
			{
				best_error = cur_error;
				best_feature = i;
			}
		}
	}
	best_features[0] = best_feature;
	best_errors[0] = best_error;
	used_features[best_feature] = true;
	feature_id_at_depth[depth] = best_feature;
}

__global__ void make_split_gpu(node* nodes, int* node_id_of_test, float* split_values,
							   int* best_feature, float* features, float* answers,
							   int tests_size, int layer_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	x = fminf(x, layer_size - 1);
	int node_id = layer_size - 1 + x;
	if (!nodes[node_id].is_exists || nodes[node_id].is_leaf)
	{
		return;
	}
	float split_value = split_values[best_feature[0] * layer_size + x];
	nodes[node_id].split_value = split_value;
	float l_sum = 0;
	float r_sum = 0;
	int l_size = 0;
	int r_size = 0;
	float l_avg = 0;
	float r_avg = 0;
	float l_err = 0;
	float r_err = 0;
	float l_tests[MAX_TESTS];
	float r_tests[MAX_TESTS];
	for (int i = 0; i < tests_size; i++)
	{
		if (node_id_of_test[i] == node_id)
		{
			if (features[best_feature[0] * tests_size + i] < split_value)
			{
				l_tests[l_size] = answers[i];
				l_sum += l_tests[l_size];
				l_size++;
			}
			else
			{
				r_tests[r_size] = answers[i];
				r_sum += r_tests[r_size];
				r_size++;
			}
		}
	}
	if (l_size == 0)
	{
		nodes[2 * node_id + 1].output_value = nodes[node_id].output_value;
		nodes[2 * node_id + 1].size = 0;
		nodes[2 * node_id + 1].is_leaf = true;
		nodes[2 * node_id + 2].output_value = nodes[node_id].output_value;
		nodes[2 * node_id + 2].size = nodes[node_id].size;
		nodes[2 * node_id + 2].sum = nodes[node_id].sum;
		nodes[2 * node_id + 2].node_mse = nodes[node_id].node_mse;
		nodes[2 * node_id + 2].is_leaf = (nodes[node_id].size <= 1) ? true : false;
		return;
	}
	if (r_size == 0)
	{
		nodes[2 * node_id + 2].output_value = nodes[node_id].output_value;
		nodes[2 * node_id + 2].size = 0;
		nodes[2 * node_id + 2].is_leaf = true;
		nodes[2 * node_id + 1].output_value = nodes[node_id].output_value;
		nodes[2 * node_id + 1].size = nodes[node_id].size;
		nodes[2 * node_id + 1].sum = nodes[node_id].sum;
		nodes[2 * node_id + 1].node_mse = nodes[node_id].node_mse;
		nodes[2 * node_id + 1].is_leaf = (nodes[node_id].size <= 1) ? true : false;
		return;
	}
	l_avg = l_sum / l_size; 
	r_avg = r_sum / r_size; 
	for (int i = 0; i < l_size; i++)
	{
		l_err += ((l_tests[i] - l_avg) * (l_tests[i] - l_avg));
	}
	for (int i = 0; i < r_size; i++)
	{
		r_err += ((r_tests[i] - r_avg) * (r_tests[i] - r_avg));
	}
	l_err = l_err / l_size;
	r_err = r_err / r_size;
	nodes[2 * node_id + 1].output_value = l_avg;
	nodes[2 * node_id + 1].size = l_size;
	nodes[2 * node_id + 1].sum = l_sum;
	nodes[2 * node_id + 1].node_mse = l_err;
	nodes[2 * node_id + 1].is_leaf = (l_size == 1) ? true : false;
	nodes[2 * node_id + 2].output_value = r_avg;
	nodes[2 * node_id + 2].size = r_size;
	nodes[2 * node_id + 2].sum = r_sum;
	nodes[2 * node_id + 2].node_mse = r_err;
	nodes[2 * node_id + 2].is_leaf = (r_size == 1) ? true : false;
}

std::pair<int, float> tree::fill_layer()
{
	int layer_size = pow(2, depth);
	int tests_per_slot = tests_size / SLOT_SIZE + 1;
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + tests_size / (1 + BLOCK_SIZE), 1);
	thrust::device_vector<int> node_id_of_test(tests_size);
	thrust::device_vector<unsigned int> is_tests_in_node(layer_size * tests_per_slot, 0);
	fill_node_id_of_test<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		feature_id_at_depth,
		features, tests_size, depth);
	cudaDeviceSynchronize();
	grid.x = 1 + tests_per_slot / (1 + BLOCK_SIZE);
	fill_is_tests_in_node<<<grid, block>>>(thrust::raw_pointer_cast(&node_id_of_test[0]), thrust::raw_pointer_cast(&is_tests_in_node[0]),
		tests_size, layer_size, tests_per_slot);
	thrust::device_vector<float> pre_errors(tests_size * features_size, INF_INT);
	thrust::device_vector<float> pre_split_values(tests_size * features_size, 0);
	grid.x = 1 + tests_size / (1 + BLOCK_SIZE);
	block.y = BLOCK_SIZE;
	grid.y = 1 + features_size / (1 + BLOCK_SIZE);
	/*int sumg = 0;
	for (int i = 0 ; i < tests_per_slot; i++) {
		for (int j = 0; j < SLOT_SIZE; j++) {
			sumg += (is_tests_in_node[i] >> (unsigned int)(j)) & (unsigned int)(1);	 
		}
	}
	for (int i = 0; i < tests_per_slot; i++)
	{
		//std::cout << is_tests_in_node[i] << std::endl;
	}
	std::cout << "dfd " << sumg << " sdsds" << std::endl;
	*/
	cudaDeviceSynchronize();
	calc_split_gpu2<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&pre_errors[0]), thrust::raw_pointer_cast(&pre_split_values[0]),
		features, answers, tests_size, used_features, features_size, layer_size, thrust::raw_pointer_cast(&is_tests_in_node[0]),
		thrust::raw_pointer_cast(&sorted_tests[0]), tests_per_slot);
	thrust::device_vector<float> errors(layer_size * features_size, INF_INT);
	thrust::device_vector<float> split_values(layer_size * features_size, INF_INT);
	block.x = 1;
	grid.x = 1;
	cudaDeviceSynchronize();
	/*if (layer_size==1)
	{
	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < tests_size; j++)
		{
			std::cout << pre_errors[i * tests_size + j] << ";" << std::endl ;

		}
		std::cout << std::endl;
	}
	}*/
	calc_min_error<<<grid, block>>>(thrust::raw_pointer_cast(&node_id_of_test[0]), thrust::raw_pointer_cast(&pre_errors[0]),
		thrust::raw_pointer_cast(&pre_split_values[0]),	thrust::raw_pointer_cast(&errors[0]),
		thrust::raw_pointer_cast(&split_values[0]),	tests_size, features_size, layer_size, used_features);
	cudaDeviceSynchronize();
	/*for (int i = 0; i < features_size; i++)
	{
		for (int j = 0; j < layer_size; j++)
		{
			std::cout << errors[i * layer_size + j] << " " ;

		}
		std::cout << std::endl;
	}*/
	thrust::replace(errors.begin(), errors.end(), INF_INT, 0);
	for (int i = 0; i < features_size; i++)
	{
		errors[i * layer_size] = thrust::reduce(errors.begin() + i * layer_size,
			errors.begin() + (i + 1) * layer_size, 0.0, thrust::plus<float>());
	}
	cudaDeviceSynchronize();
	thrust::device_vector<int> best_feature(1);
	thrust::device_vector<float> best_error(1);
	block.x = 1;
	block.y = 1;
	grid.x = 1;
	grid.y = 1;
	calc_best_feature<<<grid, block>>>(thrust::raw_pointer_cast(&errors[0]), used_features,
		thrust::raw_pointer_cast(&best_feature[0]), thrust::raw_pointer_cast(&best_error[0]), features_size, layer_size, feature_id_at_depth, depth);
	block.x = BLOCK_SIZE;
	grid.x = 1 + layer_size / (1 + BLOCK_SIZE);
	cudaDeviceSynchronize();
	make_split_gpu<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&split_values[0]), thrust::raw_pointer_cast(&best_feature[0]),
		features, answers, tests_size, layer_size);
	cudaDeviceSynchronize();
	return std::make_pair(best_feature[0], best_error[0]);
}