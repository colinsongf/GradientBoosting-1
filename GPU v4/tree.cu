#include <set>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include "tree.cuh"

#define INF 1e6
#define INF_INT 1000000
#define EPS 1e-6
#define BLOCK_SIZE 32
#define MAX_TESTS 1500

node_ptr::node_ptr() {}

node_ptr::node_ptr(const node_ptr& other) : depth(other.depth), is_leaf(other.is_leaf),
	output_value(other.output_value), split_value(other.split_value)
{
	if (!is_leaf)
	{
		left = new node_ptr(*other.left);
		right = new node_ptr(*other.right);
	}
	else
	{
		left = right = NULL;
	}
}

__host__ __device__ node::node(const node& other) : depth(other.depth), is_leaf(other.is_leaf), is_exists(other.is_exists),
	node_mse(other.node_mse), output_value(other.output_value), size(other.size), split_value(other.split_value),
	subtree_mse(other.subtree_mse), sum(other.sum) {}

__host__ __device__ node::node()
{
	depth = 0;
	is_leaf = false;
	is_exists = true;
	node_mse = 0;
	size = 0;
	sum = 0;
}

__host__ __device__ my_tuple::my_tuple(int test_id, int split_id, float feature, float answer) : test_id(test_id), split_id(split_id),
	feature(feature), answer(answer) {}

bool __host__ __device__ operator<(const my_tuple& lhs, const my_tuple& rhs)
{
	return lhs.feature < rhs.feature;
}

__host__ __device__ my_pair::my_pair(int sorted_tests_id, float error) : sorted_tests_id(sorted_tests_id), error(error) {}

__host__ __device__ my_pair::my_pair()
{
	sorted_tests_id = -1;
	error = INF_INT;
}

bool __host__ __device__ operator<(const my_pair& lhs, const my_pair& rhs)
{
	return lhs.error < rhs.error;
}
	

tree::tree(const tree& other) : features_size(other.features_size), max_depth(other.max_depth)
{
	root = new node_ptr(*other.root);
	h_feature_id_at_depth = (int*)malloc(features_size * sizeof(int));
	h_nodes = (node*)malloc((pow(2, max_depth + 1) - 1) * sizeof(node));
	memcpy(h_feature_id_at_depth, other.h_feature_id_at_depth, features_size * sizeof(int));
	memcpy(h_nodes, other.h_nodes, (pow(2, max_depth + 1) - 1) * sizeof(node));

}

__global__ void make_last_layer_gpu(node* nodes, int depth, int layer_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < layer_size)
	{
		nodes[x + layer_size - 1].is_leaf = true;
	}
}

tree::tree(data_set& train_set, int max_leafs, int max_depth) : max_depth(max_depth)
{
	features_size = train_set.features_size;
	tests_size = train_set.tests_size;
	cudaMalloc(&nodes, (pow(2, max_depth + 1) - 1) * sizeof(node));
	cudaMalloc(&feature_id_at_depth, features_size * sizeof(int));
	cudaMalloc(&used_features, features_size * sizeof(bool));
	cudaMalloc(&features, features_size * tests_size * sizeof(float));
	//cudaMalloc(&answers, tests_size * sizeof(float));
	cudaMemsetAsync(used_features, false, features_size * sizeof(bool));
	std::set<int> features_set;
	for (size_t i = 0; i < train_set.features_size; i++)
	{
		features_set.insert(i);
	}
	cudaMemcpyAsync(features, &train_set.features[0], sizeof(float) * tests_size * features_size, cudaMemcpyHostToDevice);
	//cudaMemcpyAsync(answers, &train_set.answers[0], sizeof(float) * tests_size, cudaMemcpyHostToDevice);
	std::vector<my_tuple> sorted(tests_size * features_size);
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
	sorted_tests = thrust::device_vector<my_tuple> (sorted);
	//auto start = std::chrono::high_resolution_clock::now();
	leafs = 1;
	depth = 0;
	node root;
	for (int i = 0; i < tests_size; i++)
	{
		root.size++;
		root.sum += train_set.answers[i];
		root.sum_of_squares += pow(train_set.answers[i], 2);
	}
	root.output_value = root.sum / root.size;
	root.node_mse = root.sum_of_squares / root.size - pow(root.output_value, 2);
	cudaMemcpyAsync(nodes, &root, sizeof(node), cudaMemcpyHostToDevice);
	//float new_error = root.node_mse;
	//float old_error = new_error + EPS;
	while (/*new_error < old_error &&*/ leafs < max_leafs && depth < max_depth && !features_set.empty())
	{
		cudaDeviceSynchronize();
		make_layer(depth);
		std::pair<int, float> feature_and_error = fill_layer();
		features_set.erase(feature_and_error.first);
		depth++;
		//old_error = new_error;
		//new_error = feature_and_error.second;
		//std::cout << "level " << depth << " created. training error: " << new_error << " best_feat: " << feature_and_error.first << std::endl;
	}
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + pow(2, depth) / (1 + BLOCK_SIZE), 1);
	make_last_layer_gpu<<<grid, block>>>(nodes, depth, pow(2, depth));
	cudaDeviceSynchronize();
	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = end - start;
	//std::cout << "leafs before pruning: " << leafs << std::endl;
	//std::cout << "new tree! calculating time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
	
	h_feature_id_at_depth = (int*)malloc(features_size * sizeof(int));
	h_nodes = (node*)malloc((pow(2, max_depth + 1) - 1) * sizeof(node));
	cudaMemcpyAsync(h_feature_id_at_depth, feature_id_at_depth, features_size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_nodes, nodes, (pow(2, max_depth + 1) - 1) * sizeof(node), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	make_tree_ptr();
	sorted_tests.clear();
	cudaFree(nodes);
	cudaFree(feature_id_at_depth);
	cudaFree(used_features);
	cudaFree(features);
	
	//prune(0);                                         //*******************TODO!!
	//std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
}

tree::~tree()
{
	cudaFree(nodes);
	cudaFree(feature_id_at_depth);
	cudaFree(used_features);
	cudaFree(features);
	//cudaFree(answers);
	free(h_feature_id_at_depth);
	free(h_nodes);
	//delete_node(root);
}

float tree::calculate_answer(test& _test)
{
	/*int cur_id = 0;
	node cur = h_nodes[cur_id];
	while (!cur.is_leaf)
	{
		if (_test.features[h_feature_id_at_depth[cur.depth]] < cur.split_value)
		{
			cur = h_nodes[cur_id * 2 + 1];
			cur_id = cur_id * 2 + 1;
		}
		else
		{
			cur = h_nodes[cur_id * 2 + 2];
			cur_id = cur_id * 2 + 2;
		}
	}
	return cur.output_value;
	*/

	node_ptr* cur = root;
	while (!cur->is_leaf)
	{
		cur = _test.features[h_feature_id_at_depth[cur->depth]] < cur->split_value ? cur->left : cur->right;
	}
	return cur->output_value;
}

float tree::calculate_error(data_set& test_set)
{
	float error = 0;
	for (int i = 0; i < test_set.tests_size; i++)
	{
		float ans = calculate_answer(test_set.tests[i]);
		error += ((ans - test_set.answers[i]) * (ans - test_set.answers[i]));
	}
	error /= (1.0 * test_set.tests_size);
	return error;
}

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
			nodes[2 * i + 1].is_leaf = false;
			nodes[2 * i + 2].is_leaf = false;
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

/*__device__ void calc_subtree_mse(node* nodes, int node_id, float* features, float* answers,
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
}*/

/*__global__ void prune_gpu(node* nodes, int node_id, bool* need_go_deeper, float* features, float* answers, int tests_size,
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
}*/

/*void tree::prune(int node_id)
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
}*/

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
		//printf("node %d test %d\n", cur, i);
	}
}

__global__ void fill_split_ids(int* node_id_of_test, my_tuple* sorted_tests, int tests_size, int features_size, int layer_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < tests_size && y < features_size) //used_feat??
	{
		my_tuple t = sorted_tests[y * tests_size + x];
		int id_shifted = node_id_of_test[t.test_id] - layer_size + 1;
		if (id_shifted >= 0)
		{
			sorted_tests[y * tests_size + x].split_id = 1 << id_shifted;
			//printf("test %d feat %d split_id %d\n", x, y, sorted_tests[y * tests_size + x].split_id);
		}
	}
}    

__global__ void calc_split_gpu(int* node_id_of_test, my_pair* errors, int tests_size, bool* used_features,
									 int features_size, int layer_size, my_tuple* sorted_tests)
{
	__shared__ my_tuple sorted_tests_shared[BLOCK_SIZE][BLOCK_SIZE];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < tests_size && y < features_size && !used_features[y])
	{
		my_tuple cur_my_tuple = sorted_tests[y * tests_size + x];
		int node_id = node_id_of_test[cur_my_tuple.test_id];
		if (node_id < layer_size - 1)
		{
			return;
		}
		int node_split_id_shifted = node_id - layer_size + 1;
		float l_sum = 0;
		float r_sum = 0;
		float l_sum_pow = 0;
		float r_sum_pow = 0;
		int l_size = 0;
		int r_size = 0;
		float l_avg = 0;
		float r_avg = 0;
		float l_err = 0;
		float r_err = 0;
		int parts = tests_size / BLOCK_SIZE + 1;
		for (int id = 0; id < parts; id++)
		{
			sorted_tests_shared[threadIdx.y][threadIdx.x] = sorted_tests[y * tests_size + id * BLOCK_SIZE + threadIdx.x];
			__syncthreads();
			int i = 0;
			for (; i < BLOCK_SIZE && id * BLOCK_SIZE + i <= x; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> node_split_id_shifted) & 1;
				l_sum += exists * cur_my_tuple.answer;
				l_sum_pow += exists * pow(cur_my_tuple.answer, 2);
				l_size += exists;
			}
			for (; i < BLOCK_SIZE && id * BLOCK_SIZE + i < tests_size; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> node_split_id_shifted) & 1;
				r_sum += exists * cur_my_tuple.answer;
				r_sum_pow += exists * pow(cur_my_tuple.answer, 2);
				r_size += exists;
			}
			__syncthreads();
		}
		l_avg = (l_size > 0) ? (l_sum / l_size) : 0; 
		r_avg = (r_size > 0) ? (r_sum / r_size) : 0; 
		/*for (int id = 0; id < parts; id++)
		{
			sorted_tests_shared[threadIdx.y][threadIdx.x] = sorted_tests[y * tests_size + id * BLOCK_SIZE + threadIdx.x];
			__syncthreads();
			int i = 0;
			for (; i < BLOCK_SIZE && id * BLOCK_SIZE + i <= x; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> node_split_id_shifted) & 1;
				float diff = cur_my_tuple.answer - l_avg;
				l_err += exists * pow(diff, 2); 
			}
			for (; i < BLOCK_SIZE && id * BLOCK_SIZE + i < tests_size; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> node_split_id_shifted) & 1;
				float diff = cur_my_tuple.answer - r_avg;
				r_err += exists * pow(diff, 2);
			}
			__syncthreads();
		}*/
		/*l_err = (l_size > 0) ? (l_err / l_size) : 0;
		r_err = (r_size > 0) ? (r_err / r_size) : 0;*/
		l_err = (l_size > 0) ? ((l_sum_pow - 2 * l_avg * l_sum) / l_size + pow(l_avg, 2)) : 0;
		r_err = (r_size > 0) ? ((r_sum_pow - 2 * r_avg * r_sum) / r_size + pow(r_avg, 2)) : 0;
		errors[y * tests_size + x] = my_pair(x, l_err + r_err);
		//printf("test: %d feat: %d err: %f l_s: %d r_s: %d l_err: %f r_err: %f\n", x, y, (float)(l_err + r_err), l_size, r_size, l_err, r_err);
	}
}

__global__ void calc_split_gpu2(node* nodes, my_pair* errors, int tests_size, bool* used_features,
									 int features_size, int layer_size, my_tuple* sorted_tests)
{
	//__shared__ my_tuple sorted_tests_shared[BLOCK_SIZE][BLOCK_SIZE];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < features_size && x < layer_size /*&& !used_features[y]*/)
	{
		int node_id = layer_size - 1 + x;
		node cur_node = nodes[node_id];
		/*if (!cur_node.is_exists || cur_node.is_leaf)
		{
			return;
		}*/
		my_tuple cur_my_tuple;
		float l_sum = 0;
		float r_sum = cur_node.sum;
		float l_sum_pow = 0;
		float r_sum_pow = cur_node.sum_of_squares;
		int l_size = 0;
		int r_size = cur_node.size;
		//float l_avg = 0;
		//float r_avg = 0;
		float l_err = 0;
		float r_err = 0;
		float ans_pow = 0;
		//int parts = tests_size / BLOCK_SIZE + 1;
		/*for (int id = 0; id < parts; id++)
		{
			sorted_tests_shared[threadIdx.y][threadIdx.x] = sorted_tests[y * tests_size + id * BLOCK_SIZE + threadIdx.x];
			__syncthreads();
			for (int i = 0; i < BLOCK_SIZE && id * BLOCK_SIZE + i < tests_size; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> x) & 1;
				ans_pow = pow(cur_my_tuple.answer, 2);
				l_sum += exists * cur_my_tuple.answer;
				l_sum_pow += exists * ans_pow;
				l_size += exists;
				r_sum -= exists * cur_my_tuple.answer;
				r_sum_pow -= exists * ans_pow;
				r_size -= exists;
				l_avg = (l_size > 0) ? (l_sum / l_size) : 0; 
				r_avg = (r_size > 0) ? (r_sum / r_size) : 0; 
				l_err = (l_size > 0) ? ((l_sum_pow - 2 * l_avg * l_sum) / l_size + pow(l_avg, 2)) : 0;
				r_err = (r_size > 0) ? ((r_sum_pow - 2 * r_avg * r_sum) / r_size + pow(r_avg, 2)) : 0;
				if (exists)
				{
				    errors[y * tests_size + id * BLOCK_SIZE + i] = my_pair(id * BLOCK_SIZE + i, l_err + r_err);
				}
			}
			__syncthreads();
		}*/
		for (int i = 0; i < tests_size; i++)
		{
			//cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
			cur_my_tuple = sorted_tests[y * tests_size + i];
			int exists = (cur_my_tuple.split_id >> x) & 1;
			ans_pow = pow(cur_my_tuple.answer, 2);
			l_sum += exists * cur_my_tuple.answer;
			l_sum_pow += exists * ans_pow;
			l_size += exists;
			r_sum -= exists * cur_my_tuple.answer;
			r_sum_pow -= exists * ans_pow;
			r_size -= exists;
			//l_avg = (l_size > 0) ? (l_sum / l_size) : 0;
			//r_avg = (r_size > 0) ? (r_sum / r_size) : 0;
			//l_err = (l_size > 0) ? ((l_sum_pow - 2 * l_avg * l_sum) / l_size + pow(l_avg, 2)) : 0;
			//r_err = (r_size > 0) ? ((r_sum_pow - 2 * r_avg * r_sum) / r_size + pow(r_avg, 2)) : 0;
			l_err = (l_size > 0) ? (l_sum_pow / l_size - pow(l_sum / l_size, 2)) : 0;
			r_err = (r_size > 0) ? (r_sum_pow / r_size - pow(r_sum / r_size, 2)) : 0;
			if (exists)
			{
				errors[y * tests_size + i] = my_pair(i, l_err + r_err);
			}
		}
		//l_avg = (l_size > 0) ? (l_sum / l_size) : 0; 
		//r_avg = (r_size > 0) ? (r_sum / r_size) : 0; 
		/*for (int id = 0; id < parts; id++)
		{
			sorted_tests_shared[threadIdx.y][threadIdx.x] = sorted_tests[y * tests_size + id * BLOCK_SIZE + threadIdx.x];
			__syncthreads();
			int i = 0;
			for (; i < BLOCK_SIZE && id * BLOCK_SIZE + i <= x; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> node_split_id_shifted) & 1;
				float diff = cur_my_tuple.answer - l_avg;
				l_err += exists * pow(diff, 2); 
			}
			for (; i < BLOCK_SIZE && id * BLOCK_SIZE + i < tests_size; i++)
			{
				cur_my_tuple = sorted_tests_shared[threadIdx.y][i];
				int exists = (cur_my_tuple.split_id >> node_split_id_shifted) & 1;
				float diff = cur_my_tuple.answer - r_avg;
				r_err += exists * pow(diff, 2);
			}
			__syncthreads();
		}*/
		/*l_err = (l_size > 0) ? (l_err / l_size) : 0;
		r_err = (r_size > 0) ? (r_err / r_size) : 0;*/
		/*l_err = (l_size > 0) ? ((l_sum_pow - 2 * l_avg * l_sum) / l_size + pow(l_avg, 2)) : 0;
		r_err = (r_size > 0) ? ((r_sum_pow - 2 * r_avg * r_sum) / r_size + pow(r_avg, 2)) : 0;
		errors[y * tests_size + x] = my_pair(x, l_err + r_err);*/
		//printf("test: %d feat: %d err: %f l_s: %d r_s: %d l_err: %f r_err: %f\n", x, y, (float)(l_err + r_err), l_size, r_size, l_err, r_err);
	}
}

__global__ void calc_min_error(node* nodes, my_pair* pre_errors,
							   float* errors, float* split_values, int tests_size, int features_size,
							   int layer_size, bool* used_features, my_tuple* sorted_tests, int* sorted_tests_ids)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < features_size && x < layer_size && !used_features[y])
	{
		int node_id = layer_size - 1 + x;
		node cur_node = nodes[node_id];
		if (!cur_node.is_exists || cur_node.is_leaf)
		{
			return;
		}
		for (int i = 0; i < tests_size; i++)
		{
			my_pair p1 = pre_errors[y * tests_size + i];
			int id = p1.sorted_tests_id;
			my_tuple t1 = sorted_tests[y * tests_size + id];
			my_tuple t2;
			if ((t1.split_id >> x) & 1)
			{
				int j = id + 1;
				while (j < tests_size)
				{
					t2 = sorted_tests[y * tests_size + j];
					if ((t2.split_id >> x) & 1)
					{
						break;
					}
					j++;
				}
				if (j == tests_size)
				{
					errors[y * layer_size + x] = p1.error;
					split_values[y * layer_size + x] = t1.feature + EPS;
					sorted_tests_ids[y * layer_size + x] = id;
					return;
				}
				if (t1.feature != t2.feature)
				{
					errors[y * layer_size + x] = p1.error;
					split_values[y * layer_size + x] = (t1.feature + t2.feature) / 2.0;
					sorted_tests_ids[y * layer_size + x] = id;
					return;
				}
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

__global__ void make_split_gpu(node* nodes, float* split_values,
							   int* best_feature, int tests_size, int layer_size, int* sorted_tests_ids, my_tuple* sorted_tests)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < layer_size)
	{
		int node_id = layer_size - 1 + x;
		if (!nodes[node_id].is_exists || nodes[node_id].is_leaf)
		{
			return;
		}
		int best_f = best_feature[0];
		float split_value = split_values[best_f * layer_size + x];
		nodes[node_id].split_value = split_value;
		//printf("split_val: %f\n", split_value);
		//node cur_node = nodes[node_id];
		float l_sum = 0;
		float r_sum = 0;
		float l_sum_pow = 0;
		float r_sum_pow = 0;
		int l_size = 0;
		int r_size = 0;
		float l_avg = 0;
		float r_avg = 0;
		float l_err = 0;
		float r_err = 0;
		int id = sorted_tests_ids[best_f * layer_size + x];
		my_tuple cur_my_tuple;
		/*for (int i = 0; i <= id; i++)
		{
			cur_my_tuple = sorted_tests[best_f * tests_size + i];
			int exists = (cur_my_tuple.split_id >> x) & 1;
			ans_pow = pow(cur_my_tuple.answer, 2);
			l_sum += exists * cur_my_tuple.answer;
			l_sum_pow += exists * ans_pow;
			l_size += exists;

		}
		r_sum -= l_sum;
		r_sum_pow -= l_sum_pow;
		r_size -= l_size;*/
		for (int i = 0; i <= id; i++)
		{
			cur_my_tuple = sorted_tests[best_f * tests_size + i];
			int exists = (cur_my_tuple.split_id >> x) & 1;
			l_sum += exists * cur_my_tuple.answer;
			l_sum_pow += exists * pow(cur_my_tuple.answer, 2);
			l_size += exists;
		}
		for (int i = id + 1; i < tests_size; i++)
		{
			cur_my_tuple = sorted_tests[best_f * tests_size + i];
			int exists = (cur_my_tuple.split_id >> x) & 1;
			r_sum += exists * cur_my_tuple.answer;
			r_sum_pow += exists * pow(cur_my_tuple.answer, 2);
			r_size += exists;
		}
		if (l_size == 0)
		{
			nodes[2 * node_id + 1].output_value = nodes[node_id].output_value;
			nodes[2 * node_id + 1].size = 0;
			nodes[2 * node_id + 1].is_leaf = true;
			nodes[2 * node_id + 1].node_mse = 0;
			nodes[2 * node_id + 2].output_value = nodes[node_id].output_value;
			nodes[2 * node_id + 2].size = nodes[node_id].size;
			nodes[2 * node_id + 2].sum = nodes[node_id].sum;
			nodes[2 * node_id + 2].sum_of_squares = nodes[node_id].sum_of_squares;
			nodes[2 * node_id + 2].node_mse = nodes[node_id].node_mse;
			nodes[2 * node_id + 2].is_leaf = (nodes[node_id].size <= 1) ? true : false;
			return;
		}
		if (r_size == 0)
		{
			nodes[2 * node_id + 2].output_value = nodes[node_id].output_value;
			nodes[2 * node_id + 2].size = 0;
			nodes[2 * node_id + 2].is_leaf = true;
			nodes[2 * node_id + 2].node_mse = 0;
			nodes[2 * node_id + 1].output_value = nodes[node_id].output_value;
			nodes[2 * node_id + 1].size = nodes[node_id].size;
			nodes[2 * node_id + 1].sum = nodes[node_id].sum;
			nodes[2 * node_id + 1].sum_of_squares = nodes[node_id].sum_of_squares;
			nodes[2 * node_id + 1].node_mse = nodes[node_id].node_mse;
			nodes[2 * node_id + 1].is_leaf = (nodes[node_id].size <= 1) ? true : false;
			return;
		}
		l_avg = l_sum / l_size; 
		r_avg = r_sum / r_size; 
		/*for (int i = 0; i <= id; i++)
		{
			cur_my_tuple = sorted_tests[best_f * tests_size + i];
			int exists = (cur_my_tuple.split_id >> x) & 1;
			float diff = cur_my_tuple.answer - l_avg;
			l_err += exists * pow(diff, 2); 
		}
		for (int i = id + 1; i < tests_size; i++)
		{
			cur_my_tuple = sorted_tests[best_f * tests_size + i];
			int exists = (cur_my_tuple.split_id >> x) & 1;
			float diff = cur_my_tuple.answer - r_avg;
			r_err += exists * pow(diff, 2);
		}*/
		//l_err = (l_sum_pow - 2 * l_avg * l_sum) / l_size + pow(l_avg, 2);
		//r_err = (r_sum_pow - 2 * r_avg * r_sum) / r_size + pow(r_avg, 2);
		l_err = l_sum_pow / l_size - pow(l_avg, 2);
		r_err = r_sum_pow / r_size - pow(r_avg, 2);
		//l_err = l_err / l_size;
		//r_err = r_err / r_size;
		nodes[2 * node_id + 1].output_value = l_avg;
		nodes[2 * node_id + 1].size = l_size;
		nodes[2 * node_id + 1].sum = l_sum;
		nodes[2 * node_id + 1].sum_of_squares = l_sum_pow;
		nodes[2 * node_id + 1].node_mse = l_err;
		nodes[2 * node_id + 1].is_leaf = (l_size == 1) ? true : false;
		nodes[2 * node_id + 2].output_value = r_avg;
		nodes[2 * node_id + 2].size = r_size;
		nodes[2 * node_id + 2].sum = r_sum;
		nodes[2 * node_id + 2].sum_of_squares = r_sum_pow;
		nodes[2 * node_id + 2].node_mse = r_err;
		nodes[2 * node_id + 2].is_leaf = (r_size == 1) ? true : false;
	}
}

/*__global__ void sort_helper(my_pair* errors, int tests_size, int features_size)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < features_size)
	{
		thrust::sort(thrust::cuda::par, errors + y * tests_size, errors + (y + 1) * tests_size);
	}
}*/

std::pair<int, float> tree::fill_layer()
{
	int layer_size = pow(2, depth);
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + tests_size / (1 + BLOCK_SIZE), 1);
	thrust::device_vector<int> node_id_of_test(tests_size);
	fill_node_id_of_test<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		feature_id_at_depth,
		features, tests_size, depth);
	block.y = BLOCK_SIZE;
	grid.y = 1 + features_size / (1 + BLOCK_SIZE);
	cudaDeviceSynchronize();
	fill_split_ids<<<grid, block>>>(thrust::raw_pointer_cast(&node_id_of_test[0]), thrust::raw_pointer_cast(&sorted_tests[0]),
		tests_size, features_size, layer_size);
	thrust::device_vector<my_pair> pre_errors(tests_size * features_size);
	cudaDeviceSynchronize();
	
	block.x = BLOCK_SIZE;
	grid.x = 1 + layer_size / (1 + BLOCK_SIZE);
	calc_split_gpu2<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&pre_errors[0]), tests_size, used_features, features_size, layer_size,
		thrust::raw_pointer_cast(&sorted_tests[0]));
	/*calc_split_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&pre_errors[0]), tests_size, used_features, features_size, layer_size,
		thrust::raw_pointer_cast(&sorted_tests[0]));*/
	thrust::device_vector<float> errors(layer_size * features_size, INF_INT);
	thrust::device_vector<float> split_values(layer_size * features_size, INF_INT);
	thrust::device_vector<int> sorted_tests_ids(layer_size * features_size, 0);
	cudaDeviceSynchronize();
	/*block.x = 1;
	grid.x = 1;
	sort_helper<<<grid, block>>>(thrust::raw_pointer_cast(&pre_errors[0]), tests_size, features_size);
	cudaDeviceSynchronize();*/
	for (int i = 0; i < features_size; i++)
	{
		thrust::sort(pre_errors.begin() + i * tests_size, pre_errors.begin() + (i + 1) * tests_size);
	}
	block.x = BLOCK_SIZE;
	grid.x = 1 + layer_size / (1 + BLOCK_SIZE);
	cudaDeviceSynchronize();
	calc_min_error<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&pre_errors[0]),
		thrust::raw_pointer_cast(&errors[0]),
		thrust::raw_pointer_cast(&split_values[0]),	tests_size, features_size, layer_size, used_features, thrust::raw_pointer_cast(&sorted_tests[0]),
		thrust::raw_pointer_cast(&sorted_tests_ids[0]));
	cudaDeviceSynchronize();
	thrust::replace(errors.begin(), errors.end(), INF_INT, 0);
	for (int i = 0; i < features_size; i++)
	{
		errors[i * layer_size] = thrust::reduce(errors.begin() + i * layer_size,
			errors.begin() + (i + 1) * layer_size, 0.0, thrust::plus<float>());
		//std::cout << i << " # " << errors[i * layer_size] << std::endl;
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
	make_split_gpu<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&split_values[0]), thrust::raw_pointer_cast(&best_feature[0]),
		tests_size, layer_size, thrust::raw_pointer_cast(&sorted_tests_ids[0]), thrust::raw_pointer_cast(&sorted_tests[0]));
	cudaDeviceSynchronize();
	return std::make_pair(best_feature[0], best_error[0]);
}

void tree::make_tree_ptr()
{
	root = new node_ptr();
	fill_node_ptr(root, 0);
}

void tree::fill_node_ptr(node_ptr* n, int node_id)
{
	node cur = h_nodes[node_id];
	n->depth = cur.depth;
	n->is_leaf = cur.is_leaf;
	n->output_value = cur.output_value;
	n->split_value = cur.split_value;
	if (!cur.is_leaf)
	{
		node_ptr* l = new node_ptr();
		node_ptr* r = new node_ptr();
		fill_node_ptr(l, 2 * node_id + 1);
		fill_node_ptr(r, 2 * node_id + 2);
		n->left = l;
		n->right = r;
	}
}
