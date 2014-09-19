#include <set>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <thrust\reduce.h>
#include <thrust\replace.h>
#include <thrust\execution_policy.h>
#include "device_launch_parameters.h"
#include <thrust\host_vector.h>
#include "tree.cuh"

#define INF 1e5
#define INF_INT 100000
#define EPS 1e-5
#define BLOCK_SIZE 32

double calc_root_mse(data_set& train_set,  double avg, double n)
{
	double ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (int i = 0; i < train_set.size(); i++)
	{
		ans += ((train_set[i].anwser - avg) * (train_set[i].anwser - avg));
	}
	ans /= n; 
	return ans;
}

 /*__device__ node::node(const node& other) : depth(other.depth), is_leaf(other.is_leaf), is_exists(is_exists),
	node_mse(other.node_mse), output_value(other.output_value), size(other.size), split_value(other.split_value),
	subtree_mse(other.subtree_mse), sum(other.sum) {}*/

node::node(int depth) : depth(depth)
{
	is_leaf = false;
	is_exists = true;
	node_mse = 0;
	size = 0;
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

tree::tree(data_set& train_set, int max_leafs) : max_leafs(max_leafs)
{
	features_size = train_set[0].features.size();
	tests_size = train_set.size();
	cudaMalloc(&tests_d, tests_size * sizeof(test_d));
	cudaMalloc(&nodes, (pow(2, features_size + 1) - 1) * sizeof(node));
	cudaMalloc(&feature_id_at_depth, features_size * sizeof(int));
	cudaMalloc(&used_features, features_size * sizeof(bool));
	cudaMemsetAsync(used_features, false, features_size * sizeof(bool));
	std::set<int> features;
	for (size_t i = 0; i < train_set[0].features.size(); i++)
	{
		features.insert(i);
	}
	thrust::host_vector<test_d> tests_h;
	for (data_set::iterator cur_test = train_set.begin(); cur_test != train_set.end(); cur_test++)
	{
		tests_h.push_back(test_d(*cur_test));
	}
	cudaMemcpyAsync(tests_d, &tests_h[0], sizeof(test_d) * tests_size, cudaMemcpyHostToDevice);
	leafs = 1;
	depth = 0;
	node root(0);
	root.sum = 0;
	root.size = 0;
	for (int i = 0; i < tests_size; i++)
	{
		root.size++;
		root.sum += train_set[i].anwser;
	}
	root.output_value = root.sum / root.size;
	root.node_mse = calc_root_mse(train_set, root.output_value, root.size); 
	cudaMemcpyAsync(nodes, &root, sizeof(node), cudaMemcpyHostToDevice);
	double new_error = root.node_mse;
	double old_error = new_error + EPS;
	while (/*new_error < old_error &&*/ leafs < max_leafs && !features.empty())
	{
		cudaDeviceSynchronize();
		make_layer(depth);
		std::pair<int, double> feature_and_error = fill_layer();
		cudaMemcpyAsync(feature_id_at_depth + depth, &feature_and_error.first, sizeof(int), cudaMemcpyHostToDevice);
		features.erase(feature_and_error.first);
		depth++;
		old_error = new_error;
		new_error = feature_and_error.second;
		std::cout << "level " << depth << " created. training error: " << new_error << std::endl;
	}
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + pow(2, depth) / (1 + BLOCK_SIZE), 1);
	make_last_layer_gpu<<<grid, block>>>(nodes, depth, pow(2, depth));
	cudaDeviceSynchronize();
	std::cout << "leafs before pruning: " << leafs << std::endl;
	//prune(0);                                         //*******************TODO!!
	//std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
}

tree::~tree()
{
	//delete_node(root);
}

double tree::calculate_anwser(test& _test)
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

/*double tree::calculate_error(data_set& test_set)
{
	double error = 0;
	for (data_set::iterator cur_test = test_set.begin(); cur_test != test_set.end(); cur_test++)
	{
		double ans = calculate_anwser(*cur_test);
		error += ((ans - cur_test->anwser) * (ans - cur_test->anwser));
	}
	error /= (1.0 * test_set.size());
	return error;
}*/

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

__device__ void calc_subtree_mse(node* nodes, int node_id, test_d* tests, int tests_size, int* feature_id_at_depth)
{
	double error = 0;
	double ans = 0;
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
			cur_node_id = tests[i].features[feature_id_at_depth[nodes[cur_node_id].depth]] < nodes[cur_node_id].split_value ?
				cur_node_id * 2 + 1 : cur_node_id * 2 + 2;
		}
		if (is_test_in_node)
		{
			ans = nodes[cur_node_id].output_value;
			error += ((ans - *tests[i].anwser) * (ans - *tests[i].anwser));
		}
	}
	error /= (1.0 * nodes[node_id].size);
	nodes[node_id].subtree_mse = error;
}

__global__ void prune_gpu(node* nodes, int node_id, bool* need_go_deeper, test_d* tests, int tests_size,
						  int* feature_id_at_depth, int* new_leafs)
{
	*need_go_deeper = false;
	if (!nodes[node_id].is_leaf && nodes[node_id].is_exists)
	{
		calc_subtree_mse(nodes, 2 * node_id + 1, tests, tests_size, feature_id_at_depth);
		calc_subtree_mse(nodes, 2 * node_id + 2, tests, tests_size, feature_id_at_depth);
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
	prune_gpu<<<grid, block>>>(nodes, node_id, need_go_deeper, tests_d, tests_size, feature_id_at_depth,
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
									 test_d* tests, int tests_size, int depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < tests_size)
	{
		int cur = 0;
		while (nodes[cur].depth < depth && !nodes[cur].is_leaf && nodes[cur].is_exists)
		{
			if (tests[i].features[feature_id_at_depth[nodes[cur].depth]] < nodes[cur].split_value)
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

__global__ void calc_split_gpu(node* nodes, int* node_id_of_test, double* errors, double* split_values,
									 test_d* tests, int tests_size, bool* used_features, int features_size, int depth,
									 int layer_size)
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
		double feature_value = tests[x].features[y];
		double split_value = INF;
		double l_sum = 0;
		double r_sum = 0;
		double l_size = 0;
		double r_size = 0;
		double l_avg = 0;
		double r_avg = 0;
		double l_err = 0;
		double r_err = 0;
		for (int i = 0; i < tests_size; i++)
		{
			if (node_id_of_test[i] == node_id)
			{
				if (tests[i].features[y] >= feature_value && tests[i].features[y] < split_value)
				{
					split_value = tests[i].features[y];
				}
				if (tests[i].features[y] <= feature_value)
				{
					l_sum += *tests[i].anwser;
					l_size++;
				}
				else
				{
					r_sum += *tests[i].anwser;
					r_size++;
				}
			}
		}
		split_value = (feature_value + split_value) / 2.0 + EPS;
		l_avg = (l_size > 0) ? (l_sum / l_size) : 0; 
		r_avg = (r_size > 0) ? (r_sum / r_size) : 0; 
		for (int i = 0; i < tests_size; i++)
		{
			if (node_id_of_test[i] == node_id)
			{
				if (tests[i].features[y] < split_value)
				{
					l_err += ((*tests[i].anwser - l_avg) * (*tests[i].anwser - l_avg));
				}
				else
				{
					r_err += ((*tests[i].anwser - r_avg) * (*tests[i].anwser - r_avg));
				}
			}
		}
		l_err = (l_size > 0) ? (l_err / l_size) : 0;
		r_err = (r_size > 0) ? (r_err / r_size) : 0;
		errors[y * tests_size + x] = l_err + r_err;
		split_values[y * tests_size + x] = split_value;
	}
}

__global__ void calc_min_error(int* node_id_of_test, double* pre_errors, double* pre_split_values,
							   double* errors, double* split_values, int tests_size, int features_size,
							   int layer_size)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y < features_size)
	{
		for (int i = 0; i < tests_size; i++)
		{
			int node_id = node_id_of_test[i];
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

__global__ void calc_best_feature(double* errors, bool* used_features, int* best_features, double* best_errors,
									 int features_size, int layer_size)
{
	double best_error = INF;
	int best_feature = -1;
	for (int i = 0; i < features_size; i++)
	{
		if (!used_features[i])
		{
			double cur_error = errors[i * layer_size];
			if (cur_error < best_error)
			{
				best_error = cur_error;
				best_feature = i;
			}
		}
	}
	best_features[0] = best_feature;
	best_errors[0] = best_error;
}

__global__ void make_split_gpu(node* nodes, int* node_id_of_test, double* split_values,
							   int* best_feature, test_d* tests,
									 int tests_size, int features_size, int depth, int layer_size, bool* used_features)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0)
	{
		used_features[best_feature[0]] = true;
	}
	if (x < layer_size)
	{
		int node_id = layer_size - 1 + x;
		if (!nodes[node_id].is_exists || nodes[node_id].is_leaf)
		{
			return;
		}
		double split_value = split_values[best_feature[0] * layer_size + x];
		nodes[node_id].split_value = split_value;
		double l_sum = 0;
		double r_sum = 0;
		double l_size = 0;
		double r_size = 0;
		double l_avg = 0;
		double r_avg = 0;
		double l_err = 0;
		double r_err = 0;
		for (int i = 0; i < tests_size; i++)
		{
			if (node_id_of_test[i] == node_id)
			{
				if (tests[i].features[best_feature[0]] < split_value)
				{
					l_sum += *tests[i].anwser;
					l_size++;
				}
				else
				{
					r_sum += *tests[i].anwser;
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
		for (int i = 0; i < tests_size; i++)
		{
			if (node_id_of_test[i] == node_id)
			{
				if (tests[i].features[best_feature[0]] < split_value)
				{
					l_err += ((*tests[i].anwser - l_avg) * (*tests[i].anwser - l_avg));
				}
				else
				{
					r_err += ((*tests[i].anwser - r_avg) * (*tests[i].anwser - r_avg));
				}
			}
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
}

std::pair<int, double> tree::fill_layer()
{
	int layer_size = pow(2, depth);
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + tests_size / (1 + BLOCK_SIZE), 1);
	thrust::device_vector<int> node_id_of_test(tests_size);
	fill_node_id_of_test<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]), feature_id_at_depth,
		tests_d, tests_size, depth);
	thrust::device_vector<double> pre_errors(tests_size * features_size, 0);
	thrust::device_vector<double> pre_split_values(tests_size * features_size, 0);
	block.y = BLOCK_SIZE;
	grid.y = 1 + features_size / (1 + BLOCK_SIZE);
	cudaDeviceSynchronize();
	calc_split_gpu<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&pre_errors[0]), thrust::raw_pointer_cast(&pre_split_values[0]),
		tests_d, tests_size, used_features, features_size, depth, layer_size);
	thrust::device_vector<double> errors(pow(2, depth) * features_size, INF_INT);
	thrust::device_vector<double> split_values(pow(2, depth) * features_size, INF_INT);
	block.x = 1;
	grid.x = 1;
	cudaDeviceSynchronize();
	calc_min_error<<<grid, block>>>(thrust::raw_pointer_cast(&node_id_of_test[0]), thrust::raw_pointer_cast(&pre_errors[0]),
		thrust::raw_pointer_cast(&pre_split_values[0]),	thrust::raw_pointer_cast(&errors[0]),
		thrust::raw_pointer_cast(&split_values[0]),	tests_size, features_size, layer_size);
	cudaDeviceSynchronize();
	thrust::replace(errors.begin(), errors.end(), INF_INT, 0);
	for (int i = 0; i < features_size; i++)
	{
		errors[i * layer_size] = thrust::reduce(errors.begin() + i * layer_size,
			errors.begin() + (i + 1) * layer_size, 0.0, thrust::plus<double>());
	}
	cudaDeviceSynchronize();
	thrust::device_vector<int> best_feature(1);
	thrust::device_vector<double> best_error(1);
	block.x = 1;
	block.y = 1;
	grid.x = 1;
	grid.y = 1;
	calc_best_feature<<<grid, block>>>(thrust::raw_pointer_cast(&errors[0]), used_features,
		thrust::raw_pointer_cast(&best_feature[0]), thrust::raw_pointer_cast(&best_error[0]), features_size, layer_size);
	block.x = BLOCK_SIZE;
	grid.x = 1 + pow(2, depth) / (1 + BLOCK_SIZE);
	cudaDeviceSynchronize();
	make_split_gpu<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&split_values[0]), thrust::raw_pointer_cast(&best_feature[0]),
		tests_d, tests_size, features_size, depth, layer_size, used_features);
	cudaDeviceSynchronize();
	return std::make_pair(best_feature[0], best_error[0]);
}