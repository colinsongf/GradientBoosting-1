#include <set>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include "tree.cuh"
#include "thrust\sort.h"
#include "thrust\execution_policy.h"
#include "device_launch_parameters.h"
#include "thrust\host_vector.h"

#define INF 1e10
#define BLOCK_SIZE 32
#define MAX_FEATURES 23
#define MAX_TESTS 1500

double calc_mse(data_set& train_set, int begin_id, int end_id, double avg, double n)
{
	double ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (size_t i = begin_id; i < end_id; i++)
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
}

node::node()
{
	is_leaf = false;
	is_exists = true;
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

__device__ double calc_mse_d(test_d* tests, int begin_id, int end_id, double avg, double n)
{
	double ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (size_t i = begin_id; i < end_id; i++)
	{
		ans += ((*tests[i].anwser - avg) * (*tests[i].anwser - avg));
	}
	ans /= n; 
	return ans;
}

__global__ void make_last_layer_gpu(node* nodes, int depth)
{
	int begin_id = pow((double)2, (double)depth) - 1;
	int end_id = begin_id + pow((double)2, (double)depth);
	for (int i = begin_id; i < end_id; i++)
	{
		nodes[i].is_leaf = true;
	}
}

tree::tree(data_set& train_set, int max_leafs) : max_leafs(max_leafs)
{
	tests_d_size = 0;
	cudaMalloc(&tests_d, MAX_TESTS * sizeof(test_d));
	cudaMalloc(&nodes, (pow(2, MAX_FEATURES) - 1) * sizeof(node));
	cudaMalloc(&feature_id_at_depth, MAX_FEATURES * sizeof(int));
	//cudaError_t cuda_err = cudaGetLastError();
	std::set<int> features;
	for (size_t i = 0; i < train_set[0].features.size(); i++)
	{
		features.insert(i);
	}
	for (data_set::iterator cur_test = train_set.begin(); cur_test != train_set.end(); cur_test++)
	{
		test_d temp_test(*cur_test);
		cudaMemcpy(tests_d + tests_d_size, &temp_test, sizeof(test_d), cudaMemcpyHostToDevice);
		tests_d_size++;
	}
	leafs = 1;
	depth = 0;
	node root(0);
	root.data_begin_id = 0;
	root.data_end_id = tests_d_size;
	root.sum = 0;
	root.size = 0;
	for (size_t i = root.data_begin_id; i < root.data_end_id; i++)
	{
		root.size++;
		root.sum += train_set[i].anwser;
	}
	root.output_value = root.sum / root.size;
	root.node_mse = calc_mse(train_set, root.data_begin_id, root.data_end_id, root.output_value, root.size); 
	cudaMemcpy(nodes, &root, sizeof(node), cudaMemcpyHostToDevice);
	while (leafs < max_leafs && !features.empty())
	{
		double min_error = INF;
		int best_feature = -1;
		make_layer(depth);
		for (std::set<int>::iterator cur_split_feature = features.begin(); cur_split_feature != features.end(); cur_split_feature++)
		//choose best split feature at current depth
		{
			double cur_error = split_layer(depth, *cur_split_feature);
			if (cur_error < min_error)
			{
				min_error = cur_error;
				best_feature = *cur_split_feature;
			}
		}
		split_layer(depth, best_feature);
		cudaMemcpy(feature_id_at_depth + depth, &best_feature, sizeof(int), cudaMemcpyHostToDevice);
		features.erase(best_feature);
		depth++;
		std::cout << "level " << depth << " created. training error: " << min_error << std::endl;
	}
	dim3 block(1, 1);
	dim3 grid(1, 1);
	make_last_layer_gpu<<<grid, block>>>(nodes, depth);
	cudaDeviceSynchronize();
	std::cout << "leafs before pruning: " << leafs << std::endl;
	prune(0);                                         //*******************TODO!!
	std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
	/*while (layers[depth].empty())  //***************************TODO!!!
	{
		depth--;
	}*/
}

/*tree::~tree()
{
	delete_node(root);
}*/

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

__device__ void delete_node(node* nodes, int node_id)
{
	if (!nodes[node_id].is_exists)
	{
		return;
	}
	delete_node(nodes, node_id * 2 + 1);
	delete_node(nodes, node_id * 2 + 2);
	nodes[node_id].is_exists = false;
}

__global__ void make_layer_gpu(node* nodes, int begin_id, int end_id, int* new_leafs)
{
	*new_leafs = 0;
	for (int i = begin_id; i < end_id; i++) //make children for non-leaf nodes at current depth
	{
		if (!nodes[i].is_leaf && nodes[i].is_exists)
		{
			nodes[2 * i + 1].is_exists = true;
			nodes[2 * i + 2].is_exists = true;
			nodes[2 * i + 1].depth = nodes[i].depth + 1;
			nodes[2 * i + 2].depth = nodes[i].depth + 1;
			*new_leafs = *new_leafs + 1;
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
	int* new_leafs_d;
	int new_leafs_h;
	cudaMalloc(&new_leafs_d, sizeof(int));
	//cudaError cuda_err = cudaGetLastError();
	int begin_id = pow(2, depth) - 1;
	int end_id = begin_id + pow(2, depth);
	dim3 block(1, 1);
	dim3 grid(1, 1);
	make_layer_gpu<<<grid, block>>>(nodes, begin_id, end_id, new_leafs_d);
	cudaDeviceSynchronize();
	cudaMemcpy(&new_leafs_h, new_leafs_d, sizeof(int), cudaMemcpyDeviceToHost);
	leafs += new_leafs_h;
	cudaFree(new_leafs_d);
}

__device__ void calc_subtree_mse(node* nodes, int node_id, test_d* tests, int* feature_id_at_depth)
{
	double error = 0;
	for (int i = nodes[node_id].data_begin_id; i < nodes[node_id].data_end_id; i++)
	{
		int cur_node_id = node_id;
		while (!nodes[cur_node_id].is_leaf)
		{
			cur_node_id = tests[i].features[feature_id_at_depth[nodes[cur_node_id].depth]] < nodes[cur_node_id].split_value ?
				cur_node_id * 2 + 1 : cur_node_id * 2 + 2;
		}
		double ans = nodes[cur_node_id].output_value;
		error += ((ans - *tests[i].anwser) * (ans - *tests[i].anwser));
	}
	error /= (1.0 * nodes[node_id].size);
	nodes[node_id].subtree_mse = error;
}

__global__ void prune_gpu(node* nodes, int node_id, bool* need_go_deeper, test_d* tests, int* feature_id_at_depth)
{
	*need_go_deeper = false;
	if (!nodes[node_id].is_leaf && nodes[node_id].is_exists)
	{
		calc_subtree_mse(nodes, 2 * node_id + 1, tests, feature_id_at_depth);
		calc_subtree_mse(nodes, 2 * node_id + 2, tests, feature_id_at_depth);
		if (nodes[node_id].node_mse <= nodes[2 * node_id + 1].subtree_mse + nodes[2 * node_id + 2].subtree_mse)
		{
			nodes[node_id].is_leaf = true;
			delete_node(nodes, 2 * node_id + 1);
			delete_node(nodes, 2 * node_id + 2);
		}
		else
		{
			*need_go_deeper = true;
		}
	}
}

__device__ void recalc_leafs_gpu2(node* nodes, int node_id, int* new_leafs)
{
	if (!nodes[node_id].is_exists)
	{
		return;
	}
	if (nodes[node_id].is_leaf)
	{
		*new_leafs = *new_leafs + 1;
	}
	recalc_leafs_gpu2(nodes, node_id * 2 + 1, new_leafs);
	recalc_leafs_gpu2(nodes, node_id * 2 + 2, new_leafs);
}

__global__ void recalc_leafs_gpu(node* nodes, int node_id, int* new_leafs)
{
	*new_leafs = 0;
	if (nodes[node_id].is_exists)
	{
		recalc_leafs_gpu2(nodes, node_id, new_leafs);
	}
}

void tree::prune(int node_id)
{
	int* new_leafs;
	int new_leafs_h;
	bool* need_go_deeper;
	bool need_go_deeper_h = false;
	cudaMalloc(&need_go_deeper, sizeof(bool));
	cudaMalloc(&new_leafs, sizeof(int));
	cudaError_t cuda_err = cudaGetLastError();
	dim3 block(1, 1);
	dim3 grid(1, 1);
	prune_gpu<<<grid, block>>>(nodes, node_id, need_go_deeper, tests_d, feature_id_at_depth);
	cudaDeviceSynchronize();
	cudaMemcpy(&need_go_deeper_h, need_go_deeper, sizeof(bool), cudaMemcpyDeviceToHost);
	cuda_err = cudaGetLastError();
	cudaFree(need_go_deeper);
	if (need_go_deeper_h)
	{
		prune(2 * node_id + 1);
		prune(2 * node_id + 2);
	}
	recalc_leafs_gpu<<<grid, block>>>(nodes, node_id, new_leafs);
	cudaDeviceSynchronize();
	cudaMemcpy(&new_leafs_h, new_leafs, sizeof(int), cudaMemcpyDeviceToHost);
	cuda_err = cudaGetLastError();
	leafs = new_leafs_h;
	cudaFree(new_leafs);
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

struct test_d_comparator {
	__host__ __device__ test_d_comparator(int split_feature_id) : split_feature_id(split_feature_id) {}
	__host__ __device__	bool operator()(test_d t1, test_d t2)
	{
		return t1.features[split_feature_id] < t2.features[split_feature_id];
	}
	int split_feature_id;
};

__global__ void split_node_gpu(node* nodes, test_d* tests, double* errors, int split_feature_id, int depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < (int)(pow((double)2, (double)depth)))
	{
		int id = pow((double)2, (double)depth) - 1 + i;
		if (!nodes[id].is_exists)
		{
			errors[i] = 0;
			return;
		}
		thrust::sort(thrust::seq, tests + nodes[id].data_begin_id, tests + nodes[id].data_end_id,
				test_d_comparator(split_feature_id));
		if (nodes[id].is_leaf)
		{
			errors[i] = calc_mse_d(tests, nodes[id].data_begin_id, nodes[id].data_end_id, nodes[id].output_value, nodes[id].size);
			return;
		}
		double l_sum = 0;
		double l_size = 0;
		double r_sum = nodes[id].sum;
		double r_size = nodes[id].size;
		double best_mse = INF; 
		for (size_t j = nodes[id].data_begin_id + 1; j < nodes[id].data_end_id; j++) //try all possible splits
		{
			l_sum += *tests[j - 1].anwser;
			l_size++;
			r_sum -= *tests[j - 1].anwser;
			r_size--;
			if (tests[j].features[split_feature_id] == tests[j - 1].features[split_feature_id])
			{
				continue;
			}
			double l_avg = l_sum / l_size;
			double r_avg = r_sum / r_size;
			double l_mse = calc_mse_d(tests, nodes[id].data_begin_id, j, l_avg, l_size);
			double r_mse = calc_mse_d(tests, j, nodes[id].data_end_id, r_avg, r_size);
			double cur_mse = l_mse + r_mse;
			if (cur_mse < best_mse)
			{
				best_mse = cur_mse;
				nodes[id].split_value = tests[j].features[split_feature_id];
				nodes[2 * id + 1].data_begin_id = nodes[id].data_begin_id;
				nodes[2 * id + 1].data_end_id = j;
				nodes[2 * id + 1].output_value = l_avg;
				nodes[2 * id + 1].size = l_size;
				nodes[2 * id + 1].sum = l_sum;
				nodes[2 * id + 1].node_mse = l_mse;
				nodes[2 * id + 1].is_leaf = (l_size == 1) ? true : false;
				nodes[2 * id + 2].data_begin_id = j;
				nodes[2 * id + 2].data_end_id = nodes[id].data_end_id;
				nodes[2 * id + 2].output_value = r_avg;
				nodes[2 * id + 2].size = r_size;
				nodes[2 * id + 2].sum = r_sum;
				nodes[2 * id + 2].node_mse = r_mse;
				nodes[2 * id + 2].is_leaf = (r_size == 1) ? true : false;
			}
		}
		if (best_mse == INF)
		{
			best_mse = nodes[id].node_mse;
			nodes[id].split_value = tests[nodes[id].data_begin_id + 1].features[split_feature_id];
			nodes[2 * id + 1].output_value = nodes[id].output_value;
			nodes[2 * id + 1].size = 0;
			nodes[2 * id + 1].is_leaf = true;
			nodes[2 * id + 2].data_begin_id = nodes[id].data_begin_id;
			nodes[2 * id + 2].data_end_id = nodes[id].data_end_id;
			nodes[2 * id + 2].output_value = nodes[id].output_value;
			nodes[2 * id + 2].size = nodes[id].size;
			nodes[2 * id + 2].sum = nodes[id].sum;
			nodes[2 * id + 2].node_mse = nodes[id].node_mse;
			nodes[2 * id + 2].is_leaf = (nodes[id].size <= 1) ? true : false;
		}
		errors[i] = best_mse;
	}
}


double tree::split_layer(int depth, int split_feature_id)
{
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + (int)(pow(2, depth)) / (1 + BLOCK_SIZE), 1);
	thrust::device_vector<double> errors(pow(2, depth), 0);
	split_node_gpu<<<grid, block>>>(nodes, tests_d,
		thrust::raw_pointer_cast(&errors[0]), split_feature_id, depth);
	cudaDeviceSynchronize();
	//cudaError_t cuda_err = cudaGetLastError();
	double error = 0;
	thrust::host_vector<double> errors_h = errors;
	for (size_t i = 0; i < errors_h.size(); i++)
	{
		error += errors_h[i];
	}
	return error;
}