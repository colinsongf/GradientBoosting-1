#include <set>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include "tree.cuh"
#include "thrust\sort.h"
#include <thrust\reduce.h>
#include <thrust\execution_policy.h>
#include "device_launch_parameters.h"
#include "thrust\host_vector.h"

#define INF 1e10
#define EPS 1e-5
#define BLOCK_SIZE 32
#define MAX_FEATURES 23
#define MAX_TESTS 1500

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

__host__ __device__ node::node()
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

//__device__ double calc_mse_d(test_d* tests, int begin_id, int end_id, double avg, double n)
//{
//	double ans = 0;
//	if (n == 0)
//	{
//		return ans;
//	}
//	for (size_t i = begin_id; i < end_id; i++)
//	{
//		ans += ((*tests[i].anwser - avg) * (*tests[i].anwser - avg));
//	}
//	ans /= n; 
//	return ans;
//}

__global__ void make_last_layer_gpu(node* nodes, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < pow((double)2, (double)depth))
	{
		nodes[x + (int)pow((double)2, (double)depth) - 1].is_leaf = true;
	}
	/*int begin_id = pow((double)2, (double)depth) - 1;
	int end_id = begin_id + pow((double)2, (double)depth);
	for (int i = begin_id; i < end_id; i++)
	{
		nodes[i].is_leaf = true;
	*/
}

tree::tree(data_set& train_set, int max_leafs) : max_leafs(max_leafs)
{
	tests_d_size = 0;
	cudaMalloc(&tests_d, MAX_TESTS * sizeof(test_d));
	cudaMalloc(&nodes, (pow(2, MAX_FEATURES) - 1) * sizeof(node));
	cudaMalloc(&feature_id_at_depth, MAX_FEATURES * sizeof(int));
	//cudaError_t cuda_err = cudaGetLastError();
	cudaMalloc(&used_features, MAX_FEATURES * sizeof(bool));
	features_size = train_set[0].features.size();
	bool f = false;
	for (int i = 0; i < features_size; i++)
	{
		cudaMemcpy(used_features + i, &f, sizeof(bool), cudaMemcpyHostToDevice);
	}
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
	root.sum = 0;
	root.size = 0;
	for (int i = 0; i < tests_d_size; i++)
	{
		root.size++;
		root.sum += train_set[i].anwser;
	}
	root.output_value = root.sum / root.size;
	root.node_mse = calc_root_mse(train_set, root.output_value, root.size); 
	cudaMemcpy(nodes, &root, sizeof(node), cudaMemcpyHostToDevice);
	while (leafs < max_leafs && !features.empty())
	{
		cudaError_t cuda_err = cudaGetLastError();
		make_layer(depth);
		cuda_err = cudaGetLastError();
		std::pair<int, double> p = fill_layer();
		cudaMemcpy(feature_id_at_depth + depth, &p.first, sizeof(int), cudaMemcpyHostToDevice);
		cuda_err = cudaGetLastError();
		features.erase(p.first);
		depth++;
		std::cout << "level " << depth << " created. training error: " << p.second << std::endl;
	}
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + pow(2, depth) / (1 + BLOCK_SIZE), 1);
	make_last_layer_gpu<<<grid, block>>>(nodes, depth);
	cudaDeviceSynchronize();
	std::cout << "leafs before pruning: " << leafs << std::endl;
	//prune(0);                                         //*******************TODO!!
	std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
	/*while (layers[depth].empty())  //***************************TODO!!!
	{
		depth--;
	}*/
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
	/**new_leafs = 0;
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
	}*/
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
	/*int* new_leafs_d;
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
	cudaFree(new_leafs_d);*/
}

__device__ void calc_subtree_mse(node* nodes, int node_id, test_d* tests, int* feature_id_at_depth)
{
	/*double error = 0;
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
	nodes[node_id].subtree_mse = error;*/
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

/*__global__ void split_node_gpu(node* nodes, test_d* tests, double* errors, int split_feature_id, int depth)
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
}*/

__global__ void fill_node_id_of_test(node* nodes, int* node_id_of_test, int* feature_id_at_depth,
									 test_d* tests, int tests_d_size, int depth)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < tests_d_size)
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

__device__ double calc_mse_d(test_d* tests, int* node_id_of_test, int tests_d_size, int node_id, double avg, double n)
{
	double ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (int i = 0; i < tests_d_size; i++)
	{
		if (node_id_of_test[i] == node_id)
		{
			ans += ((*tests[i].anwser - avg) * (*tests[i].anwser - avg));
		}
	}
	ans /= n; 
	return ans;
}

__global__ void calc_split_gpu(node* nodes, int* node_id_of_test, double* errors, double* split_values,
									 test_d* tests, int tests_d_size, bool* used_features, int features_size, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < tests_d_size && y < features_size && !used_features[y])
	{
		int node_id = node_id_of_test[x];
		if (node_id < pow((double)2, (double)depth) - 1)
		{
			return;
		}
		double feature_value = tests[x].features[y];
		double split_value = INF;
		/*if (!nodes[node_id].is_exists)
		{
			return;
		}*/
		if (nodes[node_id].is_leaf)
		{
			errors[y * tests_d_size + x] = nodes[node_id].node_mse;
			return;
		}
		double l_sum = 0;
		double r_sum = 0;
		double l_size = 0;
		double r_size = 0;
		double l_avg = 0;
		double r_avg = 0;
		double l_err = 0;
		double r_err = 0;
		for (int i = 0; i < tests_d_size; i++)
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
		split_value = (feature_value + split_value) / 2.0 + 0.0001;
		l_avg = (l_size > 0) ? (l_sum / l_size) : 0; 
		r_avg = (r_size > 0) ? (r_sum / r_size) : 0; 
		for (int i = 0; i < tests_d_size; i++)
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
		errors[y * tests_d_size + x] = l_err + r_err;
		split_values[y * tests_d_size + x] = split_value;
	}
}

__global__ void calc_min_error(node* nodes, int* node_id_of_test, double* pre_errors, double* pre_split_values,
							   double* errors, double* split_values,
									 int tests_d_size, int features_size, int depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int node_id = pow((double)2, (double)depth) - 1 + x;
	if (x < pow((double)2, (double)depth) && y < features_size && nodes[node_id].is_exists)
	{
		if (nodes[node_id].is_leaf)
		{
			for (int i = 0; i < tests_d_size; i++)
			{
				if (node_id_of_test[i] == node_id)
				{
					errors[y * (int)pow((double)2, (double)depth) + x] = pre_errors[y * tests_d_size + i];
					return;
				}
			}
			errors[y * (int)pow((double)2, (double)depth) + x] = 0;
			return;
		}
		double best_error = INF;
		double best_split_value = INF;
		for (int i = 0; i < tests_d_size; i++)
		{
			if (node_id_of_test[i] == node_id && pre_errors[y * tests_d_size + i] < best_error)
			{
				best_error = pre_errors[y * tests_d_size + i];
				best_split_value = pre_split_values[y * tests_d_size + i];
			}
		}
		if (best_error == INF)
		{
			int size = nodes[node_id].size;
			printf("bad %d size %d \n", node_id, size);
		}
		errors[y * (int)pow((double)2, (double)depth) + x] = best_error;
		split_values[y * (int)pow((double)2, (double)depth) + x] = best_split_value;
	}
}

__global__ void calc_best_feature(double* errors, bool* used_features, int* best_features, double* best_errors,
									 int features_size, int depth)
{
	int nodes_size = pow((double)2, (double)depth);
	double best_error = INF;
	int best_feature = -1;
	for (int i = 0; i < features_size; i++)
	{
		if (used_features[i])
		{
			continue;
		}
		double cur_error = errors[i * nodes_size];
		if (cur_error < best_error)
		{
			best_error = cur_error;
			best_feature = i;
		}
	}
	best_features[0] = best_feature;
	best_errors[0] = best_error;
}

__global__ void make_split_gpu(node* nodes, int* node_id_of_test, double* split_values,
							   int* best_feature, test_d* tests,
									 int tests_d_size, int features_size, int depth, bool* used_features)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0)
	{
		used_features[best_feature[0]] = true;
	}
	if (x < pow((double)2, (double)depth))
	{
		
		int node_id = pow((double)2, (double)depth) - 1 + x;
		if (!nodes[node_id].is_exists)
		{
			return;
		}
		double split_value = split_values[best_feature[0] * (int)pow((double)2, (double)depth) + x];
		int sz = nodes[node_id].size;
		int le = nodes[node_id].is_leaf? 1 : 0;
		printf("id %d split %f size %d leaf %d \n", node_id, split_value, sz, le);
		if (!nodes[node_id].is_exists || nodes[node_id].is_leaf)
		{
			return;
		}
		nodes[node_id].split_value = split_value;
		double l_sum = 0;
		double r_sum = 0;
		double l_size = 0;
		double r_size = 0;
		double l_avg = 0;
		double r_avg = 0;
		double l_err = 0;
		double r_err = 0;
		for (int i = 0; i < tests_d_size; i++)
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
		for (int i = 0; i < tests_d_size; i++)
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
	cudaError_t cuda_err = cudaGetLastError();
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1 + tests_d_size / (1 + BLOCK_SIZE), 1);
	thrust::device_vector<int> node_id_of_test(tests_d_size);
	fill_node_id_of_test<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]), feature_id_at_depth,
		tests_d, tests_d_size, depth);
	cudaDeviceSynchronize();
	cuda_err = cudaGetLastError();
	thrust::device_vector<double> pre_errors(tests_d_size * features_size, 0);
	thrust::device_vector<double> pre_split_values(tests_d_size * features_size, 0);
	block.y = BLOCK_SIZE;
	grid.y = 1 + features_size / (1 + BLOCK_SIZE);
	calc_split_gpu<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&pre_errors[0]), thrust::raw_pointer_cast(&pre_split_values[0]),
		tests_d, tests_d_size, used_features, features_size, depth);
	cudaDeviceSynchronize();
	cuda_err = cudaGetLastError();

	thrust::device_vector<double> errors(pow(2, depth) * features_size, 0);
	thrust::device_vector<double> split_values(pow(2, depth) * features_size, 0);
	grid.x = 1 + pow(2, depth) / (1 + BLOCK_SIZE);
	calc_min_error<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&pre_errors[0]), thrust::raw_pointer_cast(&pre_split_values[0]),
		thrust::raw_pointer_cast(&errors[0]), thrust::raw_pointer_cast(&split_values[0]),
		tests_d_size, features_size, depth);
	cudaDeviceSynchronize();
	cuda_err = cudaGetLastError();

	block.x = 1;
	block.y = 1;
	grid.x = 1;
	grid.y = 1;

	int nodes_size = pow(2, depth);
	for (int i = 0; i < features_size; i++)
	{
		errors[i * nodes_size] = thrust::reduce(thrust::device, errors.begin() + i * nodes_size,
			errors.begin() + (i + 1) * nodes_size, 0.0, thrust::plus<double>());
	}
	
	thrust::device_vector<int> best_feature(1, 0);
	thrust::device_vector<double> best_error(1, 0);
	calc_best_feature<<<grid, block>>>(thrust::raw_pointer_cast(&errors[0]), used_features,
		thrust::raw_pointer_cast(&best_feature[0]), 
		thrust::raw_pointer_cast(&best_error[0]),
		features_size,  depth);
	cudaDeviceSynchronize();
	cuda_err = cudaGetLastError();
	block.x = BLOCK_SIZE;
	grid.x = 1 + pow(2, depth) / (1 + BLOCK_SIZE);
	make_split_gpu<<<grid, block>>>(nodes, thrust::raw_pointer_cast(&node_id_of_test[0]),
		thrust::raw_pointer_cast(&split_values[0]), thrust::raw_pointer_cast(&best_feature[0]),
		tests_d, tests_d_size, features_size,  depth, used_features);
	cudaDeviceSynchronize();
	cuda_err = cudaGetLastError();
	return std::make_pair(best_feature[0], best_error[0]);
}