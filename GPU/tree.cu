#include <set>
#include <algorithm>
#include <iostream>
#include "tree.cuh"
#include "thrust\sort.h"
#include "device_launch_parameters.h"

#define INF 1e10
#define BLOCK_SIZE 32

double calc_mse(data_set::iterator data_begin, data_set::iterator data_end, double avg, double n)
{
	double ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (data_set::iterator cur_test = data_begin; cur_test != data_end; cur_test++)
	{
		ans += ((cur_test->anwser - avg) * (cur_test->anwser - avg));
	}
	ans /= n; 
	return ans;
}

void* node::operator new(size_t len)
{
	void* ptr;
	cudaMallocManaged(&ptr, len);
	return ptr;
}
 
void node::operator delete(void *ptr)
{
    cudaFree(ptr);
}

node::node(const node& other) : data_begin(other.data_begin), data_end(other.data_end), depth(other.depth), is_leaf(other.is_leaf),
	node_mse(other.node_mse), output_value(other.output_value), size(other.size), split_value(other.split_value),
	subtree_mse(other.subtree_mse), sum(other.sum)
{
	if (!is_leaf)
	{
		left = new node(*other.left);
		right = new node(*other.right);
	}
	else
	{
		left = right = NULL;
	}
}

node::node(int depth) : depth(depth)
{
	left = right = NULL;
	is_leaf = false;
}

void node::calc_avg()
{
	sum = 0;
	size = 0;
	for (data_set::iterator cur_test = data_begin; cur_test != data_end; cur_test++)
	{
		size++;
		sum += cur_test->anwser;
	}
	output_value = sum / size;
}

double node::split(int split_feature_id)
{
	if (is_leaf)
	{
		return calc_mse(data_begin, data_end, output_value, size);
	}
	std::sort(data_begin, data_end, [&split_feature_id](test t1, test t2)
	{
		return t1.features[split_feature_id] < t2.features[split_feature_id];
	});
	double l_sum = 0;
	double l_size = 0;
	double r_sum = sum;
	double r_size = size;
	double best_mse = INF; 
	for (data_set::iterator cur_test = data_begin + 1; cur_test != data_end; cur_test++) //try all possible splits
	{
		l_sum += (cur_test - 1)->anwser;
		l_size++;
		r_sum -= (cur_test - 1)->anwser;
		r_size--;
		if (cur_test->features[split_feature_id] == (cur_test - 1)->features[split_feature_id])
		{
			continue;
		}
		double l_avg = l_sum / l_size;
		double r_avg = r_sum / r_size;
		double l_mse = calc_mse(data_begin, cur_test, l_avg, l_size);
		double r_mse = calc_mse(cur_test, data_end, r_avg, r_size);
		double cur_mse = l_mse + r_mse;
		if (cur_mse < best_mse)
		{
			best_mse = cur_mse;
			split_value = cur_test->features[split_feature_id];
			left->data_begin = data_begin;
			left->data_end = cur_test;
			left->data_begin_id = data_begin_id;
			left->data_end_id = data_begin_id + (cur_test - data_begin);
			left->output_value = l_avg;
			left->size = l_size;
			left->sum = l_sum;
			left->node_mse = l_mse;
			left->is_leaf = (l_size == 1) ? true : false;
			right->data_begin = cur_test;
			right->data_end = data_end;
			right->data_begin_id = data_begin_id + (cur_test - data_begin);
			right->data_end_id = data_end_id;
			right->output_value = r_avg;
			right->size = r_size;
			right->sum = r_sum;
			right->node_mse = r_mse;
			right->is_leaf = (r_size == 1) ? true : false;
		}
	}
	if (best_mse == INF)
	{
		best_mse = node_mse;
		split_value = (data_begin + 1)->features[split_feature_id];
		left->output_value = output_value;
		left->size = 0;
		left->is_leaf = true;
		right->data_begin = data_begin;
		right->data_end = data_end;
		right->data_begin_id = data_begin_id;
		right->data_end_id = data_end_id;
		right->output_value = output_value;
		right->size = size;
		right->sum = sum;
		right->node_mse = node_mse;
		right->is_leaf = (size <= 1) ? true : false;
	}
	return best_mse;
}

double node::split(int split_feature_id, double best_split_value)
{
	if (is_leaf)
	{
		return calc_mse(data_begin, data_end, output_value, size);
	}
	if (best_split_value == INF)
	{
		split_value = (data_begin + 1)->features[split_feature_id];
		left->output_value = output_value;
		left->size = 0;
		left->is_leaf = true;
		right->data_begin = data_begin;
		right->data_end = data_end;
		right->data_begin_id = data_begin_id;
		right->data_end_id = data_end_id;
		right->output_value = output_value;
		right->size = size;
		right->sum = sum;
		right->node_mse = node_mse;
		right->is_leaf = (size <= 1) ? true : false;
		return node_mse;
	}
	std::sort(data_begin, data_end, [&split_feature_id](test t1, test t2)
	{
		return t1.features[split_feature_id] < t2.features[split_feature_id];
	});
	double l_sum = 0;
	double l_size = 0;
	double r_sum = 0;
	double r_size = 0;
	data_set::iterator split = std::lower_bound(data_begin, data_end, best_split_value, [&split_feature_id](test t, double best_split_value)
	{
		return t.features[split_feature_id] < best_split_value;
	});
	std::for_each(data_begin, split, [&l_sum, &l_size](test cur_test)
	{
		l_sum += cur_test.anwser;
		l_size++;
	});
	r_sum = sum - l_sum;
	r_size = size - l_size;
	double l_avg = l_sum / l_size;
	double r_avg = r_sum / r_size;
	double l_mse = calc_mse(data_begin, split, l_avg, l_size);
	double r_mse = calc_mse(split, data_end, r_avg, r_size);
	double cur_mse = l_mse + r_mse;
	split_value = split->features[split_feature_id];
	left->data_begin = data_begin;
	left->data_end = split;
	left->data_begin_id = data_begin_id;
	left->data_end_id = data_begin_id + (split - data_begin);
	left->output_value = l_avg;
	left->size = l_size;
	left->sum = l_sum;
	left->node_mse = l_mse;
	left->is_leaf = (l_size == 1) ? true : false;
	right->data_begin = split;
	right->data_end = data_end;
	right->data_begin_id = data_begin_id + (split - data_begin);
	right->data_end_id = data_end_id;
	right->output_value = r_avg;
	right->size = r_size;
	right->sum = r_sum;
	right->node_mse = r_mse;
	right->is_leaf = (r_size == 1) ? true : false;
	return cur_mse;
}

tree::tree(const tree& other) : feature_id_at_depth(other.feature_id_at_depth), leafs(other.leafs), max_leafs(other.max_leafs)
{
	root = new node(*other.root);
	layers.resize(feature_id_at_depth.size() + 1);
	if (!layers.empty())
	{
		fill_layers(root);
	}
}

tree::tree(data_set& train_set, int max_leafs) : max_leafs(max_leafs)
{
	std::set<int> features;
	for (size_t i = 0; i < train_set[0].features.size(); i++)
	{
		features.insert(i);
	}
	for (data_set::iterator cur_test = train_set.begin(); cur_test != train_set.end(); cur_test++)
	{
		tests_d.push_back(test_d(*cur_test));
	}
	leafs = 1;
	int depth = 0;
	root = new node(0);
	root->data_begin = train_set.begin();
	root->data_end = train_set.end();
	root->data_begin_id = 0;
	root->data_end_id = tests_d.size();
	root->calc_avg();
	root->node_mse = calc_mse(root->data_begin, root->data_end, root->output_value, root->size);
	std::vector<node*> layer;
	layer.push_back(root);
	layers.push_back(layer);
	while (leafs < max_leafs && !features.empty())
	{
		double min_error = INF;
		int best_feature = -1;
		make_layer(depth);
		for (std::set<int>::iterator cur_split_feature = features.begin(); cur_split_feature != features.end(); cur_split_feature++)
		//choose best split feature at current depth
		{
			double cur_error = split_layer(depth, *cur_split_feature);
			/*double cur_error = 0;
			for (size_t i = 0; i < layers[depth].size(); i++)
			{
				cur_error += layers[depth][i]->split(*cur_split_feature);
			}*/
			if (cur_error < min_error)
			{
				min_error = cur_error;
				best_feature = *cur_split_feature;
			}
		}
		split_layer(depth, best_feature);
		/*for (size_t i = 0; i < layers[depth].size(); i++)
		{
			layers[depth][i]->split(best_feature);
		}*/
		feature_id_at_depth.push_back(best_feature);
		features.erase(best_feature);
		depth++;
		std::cout << "level " << depth << " created. training error: " << min_error << std::endl;
	}
	for (size_t i = 0; i < layers.back().size(); i++)
	{
		layers.back()[i]->is_leaf = true;
	}
	std::cout << "leafs before pruning: " << leafs << std::endl;
	prune(root);
	std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
	while (layers.back().empty())
	{
		layers.pop_back();
	}
}

tree::~tree()
{
	delete_node(root);
}

double tree::calculate_anwser(test& _test)
{
	node* cur = root;
	while (!cur->is_leaf)
	{
		cur = _test.features[feature_id_at_depth[cur->depth]] < cur->split_value ? cur->left : cur->right;
	}
	return cur->output_value;
}

double tree::calculate_error(data_set& test_set)
{
	double error = 0;
	for (data_set::iterator cur_test = test_set.begin(); cur_test != test_set.end(); cur_test++)
	{
		double ans = calculate_anwser(*cur_test);
		error += ((ans - cur_test->anwser) * (ans - cur_test->anwser));
	}
	error /= (1.0 * test_set.size());
	return error;
}

void tree::print()
{
	std::cout << "************TREE (layers structure)**************" << std::endl;
	for (size_t i = 0; i < layers.size(); i++)
	{
		std::cout << "layer " << i << "; layer size: " << layers[i].size() << std::endl; 
	}
	std::cout << "************TREE (DFS pre-order)**************" << std::endl;
	print(root);
	std::cout << "**********************************************" << std::endl;
}

void tree::delete_node(node* n)
{
	if (n == NULL)
	{
		return;
	}
	if (!layers.empty())
	{
		layers[n->depth].erase(std::find(layers[n->depth].begin(), layers[n->depth].end(), n));
	}
	delete_node(n->left);
	delete_node(n->right);
	if(n->is_leaf)
	{
		leafs--;
	}
	delete n;
}

void tree::make_layer(int depth)
{
	std::vector<node*> new_level;
	for (size_t i = 0; i < layers[depth].size(); i++) //make children for non-leaf nodes at current depth
	{
		if (!layers[depth][i]->is_leaf)
		{
			node* l = new node(layers[depth][i]->depth + 1);
			node* r = new node(layers[depth][i]->depth + 1);
			layers[depth][i]->left = l;
			layers[depth][i]->right = r;
			new_level.push_back(l);
			new_level.push_back(r);
			leafs++;
		}
	}
	if (!new_level.empty())
	{
		layers.push_back(new_level);
	}
}

void tree::prune(node* n)
{
	if (!n->is_leaf)
	{
		calc_subtree_mse(n->left);
		calc_subtree_mse(n->right);
		if (n->node_mse <= n->left->subtree_mse + n->right->subtree_mse)
		{
			n->is_leaf = true;
			leafs++;
			delete_node(n->left);
			delete_node(n->right);
			n->left = NULL;
			n->right = NULL;
		}
		else
		{
			prune(n->left);
			prune(n->right);
		}
	}
}

void tree::calc_subtree_mse(node* n)
{
	double error = 0;
	for (data_set::iterator cur_test = n->data_begin; cur_test != n->data_end; cur_test++)
	{
		node* cur_node = n;
		while (!cur_node->is_leaf)
		{
			cur_node = cur_test->features[feature_id_at_depth[cur_node->depth]] < cur_node->split_value ? cur_node->left : cur_node->right;
		}
		double ans = cur_node->output_value;
		error += ((ans - cur_test->anwser) * (ans - cur_test->anwser));
	}
	error /= (1.0 * n->size);
	n->subtree_mse = error;
}

void tree::print(node* n)
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
}

struct test_d_comparator {
	test_d_comparator(int split_feature_id) : split_feature_id(split_feature_id) {}
	__host__ __device__	bool operator()(test_d t1, test_d t2)
	{
		return t1.features[split_feature_id] < t2.features[split_feature_id];
	}
	int split_feature_id;
};

__device__ double calc_mse_d(test_d* tests, int begin_id, int end_id, double avg, double n)
{
	double ans = 0;
	if (n == 0)
	{
		return ans;
	}
	for (size_t i = begin_id; i < end_id; i++)
	{
		ans += ((tests[i].anwser - avg) * (tests[i].anwser - avg));
	}
	ans /= n; 
	return ans;
}

__global__ void split_layer_gpu(node** layer, test_d* tests, double* split_values, int split_feature_id)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (layer[i]->is_leaf)
	{
		return;
	}
	double l_sum = 0;
	double l_size = 0;
	double r_sum = layer[i]->sum;
	double r_size = layer[i]->size;
	double best_mse = INF; 
	double split_value = 0;
	for (size_t j = layer[i]->data_begin_id + 1; j < layer[i]->data_end_id; j++) //try all possible splits
	{
		l_sum += tests[j - 1].anwser;
		l_size++;
		r_sum -= tests[j - 1].anwser;
		r_size--;
		if (tests[j].features[split_feature_id] == tests[j - 1].features[split_feature_id])
		{
			continue;
		}
		double l_avg = l_sum / l_size;
		double r_avg = r_sum / r_size;
		double l_mse = calc_mse_d(tests, layer[i]->data_begin_id, j, l_avg, l_size);
		double r_mse = calc_mse_d(tests, j, layer[i]->data_end_id, r_avg, r_size);
		double cur_mse = l_mse + r_mse;
		if (cur_mse < best_mse)
		{
			best_mse = cur_mse;
			split_value = tests[j].features[split_feature_id];
		}
	}
	if (best_mse != INF)
	{
		split_values[i] = split_value;
	}
}

double tree::split_layer(int depth, int split_feature_id)
{
	for (size_t i = 0; i < layers[depth].size(); i++)
	{
		thrust::sort(tests_d.begin() + layers[depth][i]->data_begin_id, tests_d.begin() + layers[depth][i]->data_end_id,
			test_d_comparator(split_feature_id));
	}
	thrust::device_vector<double> split_values(layers[depth].size(), INF);
	thrust::device_vector<node*> layer(layers[depth]);
	dim3 block(layers[depth].size(), 1); //*****************************************BAD!**********************
	dim3 grid(1, 1);
	split_layer_gpu<<<grid, block>>>(thrust::raw_pointer_cast(&layer[0]), thrust::raw_pointer_cast(&tests_d[0]), 
		thrust::raw_pointer_cast(&split_values[0]), split_feature_id);
	cudaDeviceSynchronize();
	double error = 0;
	double* split_values_h = (double*)malloc(layers[depth].size() * sizeof(double));
	cudaMemcpy(split_values_h, thrust::raw_pointer_cast(&split_values[0]), layers[depth].size() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < layers[depth].size(); i++)
	{
		error += layers[depth][i]->split(split_feature_id, split_values_h[i]);
	}
	free(split_values_h);
	return error;
}