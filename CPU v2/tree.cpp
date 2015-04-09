#include <set>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string.h>
#include "tree.h"

#define INF 1e6
#define INF_INT 1000000
#define EPS 1e-6
#define BLOCK_SIZE 32

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

node::node(const node& other) : depth(other.depth), is_leaf(other.is_leaf), is_exists(other.is_exists),
	node_mse(other.node_mse), output_value(other.output_value), size(other.size), split_value(other.split_value),
	subtree_mse(other.subtree_mse), sum(other.sum), sum_of_squares(other.sum_of_squares) {}

node::node()
{
	depth = 0;
	is_leaf = false;
	is_exists = true;
	node_mse = 0;
	size = 0;
	sum = 0;
	sum_of_squares = 0;
}

my_tuple::my_tuple(int test_id, int split_id, float feature, float answer) : test_id(test_id), split_id(split_id),
	feature(feature), answer(answer) {}

bool operator<(const my_tuple& lhs, const my_tuple& rhs)
{
	return lhs.feature < rhs.feature;
}

my_pair::my_pair(int sorted_tests_id, float error) : sorted_tests_id(sorted_tests_id), error(error) {}

my_pair::my_pair()
{
	sorted_tests_id = -1;
	error = INF_INT;
}

bool operator<(const my_pair& lhs, const my_pair& rhs)
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



tree::tree(data_set& train_set, int max_leafs, int max_depth) : max_depth(max_depth)
{
	features_size = train_set.features_size;
	tests_size = train_set.tests_size;
	nodes = (node*)malloc((pow(2, max_depth + 1) - 1) * sizeof(node));
	feature_id_at_depth = (int*)malloc(features_size * sizeof(int));
	used_features = (bool*)malloc(features_size * sizeof(bool));
	features = (float*)malloc(features_size * tests_size * sizeof(float));
	memset(used_features, false, features_size * sizeof(bool));

	std::set<int> features_set;
	for (size_t i = 0; i < train_set.features_size; i++)
	{
		features_set.insert(i);
	}
	memcpy(features, &train_set.features[0], sizeof(float) * tests_size * features_size);
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
	sorted_tests = std::vector<my_tuple> (sorted);
	//auto start = std::chrono::high_resolution_clock::now();
	leafs = 1;
	depth = 0;
	clock_t time = clock();
	node root;
	for (int i = 0; i < tests_size; i++)
	{
		root.size++;
		root.sum += train_set.answers[i];
		root.sum_of_squares += pow(train_set.answers[i], 2);
	}
	root.output_value = root.sum / root.size;
	root.node_mse = root.sum_of_squares / root.size - pow(root.output_value, 2);
	memcpy(nodes, &root, sizeof(node));
	float new_error = root.node_mse;
	//float old_error = new_error + EPS;
	while (/*new_error < old_error &&*/ leafs < max_leafs && depth < max_depth && !features_set.empty())
	{
		make_layer(depth);
		std::pair<int, float> feature_and_error = fill_layer();
		features_set.erase(feature_and_error.first);
		depth++;
		//old_error = new_error;
		new_error = feature_and_error.second;
		std::cout << "level " << depth << " created. training error: " << new_error << " best_feat: " << feature_and_error.first << std::endl;
	}
	int layer_size = pow(2, depth);
	for (int i = 0; i < layer_size; i++)
	{
		nodes[i + layer_size - 1].is_leaf = true;
	}
	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = end - start;
	//std::cout << "leafs before pruning: " << leafs << std::endl;
	//std::cout << "new tree! calculating time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
	
	time = clock() - time;
	printf("calc time: %f\n\n", (float)time / CLOCKS_PER_SEC);

	h_feature_id_at_depth = (int*)malloc(features_size * sizeof(int));
	h_nodes = (node*)malloc((pow(2, max_depth + 1) - 1) * sizeof(node));
	memcpy(h_feature_id_at_depth, feature_id_at_depth, features_size * sizeof(int));
	memcpy(h_nodes, nodes, (pow(2, max_depth + 1) - 1) * sizeof(node));
	make_tree_ptr();
	sorted_tests.clear();
	free(nodes);
	free(feature_id_at_depth);
	free(used_features);
	free(features);
	//prune(0);                                         //*******************TODO!!
	//std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
}

tree::~tree()
{
	free(h_feature_id_at_depth);
	free(h_nodes);
	//delete_node(root);
}

float tree::calculate_answer(test& _test)
{
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

void tree::make_layer(int depth)
{
	int new_leafs = 0;
	int begin_id = pow(2, depth) - 1;
	int end_id = begin_id + pow(2, depth);
	for (int x = 0; begin_id + x < end_id; x++)
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
			new_leafs++;
		}
		else
		{
			nodes[2 * i + 1].is_exists = false;
			nodes[2 * i + 2].is_exists = false;
		}
	}
	leafs += new_leafs;
}

void fill_node_id_of_test(node* nodes, int* node_id_of_test, int* feature_id_at_depth,
									 float* features, int tests_size, int depth, int i)
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

void fill_split_ids(int* node_id_of_test, my_tuple* sorted_tests, int tests_size, int features_size, int layer_size,
							   int x, int y)
{
	my_tuple t = sorted_tests[y * tests_size + x];
	int id_shifted = node_id_of_test[t.test_id] - layer_size + 1;
	if (id_shifted >= 0)
	{
		sorted_tests[y * tests_size + x].split_id = 1 << id_shifted;
		//printf("test %d feat %d split_id %d\n", x, y, sorted_tests[y * tests_size + x].split_id);
	}
}    

void calc_split_gpu2(node* nodes, my_pair* errors, int tests_size, /*bool* used_features,*/
									 int features_size, int layer_size, my_tuple* sorted_tests, int x, int y)
{
	int node_id = layer_size - 1 + x;
	node cur_node = nodes[node_id];
	/*if (!cur_node.is_exists || cur_node.is_leaf)
	{
		return;
	}*/
	my_tuple cur_my_tuple;
	double l_sum = 0;
	double r_sum = cur_node.sum;
	double l_sum_pow = 0;
	double r_sum_pow = cur_node.sum_of_squares;
	int l_size = 0;
	int r_size = cur_node.size;
	//float l_avg = 0;
	//float r_avg = 0;
	float l_err = 0;
	float r_err = 0;
	float ans_pow = 0;
	for (int i = 0; i < tests_size; i++)
	{
		cur_my_tuple = sorted_tests[y * tests_size + i];
		/*if (y == 0)
		{
			printf("dd %f\n", cur_my_tuple.answer);
		}*/
		int exists = (cur_my_tuple.split_id >> x) & 1;
		ans_pow = pow(cur_my_tuple.answer, 2);
		l_sum += exists * cur_my_tuple.answer;
		l_sum_pow += exists * ans_pow;
		l_size += exists;
		r_sum -= exists * cur_my_tuple.answer;
		r_sum_pow -= exists * ans_pow;
		r_size -= exists;
		l_err = (l_size > 0) ? (l_sum_pow / l_size - pow(l_sum / l_size, 2)) : 0;
		r_err = (r_size > 0) ? (r_sum_pow / r_size - pow(r_sum / r_size, 2)) : 0;
		if (exists)
		{
			errors[y * tests_size + i] = my_pair(i, l_err + r_err);
		}
	}
}

void calc_min_error(node* nodes, my_pair* pre_errors,
							   float* errors, float* split_values, int tests_size, int features_size,
							   int layer_size, bool* used_features, my_tuple* sorted_tests, int* sorted_tests_ids, int x, int y)
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

void calc_min_error2(node* nodes, my_pair* pre_errors,
							   float* errors, float* split_values, int tests_size, int features_size,
							   int layer_size, bool* used_features, my_tuple* sorted_tests, int* sorted_tests_ids, int x, int y)
{
	int node_id = layer_size - 1 + x;
	node cur_node = nodes[node_id];
	if (!cur_node.is_exists || cur_node.is_leaf)
	{
		return;
	}
	float best_error = INF_INT;
	float best_split_val;
	int best_id;
	for (int i = 0; i < tests_size; i++)
	{
		my_pair p1 = pre_errors[y * tests_size + i];
		int id = p1.sorted_tests_id;
		my_tuple t1 = sorted_tests[y * tests_size + id];
		my_tuple t2;
		if ((p1.error < best_error) && ((t1.split_id >> x) & 1))
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
				best_error = p1.error;
				best_split_val = t1.feature + EPS;
				best_id = id;
				continue;
			}
			if (t1.feature != t2.feature)
			{
				best_error = p1.error;
				best_split_val = (t1.feature + t2.feature) / 2.0;
				best_id = id;
			}
		}
	}
	errors[y * layer_size + x] = best_error;
	split_values[y * layer_size + x] = best_split_val;
	sorted_tests_ids[y * layer_size + x] = best_id;
}

void my_reduce(float* errors, int layer_size, int y)
{
	float ans = 0;
	for (int i = 0; i < layer_size; i++)
	{
		ans += errors[y * layer_size + i];
	}
	errors[y * layer_size] = ans;
}

void calc_best_feature(float* errors, bool* used_features, int* best_features, float* best_errors,
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

void make_split_gpu(node* nodes, float* split_values,
							   int* best_feature, int tests_size, int layer_size, int* sorted_tests_ids, my_tuple* sorted_tests, int x)
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
	node cur_node = nodes[node_id];
	double l_sum = 0;
	double r_sum = cur_node.sum;
	double l_sum_pow = 0;
	double r_sum_pow = cur_node.sum_of_squares;
	int l_size = 0;
	int r_size = cur_node.size;
	float l_avg = 0;
	float r_avg = 0;
	float l_err = 0;
	float r_err = 0;
	int id = sorted_tests_ids[best_f * layer_size + x];
	my_tuple cur_my_tuple;
	for (int i = 0; i <= id; i++)
	{
		cur_my_tuple = sorted_tests[best_f * tests_size + i];
		int exists = (cur_my_tuple.split_id >> x) & 1;
		l_sum += exists * cur_my_tuple.answer;
		l_sum_pow += exists * pow(cur_my_tuple.answer, 2);
		l_size += exists;
	}
	r_sum -= l_sum;
	r_sum_pow -= l_sum_pow;
	r_size -= l_size;
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
	l_err = l_sum_pow / l_size - pow(l_avg, 2);
	r_err = r_sum_pow / r_size - pow(r_avg, 2);
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

std::pair<int, float> tree::fill_layer()
{
	int layer_size = pow(2, depth);
	std::vector<int> node_id_of_test(tests_size);

	for (int i = 0; i < tests_size; i++)
	{
		fill_node_id_of_test(nodes, &node_id_of_test[0], feature_id_at_depth, 
			features, tests_size, depth, i);
	}

	for (int x = 0; x < tests_size; x++)
	{
		for (int y = 0; y < features_size; y++)
		{
			fill_split_ids(&node_id_of_test[0], &sorted_tests[0], tests_size, features_size, layer_size, x, y);
		}
	}


	std::vector<my_pair> pre_errors(tests_size * features_size);

	time_t gg = clock();
	for (int x = 0; x < layer_size; x++)
	{
		for (int y = 0; y < features_size; y++)
		{
			calc_split_gpu2(nodes, &pre_errors[0], tests_size, /*used_features,*/ features_size, layer_size,
				&sorted_tests[0], x, y);
		}
	}
	gg = clock() - gg;
	printf("calc_split_c: %f\n", (float)gg / CLOCKS_PER_SEC);

	
	
	std::vector<float> errors(layer_size * features_size, INF_INT);
	std::vector<float> split_values(layer_size * features_size, INF_INT);
	std::vector<int> sorted_tests_ids(layer_size * features_size, 0);
	/*gg = clock();
	for (int i = 0; i < features_size; i++)
	{
		std::sort(pre_errors.begin() + i * tests_size, pre_errors.begin() + (i + 1) * tests_size);
	}

	gg = clock() - gg;
	printf("sort: %f\n", (float)gg / CLOCKS_PER_SEC);
	*/

	
	//if (y < features_size && x < layer_size && !used_features[y])
	gg = clock();
	for (int x = 0; x < layer_size; x++)
	{
		for (int y = 0; y < features_size; y++)
		{
			if (!used_features[y])
			{
				calc_min_error2(nodes, &pre_errors[0], &errors[0], &split_values[0],	tests_size, features_size, layer_size,
					used_features, &sorted_tests[0], &sorted_tests_ids[0], x, y);
			}
		}
	}

	gg = clock() - gg;
	printf("calc_min_err: %f\n", (float)gg / CLOCKS_PER_SEC);

	gg = clock();
	std::replace(errors.begin(), errors.end(), INF_INT, 0);
	for (int i = 0; i < features_size; i++)
	{
		//errors[i * layer_size] = std::accumulate(errors.begin() + i * layer_size,	errors.begin() + (i + 1) * layer_size, 0.0);
		my_reduce(&errors[0], layer_size, i);	
		//std::cout << i << " # " << errors[i * layer_size] << std::endl;
	}
	gg = clock() - gg;
	printf("reduce: %f\n", (float)gg / CLOCKS_PER_SEC);

	std::vector<int> best_feature(1);
	std::vector<float> best_error(1);
	calc_best_feature(&errors[0], used_features, &best_feature[0], &best_error[0], features_size, layer_size, feature_id_at_depth, depth);
	
	for (int x = 0; x < layer_size; x++)
	{
		make_split_gpu(nodes, &split_values[0], &best_feature[0], tests_size, layer_size, &sorted_tests_ids[0], &sorted_tests[0], x);
	}

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
