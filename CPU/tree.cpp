#include <set>
#include <algorithm>
#include <iostream>
#include "tree.h"

#define INF 1e10;

double calc_error(data_set::iterator data_begin, data_set::iterator data_end, double avg, double n)
{
	double ans = 0;
	for (data_set::iterator cur_test = data_begin; cur_test != data_end; cur_test++)
	{
		ans += ((cur_test->anwser - avg) * (cur_test->anwser - avg));
	}
	ans /= n; 
	return ans;
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
		return calc_error(data_begin, data_end, output_value, size);
	}
	std::sort(data_begin, data_end, [&split_feature_id](test t1, test t2)
	{
		return t1.features[split_feature_id] < t2.features[split_feature_id];
	});
	double l_sum = 0;
	double l_size = 0;
	double r_sum = sum;
	double r_size = size;
	double best_sum = INF; 
	for (data_set::iterator cur_test = data_begin + 1; cur_test != data_end; cur_test++)
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
		double cur_sum = calc_error(data_begin, cur_test, l_avg, l_size) + calc_error(cur_test, data_end, r_avg, r_size);
		if (cur_sum < best_sum)
		{
			best_sum = cur_sum;
			split_value = cur_test->features[split_feature_id];
			left->data_begin = data_begin;
			left->data_end = cur_test;
			left->output_value = l_avg;
			left->size = l_size;
			left->sum = l_sum;
			left->is_leaf = (l_size == 1) ? true : false;
			right->data_begin = cur_test;
			right->data_end = data_end;
			right->output_value = r_avg;
			right->size = r_size;
			right->sum = r_sum;
			right->is_leaf = (r_size == 1) ? true : false;
		}
	}
	return best_sum;
}

void tree::make_layer(int old_level)
{
	std::vector<node*> new_level;
	for (size_t i = 0; i < layers[old_level].size(); i++)
	{
		if (!layers[old_level][i]->is_leaf)
		{
			node* l = new node(layers[old_level][i]->depth + 1);
			node* r = new node(layers[old_level][i]->depth + 1);
			layers[old_level][i]->left = l;
			layers[old_level][i]->right = r;
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

tree::tree(data_set& train_set, int max_leafs)
	: max_leafs(max_leafs)
{
	std::set<int> features;
	for (size_t i = 0; i < train_set[0].features.size(); i++)
	{
		features.insert(i);
	}
	leafs = 1;
	int depth = 0;
	root = new node(0);
	root->data_begin = train_set.begin();
	root->data_end = train_set.end();
	root->calc_avg();
	std::vector<node*> layer;
	layer.push_back(root);
	layers.push_back(layer);
	while (leafs < max_leafs && !features.empty())
	{
		double min_error = INF;
		int best_feature = -1;
		make_layer(depth);
		for (std::set<int>::iterator cur_split_feature = features.begin(); cur_split_feature != features.end(); cur_split_feature++)
		{
			double cur_error = 0;
			for (size_t i = 0; i < layers[depth].size(); i++)
			{
				cur_error += layers[depth][i]->split(*cur_split_feature);
			}
			if (cur_error < min_error)
			{
				min_error = cur_error;
				best_feature = *cur_split_feature;
			}
		}
		for (size_t i = 0; i < layers[depth].size(); i++)
		{
			layers[depth][i]->split(best_feature);
		}
		feature_id_at_depth.push_back(best_feature);
		features.erase(best_feature);
		depth++;
		std::cout << "level " << depth << " created. training error: " << min_error << std::endl;
	}
	for (size_t i = 0; i < layers.back().size(); i++)
	{
		layers.back()[i]->is_leaf = true;
	}
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