#include <set>
#include <algorithm>
#include <iostream>
#include "tree.h"

#define INF 1e10

float calc_mse(data_set::iterator data_begin, data_set::iterator data_end, float avg, float n)
{
	float ans = 0;
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

struct test_comparator {
	int split_feature_id;
	test_comparator(int split_feature_id) : split_feature_id(split_feature_id) {}
	bool operator()(test t1, test t2)
	{
		return t1.features[split_feature_id] < t2.features[split_feature_id];
	}
};

float node::split(int split_feature_id)
{
	if (is_leaf)
	{
		return calc_mse(data_begin, data_end, output_value, size);
	}
	std::sort(data_begin, data_end, test_comparator(split_feature_id));
	float l_sum = 0;
	float l_size = 0;
	float r_sum = sum;
	float r_size = size;
	float best_mse = INF; 
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
		float l_avg = l_sum / l_size;
		float r_avg = r_sum / r_size;
		float l_mse = calc_mse(data_begin, cur_test, l_avg, l_size);
		float r_mse = calc_mse(cur_test, data_end, r_avg, r_size);
		float cur_mse = l_mse + r_mse;
		if (cur_mse < best_mse)
		{
			best_mse = cur_mse;
			split_value = cur_test->features[split_feature_id];
			left->data_begin = data_begin;
			left->data_end = cur_test;
			left->output_value = l_avg;
			left->size = l_size;
			left->sum = l_sum;
			left->node_mse = l_mse;
			left->is_leaf = (l_size == 1) ? true : false;
			right->data_begin = cur_test;
			right->data_end = data_end;
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
		right->output_value = output_value;
		right->size = size;
		right->sum = sum;
		right->node_mse = node_mse;
		right->is_leaf = (size <= 1) ? true : false;
	}
	return best_mse;
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

tree::tree(data_set& train_set, int max_leafs, int max_depth) : max_leafs(max_leafs)
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
	root->node_mse = calc_mse(root->data_begin, root->data_end, root->output_value, root->size);
	std::vector<node*> layer;
	layer.push_back(root);
	layers.push_back(layer);
	while (leafs < max_leafs && depth < max_depth && !features.empty())
	{
		float min_error = INF;
		int best_feature = -1;
		make_layer(depth);
		for (std::set<int>::iterator cur_split_feature = features.begin(); cur_split_feature != features.end(); cur_split_feature++)
		//choose best split feature at current depth
		{
			float cur_error = 0;
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
		//std::cout << "level " << depth << " created. training error: " << min_error << " best feat: " << best_feature << " split_val: "
			//<< root->split_value << std::endl;
	}
	for (size_t i = 0; i < layers.back().size(); i++)
	{
		layers.back()[i]->is_leaf = true;
	}
	//std::cout << "leafs before pruning: " << leafs << std::endl;
	//prune(root);
	//std::cout << "new tree! leafs after pruning: " << leafs << std::endl;
	while (layers.back().empty())
	{
		layers.pop_back();
	}
}

tree::~tree()
{
	delete_node(root);
}

float tree::calculate_anwser(test& _test)
{
	node* cur = root;
	while (!cur->is_leaf)
	{
		cur = _test.features[feature_id_at_depth[cur->depth]] < cur->split_value ? cur->left : cur->right;
	}
	return cur->output_value;
}

float tree::calculate_error(data_set& test_set)
{
	float error = 0;
	for (data_set::iterator cur_test = test_set.begin(); cur_test != test_set.end(); cur_test++)
	{
		float ans = calculate_anwser(*cur_test);
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
	float error = 0;
	for (data_set::iterator cur_test = n->data_begin; cur_test != n->data_end; cur_test++)
	{
		node* cur_node = n;
		while (!cur_node->is_leaf)
		{
			cur_node = cur_test->features[feature_id_at_depth[cur_node->depth]] < cur_node->split_value ? cur_node->left : cur_node->right;
		}
		float ans = cur_node->output_value;
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