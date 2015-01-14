#ifndef TREE_H
#define TREE_H

#include <vector>
#include "data_set.h"
#include "test.h"

struct node
{
	node(const node& other);
	node(int depth);
	void calc_avg();
	float split(int split_feature_id);
	node* left;
	node* right;
	int depth;
	float split_value;
	float output_value;
	float sum;
	float size;
	float node_mse;
	float subtree_mse;
	bool is_leaf;
	data_set::iterator data_begin;
	data_set::iterator data_end;
};

class tree
{
public:
	tree(const tree& other);
	tree(data_set& train_set, int max_leafs, int max_depth);
	~tree();
	float calculate_anwser(test& _test);
	float calculate_error(data_set& test_set);
	void print();
private:
	void delete_node(node* n);
	void make_layer(int depth);
	void prune(node* n);
	void calc_subtree_mse(node* n);
	void print(node* n);
	void fill_layers(node* n);
	int max_leafs;
	int leafs;
	node* root;
	std::vector<int> feature_id_at_depth; 
	std::vector<std::vector<node*>> layers;
};
#endif // TREE_H