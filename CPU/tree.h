#ifndef TREE_H
#define TREE_H

#include <vector>
#include "data_set.h"
#include "test.h"

struct node
{
	node(int depth);
	void calc_avg();
	double split(int split_feature_id);
	node* left;
	node* right;
	int depth;
	double split_value;
	double output_value;
	double sum;
	double size;
	bool is_leaf;
	data_set::iterator data_begin;
	data_set::iterator data_end;
};

class tree
{
public:
	tree(data_set& train_set, int max_leafs);
	double calculate_anwser(test& _test);
	double calculate_error(data_set& test_set);
private:
	void make_layer(int depth);
	int max_leafs;
	int leafs;
	node* root;
	std::vector<int> feature_id_at_depth; 
	std::vector<std::vector<node*>> layers;
};
#endif // TREE_H