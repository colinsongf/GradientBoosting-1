#ifndef TREE_H
#define TREE_H

#include <vector>

#include "data_set.h"
#include "test.h"

struct node
{
	node(int level);
	void calc_avg();
	node* left;
	node* right;
	int level;
	double split_value;
	double output_value;
	double sum;
	double size;
	bool is_terminal;
	data_set::iterator data_begin;
	data_set::iterator data_end;
};

class tree
{
public:
	tree(data_set& train_set, int max_terminal_nodes);
	double calculate(test& _test);
	double calculate(data_set& test_set);
private:
	double split_node(node* n, int feature);
	void make_level(int level);
	int max_terminal_nodes;
	int terminal_nodes;
	node* root;
	std::vector<int> feature_id_at_level; 
	std::vector<std::vector<node*>> levels;
};
#endif // TREE_H