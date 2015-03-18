#ifndef TREE_H
#define TREE_H

#include <vector>
#include "data_set.h"
#include "test.h"

struct node
{
	node(const node& other);
	node();
	int depth;
	float split_value;
	float output_value;
	double sum;
	double sum_of_squares;
	float size;
	float node_mse;
	float subtree_mse;
	bool is_leaf;
	bool is_exists;
};

struct node_ptr
{
	node_ptr();
	node_ptr(const node_ptr& other);
	node_ptr* left;
	node_ptr* right;
	int depth;
	float split_value;
	float output_value;
	bool is_leaf;
};

struct my_tuple
{
	my_tuple(int test_id, int split_id, float feature, float answer);
	my_tuple() {};
	friend bool operator<(const my_tuple& lhs, const my_tuple& rhs);
	int test_id;
	int split_id;
	float feature;
	float answer;
};

struct my_pair
{
	my_pair(int sorted_tests_id, float error);
	my_pair();
	friend bool operator<(const my_pair& lhs, const my_pair& rhs);
	int sorted_tests_id;
	float error;
};

class tree
{
public:
	tree(const tree& other);
	tree(data_set& train_set, int max_leafs, int max_depth);
	~tree();
	float calculate_answer(test& _test);
	float calculate_error(data_set& test_set);
	void print();

private:
	void delete_node(node_ptr* n);
	void make_layer(int depth);
	node_ptr* root;
	void make_tree_ptr();
	void fill_node_ptr(node_ptr* n, int node_id);
	std::pair<int, float> fill_layer();
	std::vector<my_tuple> sorted_tests;
	float* features;
	int* feature_id_at_depth;
	int* h_feature_id_at_depth;
	node* nodes; 
	node* h_nodes;
	bool* used_features;
	int leafs;
	int depth;
	int tests_size;
	int features_size;
	int max_depth;
};
#endif // TREE_H