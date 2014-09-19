#ifndef TREE_H
#define TREE_H

#include <vector>
#include "data_set.h"
#include "test.h"
#include "test_d.cuh"
#include "thrust\device_vector.h"

struct node
{
	void* operator new(size_t len);
	void operator delete(void *ptr);
	node(const node& other);
	node(int depth);
	void calc_avg();
	double split(int split_feature_id);
	double split(int split_feature_id, double split_value);
	node* left;
	node* right;
	int depth;
	double split_value;
	double output_value;
	double sum;
	double size;
	double node_mse;
	double subtree_mse;
	bool is_leaf;
	data_set::iterator data_begin;
	data_set::iterator data_end;
	int data_begin_id;
	int data_end_id;
};

class tree
{
public:
	tree(const tree& other);
	tree(data_set& train_set, int max_leafs);
	~tree();
	double calculate_anwser(test& _test);
	double calculate_error(data_set& test_set);
	void print();
private:
	void delete_node(node* n);
	void make_layer(int depth);
	void prune(node* n);
	void calc_subtree_mse(node* n);
	void print(node* n);
	void fill_layers(node* n);
	double split_layer(int depth, int split_feature_id);
	int max_leafs;
	int leafs;
	node* root;
	std::vector<int> feature_id_at_depth; 
	std::vector<std::vector<node*>> layers;
	thrust::device_vector<test_d> tests_d;
};
#endif // TREE_H