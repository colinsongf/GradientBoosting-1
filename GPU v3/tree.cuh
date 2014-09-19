#ifndef TREE_H
#define TREE_H

#include "thrust\device_vector.h"
#include "data_set.h"
#include "test.h"
#include "test_d.cuh"

struct node
{
	 //__device__ node(const node& other);
	node(int depth = 0);
	int depth;
	double split_value;
	double output_value;
	double sum;
	double size;
	double node_mse;
	double subtree_mse;
	bool is_leaf;
	bool is_exists;
};

class tree
{
public:
	//tree(const tree& other);
	tree(data_set& train_set, int max_leafs);
	~tree();
	double calculate_anwser(test& _test);
	//double calculate_error(data_set& test_set);
	void print();
private:
	void make_layer(int depth);
	void prune(int node_id);
	//void print(node* n);
	//void fill_layers(node* n);
	std::pair<int, double> fill_layer();
	int* feature_id_at_depth;
	node* nodes; 
	test_d* tests_d;
	bool* used_features;
	int max_leafs;
	int leafs;
	int depth;
	int tests_size;
	int features_size;
};
#endif // TREE_H