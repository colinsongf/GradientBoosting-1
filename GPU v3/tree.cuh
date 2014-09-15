#ifndef TREE_H
#define TREE_H

#include "thrust\device_vector.h"
#include "data_set.h"
#include "test.h"
#include "test_d.cuh"

struct node
{
	 //__device__ node(const node& other);
	node(int depth);
	__host__ __device__ node();
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
	//double split_layer(int depth, int split_feature_id);
	std::pair<int, double> fill_layer();
	int max_leafs;
	int leafs;
	int depth;
	int* feature_id_at_depth;
	node* nodes; 
	test_d* tests_d;
	int tests_d_size;
	int features_size;
	bool* used_features;
};
#endif // TREE_H