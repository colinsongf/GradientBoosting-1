#ifndef TREE_H
#define TREE_H

#include "thrust\device_vector.h"
#include "data_set.h"
#include "test.h"

struct node
{
	__host__ __device__ node(const node& other);
	__host__ __device__ node(int depth = 0);
	int depth;
	float split_value;
	float output_value;
	float sum;
	float size;
	float node_mse;
	float subtree_mse;
	bool is_leaf;
	bool is_exists;
};

class tree
{
public:
	//tree(const tree& other);
	tree(data_set& train_set, int max_leafs);
	~tree();
	float calculate_answer(test& _test);
	//float calculate_error(data_set& test_set);
	void print();
private:
	void make_layer(int depth);
	void prune(int node_id);
	//void print(node* n);
	//void fill_layers(node* n);
	std::pair<int, float> fill_layer();
	float* features;
	float* answers;
	int* feature_id_at_depth;
	node* nodes; 
	bool* used_features;
	int max_leafs;
	int leafs;
	int depth;
	int tests_size;
	int features_size;
};
#endif // TREE_H