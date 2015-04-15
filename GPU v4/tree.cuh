#ifndef TREE_H
#define TREE_H

#include "thrust/device_vector.h"
#include "data_set.h"
#include "test.h"

struct node
{
	__host__ __device__ node(const node& other);
	__host__ __device__ node();
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
	//void prune(int node_id);
	//void print(node* n);
	//void fill_layers(node* n);
	std::pair<int, float> fill_layer();
	//thrust::device_vector<my_tuple> sorted_tests;
	float* features;
	//float* answers;
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

	thrust::device_vector<int> tuple_test_id;
	thrust::device_vector<int> tuple_split_id;
	thrust::device_vector<float> tuple_feature;
	thrust::device_vector<float> tuple_answer;
};
#endif // TREE_H
