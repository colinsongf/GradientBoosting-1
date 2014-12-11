#ifndef TREE_H
#define TREE_H

#include "thrust\device_vector.h"
#include "data_set.h"
#include "test.h"

struct node
{
	__host__ __device__ node(const node& other);
	__host__ __device__ node();
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

struct my_tuple
{
	__host__ __device__ my_tuple(int test_id, int split_id, float feature, float answer);
	__host__ __device__ my_tuple() {};
	friend bool __host__ __device__ operator<(const my_tuple& lhs, const my_tuple& rhs);
	int test_id;
	int split_id;
	float feature;
	float answer;
};

struct my_pair
{
	__host__ __device__ my_pair(int sorted_tests_id, float error);
	__host__ __device__ my_pair();
	friend bool __host__ __device__ operator<(const my_pair& lhs, const my_pair& rhs);
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
	//float calculate_error(data_set& test_set);
	void print();
private:
	void make_layer(int depth);
	//void prune(int node_id);
	//void print(node* n);
	//void fill_layers(node* n);
	std::pair<int, float> fill_layer();
	thrust::device_vector<my_tuple> sorted_tests;
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
};
#endif // TREE_H