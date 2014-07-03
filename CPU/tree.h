#ifndef TREE_H
#define TREE_H

#include <vector>

struct node
{
	node();
	node* left;
	node* right;
	int depth;
	double split_value;
	double output_value;
};

class tree
{
public:
	tree(std::vector<std::pair<double, std::vector<double> > >& train_set, int terminal_nodes);
private:
	int terminal_nodes;
	node* root;
	std::vector<int> feature_id_at_depth; 
	std::vector<std::vector<node*>> layers;
};
	
#endif // TREE_H