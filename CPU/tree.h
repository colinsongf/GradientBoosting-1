#ifndef TREE_H
#define TREE_H

#include <vector>

struct node
{
	node(int depth);
	void calc_avg();
	node* left;
	node* right;
	int depth;
	double split_value;
	double output_value;
	double sum;
	double size;
	bool is_terminal;
	std::vector<std::pair<double, std::vector<double> > >::iterator b;
	std::vector<std::pair<double, std::vector<double> > >::iterator e;
};

class tree
{
public:
	tree(std::vector<std::pair<double, std::vector<double> > >& train_set, int max_terminal_nodes);
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