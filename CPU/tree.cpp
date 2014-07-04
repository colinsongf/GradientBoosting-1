#include <set>
#include <algorithm>
#include <iostream>


#include "tree.h"

#define INF 1e10;

double calc_error(std::vector<std::pair<double, std::vector<double> > >::iterator b,
				  std::vector<std::pair<double, std::vector<double> > >::iterator e, double avg, double n)
{
	double ans = 0;
	for (std::vector<std::pair<double, std::vector<double> > >::iterator it = b; it != e; it++)
	{
		ans += ((it->first - avg) * (it->first - avg));
	}
	ans /= n; 
	return ans;
}


node::node(int depth) : depth(depth)
{
	left = right = NULL;
}

void node::calc_avg()
{
	sum = 0;
	size = 0;
	for (std::vector<std::pair<double, std::vector<double> > >::iterator it = b; it != e; it++)
	{
		size++;
		sum += it->first;
	}
	output_value = sum / size;
}

double tree::split_node(node* n, int feature)
{
	std::sort(n->b, n->e, [&feature](std::pair<double, std::vector<double> > p1, std::pair<double, std::vector<double> > p2)
	{
		return p1.second[feature] < p2.second[feature];
	});
	double l_sum = 0;
	double l_size = 0;
	double r_sum = n->sum;
	double r_size = n->size;
	double best_sum = INF; 
	for (std::vector<std::pair<double, std::vector<double> > >::iterator it = n->b + 1; it != n->e; it++)
	{
		l_sum += (it - 1)->first;
		l_size++;
		r_sum -= (it - 1)->first;
		r_size--;
		if (it->second[feature] == (it - 1)->second[feature])
		{
			continue;
		}
		double l_avg = l_sum / l_size;
		double r_avg = r_sum / r_size;
		double cur_sum = calc_error(n->b, it, l_avg, l_size) + calc_error(it, n->e, r_avg, r_size);
		//TODO: ...

	}
	return 0;
}

void tree::make_level(int old_level)
{
	std::vector<node*> new_level;
	for (size_t i = 0; i < levels[old_level].size(); i++)
	{
		if (levels[old_level][i]->left == NULL && levels[old_level][i]->right == NULL) // TODO: remove that
		{
			node* l = new node(levels[old_level][i]->depth + 1);
			node* r = new node(levels[old_level][i]->depth + 1);
			levels[old_level][i]->left = l;
			levels[old_level][i]->right = r;
			new_level.push_back(l);
			new_level.push_back(r);
		}
	}
	if (!new_level.empty())
	{
		levels.push_back(new_level);
	}
}

tree::tree(std::vector<std::pair<double, std::vector<double> > >& train_set, int max_terminal_nodes)
	: max_terminal_nodes(max_terminal_nodes)
{
	std::set<int> features;
	std::set<int>::iterator it;
	for (size_t i = 0; i < train_set[0].second.size(); i++)
	{
		features.insert(i);
	}
	terminal_nodes = 0;
	int level = 0;
	root = new node(0);
	root->b = train_set.begin();
	root->e = train_set.end();
	root->calc_avg();
	std::vector<node*> layer;
	layer.push_back(root);
	levels.push_back(layer);
	while (terminal_nodes < max_terminal_nodes)
	{
		make_level(level);
		for (it = features.begin(); it != features.end(); it++)
		{
			double error = 0;
			for (size_t i = 0; i < levels[level].size(); i++)
			{
				error += split_node(levels[level][i], *it);
			}
		}
	}
}