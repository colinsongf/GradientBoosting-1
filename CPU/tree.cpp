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
	is_terminal = false;
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
	if (n->is_terminal)
	{
		return calc_error(n->b, n->e, n->output_value, n->size);
	}
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
		if (cur_sum < best_sum)
		{
			best_sum = cur_sum;
			n->split_value = it->second[feature];
			n->left->b = n->b;
			n->left->e = it;
			n->left->output_value = l_avg;
			n->left->size = l_size;
			n->left->sum = l_sum;
			n->left->is_terminal = (l_size == 1) ? true : false;
			n->right->b = it;
			n->right->e = n->e;
			n->right->output_value = r_avg;
			n->right->size = r_size;
			n->right->sum = r_sum;
			n->right->is_terminal = (r_size == 1) ? true : false;
		}
	}
	return best_sum;
}

void tree::make_level(int old_level)
{
	std::vector<node*> new_level;
	for (size_t i = 0; i < levels[old_level].size(); i++)
	{
		if (!levels[old_level][i]->is_terminal)
		{
			node* l = new node(levels[old_level][i]->depth + 1);
			node* r = new node(levels[old_level][i]->depth + 1);
			levels[old_level][i]->left = l;
			levels[old_level][i]->right = r;
			new_level.push_back(l);
			new_level.push_back(r);
			terminal_nodes++;
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
	for (size_t i = 0; i < train_set[0].second.size(); i++)
	{
		features.insert(i);
	}
	terminal_nodes = 1;
	int level = 0;
	root = new node(0);
	root->b = train_set.begin();
	root->e = train_set.end();
	root->calc_avg();
	std::vector<node*> layer;
	layer.push_back(root);
	levels.push_back(layer);
	double min_error = INF;
	while (terminal_nodes < max_terminal_nodes && !features.empty())
	{
		int best_feature = -1;
		make_level(level);
		for (std::set<int>::iterator it = features.begin(); it != features.end(); it++)
		{
			double cur_error = 0;
			for (size_t i = 0; i < levels[level].size(); i++)
			{
				cur_error += split_node(levels[level][i], *it);
			}
			if (cur_error < min_error)
			{
				min_error = cur_error;
				best_feature = *it;
			}
		}
		if (best_feature == -1)
		{
			for (size_t i = 0; i < levels[level].size(); i++)
			{
				levels[level][i]->is_terminal = true;
			}
			//TODO: GC
			break;
		}
		for (size_t i = 0; i < levels[level].size(); i++)
		{
			split_node(levels[level][i], best_feature);
		}
		features.erase(best_feature);
		level++;
		std::cout << "level " << level << " created. training error: " << min_error << std::endl;
	}
}