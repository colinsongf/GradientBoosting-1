#ifndef DATA_SET_H
#define DATA_SET_H

#include <string>
#include <vector>

class data_set
{
public:
	typedef std::vector<std::pair<double, std::vector<double> > >::iterator iterator;
	data_set(std::string file_name); //format: ans feature_1 feature_2 ... feature_n
	std::pair<double, std::vector<double> >& operator[](int index);
	iterator begin();
	iterator end();
	int size();
private:
	std::vector<std::pair<double, std::vector<double> > > data;
};

#endif // DATA_SET_H