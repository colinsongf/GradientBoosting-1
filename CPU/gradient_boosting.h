#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

#include "data_set.h"

class gradient_boosting
{
public:
	gradient_boosting(data_set& train_set, int iterations, int max_leafs);
	double calculate_anwser(test& _test);
	double calculate_error(data_set& test_set);
private:

};
#endif // GRADIENT_BOOSTING_H