#ifndef TEST_D_H
#define TEST_D_H

#include "test.h"
#include "device_launch_parameters.h"
#include "thrust\device_vector.h"

struct test_d
{
	test_d(test t);
	test_d();
	~test_d();
	double* features;
	double* anwser;
};
#endif // TEST_D_H