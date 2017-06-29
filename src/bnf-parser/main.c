#include "bnf-wrapper.h"



int main(void)
{

	run_tests_with_file("../benchmark_files/dog.bif", 1);
	run_tests_with_file("../benchmark_files/medium/alarm.bif", 1);

	return 0;
}
