#!/usr/bin/python
import random

NUM_BELIEFS = 32

for j in range(NUM_BELIEFS):
    values = []
    sum = 0.0
    print("// row {}".format(j))
    for i in range(NUM_BELIEFS):
        num = random.random()
        sum += num
        values.append(num)

    for i in range(NUM_BELIEFS):
        values[i] = values[i] / sum
        print("edge_joint_probability->data[{}][{}] = {}f;".format(j, i, values[i]))

    print("")
