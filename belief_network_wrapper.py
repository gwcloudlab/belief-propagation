#!/usr/bin/python

import subprocess

i = 100000
while i < 1000000:
    if i != 100000 and i != 500000:
        num_nodes = int(i)
        num_arcs = 2 * num_nodes
        file_name = 'src/benchmark_files/xml2/%d_%d.xml' % (num_nodes, num_arcs)

        subprocess.call(['python', 'generate_belief_network.py', '--nodes', str(num_nodes), '--arcs', str(num_arcs), '--file', file_name])

    i += 100000
