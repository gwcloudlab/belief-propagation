#!/usr/bin/python

import re
import random
import argparse
import os
from string import Template

REGEX_NODE_DATA = re.compile('#\s+Nodes:\s+(?P<num_nodes>\d+)\s+Edges:\s+(?P<num_edges>\d+)')
REGEX_NODE_DATA_2 = re.compile('\s*(?P<num_nodes>\d+)\s+(?P<num_nodes_2>\d+)\s+(?P<num_edges>\d+)')
REGEX_EDGE = re.compile('(?P<src>\d+)\s+(?P<dest>\d+)')

DEFAULT_STATE = 1.0

TEMPLATE_DATA = Template('% Nodes: $num_nodes Edges: $num_edges Beliefs: $num_beliefs Belief States: $num_belief_states')

def read_snap_file(read_path, write_edges_path, write_observed_nodes_path, num_beliefs, num_belief_states, seed, pct_of_observed):
    random.seed(seed)

    num_nodes = None
    num_edges = None
    read_nodes = set()
    with open(read_path, 'r') as read_fp:
        with open(write_edges_path, 'w') as write_fp:
            for line in read_fp:
                if num_nodes is None or line.startswith('#') or line.startswith('%'):
                    match = REGEX_NODE_DATA.match(line)
                    if match is None and num_nodes is None:
                        match = REGEX_NODE_DATA_2.match(line)
                    if match is None:
                        write_fp.write(line)
                    else:
                        num_nodes = int(match.group('num_nodes'))
                        num_edges = int(match.group('num_edges'))
                        print "Num nodes: %d; Num edges: %d" % (num_nodes, num_edges)
                        replacement_line = TEMPLATE_DATA.substitute(num_nodes=num_nodes, num_edges=num_edges, num_beliefs=num_beliefs, num_belief_states=num_belief_states)
                        write_fp.write(replacement_line + '\n')
                        write_fp.write('{}\t{}\t{}'.format(num_nodes, num_nodes, num_edges))
                else:
                    edge_match = REGEX_EDGE.match(line)
                    if edge_match is None:
                        write_fp.write(line)
                    else:
                        assert num_nodes is not None, "Number of nodes is None still; invalid file"
                        assert num_edges is not None, "Number of edges is None still; invalid file"
                        src = edge_match.group('src')
                        dest = edge_match.group('dest')
                        read_nodes.add(src)
                        read_nodes.add(dest)
                        joint_probabilities = []
                        for i in range(num_belief_states):
                            joint_probabilities += generate_prob_list(num_belief_states)
                        new_line_data = [src, dest] + joint_probabilities
                        write_fp.write('\t'.join(new_line_data) + '\n')
    num_observed_nodes = pct_of_observed * num_nodes
    observed_nodes = set(random.sample(read_nodes, int(num_observed_nodes)))
    with open(write_observed_nodes_path, 'w') as write_node_fp:
        write_node_fp.write('% Belief network generated from mtx file: {}\n'.format(read_path))
        write_node_fp.write('{}\t{}\t{}\n'.format(num_nodes, num_nodes, num_edges))
        for node in read_nodes:
            if node in observed_nodes:
                prob_list = generate_prob_list(num_belief_states)
            else:
                prob_list = [str(DEFAULT_STATE) for _ in range(num_belief_states)]
            new_node_line_data = [str(node), str(node)] + prob_list
            write_node_fp.write('\t'.join(new_node_line_data) + '\n')


def generate_prob_list(num_belief_states):
    max_prob = 1.0
    sum = 0.0
    probabilities = []
    for i in range(num_belief_states - 1):
        new_prob = random.random() * max_prob
        probabilities.append(str(new_prob))
        max_prob *= new_prob
        sum += new_prob
    probabilities.append(str(1.0 - sum))
    return probabilities


def parse_args():
    parser = argparse.ArgumentParser(description="Generates a Bayesian network graph from a SNAP edge graph file")
    parser.add_argument('-i', '--input', required=True, help='The input file for the SNAP edge graph')
    parser.add_argument('-oe', '--output-edges', required=True, help='The output file to write the Bayesian network edge file')
    parser.add_argument('-on', '--output-nodes', required=True, help='The output file to write the Bayesian network observed node file')
    parser.add_argument('-s', '--seed', required=False, type=int, default=0, help='The initial seed value; defaults to 0')
    parser.add_argument('-nb', '--num-beliefs', required=False, type=int, default=2, help='The number of beliefs in the graph; defaults to 2')
    parser.add_argument('-ns', '--num-states', required=False, type=int, default=2, help='The number of states per belief; defaults to 2')
    parser.add_argument('-p', '--pct-observed', required=False, type=float, default=0.3, help='The percentage of nodes which are observed')

    args = parser.parse_args()

    # validate
    assert os.path.exists(args.input), "Input path (%s) does not exist" % args.input
    assert os.path.isfile(args.input), "Input path (%s) is not a file" % args.input

    output_edges_dirname = os.path.dirname(args.output_edges)
    if output_edges_dirname != '':
        assert os.path.exists(output_edges_dirname), "Output edge path (%s) does not exist" % args.output_edges

    output_nodes_dirname = os.path.dirname(args.output_nodes)
    if output_nodes_dirname != '':
        assert os.path.exists(output_nodes_dirname), "Output node path (%s) does not exist" % args.output_nodes

    assert args.num_beliefs > 0, "Number of beliefs (%d) must be greater than 0" % args.num_beliefs
    assert args.num_states > 0, "Number of belief states (%s) must be greater than 0" % args.num_states

    assert 0 <= args.pct_observed <= 1, "Percentage observed (%f) must be in the range of [0.0, 1.0]" % args.pct_observed

    return args


def main():
    args = parse_args()
    read_snap_file(args.input, args.output_edges, args.output_nodes, args.num_beliefs, args.num_states, args.seed, args.pct_observed)

if __name__ == '__main__':
    main()

