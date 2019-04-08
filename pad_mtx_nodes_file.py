#!/usr/bin/python

import argparse
import os
import re

REGEX_COMMENT = re.compile(r'^[%|#].*')
REGEX_NODE_DATA = re.compile(r'(?P<num_nodes_1>\d+)\s+(?P<num_nodes_2>\d+)\s+(?P<num_edges>\d+)$')
REGEX_NODE_LINE = re.compile(r'(?P<node_id_1>\d+)\t(?P<node_id_2>\d+)')

def parse_args():
    parser = argparse.ArgumentParser(description='Pads a nodes mtx file with additional node info')
    parser.add_argument('-i', '--input', required=True, help='The input file to read')
    args = parser.parse_args()

    # validate
    assert os.path.exists(args.input), "Input path (%s) does not exist" % args.input
    assert os.path.isfile(args.input), "Input path (%s) is not a file" % args.input
    assert args.input.endswith('.nodes.mtx')

    return args.input

def read_input_file(file_name):
    temp_file = file_name + '.tmp'

    seen_nodes = {}

    num_beliefs = -1
    num_nodes = -1

    with open(temp_file, 'w') as out_file:
        with open(file_name, 'r') as in_file:
            for line in in_file:
                if REGEX_COMMENT.match(line):
                    continue
                if num_nodes < 0 and REGEX_NODE_DATA.match(line):
                    match = REGEX_NODE_DATA.match(line)
                    num_nodes = int(match.group('num_nodes_1'))
                    for i in range(1, num_nodes):
                        seen_nodes[i] = False
                else:
                    if num_beliefs < 0:
                        num_beliefs = len(line.split('\t')) - 2
                    match = REGEX_NODE_LINE.match(line)
                    assert match is not None
                    node_id_1 = int(match.group('node_id_1'))
                    node_id_2 = int(match.group('node_id_2'))
                    assert node_id_1 == node_id_2
                    seen_nodes[node_id_1] = True
                out_file.write(line)
            for key, value in iter(sorted(seen_nodes.items())):
                if value:
                    continue
                print("Writing missing value: {}".format(key))
                default_beliefs = ['1.0' for i in range(num_beliefs)]
                out_file.write('{}\t{}\t{}\n'.format(key, key, '\t'.join(default_beliefs)))



def main():
    file_name = parse_args()
    read_input_file(file_name)

if __name__ == '__main__':
    main()

