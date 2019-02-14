#!/usr/bin/python

import argparse
import re
import os

REGEX_COMMENT = re.compile(r'^[%|#].*')
REGEX_NODE_DATA = re.compile(r'(?P<num_nodes_1>\d+)\s+(?P<num_nodes_2>\d+)\s+(?P<num_edges>\d+)$')
REGEX_NODE_LINE = re.compile(r'(?P<node_id_1>\d+)\t(?P<node_id_2>\d+)')

def parse_args():
    parser = argparse.ArgumentParser(description='Fixes the percents')
    parser.add_argument('-i', '--input', required=True, help='The input file to read')
    args = parser.parse_args()

    # validate
    assert os.path.exists(args.input), "Input path (%s) does not exist" % args.input
    assert os.path.isfile(args.input), "Input path (%s) is not a file" % args.input
    assert args.input.endswith('.nodes.mtx')

    return args.input


def read_input_file(file_name):
    temp_file = file_name + '.tmp'

    num_beliefs = -1
    num_nodes = -1

    with open(temp_file, 'w') as out_file:
        with open(file_name, 'r') as in_file:
            for line in in_file:
                if REGEX_COMMENT.match(line):
                    out_file.write(line)
                    continue
                if num_nodes < 0 and REGEX_NODE_DATA.match(line):
                    match = REGEX_NODE_DATA.match(line)
                    num_nodes = int(match.group('num_nodes_1'))
                    out_file.write(line)
                else:
                    print("Looking at line: " + line)
                    parts = line.split()
                    new_parts = []
                    for idx, part in enumerate(parts):
                        if idx < 2:
                            new_parts.append(part)
                            continue
                        prob = float(part)
                        while prob > 1.0:
                            prob -= 1.0
                        while prob < 0.0:
                            prob += 1
                        new_parts.append(str(prob))
                    out_file.write('\t'.join(new_parts) + '\n')

def main():
    file_name = parse_args()
    read_input_file(file_name)

if __name__ == '__main__':
    main()