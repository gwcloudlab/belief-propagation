#!/usr/bin/python
import re
import shutil
import math

REGEX_LINE_INFO = re.compile(r'\d+\s+\d+\s+\d+')

FILE_NAME = '/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal_32_beliefs.nodes.mtx'
TMP_FILE = 'nodes.tmp'

with open(FILE_NAME, 'r') as file_in:
    with open(TMP_FILE, 'w') as out:
        read_line_info = False
        for line in file_in:
            if line.startswith('%'):
                out.write(line)
                # comment
                continue
            elif not read_line_info and REGEX_LINE_INFO.match(line):
                read_line_info = True
                out.write(line)
                continue
            else:
                splits = line.split()
                for i in range(2, len(splits)):
                    num = float(splits[i])
                    if num < 0.0:
                        num = math.fabs(num)
                    while num > 1.0:
                        num -= 1.0
                    splits[i] = str(num)
                out.write('\t'.join(splits))
                out.write('\n')

shutil.move(TMP_FILE, FILE_NAME)

