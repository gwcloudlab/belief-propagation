#!/usr/bin/python

import argparse
import random
import logging
from string import Template

log = logging.getLogger('generate_belief_network')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

VARIABLE_TEMPLATE = Template("""<VARIABLE TYPE="nature">
<NAME>Node$num</NAME>
<OUTCOME>Value1</OUTCOME>
<OUTCOME>Value2</OUTCOME>
<PROPERTY>position = (0,0)</PROPERTY>
</VARIABLE>""")

OBSERVED_NODE_TEMPLATE = Template("""<DEFINITION>
<FOR>Node$num</FOR>
<TABLE>
$prob_1 $prob_2
</TABLE>
</DEFINITION>""")

EDGE_TEMPLATE = Template("""<DEFINITION>
<FOR>Node$for_node</FOR>
<GIVEN>Node$given_node</GIVEN>
<TABLE>
$prob_1 $prob_2
$prob_3 $prob_4
</TABLE>
</DEFINITION>""")

HEADER_TEMPLATE = Template("""<?xml version="1.0"?>
<!-- DTD for the XMLBIF 0.3 format -->
<!DOCTYPE BIF [
    <!ELEMENT BIF ( NETWORK )*>
          <!ATTLIST BIF VERSION CDATA #REQUIRED>
    <!ELEMENT NETWORK ( NAME, ( PROPERTY | VARIABLE | DEFINITION )* )>
    <!ELEMENT NAME (#PCDATA)>
    <!ELEMENT VARIABLE ( NAME, ( OUTCOME |  PROPERTY )* ) >
          <!ATTLIST VARIABLE TYPE (nature|decision|utility) "nature">
    <!ELEMENT OUTCOME (#PCDATA)>
    <!ELEMENT DEFINITION ( FOR | GIVEN | TABLE | PROPERTY )* >
    <!ELEMENT FOR (#PCDATA)>
    <!ELEMENT GIVEN (#PCDATA)>
    <!ELEMENT TABLE (#PCDATA)>
    <!ELEMENT PROPERTY (#PCDATA)>
]>


<BIF VERSION="0.3">
<NETWORK>
<NAME>RandomNet</NAME>""")


FOOTER_TEMPLATE = Template("""</NETWORK>
</BIF>""")

num_arcs_written = 0

def parse_arguments():
    """
    Parses the command line arguments
    :return: An args object holding this information
    """
    parser = argparse.ArgumentParser(description="Generate a random bayesian network")
    parser.add_argument("--nodes", type=int, required=True, help="The number of nodes to generate")
    parser.add_argument("--arcs", type=int, required=True, help="The number of arcs to generate")
    parser.add_argument("--seed", type=int, default=1, help="The seed to use for generation")
    parser.add_argument("--file", type=str, required=True, help="The file to write to")
    parser.add_argument("--observed-probability", type=float, default=0.3, help="The probability that a given node is observed")
    args = parser.parse_args()
    assert args.arcs >= args.nodes - 1, "Arcs (%d) must be >= Nodes - 1 (%d)" % (args.arcs, args.nodes - 1)
    assert args.arcs <= args.nodes * (args.nodes - 1) / 2, "Arcs (%d) must be <= Nodes * (Nodes - 1) / 2 (%d)" % (args.arcs, args.nodes * (args.nodes - 1) / 2)
    assert args.nodes > 0, "The number of nodes (%d) must be greater than 0." % args.nodes
    return args


def write_header(out):
    out.write(HEADER_TEMPLATE.substitute())


def write_footer(out):
    out.write(FOOTER_TEMPLATE.substitute())


def write_variables(out, args):
    for i in range(0, args.nodes):
        out.write(VARIABLE_TEMPLATE.substitute(num=i))
        log.info("Num variables written: %d/%d" % (i + 1, args.nodes))


def write_observed_nodes(out, args):
    k = int(args.observed_probability * args.nodes)
    population = [i for i in range(0, args.nodes)]
    for i in random.sample(population, k):
        prob = random.random()
        out.write(OBSERVED_NODE_TEMPLATE.substitute(num=i, prob_1=prob, prob_2=1.0-prob))


def write_edge(out, src, dest):
    prob_1 = random.random()
    prob_2 = 1 - prob_1

    prob_3 = random.random()
    prob_4 = 1 - prob_3

    out.write(EDGE_TEMPLATE.substitute(given_node=src, for_node=dest, prob_1=prob_1, prob_2=prob_2, prob_3=prob_3, prob_4=prob_4))



def write_and_build_tree(out, args):
    parents = {}
    unconnected_nodes = [i for i in range(0, args.nodes)]

    # based off of bayes net generator from weka: http://grepcode.com/file/repository.pentaho.org/artifactory/pentaho/pentaho.weka/pdm-3.7-ce/3.7.7.2/weka/classifiers/bayes/net/BayesNetGenerator.java#BayesNetGenerator.generateRandomNetworkStructure%28int%2Cint%29
    node_1 = random.choice(unconnected_nodes)
    unconnected_nodes.remove(node_1)
    node_2 = random.choice(unconnected_nodes)
    unconnected_nodes.remove(node_2)

    parents[node_2] = [node_1]
    parents[node_1] = []

    if node_1 == node_2:
        node_2 = (node_1 + 1) % args.nodes
    if node_2 > node_2:
        temp = node_2
        node_1 = node_2
        node_2 = temp
    write_edge(out, node_1, node_2)

    for i in range(2, args.nodes):
        connected_node = random.choice(parents.keys())
        unconnected_node = random.choice(unconnected_nodes)
        unconnected_nodes.remove(unconnected_node)
        src_node = None
        dest_node = None
        if unconnected_node < connected_node:
            src_node = unconnected_node
            dest_node = connected_node
        else:
            src_node = connected_node
            dest_node = unconnected_node
        # assert src_node != dest_node
        if dest_node not in parents:
            parents[src_node] = [dest_node]
            parents[dest_node] = []
        else:
            parents[src_node] = [dest_node] + parents[dest_node]
        log.info("Num edges written: (%d/%d)" % (i, args.arcs))
    return parents


def write_edges(out, args):
    parents = write_and_build_tree(out, args)
    for i in range(args.nodes - 1, args.arcs):
        new_edge = False
        while not new_edge:
            node_1 = random.choice(parents.keys())
            node_2 = random.choice(parents.keys())
            if node_1 == node_2:
                node_2 = (node_1 + 1) % args.nodes
            src = None
            dest = None
            if node_2 < node_1:
                src = node_2
                dest = node_1
            else:
                src = node_1
                dest = node_2
            # assert src != dest
            if src not in parents[dest]:
                write_edge(out, src, dest)
                parents[dest].append(src)
                new_edge = True
                log.info("Num edges written: (%d/%d)" % (i, args.arcs))


def generate_file(args):
    random.seed(args.seed)

    with open(args.file, 'w') as f:
        # write header
        write_header(f)
        write_variables(f, args)
        write_observed_nodes(f, args)
        write_edges(f, args)
        write_footer(f)


def main():
    args = parse_arguments()
    generate_file(args)

if __name__ == '__main__':
    main()
