import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=argparse.FileType('r'))
    parser.add_argument('out_edge_file', type=argparse.FileType('w'))
    return parser.parse_args()


def get_edges_and_nodes(in_fp):
    edges = []
    nodes = set()

    for line in in_fp:
        nodes_in_line = line.strip().split()
        if len(nodes_in_line) > 1:
            src = nodes_in_line[0]
            nodes.add(src)
            for dest in nodes_in_line[1:]:
                nodes.add(dest)
                edges.append((src, dest))
    in_fp.close()

    return edges, nodes


def write_output(edges, nodes, out_fp):
    # write comments
    out_fp.write('# Directed graph (each unordered pair of nodes is saved once):\n')
    out_fp.write('# Paper citation network of Arxiv High Energy Physics category\n')
    out_fp.write('# Nodes: %d Edges: %d Beliefs: 1 Belief States: 1\n' % (len(nodes), 2*len(edges)))
    out_fp.write('# FromNodeId    ToNodeId\n')
    for src, dest in edges:
        out_fp.write('%s\t%s\t1.0\n' % (src, dest))
    out_fp.close()


def main():
    args = parse_args()
    edges, nodes = get_edges_and_nodes(args.in_file)
    write_output(edges, nodes, args.out_edge_file)

if __name__ == '__main__':
    main()
