import os
import numpy as np
import networkx as nx
from tqdm import tqdm


def generate_dataset(generators, size, name):
    assert type(generators) == list

    all_node_labels = []
    all_edges = []
    all_graph_indicators = []
    all_graph_labels = []

    counter = 0
    num_nodes = 0
    for i in tqdm(range(size)):
        generator = generators[counter]
        n, edges, node_labels = generator.generate_graph()

        all_node_labels.extend(node_labels)
        all_edges.extend((edges + num_nodes))
        all_graph_indicators.extend([i + 1] * n)
        all_graph_labels.extend([counter + 1])

        counter = (counter + 1) % len(generators)
        num_nodes += n

    dir = os.path.join("./generated_datasets", name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    all_edges = np.array(all_edges)
    all_node_labels = np.array(all_node_labels)
    all_graph_indicators = np.array(all_graph_indicators)
    all_graph_labels = np.array(all_graph_labels)

    np.savetxt(dir + "/%s_A.txt" % name, all_edges, delimiter=',', fmt='%i')
    np.savetxt(dir + "/%s_graph_indicator.txt" % name, all_graph_indicators, delimiter=',', fmt='%i')
    np.savetxt(dir + "/%s_graph_labels.txt" % name, all_graph_labels, delimiter=',', fmt='%i')
    np.savetxt(dir + "/%s_node_labels.txt" % name, all_node_labels, delimiter=',', fmt='%i')


class GraphGenerator:
    def __init__(self):
        pass

    def generate_graph(self):
        raise NotImplementedError


class ErdosRenyiGenerator(GraphGenerator):
    """
    These graphs consist of n nodes.
    Each edge between any pair of nodes has a probability p of being chosen independently of other edges
    No node labels are used for the graph.
    """

    def __init__(self, n, p):
        super(ErdosRenyiGenerator, self).__init__()
        self.n = n
        self.p = p

    def generate_graph(self):
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.random.uniform(0, 1) < self.p:
                    edges.append([i + 1, j + 1])

        nodes_labels = []
        for i in range(self.n):
            nodes_labels.append(0)

        edges = np.array(edges)
        nodes_labels = np.array(nodes_labels)

        return self.n, edges, nodes_labels


class BarabasiAlbert(GraphGenerator):
    def __init__(self, n, m):
        super(BarabasiAlbert, self).__init__()
        self.n = n
        self.m = m

    def generate_graph(self):
        graph = nx.generators.barabasi_albert_graph(self.n, self.m)

        n = self.n
        edges = np.array(graph.edges)
        node_labels = np.array([0 for _ in range(self.n)])

        return n, edges, node_labels


class RandomRegular(GraphGenerator):
    def __init__(self, n, d):
        super(RandomRegular, self).__init__()
        self.n = n
        self.d = d

    def generate_graph(self):
        graph = nx.generators.random_regular_graph(self.d, self.n)

        n = self.n
        edges = np.array(graph.edges)
        node_labels = np.array([0 for _ in range(self.n)])

        return n, edges, node_labels


class DualBarabasiAlbert(GraphGenerator):
    def __init__(self, n, m1, m2, p):
        super(DualBarabasiAlbert, self).__init__()
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.p = p

    def generate_graph(self):
        graph = nx.generators.dual_barabasi_albert_graph(self.n, self.m1, self.m2, self.p)

        n = self.n
        edges = np.array(graph.edges)
        node_labels = np.array([0 for _ in range(self.n)])

        return n, edges, node_labels


class WattsStrogatz(GraphGenerator):
    def __init__(self, n, k, p):
        super(WattsStrogatz, self).__init__()
        self.n = n
        self.k = k
        self.p = p

    def generate_graph(self):
        graph = nx.generators.watts_strogatz_graph(self.n, self.k, self.p)

        n = self.n
        edges = np.array(graph.edges)
        node_labels = np.array([0 for _ in range(self.n)])

        return n, edges, node_labels


class RandomLobster(GraphGenerator):
    def __init__(self, n, p1, p2):
        super(RandomLobster, self).__init__()
        self.n = n
        self.p1 = p1
        self.p2 = p2

    def generate_graph(self):
        graph = nx.generators.random_lobster(self.n, self.p1, self.p2)

        n = self.n
        edges = np.array(graph.edges)
        node_labels = np.array([0 for _ in range(self.n)])

        return n, edges, node_labels


class RandomTree(GraphGenerator):
    def __init__(self, n=None, min_n=None, max_n=None):
        super(RandomTree, self).__init__()
        self.n = n
        self.min_n = min_n
        self.max_n = max_n

    def get_n(self):
        if self.n is not None:
            return self.n
        elif self.min_n is not None and self.max_n is not None:
            return np.random.randint(self.min_n, self.max_n)
        else:
            return 0

    def generate_graph(self):
        n = self.get_n()

        graph = nx.generators.random_tree(n)

        edges = np.array(graph.edges)
        node_labels = np.array([0 for _ in range(n)])

        return n, edges, node_labels


class Grid(GraphGenerator):
    def __init__(self):
        super(Grid, self).__init__()
        self.i = 10
        self.j = 10

        self.max_i = 20
        self.max_j = 20

    def generate_graph(self):
        graph = nx.grid_2d_graph(self.i, self.j)

        n = self.i * self.j
        edges_2d = list(graph.edges)

        edges = []
        for a, b in list(edges_2d):
            a = a[0] * self.j + a[1]
            b = b[0] * self.j + b[1]

            edges.append((a, b))

        edges = np.array(edges)

        node_labels = np.array([0 for _ in range(n)])

        self.j += 1
        if self.j >= self.max_j:
            self.j = 10

            self.i += 1
            if self.i >= self.max_i:
                self.i = 10

        return n, edges, node_labels


if __name__ == "__main__":
    generator_0 = Grid()
    generator_1 = RandomTree(min_n=100, max_n=400)

    generators = [generator_0, generator_1]
    generate_dataset(generators, 200, "GridVSTree")
