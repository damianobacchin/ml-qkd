import numpy as np
from collections import deque
from config import n_total_channels


class Network:
    def __init__(self):
        self.nodes = []
        self.traffic = 1
    
    def breadth_first_search(self, source, destination):
        queue = deque([(source, [])])
        visited = set([])
        while queue:
            node, path = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node == destination:
                    return path + [node]
                
                euristic_links = sorted(node.links, key=lambda link: link.config.sum(), reverse=True)
                for link in euristic_links:
                    if link.config.sum() < n_total_channels:
                        queue.append((link.destination, path + [node]))
        return None
    



class Node:
    def __init__(self, idx):
        self.idx = idx
        self.links = []
        self.cross_connect = True

    def __str__(self):
        return str(self.idx)
    

class Link:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.config = np.zeros(n_total_channels, dtype=int)


if __name__ == "__main__":
    network = Network()

    incidence_matrix = (
        (0, 1, 1, 1, 0, 0),
        (1, 0, 0, 1, 0, 0),
        (1, 0, 0, 1, 0, 1),
        (1, 1, 1, 0, 1, 1),
        (0, 0, 0, 1, 0, 1),
        (0, 0, 1, 1, 1, 0),
    )

    for elem in incidence_matrix:
        node = Node(incidence_matrix.index(elem))
        network.nodes.append(node)
    
    for elem in incidence_matrix:
        for i in range(len(elem)):
            if elem[i] == 1:
                link = Link(network.nodes[incidence_matrix.index(elem)], network.nodes[i])
                node = network.nodes[incidence_matrix.index(elem)]
                node.links.append(link)

    solution = network.breadth_first_search(network.nodes[0], network.nodes[5])
    print(" - ".join([str(node) for node in solution]))

