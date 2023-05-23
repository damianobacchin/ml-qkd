import numpy as np
from collections import deque
from random import randint
from config import n_total_channels


class Network:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.traffic = 1
    
    def breadth_first_search(self, source, destination): # Con mappa dei predecessori
        queue = deque([(source, [])])
        visited = set([])
        while queue:
            node, path = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node == destination:
                    return path
                
                euristic_links = sorted(node.links, key=lambda link: link.config.sum())
                for link in euristic_links:
                    if link.config.sum() < n_total_channels:
                        queue.append((link.destination, path + [link]))
        return None

    def generate_traffic(self, t=100, r=0.8):
        nodes_number = len(self.nodes)
        time = np.zeros(t, dtype=list)

        for i in range(t):
            requests = randint(0, 100*r)
            timeslot = []
            for j in range(requests):
                source = randint(0, nodes_number-1)
                while True:
                    destination = randint(0, nodes_number-1)
                    if destination != source:
                        break
                timeslot.append((source, destination))
            time[i] = timeslot
        return time

    
    def run_simulation(self, traffic, target):
        link_traffic = []
        for t in traffic:
            for link in self.links:
                link.config = np.zeros(n_total_channels, dtype=int)
            for request in t:
                source = self.nodes[request[0]]
                destination = self.nodes[request[1]]
                solution = self.breadth_first_search(source, destination)
                if solution is not None:
                    for link in solution:
                        pos = np.where(link.config == 0)[0][0]
                        link.config[pos] = 1
            link_traffic.append(target.config)
        return link_traffic





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

    def __str__(self):
        return f"{self.source}>{self.destination}"


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
                network.links.append(link)
                node = network.nodes[incidence_matrix.index(elem)]
                node.links.append(link)

    #solution = network.breadth_first_search(network.nodes[0], network.nodes[5])
    #print(" - ".join([str(node) for node in solution]))
    # for link in solution:
    #     pos = np.where(link.config == 0)[0][0]
    #     link.config[pos] = 1
    
    time = network.generate_traffic()
    tr = network.run_simulation(traffic=time, target=network.links[3])
    print(tr)