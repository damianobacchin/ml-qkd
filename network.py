import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from random import randint, random
from config import n_total_channels
from main import simulated_annealing, fitness_function


class Network:
    def __init__(self):
        self.nodes = []
        self.links = []
    
    def breadth_first_search(self, source, destination): # Con mappa dei predecessori
        queue = deque([(source, [])])
        visited = set([])
        while queue:
            node, path = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node == destination:
                    return path
                
                euristic_links = sorted(node.links, key=lambda link: link.channels)
                for link in euristic_links:
                    if link.channels < n_total_channels-1:
                        queue.append((link.destination, path + [link]))
        return None
    
    def generate_routing_table(self):
        routing_table = dict()
        for source in self.nodes:
            routing_table[source] = dict()
            for destination in self.nodes:
                if source != destination:
                    queue = deque([(source, [])])
                    visited = set([])
                    solutions = []
                    while queue:
                        node, path = queue.popleft()
                        if node not in visited:
                            visited.add(node)
                            if node == destination:
                                solutions.append(path)
                            for link in node.links:
                                queue.append((link.destination, path + [link]))
                    routing_table[source][destination] = solutions
        self.routing_table = routing_table
        return routing_table

    def generate_traffic(self, t=100, r=0.2):
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
                if random() < 0.8:
                    for weigth in range(randint(1, 3)):
                        timeslot.append((destination, source))
            time[i] = timeslot
        return time

    
    def run_simulation(self, traffic, target):
        link_traffic = []
        for t in traffic:
            for link in self.links:
                link.channels = 0
            for request in t:
                source = self.nodes[request[0]]
                destination = self.nodes[request[1]]
                solution = self.breadth_first_search(source, destination)
                if solution is not None:
                    for link in solution:
                        link.channels += 1
            link_traffic.append(target.channels)
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
        self.channels = 0

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
    tr = network.run_simulation(traffic=time, target=network.links[4])
    
    # Generate hystogram
    # axis = []
    # for i in tr:
    #     axis.append(i.sum())
    # counts, bins = np.histogram(axis, bins=n_total_channels)
    # print(counts, bins)

    config_sum_channel = np.zeros(n_total_channels)
    config_sum_quantum = np.zeros(n_total_channels)

    # Make hist
    counter = {}
    for i in range(n_total_channels):
        counter[i] = tr.count(i)

    tot_freq = sum(counter.values())

    # Ottimizzazione quantum channel
    for n_ch in range(4, n_total_channels):
        config = np.zeros(n_total_channels)
        config[:n_ch] = 1
        config[n_ch] = 2

        annealing_result = simulated_annealing(config)
        print(annealing_result)

        config_quantum = annealing_result.copy()
        config_quantum[annealing_result == 1] = 0
        config_quantum[annealing_result == 2] = 1

        config_sum_quantum += config_quantum * counter[n_ch] / tot_freq
    
    print(config_sum_quantum)

    config = np.zeros(n_total_channels)
    config[np.argmax(config_sum_quantum)] = 2

    for n_ch in range(1, n_total_channels-2):
        # Creazione config
        new_config = config.copy()
        for i in range(n_ch):
            if new_config[i] == 0:
                new_config[i] = 1
            else:
                new_config[n_total_channels-1] = 1
        
        # Simulated annealing
        annealing_result = simulated_annealing(new_config, mod=True)
        config_classic = annealing_result.copy()
        config_classic[annealing_result == 2] = 0

        print(config_classic)

        config_sum_channel += config_classic * counter[n_ch] / tot_freq
    
    print(config_sum_channel)

    hist_pos = {}
    for i in range(n_total_channels):
        hist_pos[i] = config_sum_channel[i]
    sorted_hist = { k: v for k, v in sorted(hist_pos.items(), key=lambda item: item[1], reverse=True) }
    del sorted_hist[np.argmax(config_sum_quantum)]

    keys = list(sorted_hist.keys())
    print(keys)

    perc = []
    for i in range(1, n_total_channels-1):
        # Optimized configuration
        config_opt = np.zeros(n_total_channels)
        config_opt[np.argmax(config_sum_quantum)] = 2
        for j in range(i):
            config_opt[keys[j]] = 1
        obj_function = fitness_function(config_opt)

        # Random configuration
        results_rnd = []
        config_a = np.zeros(n_total_channels)
        config_a[:i] = 1
        config_a[i] = 2
        for k in range(10):
            np.random.shuffle(config_a)
            results_rnd.append(fitness_function(config_a))
        obj_function_rnd = np.mean(results_rnd)

        # Calc percentage
        p = (obj_function_rnd - obj_function) / obj_function_rnd
        perc.append(p)

    print(perc)
    print(np.mean(perc))