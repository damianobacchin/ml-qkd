import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from collections import deque
from random import randint, random
from config import n_total_channels
from main import simulated_annealing, fitness_function


class Network:
    def __init__(self):
        self.nodes = []
        self.links = []

    def breadth_first_search(self, source, destination):  # Con mappa dei predecessori
        queue = deque([(source, [])])
        visited = set([])
        while queue:
            node, path = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node == destination:
                    return path

                euristic_links = sorted(
                    node.links, key=lambda link: link.channels)
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

    def generate_traffic(self, t=1000, lam=30, s=500, w=1):
        '''
        t = total time (ms)
        lam = lambda poisson distribution
        s = transmission speed (Mbps)
        w = mean requests weight (MB)
        '''

        ts = 10  # timeslot (ms)

        nodes_number = len(self.nodes)
        time = []
        for i in range(int(t/ts)):
            time.append([])

        for i in range(int(t/ts)):
            requests_num = np.random.poisson(lam)
            for req in range(requests_num):
                source = randint(0, nodes_number-1)
                while True:
                    destination = randint(0, nodes_number-1)
                    if destination != source:
                        break
                weight = np.random.poisson(w) * 1024**2 * 8
                bps_speed = s * 10**6
                req_time = int(weight / bps_speed / ts*10**3) + 1
                
                for slot in range(req_time):
                    try:
                        time[i + slot].append((source, destination))
                    except IndexError:
                        break
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
                link = Link(
                    network.nodes[incidence_matrix.index(elem)], network.nodes[i])
                network.links.append(link)
                node = network.nodes[incidence_matrix.index(elem)]
                node.links.append(link)

    print('Traffic generation...')
    time = network.generate_traffic()
    tr = network.run_simulation(traffic=time, target=network.links[14])

    config_sum_channel = np.zeros(n_total_channels)
    config_sum_quantum = np.zeros(n_total_channels)

    # Make hist
    counter = {}
    for i in range(n_total_channels):
        counter[i] = tr.count(i)

    # print(counter)

    keys = list(counter.keys())
    values = list(counter.values())

    # plt.figure(figsize=(8, 5))
    # plt.bar(keys, values)
    # plt.grid(True, linestyle = '--', linewidth = 0.5)
    # plt.xlabel('Numero di canali classici occupati')
    # plt.ylabel('Frequenza assoluta')
    # plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
    # plt.show()

    tot_freq = sum(counter.values())

    print('Quantum channel optimal allocation...')

    # Ottimizzazione quantum channel
    for n_ch in range(4, n_total_channels):
        config = np.zeros(n_total_channels)
        config[:n_ch] = 1
        config[n_ch] = 2

        annealing_result = simulated_annealing(config)
        #print(annealing_result)

        config_quantum = annealing_result.copy()
        config_quantum[annealing_result == 1] = 0
        config_quantum[annealing_result == 2] = 1

        config_sum_quantum += config_quantum * counter[n_ch] / tot_freq

    print(config_sum_quantum)

    config = np.zeros(n_total_channels)
    config[np.argmax(config_sum_quantum)] = 2

    print('Data channels allocation optimization...')

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

        config_sum_channel += config_classic * counter[n_ch] / tot_freq

    print(config_sum_channel)

    #config_sum_channel = np.array([ 0, 0, 0, 0.35, 0.17, 0.17, 0, 0, 0, 0, 0, 0 ])
    #config_sum_quantum = np.array([ 0.68, 0.51, 0, 0, 0.08, 0.99, 0.97, 0.83, 0.3, 0.16, 0.04, 0 ])

    hist_pos = {}
    for i in range(n_total_channels):
        hist_pos[i] = config_sum_channel[i]
    sorted_hist = { k: v for k, v in sorted(hist_pos.items(), key=lambda item: item[1], reverse=True) }
    del sorted_hist[np.argmax(config_sum_quantum)]

    keys = list(sorted_hist.keys())

    print('Evaluating optimization percentage...')

    perc = []
    for i in tr:
        if i == 0: continue
        # Optimized configuration
        config_opt = np.zeros(n_total_channels)
        config_opt[np.argmax(config_sum_quantum)] = 2
        for j in range(i):
            config_opt[keys[j]] = 1
        obj_function = fitness_function(config_opt)

        # Random configuration
        obj_function_rnd = []
        config_rnd = np.zeros(n_total_channels, dtype=int)
        config_rnd[:i] = 1
        config_rnd[i] = 2
        for j in range(10):
            np.random.shuffle(config_rnd)
            obj_function_rnd.append(fitness_function(config_rnd))
        obj_function_tot = np.mean(obj_function_rnd)

        # Calc percentage
        p = (obj_function - obj_function_rnd) / obj_function * 100
        perc.append(p)

    print(np.mean(perc))
