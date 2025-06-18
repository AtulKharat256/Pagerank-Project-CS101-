import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def create_graph(path):
    #creating a directed graph using networkx library
    g = nx.DiGraph()  
    #reading the data from the csv file
    data = pd.read_csv(path)  
    #making a list of the first column is the email address of the person taking impressions
    main = data['Email Address'].tolist()
    #adding nodes to the graph  
    g.add_nodes_from(main)  
    for i in range(len(data)):
    #impressions contain data of each row
        impressions = list(data.iloc[i])
    #now if impression is not empty then adding a edge between the person taking impressions and the person who is being impressed
        for j in range(1, len(impressions)):
            if impressions[j] != 0:
                g.add_edge(impressions[0], impressions[j])  
    #returning the graph
    #plot the graph
    nx.draw(g, with_labels=True)
    plt.show()
    return g

G = create_graph('C:\\Users\\a201t\\Desktop\\data1.csv')



def random_walk_simulation(graph, iterations=100000):
    #creating a dictionary to store the number of visits to each node
    node_visits = {node: 0 for node in graph.nodes} 
    #starting from a random node 

    current_node = random.choice(list(graph.nodes))  
    #iterating over the number of iterations
    for i in range(iterations):  
        #incrementing the visit count for the current node
        node_visits[current_node] += 1  
        #getting the neighbors of the current node
        neighbors = list(graph.neighbors(current_node))
        #if there are neighbors then choosing a random neighbor
        if neighbors:  
            #choosing a random neighbor
            p = random.uniform(0, 1)
            if p < 0.85:
                next_node = random.choice(neighbors)
            #if the next node is in the graph then moving to the next node
                current_node = next_node if next_node in graph.nodes else random.choice(list(graph.nodes))
        else:
            #if there are no neighbors then moving to a random node
            current_node = random.choice(list(graph.nodes))
    #sorting the nodes based on the number of visits
    sorted_visits = dict(sorted(node_visits.items(), key=lambda item: item[1], reverse=True))
    sorted_visits.pop('0', None)
    #returning the sorted visits
    return sorted_visits
def missing_link(graph):
    #getting the nodes of the graph
    n = sorted(graph.nodes())
    #setting k to the number of the nodes
    k = len(n)
    #making adjacency matrix of the graph
    m = nx.to_numpy_array(graph, n)
    #creating a list to store the missing links
    links=[]
    #iterating over every element of the adjacency matrix
    for i in range(len(n)):
        for j in range(len(n)):
    #if the element is not the diagonal element and the element is 0 and the element at the transpose of the element is 0
            if i!=j and m[i,j] == 0 and m[j,i] == 0:
                c = m.copy()
    #deleting the column which contain the element
                a1 = np.delete(c[i], j)
    #deleting the row which contain the element
                a2 = np.delete(c[:,j], i) 
                c = np.delete(c,(i),axis = 0)
                c = np.delete(c,(j),axis = 1)
    #solving the linear equation, where the element is the linear combination of the other elements in column
                x = np.linalg.lstsq(c, a2, rcond = None)[0]
    #taking the dot product of the solution and the column
                k = np.dot(x, a1)
    #if the dot product is greater than 0.5 then adding the missing link to the list
                if k > 0.5:
                    links.append((n[i], n[j]))
    print("Missing links are:", links,"Number of missing links in the graph are:", len(links))

def cycle(graph):
    #visited keep track of visited vertices
    visited = set()
    #current keeps track of all the nodes in the current path, as soon as we find similar node in it, we have a cycle
    current = set()

    #defining a function to perform depth first search
    def dfs(node):
        #starting from a node, adding it to visited and current path
        visited.add(node)
        current.add(node)

        #now iterating over all the neighbours of the node
        for neighbour in graph.neighbors(node):
            #if the neighbour is in the current path, then we have a cycle
            if neighbour in current:
                return True
            #if the neighbour is not visited, recursively explore it
            elif neighbour not in visited:
                if dfs(neighbour):
                    return True

        #remove the node from the current path as we have explored all its neighbours and no cycle was found
        current.remove(node)
        return False

    #iterate over all vertices and start DFS from each unvisited vertex
    for node in graph.nodes():
        if node not in visited:
            if dfs(node):
                return True

    return False


