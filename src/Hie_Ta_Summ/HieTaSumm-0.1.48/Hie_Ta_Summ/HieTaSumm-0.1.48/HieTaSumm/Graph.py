# import networkx as nx
# import numpy as np

# import sys
# sys.path.append("/Users/palmer/.pyenv/versions/3.9.6/lib/python3.9/site-packages")

# #import higra  # Now it should work!
# import higra as hg

# class Graph:
#     def __init__(self, is_binary, hierarchy):
#         self.is_binary = is_binary
#         if hierarchy in ['watershed_hierarchy_by_attribute', 
#                       'watershed_hierarchy_by_minima_ordering', 
#                       'watershed_hierarchy_by_volume', 
#                       'watershed_hierarchy_by_area',
#                       'watershed_hierarchy_by_dynamics', 
#                       'watershed_hierarchy_by_number_of_parents']:
#           self.hierarchy = hierarchy
#         else: 
#           self.hierarchy = 'watershed_hierarchy_by_area'

#     def cut_graph(self, file, cut_graph_file, cutNumber):
#       RG = nx.Graph()
#       cut_list = []

#       with open(file.file) as f:
#         lines = f.readlines()

#       line = len(lines)-1
#       cut_list.append(int(lines[line].split(", ")[0]))

#       while(line >= 0):
#         cut = lines[line]
#         v1 = int(cut.split(", ")[0]) # node 1
#         v2 = int(cut.split(", ")[1]) # node 2
#         w = float(cut.split(", ")[2]) # weight

#         if (v2 in cut_list):
#           if cutNumber > 0:
#             cutNumber -= 1
#             cut_list.append(v1)
#           else:
#             cut_graph_file.save_graph_data(v1, v2, w)
#             RG.add_node(v1)
#         else:
#           cut_graph_file.save_graph_data(v1, v2, w)
#           RG.add_edge(v1, v2, weight = w)
#         line -= 1

#       return RG

#     def compute_hierarchy(self, input_g, input_higra):
#       leaf_list = []
#       #print("DEBUG: Checking input graph nodes and edges...")
#       #print("Nodes:", input_g.nodes())
#       #print("Edges:", input_g.edges(data=True))
      
#       if len(input_g.nodes) == 0:
#           raise ValueError("Error: input_g is empty. Ensure the graph has nodes before computing hierarchy.")

#       graph = hg.UndirectedGraph()
#       graph.add_vertices(max(input_g.nodes)+1)   #creating the nodes
#       edge_list = list(input_g.edges())
#       size_threshold = 20
      
#       #print("DEBUG: Adding edges to HiGra graph...")
#       for i in range(len(edge_list)):
#           graph.add_edge(edge_list[i][0], edge_list[i][1]) #Adding nodes to HiGra graph
#       edge_weights = np.empty(shape=len(edge_list))
#       sources, targets = graph.edge_list()

#       for i in range(len(sources)):
#         edge_weights[i] = int(input_g.adj[sources[i]][targets[i]]['weight'])
      
#       #print("DEBUG: Computing hierarchy using", self.hierarchy)
#       #print("DEBUG: HiGra Edge List (First 20 edges)")
#       sources, targets = graph.edge_list()
#       print(list(zip(sources[:20], targets[:20], edge_weights[:20])))

#       #print("DEBUG: Edge List Sorting Check")
#       #print(f" - Is sorted? {all(sources[i] <= sources[i + 1] for i in range(len(sources) - 1))}")

#       if self.hierarchy == 'watershed_hierarchy_by_attribute':
#         nb_tree, nb_altitudes = hg.watershed_hierarchy_by_attribute(graph, edge_weights)
#       elif self.hierarchy == 'watershed_hierarchy_by_minima_ordering':
#         nb_tree, nb_altitudes = hg.watershed_hierarchy_by_minima_ordering(graph, edge_weights)
#       elif self.hierarchy == 'watershed_hierarchy_by_volume':
#         nb_tree, nb_altitudes = hg.watershed_hierarchy_by_volume(graph, edge_weights)
#       elif self.hierarchy == 'watershed_hierarchy_by_area':
#         nb_tree, nb_altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
#       elif self.hierarchy == 'watershed_hierarchy_by_dynamics':
#         nb_tree, nb_altitudes = hg.watershed_hierarchy_by_dynamics(graph, edge_weights)
#       elif self.hierarchy == 'watershed_hierarchy_by_number_of_parents':
#         nb_tree, nb_altitudes = hg.watershed_hierarchy_by_number_of_parents(graph, edge_weights)
      
#       if self.is_binary:
#         tree, node_map = hg.tree_2_binary_tree(nb_tree)
#         altitudes = nb_altitudes[node_map]
#       else:
#         tree = nb_tree
#         altitudes = nb_altitudes
      
#       #print("DEBUG: Processing hierarchy tree...")
#       for n in tree.leaves_to_root_iterator():
#         leaf = -2 # It's cod is used for the node that is not a leaf
#         if tree.is_leaf(n):
#           leaf = -1 # It's cod is used for the node that is a leaf
#           leaf_list.append(n)
#         input_higra.save_graph_data(n, tree.parent(n), leaf)
      
#       #print("DEBUG: Hierarchy computation finished.")
#       return leaf_list
import networkx as nx
import numpy as np
import higra as hg

class Graph:
    def __init__(self, is_binary, hierarchy):
        self.is_binary = is_binary
        if hierarchy in ['watershed_hierarchy_by_attribute', 
                         'watershed_hierarchy_by_minima_ordering', 
                         'watershed_hierarchy_by_volume', 
                         'watershed_hierarchy_by_area',
                         'watershed_hierarchy_by_dynamics', 
                         'watershed_hierarchy_by_number_of_parents']:
            self.hierarchy = hierarchy
        else: 
            self.hierarchy = 'watershed_hierarchy_by_area'

    def cut_graph(self, file, cut_graph_file, cutNumber):
        RG = nx.Graph()
        cut_list = []

        # Instead of reading from disk, use the in-memory content.
        lines = file.content.splitlines()

        line = len(lines) - 1
        # We assume there is at least one line; if not, RG remains empty.
        cut_list.append(int(lines[line].split(", ")[0]))

        while line >= 0:
            cut_line = lines[line]
            parts = cut_line.split(", ")
            if len(parts) < 3:
                line -= 1
                continue
            v1 = int(parts[0])
            v2 = int(parts[1])
            w = float(parts[2])

            if v2 in cut_list:
                if cutNumber > 0:
                    cutNumber -= 1
                    cut_list.append(v1)
                else:
                    cut_graph_file.save_graph_data(v1, v2, w)
                    RG.add_node(v1)
            else:
                cut_graph_file.save_graph_data(v1, v2, w)
                RG.add_edge(v1, v2, weight=w)
            line -= 1

        return RG

    def compute_hierarchy(self, input_g, input_higra):
        leaf_list = []
      
        if len(input_g.nodes) == 0:
            raise ValueError("Error: input_g is empty. Ensure the graph has nodes before computing hierarchy.")

        # Create a HiGra graph.
        graph = hg.UndirectedGraph()
        graph.add_vertices(max(input_g.nodes) + 1)
        edge_list = list(input_g.edges())
        size_threshold = 20

        # Add edges to the HiGra graph.
        for i in range(len(edge_list)):
            graph.add_edge(edge_list[i][0], edge_list[i][1])
        edge_weights = np.empty(shape=len(edge_list))
        sources, targets = graph.edge_list()

        for i in range(len(sources)):
            edge_weights[i] = int(input_g.adj[sources[i]][targets[i]]['weight'])

        sources, targets = graph.edge_list()
        print(list(zip(sources[:20], targets[:20], edge_weights[:20])))

        if self.hierarchy == 'watershed_hierarchy_by_attribute':
            nb_tree, nb_altitudes = hg.watershed_hierarchy_by_attribute(graph, edge_weights)
        elif self.hierarchy == 'watershed_hierarchy_by_minima_ordering':
            nb_tree, nb_altitudes = hg.watershed_hierarchy_by_minima_ordering(graph, edge_weights)
        elif self.hierarchy == 'watershed_hierarchy_by_volume':
            nb_tree, nb_altitudes = hg.watershed_hierarchy_by_volume(graph, edge_weights)
        elif self.hierarchy == 'watershed_hierarchy_by_area':
            nb_tree, nb_altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
        elif self.hierarchy == 'watershed_hierarchy_by_dynamics':
            nb_tree, nb_altitudes = hg.watershed_hierarchy_by_dynamics(graph, edge_weights)
        elif self.hierarchy == 'watershed_hierarchy_by_number_of_parents':
            nb_tree, nb_altitudes = hg.watershed_hierarchy_by_number_of_parents(graph, edge_weights)

        if self.is_binary:
            tree, node_map = hg.tree_2_binary_tree(nb_tree)
            altitudes = nb_altitudes[node_map]
        else:
            tree = nb_tree
            altitudes = nb_altitudes

        # Process the hierarchy and store information in the in-memory input_higra.
        for n in tree.leaves_to_root_iterator():
            leaf = -2  # Default code for non-leaf
            if tree.is_leaf(n):
                leaf = -1  # Code for leaf
                leaf_list.append(n)
            input_higra.save_graph_data(n, tree.parent(n), leaf)

        return leaf_list
