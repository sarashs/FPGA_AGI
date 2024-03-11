import matplotlib.pyplot as plt
import networkx as nx
import json
from typing import Any

from FPGA_AGI.agents import HierarchicalResponse

def plot_graph(hierarchicalmodules: HierarchicalResponse, save_path: Any = None):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with the module name as the node ID
    for module in hierarchicalmodules.graph:
        G.add_node(module.name, description=module.description)

    # Add edges based on the 'connections' in each module
    for module in hierarchicalmodules.graph:
        for connection in module.connections:
            G.add_edge(module.name, connection)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=7000, edge_color='gray', linewidths=0.5, width=2, font_size=14)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): '' for u, v in G.edges()}, font_color='red')
    plt.title('Module Connections Graph')
    if save_path:
        plt.savefig(save_path, dpi=300) 
    plt.show()