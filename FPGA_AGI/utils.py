import matplotlib.pyplot as plt
import networkx as nx
import json
from typing import Any
from FPGA_AGI.parameters import LANGS

def find_extension(language: str):
    """finds the file extension for the language"""
    for item in LANGS.keys():
        if language in LANGS[item]:
            return item
    raise ValueError(f"{language} is not a valid language.")

def extract_codes_from_string(string):
    lower_string = string.lower()  # Convert the entire string to lowercase
    code = None
    langs = []
    for item in LANGS.keys():
        langs += LANGS[item]
    for lang in langs:
        # Create the search pattern for each language in lowercase
        start_pattern = f'```{lang.lower()}\n'
        end_pattern = '\n```'

        # Find the start and end indices for each code block
        start_index = lower_string.find(start_pattern)
        end_index = lower_string.rfind(end_pattern)

        if start_index != -1 and end_index != -1:
            # Calculate the actual start index in the original string
            actual_start_index = start_index + len(start_pattern)

            # Extract and store the code block from the original string
            code = string[actual_start_index:end_index]
            break

    # Check if any code blocks have been extracted
    if not code:
        # If no code blocks are found, return the original string
        return string
    else:
        return code

def plot_graph(hierarchicalmodules: Any, save_path: Any = None):
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