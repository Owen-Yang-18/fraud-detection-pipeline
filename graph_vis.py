import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_heterogeneous_bipartite_graph():
    """
    Create a heterogeneous bipartite graph for Android security analysis.
    Applications connect to all other node types but not to each other.
    """
    
    # Create an empty graph
    G = nx.Graph()
    
    # Define node types and their instances
    node_types = {
        'application': ['benign', 'riskware', 'adware', 'banking', 'sms'],
        'syscall': ['open', 'pipe'],
        'binder_call': ['getDisplayInfo', 'getPackageInfo'],
        'composite_call': ['FS_ACCESS', 'NETWORK_ACCESS']
    }
    
    # Add nodes with type attributes
    for node_type, nodes in node_types.items():
        for node in nodes:
            G.add_node(node, node_type=node_type)
    
    # Create bipartite edges: applications connect to all other types
    application_nodes = node_types['application']
    other_nodes = (node_types['syscall'] + 
                   node_types['binder_call'] + 
                   node_types['composite_call'])
    
    # Add edges from each application to all other node types
    for app_node in application_nodes:
        for other_node in other_nodes:
            G.add_edge(app_node, other_node)
    
    return G, node_types

def plot_heterogeneous_graph(G, node_types):
    """
    Create a beautiful visualization of the heterogeneous bipartite graph.
    """
    
    # Set up the figure with high DPI for crisp output
    plt.figure(figsize=(16, 12), dpi=300)
    plt.style.use('default')
    
    # Define colors for each node type
    colors = {
        'application': {
            'benign': '#1976d2',
            'riskware': '#d32f2f',
            'adware': '#f57c00',
            'banking': '#388e3c',
            'sms': '#7b1fa2'
        },
        'syscall': '#4caf50',
        'binder_call': '#ff9800',
        'composite_call': '#e91e63'
    }
    
    # Define node shapes for each type
    node_shapes = {
        'application': 'o',      # circle
        'syscall': 's',          # square
        'binder_call': '^',      # triangle
        'composite_call': 'D'    # diamond
    }
    
    # Define custom positions for better layout
    pos = {}
    
    # Position application nodes on the left
    app_nodes = node_types['application']
    for i, node in enumerate(app_nodes):
        pos[node] = (0, i * 2)
    
    # Position syscall nodes
    syscall_nodes = node_types['syscall']
    for i, node in enumerate(syscall_nodes):
        pos[node] = (3, i * 2 + 2)
    
    # Position binder call nodes
    binder_nodes = node_types['binder_call']
    for i, node in enumerate(binder_nodes):
        pos[node] = (6, i * 2 + 2)
    
    # Position composite call nodes
    composite_nodes = node_types['composite_call']
    for i, node in enumerate(composite_nodes):
        pos[node] = (9, i * 2 + 2)
    
    # Draw edges first (so they appear behind nodes)
    edge_colors = []
    for edge in G.edges():
        source, target = edge
        source_type = G.nodes[source]['node_type']
        if source_type == 'application':
            edge_colors.append(colors['application'][source])
        else:
            edge_colors.append('#666666')
    
    nx.draw_networkx_edges(G, pos, 
                          edge_color=edge_colors,
                          alpha=0.6,
                          width=2,
                          style='-')
    
    # Draw nodes by type with different shapes and colors
    for node_type, nodes in node_types.items():
        node_list = [node for node in nodes if node in G.nodes()]
        
        if node_type == 'application':
            # Draw application nodes with individual colors
            for node in node_list:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=[node],
                                     node_color=colors['application'][node],
                                     node_shape=node_shapes[node_type],
                                     node_size=2000,
                                     alpha=0.9,
                                     edgecolors='white',
                                     linewidths=3)
        else:
            # Draw other node types with uniform colors per type
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=node_list,
                                 node_color=colors[node_type],
                                 node_shape=node_shapes[node_type],
                                 node_size=2000,
                                 alpha=0.9,
                                 edgecolors='white',
                                 linewidths=3)
    
    # Add node labels with better formatting
    labels = {}
    for node in G.nodes():
        node_type = G.nodes[node]['node_type']
        if node_type == 'application':
            labels[node] = node.upper()
        else:
            labels[node] = node.replace('_', '\n')
    
    # Draw labels with white text and bold font
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=8,
                           font_weight='bold',
                           font_color='white',
                           font_family='sans-serif')
    
    # Create custom legend
    legend_elements = []
    
    # Application nodes legend
    for app_type, color in colors['application'].items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=12,
                                        label=f'{app_type.title()} App', 
                                        markeredgecolor='white', markeredgewidth=2))
    
    # Other node types legend
    for node_type in ['syscall', 'binder_call', 'composite_call']:
        shape = node_shapes[node_type]
        color = colors[node_type]
        display_name = node_type.replace('_', ' ').title()
        legend_elements.append(plt.Line2D([0], [0], marker=shape, color='w',
                                        markerfacecolor=color, markersize=12,
                                        label=display_name,
                                        markeredgecolor='white', markeredgewidth=2))
    
    # Add legend to the plot
    plt.legend(handles=legend_elements, 
              loc='upper right', 
              bbox_to_anchor=(1.0, 1.0),
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=10)
    
    # Customize the plot
    plt.title('Heterogeneous Bipartite Graph: Android Security Analysis\n' + 
              'Applications → Syscalls, Binder Calls, and Composite Calls',
              fontsize=16, fontweight='bold', pad=20)
    
    # Remove axes
    plt.axis('off')
    
    # Set equal aspect ratio and adjust layout
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    # Add text annotations for graph properties
    plt.figtext(0.02, 0.02, 
               f'Graph Properties:\n'
               f'• Nodes: {G.number_of_nodes()}\n'
               f'• Edges: {G.number_of_edges()}\n'
               f'• Bipartite: Applications ↔ Other Types\n'
               f'• No connections within node types',
               fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    return plt

def analyze_graph_properties(G, node_types):
    """
    Analyze and print key properties of the heterogeneous graph.
    """
    print("=== Heterogeneous Bipartite Graph Analysis ===\n")
    
    # Basic graph properties
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Graph density: {nx.density(G):.3f}")
    
    # Node type distribution
    print("\nNode Type Distribution:")
    for node_type, nodes in node_types.items():
        count = len([n for n in nodes if n in G.nodes()])
        print(f"  {node_type.replace('_', ' ').title()}: {count} nodes")
    
    # Degree analysis
    print("\nDegree Analysis:")
    degrees = dict(G.degree())
    
    for node_type, nodes in node_types.items():
        type_degrees = [degrees[node] for node in nodes if node in G.nodes()]
        if type_degrees:
            avg_degree = np.mean(type_degrees)
            print(f"  {node_type.replace('_', ' ').title()}: avg degree = {avg_degree:.1f}")
    
    # Verify bipartite structure
    print(f"\nBipartite Structure Verification:")
    app_nodes = set(node_types['application'])
    other_nodes = set()
    for node_type in ['syscall', 'binder_call', 'composite_call']:
        other_nodes.update(node_types[node_type])
    
    # Check if it's truly bipartite
    is_bipartite = True
    
    # Check no edges within application nodes
    app_internal_edges = 0
    for app1 in app_nodes:
        for app2 in app_nodes:
            if app1 != app2 and G.has_edge(app1, app2):
                app_internal_edges += 1
    
    # Check no edges within other node types
    other_internal_edges = 0
    for node1 in other_nodes:
        for node2 in other_nodes:
            if node1 != node2 and G.has_edge(node1, node2):
                other_internal_edges += 1
    
    if app_internal_edges == 0 and other_internal_edges == 0:
        print("  ✓ Perfect bipartite structure maintained")
        print("  ✓ No edges within application nodes")
        print("  ✓ No edges within syscall/binder/composite nodes")
    else:
        print(f"  ✗ Found {app_internal_edges} internal app edges")
        print(f"  ✗ Found {other_internal_edges} internal other edges")
    
    # Edge type analysis
    print(f"\nEdge Analysis:")
    expected_edges = len(app_nodes) * len(other_nodes)
    actual_edges = G.number_of_edges()
    print(f"  Expected edges (bipartite): {expected_edges}")
    print(f"  Actual edges: {actual_edges}")
    print(f"  Complete bipartite: {'Yes' if expected_edges == actual_edges else 'No'}")

def main():
    """
    Main function to create and visualize the heterogeneous bipartite graph.
    """
    
    # Create the graph
    G, node_types = create_heterogeneous_bipartite_graph()
    
    # Analyze graph properties
    analyze_graph_properties(G, node_types)
    
    # Create visualization
    plt_obj = plot_heterogeneous_graph(G, node_types)
    
    # Show the plot
    plt.show()
    
    # Optionally save the plot
    # plt.savefig('heterogeneous_bipartite_graph.png', dpi=300, bbox_inches='tight')
    
    return G, node_types

if __name__ == "__main__":
    # Run the main function
    graph, node_types = main()
    
    print("\n=== Node List by Type ===")
    for node_type, nodes in node_types.items():
        print(f"{node_type.replace('_', ' ').title()}: {nodes}")