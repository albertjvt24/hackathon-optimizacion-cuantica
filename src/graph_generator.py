import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generar_red_antenas(n_antenas: int, radio_interferencia: float = 0.6, seed: int = 42) -> nx.Graph:
    G = nx.random_geometric_graph(n_antenas, radio_interferencia, seed=seed)
    
    for u, v in G.edges():
        pos_u = np.array(G.nodes[u]['pos'])
        pos_v = np.array(G.nodes[v]['pos'])
        distancia = np.linalg.norm(pos_u - pos_v)
        
        if distancia < 0.05:
            distancia = 0.05 
            
        peso_interferencia = round(1.0 / (distancia ** 2), 2)
        
        G.edges[u, v]['weight'] = peso_interferencia
        
    return G

def dibujar_red_base(G: nx.Graph, titulo: str = "Topología de la Red de Antenas"):

    posiciones = nx.get_node_attributes(G, 'pos')
    pesos = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(8, 6))
    
    nx.draw(G, posiciones, with_labels=True, node_color='lightblue', 
            node_size=600, font_weight='bold', font_color='black')
    
    nx.draw_networkx_edge_labels(G, posiciones, edge_labels=pesos, font_size=9)
    
    plt.title(titulo)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":

    grafo_prueba = generar_red_antenas(n_antenas=5, radio_interferencia=0.8, seed=10)
    print(f"Nodos generados: {grafo_prueba.nodes()}")
    print(f"Conexiones y pesos de interferencia (J_ij): {grafo_prueba.edges(data=True)}")
    dibujar_red_base(grafo_prueba)
