import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from generator.py import generar_red_demo
from classical_solver.py import comparar, componentes 
from quantum_solver.py import resolver_qaoa  # 

st.set_page_config(page_title="Max-Cut Híbrido", layout="wide")
st.title("Optimización de Antenas: Clásico vs Cuántico")

def dibujar_grafo(red, asignacion, titulo, ax):
    """Renderiza la topología mostrando interferencias en rojo."""
    G = nx.Graph()
    for i in range(red['n']):
        G.add_node(i, pos=red['pos'][i])
    for i, j, w in red['interf']:
        G.add_edge(i, j)
        
    posiciones = nx.get_node_attributes(G, 'pos')
    colores_nodos = ['#1f77b4' if c == 0 else '#ff7f0e' for c in asignacion]
    colores_aristas = ['red' if asignacion[u] == asignacion[v] else 'lightgray' for u, v in G.edges()]

    nx.draw(G, posiciones, ax=ax, with_labels=True, node_color=colores_nodos, 
            edge_color=colores_aristas, node_size=500, font_weight='bold', font_color='white')
    ax.set_title(titulo)

st.sidebar.header("Configuración")
n_antenas = st.sidebar.slider("Número de Antenas", 3, 10, 5)
lam = st.sidebar.slider("Penalización por Cambio (λ)", 0.0, 3.0, 1.0, 0.5)

red = generar_red_demo(n_antenas)
red['lambda_cambio'] = lam

if st.sidebar.button("Ejecutar Benchmark"):
    with st.spinner("Simulando enfoques clásicos y circuito cuántico (QAOA)..."):

        resultados = comparar(red, lam)

        res_qaoa = resolver_qaoa(red)

        i_qaoa, c_qaoa, nc_qaoa = componentes(red, res_qaoa['x'])
        total_qaoa = i_qaoa + (lam * c_qaoa)
        
        fila_qaoa = {
            'metodo': res_qaoa['metodo'],
            'x': res_qaoa['x'],
            'interferencia': i_qaoa,
            'costo_cambio': c_qaoa,
            'penalizacion': lam * c_qaoa,
            'cambios': nc_qaoa,
            'total': total_qaoa,
            'tiempo_s': res_qaoa['tiempo_s'],
            'brecha': total_qaoa - resultados[1]['total'] 
        }
        resultados.append(fila_qaoa)
        
        df = pd.DataFrame(resultados)
        columnas = ['metodo', 'interferencia', 'cambios', 'total', 'brecha', 'tiempo_s']
        st.dataframe(df[columnas].style.format({"tiempo_s": "{:.4f}", "total": "{:.2f}", "brecha": "{:.2f}"}), use_container_width=True)
        
        # 5. Visualización topológica
        estado_inicial = next(r for r in resultados if r['metodo'] == 'Inicial')
        estado_exacto = next(r for r in resultados if r['metodo'] == 'Exacto')
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        dibujar_grafo(red, estado_inicial['x'], f"Estado Inicial\nCosto: {estado_inicial['total']:.2f}", ax1)
        dibujar_grafo(red, estado_exacto['x'], f"Óptimo Exacto (Clásico)\nCosto: {estado_exacto['total']:.2f}", ax2)
        dibujar_grafo(red, fila_qaoa['x'], f"Solución QAOA (Cuántico)\nCosto: {fila_qaoa['total']:.2f}", ax3)
        st.pyplot(fig)
        
        # Mostrar métricas del ansatz
        st.caption(f"Detalles QAOA: {res_qaoa['extra']['qubits']} qubits, Profundidad del circuito: {res_qaoa['extra']['profundidad']}, Evaluaciones de energía: {res_qaoa['extra']['evaluaciones']}")
