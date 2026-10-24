# 📡 Optimización Cuántica de Redes Inalámbricas

Este repositorio contiene un entorno interactivo diseñado para resolver problemas de optimización combinatoria en telecomunicaciones, específicamente la asignación óptima de canales de frecuencia en una red de antenas para minimizar la interferencia de señales.

## ⚛️ El Enfoque Físico-Matemático

El problema se formula matemáticamente como una partición de grafos (*Max-Cut*) y se mapea directamente a un observable de la mecánica cuántica utilizando el **Modelo de Ising Antiferromagnético**. 

Cada antena se modela como una variable de espín $\sigma_z^{(i)} \in \{+1, -1\}$, representando los dos canales de frecuencia disponibles. El Hamiltoniano de interferencia del sistema se define como:

$$ H = \sum_{\langle i,j \rangle} J_{ij} \sigma_z^{(i)} \sigma_z^{(j)} $$

Donde $J_{ij} > 0$ es el riesgo de interferencia entre antenas vecinas. Al utilizar el Algoritmo de Optimización Cuántica Aproximada (QAOA), el sistema busca el estado fundamental (*ground state*) del Hamiltoniano. Esto resuelve inherentemente la "frustración geométrica" de la red al distribuir las interferencias inevitables (en ciclos topológicos impares) hacia los enlaces con menor penalidad $J_{ij}$.

## 🛠️ Arquitectura del Entorno

El proyecto está dividido en tres módulos principales desacoplados:

1. **Backend Cuántico (`quantum_solver.py`):** Utiliza Qiskit para transformar la topología de la red en un problema QUBO, generar el circuito parametrizado Ansatz de QAOA y optimizar las compuertas para encontrar el autoestado de mínima energía.
2. **Backend Clásico (`classical_solver.py`):** Proporciona el *Ground Truth* mediante búsqueda exhaustiva (Fuerza Bruta) y heurística física (Simulated Annealing) para validar rigurosamente la precisión del solucionador cuántico.
3. **Frontend Interactivo (`app.py`):** Interfaz web construida con Streamlit que permite al usuario definir la topología de las antenas, ejecutar los solucionadores en paralelo y visualizar las métricas de rendimiento y la coloración del grafo.

## 💻 Stack Tecnológico
* **Computación Cuántica:** `qiskit`, `qiskit-optimization`, `qiskit-aer`
* **Cálculo y Redes:** `numpy`, `scipy`, `networkx`
* **Interfaz Visual:** `streamlit`, `matplotlib`
