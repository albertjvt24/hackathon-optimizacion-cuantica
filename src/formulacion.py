import rustworkx as rx
from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising


def interferencia(red, x):
    return float(sum(w for i, j, w in red["interf"] if x[i] == x[j]))


def interferencia_total_posible(red):
    return float(sum(w for _, _, w in red["interf"]))


def grafo_interferencia(red):
    g = rx.PyGraph()
    g.add_nodes_from(range(red["n"]))
    g.add_edges_from([(i, j, float(w)) for i, j, w in red["interf"]])
    return g


def hamiltoniano(red):
    problema = Maxcut(grafo_interferencia(red)).to_optimization_problem()
    qubo = OptimizationProblemToQubo().convert(problema)
    H, offset = to_ising(qubo)
    H = H.simplify() 
    return problema, H, offset
