"""PERSONA 1 - Traducir la red al lenguaje del computador cuantico."""
import rustworkx as rx
from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising


def interferencia(red, x):
    """Suma los pesos de las parejas que se estorban y tienen el MISMO canal. Menos es mejor."""
    return float(sum(w for i, j, w in red["interf"] if x[i] == x[j]))


def interferencia_total_posible(red):
    """W: la interferencia si todas las antenas usaran el mismo canal (el peor caso)."""
    return float(sum(w for _, _, w in red["interf"]))


def grafo_interferencia(red):
    g = rx.PyGraph()
    g.add_nodes_from(range(red["n"]))
    g.add_edges_from([(i, j, float(w)) for i, j, w in red["interf"]])
    return g


def hamiltoniano(red):
    """Red -> MaxCut ponderado -> QUBO -> hamiltoniano de Ising (receta del tutorial de IBM).
    Funciona porque: interferencia = W - corte."""
    problema = Maxcut(grafo_interferencia(red)).to_optimization_problem()
    qubo = OptimizationProblemToQubo().convert(problema)
    H, offset = to_ising(qubo)
    H = H.simplify()          # quita terminos con coeficiente ~0 (errores de redondeo)
    return problema, H, offset
