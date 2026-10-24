"""PERSONA 2 - QAOA: algoritmo cuantico-hibrido, en simulador local."""
import time
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from .formulacion import hamiltoniano, interferencia

# ===================== TUS PARAMETROS (Persona 2) =====================
PARAMETROS = {
    "REPS": 3,              # capas del circuito: mas capas = mejor calidad, circuito mas largo
    "SHOTS": 4000,          # cuantas veces se mide el circuito al final
    "OPTIMIZADOR": "COBYLA",  # optimizador clasico de scipy (prueba tambien "Powell")
    "MAX_ITER": 300,        # tope de iteraciones del optimizador
    "SEMILLA": 7,           # fija el azar para que el resultado se pueda repetir
}
# =====================================================================


def bits_a_canales(bitstring):
    """Qiskit escribe el qubit 0 a la DERECHA de la cadena: hay que darle la vuelta."""
    return [int(b) for b in reversed(bitstring)]


def resolver_qaoa(red, params=PARAMETROS):
    t0 = time.perf_counter()
    # 1) El problema en lenguaje cuantico (lo entrega la Persona 1)
    _, H, _ = hamiltoniano(red)
    # 2) El circuito QAOA con REPS capas; tiene 2*REPS angulos por ajustar
    ansatz = QAOAAnsatz(cost_operator=H, reps=params["REPS"])
    estimator = StatevectorEstimator()
    historial = []

    # 3) Funcion de costo: el circuito mide la "energia" con ciertos angulos
    def energia(angulos):
        ev = float(estimator.run([(ansatz, H, angulos)]).result()[0].data.evs)
        historial.append(ev)
        return ev

    # 4) El optimizador clasico cambia los angulos para bajar la energia (parte hibrida)
    rng = np.random.default_rng(params["SEMILLA"])
    inicio = rng.uniform(0, np.pi, ansatz.num_parameters)
    opt = minimize(energia, inicio, method=params["OPTIMIZADOR"], tol=1e-3,
                   options={"maxiter": params["MAX_ITER"]})

    # 5) Con los mejores angulos, se mide el circuito SHOTS veces
    circuito = ansatz.assign_parameters(opt.x)
    circuito.measure_all()
    sampler = StatevectorSampler(seed=params["SEMILLA"])
    counts = sampler.run([circuito], shots=params["SHOTS"]).result()[0].data.meas.get_counts()

    # 6) Traducir lo medido a repartos de canales y quedarnos con dos respuestas
    muestras = [(bits_a_canales(b), c / params["SHOTS"]) for b, c in counts.items()]
    x_probable = max(muestras, key=lambda m: m[1])[0]                 # la que mas salio
    x_mejor = min(muestras, key=lambda m: interferencia(red, m[0]))[0]  # la mejor que salio
    return {"metodo": "QAOA", "x": x_mejor, "costo": interferencia(red, x_mejor),
            "tiempo_s": time.perf_counter() - t0,
            "extra": {"x_mas_probable": x_probable,
                      "costo_mas_probable": interferencia(red, x_probable),
                      "muestras": muestras, "historial": historial,
                      "qubits": H.num_qubits, "profundidad": ansatz.decompose().depth(),
                      "evaluaciones": len(historial), "parametros": dict(params)}}
