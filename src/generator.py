import numpy as np

def generar_red_demo(n):
    rng = np.random.default_rng(42)
    pos = rng.uniform(0, 10, (n, 2)).tolist()

    inicial = rng.integers(0, 2, n).tolist()
    
    costo_cambio = rng.uniform(0.5, 2.0, n).tolist()
    
    interf = []
    com = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(np.array(pos[i]) - np.array(pos[j])))
            if d < 4.0:
                interf.append([i, j, float(round(10/d, 2))])
            com.append([i, j, float(round(d, 2))])
            
    return {
        'n': n, 
        'inicial': inicial, 
        'costo_cambio': costo_cambio,
        'pos': pos, 
        'interf': interf, 
        'com': com,
        'origen': 0, 
        'destino': n - 1,
        'lambda_cambio': 1.0, 
        'alpha': 0.5
    }
