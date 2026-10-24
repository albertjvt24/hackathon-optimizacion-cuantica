import json
import math
import time
from pathlib import Path
import numpy as np
import networkx as nx


def cargar(path):
    red = json.loads(Path(path).read_text(encoding='utf-8'))
    validar(red)
    return red


def validar(red):
    n = red.get('n')
    if type(n) is not int or not 1 <= n <= 12:
        raise ValueError('Esta demo admite entre 1 y 12 antenas (enumeración exponencial).')
    for key in ('inicial', 'costo_cambio', 'pos'):
        if len(red.get(key, [])) != n:
            raise ValueError(f'{key}: se necesita un elemento por antena.')
    if any(type(x) is not int or x not in (0, 1) for x in red['inicial']):
        raise ValueError('Los canales iniciales deben ser enteros 0 o 1.')
    for c in red['costo_cambio']:
        no_negativo(c, 'costo_cambio')
    for p in red['pos']:
        if len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
            raise ValueError('Cada posición debe contener dos números finitos.')
    for key in ('interf', 'com'):
        seen = set()
        for edge in red.get(key, []):
            if len(edge) != 3:
                raise ValueError(f'Arista incorrecta en {key}.')
            i, j, w = edge
            if any(type(v) is not int or not 0 <= v < n for v in (i, j)) or i == j:
                raise ValueError(f'Índices incorrectos en {key}.')
            pair = tuple(sorted((i, j)))
            if pair in seen:
                raise ValueError(f'Arista duplicada en {key}: {pair}.')
            seen.add(pair)
            no_negativo(w, key)
    for key in ('origen', 'destino'):
        if type(red.get(key)) is not int or not 0 <= red[key] < n:
            raise ValueError(f'{key}: índice fuera de rango.')
    no_negativo(red.get('lambda_cambio', 1), 'lambda_cambio')
    no_negativo(red.get('alpha', .5), 'alpha')


def no_negativo(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError(f'{name} debe ser un número finito no negativo.')


def componentes(red, x):
    x = np.asarray(x)
    if x.shape != (red['n'],) or not np.isin(x, [0, 1]).all():
        raise ValueError('Asignación inválida.')
    interferencia = sum(w for i, j, w in red['interf'] if x[i] == x[j])
    cambios = x != np.array(red['inicial'])
    return float(interferencia), float(np.dot(cambios, red['costo_cambio'])), int(cambios.sum())


def costo(red, x, lam):
    i, c, _ = componentes(red, x)
    return i + lam * c


def enumerar(red):
    n = red['n']
    xs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    inter = np.zeros(len(xs))
    for i, j, w in red['interf']:
        inter += w * (xs[:, i] == xs[:, j])
    cambios = (xs != np.array(red['inicial'])) @ np.array(red['costo_cambio'], dtype=float)
    return xs, inter, cambios


def resultado(red, x, lam, nombre, tiempo=0):
    i, c, nc = componentes(red, x)
    return dict(metodo=nombre, x=list(map(int, x)), interferencia=i,
                costo_cambio=c, penalizacion=lam*c, cambios=nc, total=i+lam*c,
                tiempo_s=tiempo)


def exacto(red, lam):
    validar(red)
    no_negativo(lam, 'lambda')
    t = time.perf_counter()
    xs, inter, cambios = enumerar(red)
    f = inter + lam*cambios
    # En empate numérico: preferir menor costo de cambio, luego menos cambios.
    candidatos = np.flatnonzero(np.isclose(f, f.min(), atol=1e-10, rtol=0))
    k = min(candidatos, key=lambda k: (cambios[k], sum(xs[k] != red['inicial']), int(k)))
    return resultado(red, xs[k], lam, 'Exacto', time.perf_counter()-t)


def local(red, lam, semilla=7, reinicios=20):
    validar(red)
    no_negativo(lam, 'lambda')
    if type(reinicios) is not int or reinicios < 1:
        raise ValueError('reinicios debe ser un entero positivo.')
    t = time.perf_counter()
    rng = np.random.default_rng(semilla)
    best = np.array(red['inicial'])
    for r in range(reinicios):
        x = np.array(red['inicial']) if r == 0 else rng.integers(0, 2, red['n'])
        while True:
            actual = costo(red, x, lam)
            propuestas = [x.copy() for _ in x]
            for i, y in enumerate(propuestas):
                y[i] ^= 1
            y = min(propuestas, key=lambda v: costo(red, v, lam))
            if costo(red, y, lam) >= actual - 1e-10:
                break
            x = y
        if costo(red, x, lam) < costo(red, best, lam):
            best = x.copy()
    return resultado(red, best, lam, 'Búsqueda local', time.perf_counter()-t)


def voraz(red, lam):
    validar(red)
    no_negativo(lam, 'lambda')
    t = time.perf_counter()
    n = red['n']
    grado = [sum(w for i,j,w in red['interf'] if v in (i,j)) for v in range(n)]
    x = [None]*n
    for v in sorted(range(n), key=lambda v: (-grado[v],v)):
        def parcial(c):
            conflictos = sum(w for i,j,w in red['interf']
                             if (i == v and x[j] == c) or (j == v and x[i] == c))
            return conflictos + lam*red['costo_cambio'][v]*(c != red['inicial'][v])
        x[v] = min((0,1), key=lambda c: (parcial(c), c != red['inicial'][v], c))
    return resultado(red,x,lam,'Voraz',time.perf_counter()-t)


def ruta(red, x):
    molestias = [0.]*red['n']
    for i,j,w in red['interf']:
        if x[i] == x[j]:
            molestias[i] += w
            molestias[j] += w
    g = nx.Graph()
    g.add_nodes_from(range(red['n']))
    for i,j,base in red.get('com',[]):
        g.add_edge(i,j,weight=base+red.get('alpha',.5)*(molestias[i]+molestias[j]))
    try:
        distancia,camino = nx.single_source_dijkstra(g,red['origen'],red['destino'],weight='weight')
        return {'camino':camino,'costo':float(distancia)}
    except nx.NetworkXNoPath:
        return {'camino':[],'costo':None}


def comparar(red, lam, semilla=7, reinicios=20):
    validar(red)
    no_negativo(lam,'lambda')
    filas = [resultado(red,red['inicial'],lam,'Inicial'),exacto(red,lam),
             voraz(red,lam),local(red,lam,semilla,reinicios)]
    for r in filas:
        r['brecha'] = r['total']-filas[1]['total']
        r['ruta'] = ruta(red,r['x'])
    return filas
