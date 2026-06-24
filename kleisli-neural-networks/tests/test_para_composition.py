"""
test_para_composition.py
========================

Verifica que la composición de morfismos en Para(Lens) cumple
las propiedades estructurales esperadas:

  1. La composición es asociativa.
  2. Los espacios de parámetros se acumulan como producto tensorial.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from para_lens import ParaLens


def capa_lineal(n_in: int, n_out: int, seed: int) -> ParaLens:
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((n_out, n_in)) * 0.3
    b0 = rng.standard_normal(n_out) * 0.1

    def fwd(a, p):
        W, b = p
        return W @ a + b

    def bwd(a, p, b_star):
        W, _ = p
        return W.T @ b_star, (np.outer(b_star, a), b_star)

    return ParaLens(fwd, bwd, (W0, b0))


def contar_parametros(params):
    """Cuenta el número total de escalares en una tupla anidada de arrays."""
    if isinstance(params, tuple):
        return sum(contar_parametros(p) for p in params)
    return params.size


def test_acumulacion_de_parametros():
    """ P(f >> g) = P(f) x P(g) :  los parámetros se acumulan estructuralmente. """
    f = capa_lineal(2, 4, seed=1)   # 4*2 + 4 = 12 parámetros
    g = capa_lineal(4, 3, seed=2)   # 3*4 + 3 = 15 parámetros

    compuesta = f >> g
    n_total = contar_parametros(compuesta.init_params)
    assert n_total == 12 + 15, f"esperaba 27, obtenido {n_total}"


def test_asociatividad():
    """(f >> g) >> h  evaluado igual que  f >> (g >> h). """
    f = capa_lineal(2, 3, seed=11)
    g = capa_lineal(3, 4, seed=22)
    h = capa_lineal(4, 2, seed=33)

    # Composición por la izquierda
    izq = (f >> g) >> h
    # Por la derecha
    der = f >> (g >> h)

    x = np.array([0.5, -1.0])

    # Hay que reorganizar los parámetros: la asociatividad de las tuplas
    # difiere por la forma anidada, pero el "contenido" es el mismo.
    # Comparamos sobre el forward usando los parámetros nativos de cada
    # ordenación: ambos resultados deben coincidir.
    y_izq = izq.forward(x, izq.init_params)
    y_der = der.forward(x, der.init_params)

    assert np.allclose(y_izq, y_der), "Forward no asociativo"


if __name__ == "__main__":
    test_acumulacion_de_parametros()
    test_asociatividad()
    print("test_para_composition.py: composición de Para verificada.")
