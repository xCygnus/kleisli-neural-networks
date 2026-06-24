"""
test_monad_laws.py
==================

Verificación numérica de las leyes de la mónada de estado.

Para cualquier mónada (T, eta, mu), las tres leyes son:

  (M1) Identidad izquierda :   eta(a) >>= g   ==   g(a)
  (M2) Identidad derecha   :   m >>= eta       ==   m
  (M3) Asociatividad       :   (m >>= g) >>= h ==   m >>= (\\a. g(a) >>= h)

Como en Python no hay igualdad extensional de funciones, comprobamos
igualdad punto-a-punto sobre un conjunto representativo de estados.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from state_monad import eta, bind_directo as bind


def igual_state_m(m1, m2, estados):
    """Igualdad punto-a-punto sobre una muestra de estados."""
    return all(m1(s) == m2(s) for s in estados)


# Muestras
ESTADOS = list(range(-5, 6))
VALORES = list(range(-5, 6))


def test_identidad_izquierda():
    """ eta(a) >>= g  ==  g(a) """
    def g(x):
        return lambda s: (x * 2, s + 1)

    for a in VALORES:
        izq = bind(eta(a), g)
        der = g(a)
        assert igual_state_m(izq, der, ESTADOS), f"(M1) falla para a={a}"


def test_identidad_derecha():
    """ m >>= eta  ==  m """
    def m(s):
        return (s * 3 + 1, s - 2)

    izq = bind(m, eta)
    assert igual_state_m(izq, m, ESTADOS), "(M2) falla"


def test_asociatividad():
    """ (m >>= g) >>= h  ==  m >>= (\\a. g(a) >>= h) """
    def m(s):
        return (s + 1, s * 2)

    def g(x):
        return lambda s: (x * 10, s + 7)

    def h(y):
        return lambda s: (y - 3, s * -1)

    izq = bind(bind(m, g), h)
    der = bind(m, lambda a: bind(g(a), h))

    assert igual_state_m(izq, der, ESTADOS), "(M3) falla"


if __name__ == "__main__":
    test_identidad_izquierda()
    test_identidad_derecha()
    test_asociatividad()
    print("test_monad_laws.py: las tres leyes monádicas se cumplen.")
