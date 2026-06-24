"""
para_lens.py
============

Morfismos parametrizados de lentes en Para(Lens(C)).

Correspondencia con la tesis:

  - Definición \\ref{def:Para}: categoría parametrizada Para(C)
  - Definición \\ref{def:Lens}: lentes (forward / backward)
  - Definición \\ref{def:Para-Lens}: morfismos en Para(Lens(C))
  - Listado \\ref{lst:para-lens}: este archivo
  - Nota \\ref{nota:cadena-emergente}: la regla de la cadena emerge de
    la composición, no se programa a mano

Un ParaLens representa un morfismo

    (A, A*) --(P, P*)--> (B, B*)

descompuesto en dos mapas:

    forward  : A x P            -> B           (predicción)
    backward : A x P x B*       -> A* x P*     (aprendizaje)

La firma del backward es la forma no-currificada del morfismo cuyo
currificado vía la adjunción tensor-hom da

    B* -> [A x P, A* x P*]

(véase Sección \\ref{sec:lentes} de la tesis).
"""

from __future__ import annotations
from typing import Callable, Any


class ParaLens:
    """Morfismo parametrizado de lentes.

    Atributos
    ---------
    forward     : (a, p) -> b
    backward    : (a, p, b_star) -> (a_star, p_star)
    init_params : valor inicial del espacio de parámetros P
    """

    def __init__(
        self,
        forward: Callable[[Any, Any], Any],
        backward: Callable[[Any, Any, Any], tuple],
        init_params: Any,
    ):
        self.forward = forward
        self.backward = backward
        self.init_params = init_params

    # ------------------------------------------------------------------
    # Composición secuencial en Para(Lens(C)).
    # Acumula parámetros como P x Q (Definición \ref{def:Para}).
    # ------------------------------------------------------------------
    def __rshift__(self, otro: "ParaLens") -> "ParaLens":
        """Composición  self >> otro : (A, A*) --(P x Q, P* x Q*)--> (C, C*).

        El backward compuesto NO programa la regla de la cadena: la regla
        de la cadena es el contenido semántico de la composición de
        lentes, y se obtiene aquí simplemente encadenando los dos
        backwards en el orden inverso al de los forwards.
        """
        f, finv = self.forward, self.backward
        g, ginv = otro.forward, otro.backward

        def fwd(a, params):
            p, q = params
            b = f(a, p)
            c = g(b, q)
            return c

        def bwd(a, params, c_star):
            p, q = params
            b = f(a, p)                          # recompute para el backward
            b_star, q_star = ginv(b, q, c_star)  # primero g*
            a_star, p_star = finv(a, p, b_star)  # luego f*
            return a_star, (p_star, q_star)      # gradiente acumulado P* x Q*

        return ParaLens(fwd, bwd, (self.init_params, otro.init_params))


if __name__ == "__main__":
    import numpy as np

    # Test mínimo: composición de dos capas afines sin activación.
    def afin(n_in, n_out, seed=0):
        rng = np.random.default_rng(seed)
        W0 = rng.standard_normal((n_out, n_in)) * 0.3
        b0 = np.zeros(n_out)

        def fwd(a, p):
            W, b = p
            return W @ a + b

        def bwd(a, p, b_star):
            W, _ = p
            return W.T @ b_star, (np.outer(b_star, a), b_star)

        return ParaLens(fwd, bwd, (W0, b0))

    red = afin(3, 4, seed=1) >> afin(4, 2, seed=2)
    x = np.array([1.0, -2.0, 0.5])
    y = red.forward(x, red.init_params)
    a_star, p_star = red.backward(x, red.init_params, np.array([1.0, -1.0]))

    assert y.shape == (2,)
    assert a_star.shape == (3,)
    print("para_lens.py: OK,  y =", y)
