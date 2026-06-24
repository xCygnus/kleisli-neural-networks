"""
state_monad.py
==============

Implementación de la mónada de estado T(A) = S -> (A x S) sobre Set.

Correspondencia con la tesis (Capítulo de implementación):

  - Definición \\ref{def:state}: la mónada (T, eta, mu)
  - Proposición \\ref{prop:bind-state}: la forma operacional de bind
  - Listado \\ref{lst:impl-state}: este mismo archivo

Cada función está en correspondencia uno-a-uno con un elemento de la
definición. No se usan variables globales mutables: el estado fluye
explícitamente como argumento.
"""

from __future__ import annotations
from typing import Callable, Tuple, TypeVar

A = TypeVar("A")
B = TypeVar("B")
S = TypeVar("S")

# Un elemento de T(A) es una función  s |-> (a, s')
StateM = Callable[[S], Tuple[A, S]]


def eta(a: A) -> StateM:
    """Unidad de la mónada de estado.

        eta_A : A -> T(A),     eta(a) = (s |-> (a, s))

    Mete `a` en la mónada sin tocar el estado.

    >>> eta(7)(100)
    (7, 100)
    """
    return lambda s: (a, s)


def T_map(g: Callable[[A], B], m: StateM) -> StateM:
    """Acción del funtor T sobre morfismos.

        T(g) : T(A) -> T(B)
        T(g)(m) = (s |-> sea (a, s') = m(s); (g(a), s'))

    Ejecuta `m`, aplica `g` solo al valor producido y deja el estado intacto.

    >>> m = lambda s: (s * 2, s)
    >>> T_map(lambda a: a + 1, m)(5)
    (11, 5)
    """
    def resultado(s):
        a, s_prime = m(s)
        return (g(a), s_prime)
    return resultado


def mu(M: Callable[[S], Tuple[StateM, S]]) -> StateM:
    """Multiplicación de la mónada de estado.

        mu_A : T(T(A)) -> T(A)
        mu(M) = (s |-> sea (m, s') = M(s); m(s'))

    Aplana ejecutando la receta interna con el estado intermedio.
    """
    def resultado(s):
        m, s_prime = M(s)
        return m(s_prime)
    return resultado


def bind(m: StateM, g: Callable[[A], StateM]) -> StateM:
    """Composición de Kleisli vía la definición canónica.

        m >>= g  :=  mu( T(g)(m) )
    """
    return mu(T_map(g, m))


def bind_directo(m: StateM, g: Callable[[A], StateM]) -> StateM:
    """Forma operacional desarrollada (Proposición \\ref{prop:bind-state}).

        (m >>= g)(s) = sea (a, s') = m(s); g(a)(s')

    Es punto-a-punto idéntica a `bind`, pero más legible.
    """
    def resultado(s):
        a, s1 = m(s)
        return g(a)(s1)
    return resultado


# ---------------------------------------------------------------------------
# Demostración: contador puro
# ---------------------------------------------------------------------------
def incrementar() -> StateM:
    """Suma 1 al estado, devuelve None como valor."""
    return lambda s: (None, s + 1)


def leer_estado() -> StateM:
    """Expone el estado actual como valor (sin modificarlo)."""
    return lambda s: (s, s)


if __name__ == "__main__":
    # Tres incrementos encadenados vía bind. La transparencia referencial
    # se manifiesta en que ejecutar `programa` dos veces con el mismo
    # estado inicial da exactamente el mismo resultado.
    programa = bind_directo(
        incrementar(),
        lambda _: bind_directo(
            incrementar(),
            lambda _: bind_directo(incrementar(), lambda _: leer_estado()),
        ),
    )

    assert programa(0) == (3, 3)
    assert programa(0) == (3, 3)   # idempotencia: pureza
    assert programa(10) == (13, 13)
    print("state_monad.py: OK")
