"""
kleisli_train.py
================

Entrenamiento de una red ParaLens expresado como morfismo de Kleisli
de la mónada de estado.

Correspondencia con la tesis:

  - Definición \\ref{def:capa-kleisli}: capa como morfismo de Kleisli
  - Listado \\ref{lst:kleisli-train}: este archivo

El estado S contiene la tupla anidada de todos los pesos de la red.
Cada paso de entrenamiento es un elemento de

    T(R) = S -> (R x S),

que devuelve (loss, estado_actualizado). La composición de pasos a lo
largo de una época es la composición de Kleisli  fish  desarrollada
operacionalmente como  (m, s) |-> (loss_acc, s_nuevo).

NO se mutan variables globales: el estado se pasa explícitamente.
"""

from __future__ import annotations
from typing import Callable, Tuple, Any, List
import numpy as np

from para_lens import ParaLens

Estado = Any                                    # estado anidado de pesos
PasoKleisli = Callable[[Estado], Tuple[float, Estado]]


def aplicar_gradiente(s: Estado, grad: Estado, lr: float) -> Estado:
    """Resta lr * grad de cada peso, respetando la estructura anidada.

    Funciona sobre tuplas anidadas porque la composición secuencial de
    ParaLens acumula los espacios de parámetros como productos tensoriales
    P x Q (Definición \\ref{def:Para}).
    """
    if isinstance(s, tuple):
        return tuple(aplicar_gradiente(si, gi, lr) for si, gi in zip(s, grad))
    return s - lr * grad


def paso_entrenamiento(
    red: ParaLens,
    x: np.ndarray,
    y_obj: np.ndarray,
    lr: float = 0.1,
) -> PasoKleisli:
    """Construye un morfismo de Kleisli  S -> (loss x S) para un par (x, y).

    El gradiente inicial del backward es el de la pérdida cuadrática media
    d/dy_pred [(y_pred - y_obj)^2] = 2 (y_pred - y_obj).
    """
    def kleisli_morfismo(s: Estado) -> Tuple[float, Estado]:
        y_pred = red.forward(x, s)                # 1. forward
        b_star = 2.0 * (y_pred - y_obj)           # 2. error de la pérdida
        _, p_star = red.backward(x, s, b_star)    # 3. backward
        s_nuevo = aplicar_gradiente(s, p_star, lr)  # 4. nuevo estado, sin mutar s
        loss = float(np.mean((y_pred - y_obj) ** 2))
        return (loss, s_nuevo)
    return kleisli_morfismo


def entrenar(
    red: ParaLens,
    datos: List[Tuple[np.ndarray, np.ndarray]],
    epocas: int,
    pesos_iniciales: Estado,
    lr: float = 0.1,
    verbose: bool = False,
) -> Tuple[Estado, List[float]]:
    """Entrena la red encadenando pasos vía la composición de Kleisli.

    Cada par (x, y) del dataset produce un paso de Kleisli; los pasos se
    encadenan por la pauta operacional  `loss, estado = m(estado)`, que es
    exactamente  (m fish g)(s) = g(a)(s')  de la Proposición \\ref{prop:bind-state}.
    """
    estado = pesos_iniciales
    historico: List[float] = []

    for epoca in range(epocas):
        loss_total = 0.0
        for x, y in datos:
            m = paso_entrenamiento(red, x, y, lr)
            loss, estado = m(estado)              # composición de Kleisli
            loss_total += loss
        loss_media = loss_total / len(datos)
        historico.append(loss_media)

        if verbose and (epoca % max(1, epocas // 10) == 0 or epoca == epocas - 1):
            print(f"  época {epoca:5d}   loss = {loss_media:.6f}")

    return estado, historico
