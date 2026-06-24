"""
02_xor_completo.py
==================

Red XOR entrenada como evaluación de un diagrama de cuerdas de DisCoPy.

Reúne las tres capas teóricas de la tesis:

  1. Sintaxis     -> diagrama de DisCoPy en la categoría libre Cat_libre(Σ)
  2. Semántica    -> funtor a Para(Lens(Set))
  3. Ejecución    -> entrenamiento como morfismo de Kleisli de la mónada
                     de estado (no hay variables globales mutables)

Correspondencia con la tesis:
  - Listado \\ref{lst:xor-completo}
  - Nota \\ref{nota:recap-impl}

Ejecutar:
    python examples/02_xor_completo.py
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from discopy.symmetric import Ty, Box

from para_lens import ParaLens
from kleisli_train import entrenar


# ---------------------------------------------------------------------------
# (1) SINTAXIS: arquitectura como diagrama de cuerdas
# ---------------------------------------------------------------------------
R2, R4, R1 = Ty("R2"), Ty("R4"), Ty("R1")

caja_oculta = Box("oculta", R2, R4)
caja_salida = Box("salida", R4, R1)

red_diagrama = caja_oculta >> caja_salida
print("Diagrama sintáctico:", red_diagrama)


# ---------------------------------------------------------------------------
# (2) SEMÁNTICA: funtor (manual) a Para(Lens(Set))
#
# Nota: en discopy 1.2.2 los Functor a categorías arbitrarias requieren
# definir la categoría destino con su tipo de morfismo. Para mantener el
# código pedagógico, aplicamos aquí el funtor "a mano": componemos los
# ParaLens correspondientes a cada caja con el operador >> de ParaLens.
# Esto es estrictamente lo mismo que la imagen funtorial del diagrama,
# por la propiedad universal de Cat_libre(Σ).
# ---------------------------------------------------------------------------
def capa_densa_sigmoide(n_in: int, n_out: int, seed: int = 0) -> ParaLens:
    """Capa afín con activación sigmoide como morfismo de Para(Lens)."""
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    dsig = lambda y: y * (1.0 - y)

    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((n_out, n_in)) * 0.7
    b0 = np.zeros(n_out)

    def fwd(a, p):
        W, b = p
        return sig(W @ a + b)

    def bwd(a, p, b_star):
        W, b = p
        y = sig(W @ a + b)
        dz = b_star * dsig(y)                # delta local
        return W.T @ dz, (np.outer(dz, a), dz)

    return ParaLens(fwd, bwd, (W0, b0))


# Asignación de cajas a morfismos de Para(Lens): el "diccionario ar" del funtor.
semantica_cajas = {
    caja_oculta: capa_densa_sigmoide(2, 4, seed=1),
    caja_salida: capa_densa_sigmoide(4, 1, seed=2),
}


def aplicar_funtor(diagrama, ar):
    """Implementación didáctica de F(diagrama) por recursión estructural.

    Esto es exactamente lo que hace discopy.symmetric.Functor cuando la
    categoría destino soporta >> y @: recorre el diagrama y compone las
    imágenes de cada caja.
    """
    cajas = diagrama.boxes
    img = [ar[c] for c in cajas]
    resultado = img[0]
    for siguiente in img[1:]:
        resultado = resultado >> siguiente
    return resultado


red = aplicar_funtor(red_diagrama, semantica_cajas)


# ---------------------------------------------------------------------------
# (3) EJECUCIÓN: entrenamiento como morfismo de Kleisli
# ---------------------------------------------------------------------------
XOR = [
    (np.array([0.0, 0.0]), np.array([0.0])),
    (np.array([0.0, 1.0]), np.array([1.0])),
    (np.array([1.0, 0.0]), np.array([1.0])),
    (np.array([1.0, 1.0]), np.array([0.0])),
]

pesos_0 = red.init_params              # estado inicial S_0
print("\nEntrenando XOR durante 5000 épocas...")
pesos_f, historia = entrenar(
    red, XOR, epocas=5000, pesos_iniciales=pesos_0, lr=0.5, verbose=True
)

print("\nPredicciones finales:")
for x, y in XOR:
    pred = red.forward(x, pesos_f)
    print(f"  x = {x}   ->  pred = {float(pred[0]):.4f}   (objetivo {float(y[0]):.0f})")

# Transparencia referencial: evaluar dos veces con el mismo estado da el
# mismo resultado. No hay estado oculto en `red`; todo vive en `pesos_f`.
p1 = red.forward(XOR[1][0], pesos_f)
p2 = red.forward(XOR[1][0], pesos_f)
assert np.allclose(p1, p2)
print("\nTransparencia referencial: OK (mismas entradas, mismas salidas).")
