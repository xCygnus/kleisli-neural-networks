"""
test_gradient_correctness.py
============================

Verifica numéricamente que el backward de un ParaLens compuesto coincide
con la diferenciación por diferencias finitas. Esto certifica que la
"regla de la cadena emergente" de la composición de lentes es
analíticamente correcta — no por construcción manual, sino como
consecuencia de la composición categórica.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from para_lens import ParaLens


def capa_tanh(n_in: int, n_out: int, seed: int) -> ParaLens:
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((n_out, n_in)) * 0.3
    b0 = rng.standard_normal(n_out) * 0.1

    def fwd(a, p):
        W, b = p
        return np.tanh(W @ a + b)

    def bwd(a, p, b_star):
        W, b = p
        z = W @ a + b
        y = np.tanh(z)
        dz = b_star * (1.0 - y ** 2)
        return W.T @ dz, (np.outer(dz, a), dz)

    return ParaLens(fwd, bwd, (W0, b0))


def gradiente_numerico(red, x, params, idx_param, idx_componente, eps=1e-6):
    """Derivada parcial de la suma de la salida respecto a un componente."""
    # idx_param: 0 -> W, 1 -> b   en el último nivel de la tupla anidada
    def aplanar(p, ruta):
        if isinstance(p, tuple):
            return [(ruta + (i,), q) for i, sub in enumerate(p) for ruta_q, q in aplanar(sub, ruta + (i,))]
        return [(ruta, p)]

    # Localizamos el array y su posición
    rutas = aplanar(params, ())
    ruta, arr = rutas[idx_param]
    arr_flat = arr.flatten()
    valor_original = arr_flat[idx_componente]

    def set_componente(params, ruta, idx, valor):
        # Reconstruye la tupla con un valor cambiado
        if not ruta:
            nuevo = params.copy()
            nuevo_flat = nuevo.flatten()
            nuevo_flat[idx] = valor
            return nuevo_flat.reshape(params.shape)
        i, *resto = ruta
        return tuple(
            set_componente(p, tuple(resto), idx, valor) if j == i else p
            for j, p in enumerate(params)
        )

    p_mas = set_componente(params, ruta, idx_componente, valor_original + eps)
    p_men = set_componente(params, ruta, idx_componente, valor_original - eps)

    f_mas = red.forward(x, p_mas).sum()
    f_men = red.forward(x, p_men).sum()
    return (f_mas - f_men) / (2 * eps)


def test_gradiente_composicion():
    """Compara backward analítico vs. diferencias finitas para una red 2->3->2."""
    red = capa_tanh(2, 3, seed=10) >> capa_tanh(3, 2, seed=20)

    x = np.array([0.7, -0.3])
    params = red.init_params

    # Gradiente analítico (vector de unos en la salida -> derivada de la suma)
    salida = red.forward(x, params)
    a_star, p_star = red.backward(x, params, np.ones_like(salida))

    # Aplanar p_star en el mismo orden que aplanamos params
    # Estructura: ((W1, b1), (W2, b2))   -> 4 arrays
    arrays_grad = [p_star[0][0], p_star[0][1], p_star[1][0], p_star[1][1]]

    # Verificar componente a componente
    rutas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    errores = []
    for k, (i, j) in enumerate(rutas):
        arr_grad = arrays_grad[k]
        for idx in range(arr_grad.size):
            # Construir gradiente numérico
            params_lista = [params[0][0], params[0][1], params[1][0], params[1][1]]
            arr = params_lista[k]
            val_orig = arr.flat[idx]

            arr.flat[idx] = val_orig + 1e-6
            f_mas = red.forward(x, params).sum()
            arr.flat[idx] = val_orig - 1e-6
            f_men = red.forward(x, params).sum()
            arr.flat[idx] = val_orig

            num = (f_mas - f_men) / 2e-6
            ana = arr_grad.flat[idx]
            errores.append(abs(num - ana))

    err_max = max(errores)
    print(f"Error máximo entre backward analítico y numérico: {err_max:.2e}")
    assert err_max < 1e-6, "El gradiente analítico discrepa de las diferencias finitas"


if __name__ == "__main__":
    test_gradiente_composicion()
    print("test_gradient_correctness.py: backward correcto al nivel de diferencias finitas.")
