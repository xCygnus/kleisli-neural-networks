"""
01_discopy_hello.py
===================

Ejemplo introductorio de DisCoPy (Ejemplo \\ref{ej:discopy-hello} de la tesis).

Construye el diagrama más simple posible — dos cajas f, g compuestas
secuencialmente — y lo evalúa bajo dos funtores distintos para ilustrar
la separación sintaxis / semántica.

Ejecutar:
    python examples/01_discopy_hello.py
"""

from discopy.symmetric import Ty, Box, Functor
from discopy.cat import Category
from discopy.python import Function


# ---------------------------------------------------------------------------
# (1) SINTAXIS: tipos y cajas generan la categoría monoidal libre Cat_libre(Σ).
# Aquí Σ_0 = {A, B, C} y Σ_1 = {f, g}.
# ---------------------------------------------------------------------------
A, B, C = Ty("A"), Ty("B"), Ty("C")

f = Box("f", A, B)
g = Box("g", B, C)

diagrama = f >> g          # composición secuencial en la categoría libre
print("Diagrama sintáctico:", diagrama)
print("  dominio :", diagrama.dom)
print("  codominio:", diagrama.cod)


# ---------------------------------------------------------------------------
# (2) PRIMERA SEMÁNTICA: funtor a la categoría de funciones de Python.
#     F envía f a "sumar 1" y g a "multiplicar por 2".
# ---------------------------------------------------------------------------
F = Functor(
    ob={A: (int,), B: (int,), C: (int,)},
    ar={f: lambda x: (x + 1,),
        g: lambda x: (x * 2,)},
    cod=Category(tuple[type, ...], Function),
)

programa = F(diagrama)
print("\nF(diagrama)(3) =", programa(3), "   # esperado: 8 = (3+1)*2")


# ---------------------------------------------------------------------------
# (3) SEGUNDA SEMÁNTICA: el mismo diagrama, distinta interpretación.
#     F' envía f a "duplicar" y g a "restar 5".
# ---------------------------------------------------------------------------
F_alt = Functor(
    ob={A: (int,), B: (int,), C: (int,)},
    ar={f: lambda x: (x * 2,),
        g: lambda x: (x - 5,)},
    cod=Category(tuple[type, ...], Function),
)

print("F'(diagrama)(3) =", F_alt(diagrama)(3), "   # esperado: 1 = 3*2 - 5")
print("\nLa MISMA arquitectura sintáctica produce DOS programas distintos")
print("según el funtor que la interpreta. Esta es la propiedad universal")
print("de la categoría libre Cat_libre(Σ).")
