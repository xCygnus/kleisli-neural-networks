# kleisli-neural-networks

> **Redes neuronales como morfismos de Kleisli: una implementación categórica con DisCoPy.**
>
> Código de acompañamiento de la tesis de licenciatura en Matemáticas
> *"[Título de la tesis]"*, Sebastián [Apellido], 2026.

Este repositorio contiene la implementación de referencia de la tesis.
Su propósito es permitir a cualquier lector **verificar
computacionalmente** que las construcciones categóricas del texto —mónada
de estado, $\mathbf{Para}(\mathcal{C})$, lentes como adjunción, evaluación
funtorial vía DisCoPy— se traducen en código ejecutable cuyo entrenamiento
de una red neuronal es una consecuencia de la composición categórica, no
de un grafo imperativo de operaciones.

---

## Idea en una frase

Una capa neuronal es un **morfismo de Kleisli** de la mónada de estado
$T(A)=S\to(A\times S)$, una red entera es un **diagrama de cuerdas** en
$\mathbf{Para}(\mathbf{Lens}(\mathbf{Set}))$, y su entrenamiento es la
**evaluación funtorial** de ese diagrama. La regla de la cadena no se
programa: emerge como teorema sobre la composición de lentes.

---

## Instalación

Requiere Python ≥ 3.10. Las dependencias se reducen a `discopy` (probado
con la versión 1.2.2) y `numpy`.

```bash
git clone https://github.com/xCygnus/kleisli-neural-networks.git
cd kleisli-neural-networks
python -m venv .venv
source .venv/bin/activate         # en Windows:  .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Verificación rápida (≈ 10 segundos)

Para reproducir los tres certificados numéricos que sustentan los
resultados de la tesis:

```bash
python tests/test_monad_laws.py
python tests/test_gradient_correctness.py
python tests/test_para_composition.py
```

Salida esperada:

```
test_monad_laws.py: las tres leyes monádicas se cumplen.
Error máximo entre backward analítico y numérico: 2.24e-11
test_gradient_correctness.py: backward correcto al nivel de diferencias finitas.
test_para_composition.py: composición de Para verificada.
```

Cada test corresponde a una afirmación matemática del texto:

| Test | Afirmación que verifica | Sección de la tesis |
|------|-------------------------|---------------------|
| `test_monad_laws.py` | $(T,\eta,\mu)$ satisface las tres leyes de mónada | Definición de mónada de estado |
| `test_gradient_correctness.py` | El *backward* compuesto coincide con la diferenciación numérica | Nota *"la regla de la cadena como teorema"* |
| `test_para_composition.py` | La composición secuencial acumula parámetros como $P\otimes Q$ | Definición de $\mathbf{Para}(\mathcal{C})$ |

---

## Ejemplos ejecutables

```bash
python examples/01_discopy_hello.py     # diagrama mínimo f >> g, dos funtores
python examples/02_xor_completo.py      # XOR entrenado como evaluación funtorial
```

El segundo entrena la red XOR durante 5000 épocas y debe converger a
una pérdida cuadrática del orden de $10^{-4}$, con predicciones cercanas
a los valores correctos $\{0,1,1,0\}$.

---

## Mapa del repositorio a la tesis

```
src/
├── state_monad.py              Mónada de estado T(A) = S → (A × S):
│                               eta, mu, T(g), bind. Implementación 1-a-1
│                               de la Definición y el Listado de la tesis.
│
├── para_lens.py                Clase ParaLens: morfismo parametrizado
│                               de lentes (A,A*) →(P,P*)→ (B,B*).
│                               El método __rshift__ codifica la
│                               composición secuencial en Para(Lens(C));
│                               la regla de la cadena emerge sin
│                               escribir backpropagation a mano.
│
└── kleisli_train.py            Entrenamiento como morfismo de Kleisli:
                                cada paso es un elemento de T(R), y la
                                composición de pasos es la composición
                                de Kleisli desplegada operacionalmente.

examples/
├── 01_discopy_hello.py         Ejemplo introductorio de DisCoPy: el
│                               mismo diagrama sintáctico interpretado
│                               por dos funtores distintos ilustra la
│                               propiedad universal de la categoría libre.
│
└── 02_xor_completo.py          Implementación completa: diagrama de
                                DisCoPy → Para(Lens) → entrenamiento
                                Kleisli. Las tres capas teóricas
                                aparecen explícitamente en el código.

tests/
├── test_monad_laws.py          Verificación de identidad izquierda,
│                               identidad derecha y asociatividad.
│
├── test_gradient_correctness.py  Comparación entre el backward
│                                 derivado por composición categórica
│                                 y diferencias finitas.
│
└── test_para_composition.py    Acumulación de parámetros y
                                asociatividad en Para(Lens).
```

---

## Estructura conceptual

El diseño del código sigue tres capas, cada una correspondiente a una
construcción categórica:

```
┌─────────────────────────────────────────────────────────────────┐
│  Capa 1: SINTAXIS                                                │
│  Diagrama de cuerdas en la categoría monoidal libre Cat_libre(Σ) │
│  (DisCoPy: Ty, Box, >> y @)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  Funtor F (semantica)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Capa 2: SEMÁNTICA                                               │
│  Morfismos en Para(Lens(Set)): cada caja es un par                │
│  (forward, backward) con su propio espacio de parámetros.         │
│  (Módulo para_lens.py)                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  Ejecución como morfismo de Kleisli
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Capa 3: EJECUCIÓN                                                │
│  Mónada de estado T(A) = S → (A × S): los pesos viven en S,       │
│  no hay variables globales mutables, la transparencia              │
│  referencial es estructural.                                      │
│  (Módulos state_monad.py + kleisli_train.py)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Limitaciones intencionales

El código está optimizado para **legibilidad y correspondencia uno-a-uno
con la tesis**, no para rendimiento. En particular:

- El *backward* de cada capa recomputa el *forward* en lugar de cachearlo;
  conceptualmente es el comportamiento correcto del lente, y la
  optimización por caché se discute en la sección de direcciones futuras
  de la tesis.
- La composición secuencial en `ParaLens.__rshift__` anida los parámetros
  como tuplas; no se usa ninguna estructura plana porque la anidación
  refleja la asociatividad de la composición de $\mathbf{Para}$.
- No hay soporte para arquitecturas con compartición de pesos
  (convoluciones, *transformers*); su formalización categórica es una de
  las direcciones de investigación abiertas que se discuten en la tesis.

---

## Citar este trabajo

Si esta implementación o el marco teórico te resultan útiles, por favor
cita la tesis:

```bibtex
@thesis{[clave]2026kleisli,
    author  = {Sebastián [Apellido]},
    title   = {[Título de la tesis]},
    school  = {[Institución]},
    year    = {2026},
    type    = {Tesis de Licenciatura en Matemáticas},
    url     = {https://github.com/xCygnus/kleisli-neural-networks}
}
```

---

## Referencias teóricas

El marco categórico se apoya principalmente en:

- B. Fong, D. Spivak, R. Tuyéras. *Backprop as Functor: A compositional
  perspective on supervised learning.* LICS 2019.
- B. Gavranović. *Categorical Deep Learning: An Algebraic Theory of
  Architectures.* Tesis doctoral, 2024.
- G. de Felice, A. Toumi, B. Coecke. *DisCoPy: Monoidal Categories in
  Python.* ACT 2020.
- G. Cruttwell, B. Gavranović, N. Ghani, P. Wilson, F. Zanasi.
  *Categorical Foundations of Gradient-Based Learning.* ESOP 2022.

Las citas precisas aparecen en el capítulo correspondiente de la tesis.

---

## Licencia

MIT. Véase [LICENSE](LICENSE).
