#!/usr/bin/env bash
# verify_all.sh
# Ejecuta todos los tests y ejemplos. Útil para que un lector de la tesis
# certifique en una sola orden que el código se comporta como se afirma.

set -e

echo "============================================================"
echo " Verificación de kleisli-neural-networks"
echo "============================================================"

echo
echo "[1/3] Módulos base..."
python src/state_monad.py
python src/para_lens.py

echo
echo "[2/3] Tests..."
python tests/test_monad_laws.py
python tests/test_gradient_correctness.py
python tests/test_para_composition.py

echo
echo "[3/3] Ejemplos..."
python examples/01_discopy_hello.py
echo
python examples/02_xor_completo.py

echo
echo "============================================================"
echo " Todo correcto."
echo "============================================================"
