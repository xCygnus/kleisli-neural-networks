# Cómo publicar este repositorio en GitHub

Estos pasos asumen que ya tienes la cuenta `xCygnus` de GitHub y `git`
instalado en tu máquina. Si el repositorio `kleisli-neural-networks` ya
existe en tu cuenta, salta al paso 3.

## 1. Crear el repositorio en GitHub

En la interfaz web de GitHub:

1. Ve a https://github.com/new
2. **Owner:** `xCygnus`
3. **Repository name:** `kleisli-neural-networks`
4. **Description:** *Redes neuronales como morfismos de Kleisli — código de la tesis.*
5. Visibilidad: **Public** (para que el lector de la tesis pueda acceder).
6. **NO** marques "Add a README", "Add .gitignore" ni "Choose a license"
   — los archivos ya están en este directorio.
7. Pulsa **Create repository**.

## 2. Inicializar y subir desde local

Desde la carpeta `kleisli-neural-networks/`:

```bash
git init
git add .
git commit -m "Versión inicial: implementación categórica completa con tests"
git branch -M main
git remote add origin https://github.com/xCygnus/kleisli-neural-networks.git
git push -u origin main
```

Si GitHub te pide credenciales y usas autenticación por token:
- Usuario: `xCygnus`
- Contraseña: tu Personal Access Token (no la contraseña web).

## 3. Si el repositorio ya existe y quieres sobrescribirlo

```bash
cd kleisli-neural-networks
git init
git remote add origin https://github.com/xCygnus/kleisli-neural-networks.git
git fetch origin
git checkout -b main
git add .
git commit -m "Reorganización completa para acompañar la tesis"
git push --force origin main      # CUIDADO: borra historia previa
```

Si prefieres conservar la historia previa, usa `git pull --rebase
origin main` antes de `git push`.

## 4. Etiquetar la versión usada en la tesis

Para que la tesis cite una versión inmutable del código:

```bash
git tag -a v1.0-tesis -m "Versión congelada para la tesis"
git push origin v1.0-tesis
```

Luego en la tesis puedes citar:
`https://github.com/xCygnus/kleisli-neural-networks/tree/v1.0-tesis`

## 5. (Opcional) Activar GitHub Actions para CI

Crea el archivo `.github/workflows/test.yml` con:

```yaml
name: tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: bash verify_all.sh
```

Así, cada *push* ejecuta automáticamente los tests y aparece un badge
verde junto al repositorio.

## 6. (Opcional) Obtener un DOI con Zenodo

Para que la tesis pueda citar el código con un DOI permanente:

1. Crear cuenta en https://zenodo.org y conectarla con GitHub.
2. Activar el repositorio en la lista de Zenodo.
3. Crear una *release* en GitHub (a partir del tag `v1.0-tesis`).
4. Zenodo te asignará automáticamente un DOI como `10.5281/zenodo.XXXXXXX`.

Ese DOI sí es citable formalmente en la bibliografía de la tesis.
