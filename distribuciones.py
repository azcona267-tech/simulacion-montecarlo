"""
distribuciones.py
=================
Generadores de números aleatorios ESTANDARIZADOS: todas las distribuciones
devuelven variables Z con media 0 y varianza 1.

¿Por qué estandarizar? El simulador construye los retornos como

        r_t = μ·Δt + σ·√Δt · Z

Si todas las Z tienen media 0 y varianza 1, el usuario controla μ y σ y la
distribución elegida solo cambia la FORMA (asimetría, curtosis, colas). Así
las diferencias de VaR/CVaR entre distribuciones son comparables y atribuibles
a la forma, no a una media o varianza distinta.

Lógica pura NumPy/SciPy: no depende de la UI ni de yfinance.
"""
from __future__ import annotations

from typing import Callable, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

Matriz = NDArray[np.float64]
Forma = int | tuple[int, ...]


class ErrorDistribucion(ValueError):
    """Parámetros inválidos o distribución desconocida (se muestra en la UI)."""


def normal(rng: np.random.Generator, forma: Forma) -> Matriz:
    """Normal estándar N(0, 1)."""
    return rng.standard_normal(forma)


def uniforme(rng: np.random.Generator, forma: Forma) -> Matriz:
    """Uniforme centrada con varianza 1.

    Var[U(a, b)] = (b − a)² / 12. Con a = −w y b = w:  (2w)²/12 = 1  →  w = √3.
    """
    w = np.sqrt(3.0)
    return rng.uniform(-w, w, forma)


def triangular(rng: np.random.Generator, forma: Forma, moda: float = 0.0) -> Matriz:
    """Triangular estandarizada sobre [−1, 1] con moda configurable.

    Para extremos a, b y moda c:
        E[X]   = (a + b + c) / 3
        Var[X] = (a² + b² + c² − ab − ac − bc) / 18
    Se devuelve (X − E[X]) / √Var[X]. Una moda ≠ 0 introduce asimetría.

    Args:
        moda: Posición de la moda en [−1, 1] (0 = simétrica).
    """
    if not -1.0 <= moda <= 1.0:
        raise ErrorDistribucion("La moda de la triangular debe estar entre -1 y 1.")
    a, c, b = -1.0, float(moda), 1.0
    media = (a + b + c) / 3.0
    var = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    return (rng.triangular(a, c, b, forma) - media) / np.sqrt(var)


def lognormal(rng: np.random.Generator, forma: Forma, sigma_log: float = 0.5) -> Matriz:
    """Lognormal estandarizada (asimetría positiva).

    Si X = exp(s·N(0,1)):
        E[X]   = e^{s²/2}
        Var[X] = (e^{s²} − 1)·e^{s²}
    Se devuelve (X − E[X]) / √Var[X].

    Args:
        sigma_log: Parámetro s de forma (> 0). Mayor s = más asimetría.
    """
    if sigma_log <= 0:
        raise ErrorDistribucion("sigma_log debe ser positivo.")
    s2 = sigma_log**2
    media = np.exp(s2 / 2.0)
    var = (np.exp(s2) - 1.0) * np.exp(s2)
    return (rng.lognormal(0.0, sigma_log, forma) - media) / np.sqrt(var)


def t_student(rng: np.random.Generator, forma: Forma, gl: float = 5.0) -> Matriz:
    """t de Student estandarizada (colas pesadas).

    Var[t_ν] = ν / (ν − 2) para ν > 2, así que se divide por √(ν/(ν−2)).
    Con pocos grados de libertad produce más eventos extremos que la normal,
    que es lo que se observa en retornos bursátiles reales.

    Args:
        gl: Grados de libertad ν (> 2).
    """
    if gl <= 2:
        raise ErrorDistribucion("Los grados de libertad deben ser mayores a 2.")
    return rng.standard_t(gl, forma) / np.sqrt(gl / (gl - 2.0))


def poisson(rng: np.random.Generator, forma: Forma, lam: float = 4.0) -> Matriz:
    """Poisson estandarizada: E = Var = λ  →  (X − λ) / √λ.

    Args:
        lam: Tasa λ (> 0).
    """
    if lam <= 0:
        raise ErrorDistribucion("lam debe ser positivo.")
    return (rng.poisson(lam, forma).astype(np.float64) - lam) / np.sqrt(lam)


def exponencial(rng: np.random.Generator, forma: Forma) -> Matriz:
    """Exponencial estandarizada: con escala 1, E = Var = 1  →  X − 1."""
    return rng.exponential(1.0, forma) - 1.0


def beta(rng: np.random.Generator, forma: Forma, a: float = 2.0, b: float = 5.0) -> Matriz:
    """Beta(a, b) estandarizada.

        E[X]   = a / (a + b)
        Var[X] = a·b / ((a + b)²·(a + b + 1))
    """
    if a <= 0 or b <= 0:
        raise ErrorDistribucion("a y b deben ser positivos.")
    media = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return (rng.beta(a, b, forma) - media) / np.sqrt(var)


def gamma(rng: np.random.Generator, forma: Forma, k: float = 2.0) -> Matriz:
    """Gamma(k, θ=1) estandarizada: E = Var = k  →  (X − k) / √k."""
    if k <= 0:
        raise ErrorDistribucion("k debe ser positivo.")
    return (rng.gamma(k, 1.0, forma) - k) / np.sqrt(k)


def personalizada(rng: np.random.Generator, forma: Forma, muestra: Matriz | None = None) -> Matriz:
    """Distribución empírica por bootstrap de una muestra del usuario.

    Estandariza la muestra (media 0, desviación 1, ddof=0 para que el
    remuestreo tenga varianza exactamente 1) y sortea con reemplazo. Conserva
    la asimetría y la curtosis REALES de los datos, sin asumir ninguna forma
    paramétrica. Típicamente se alimenta con los retornos de yfinance.

    Args:
        muestra: Observaciones (≥ 30), p. ej. retornos logarítmicos históricos.
    """
    if muestra is None:
        raise ErrorDistribucion("La distribución personalizada requiere una muestra de datos.")
    m = np.asarray(muestra, dtype=np.float64)
    if m.ndim != 1 or m.size < 30 or not np.all(np.isfinite(m)):
        raise ErrorDistribucion("La muestra debe ser un vector finito de al menos 30 datos.")
    sd = m.std(ddof=0)
    if sd == 0:
        raise ErrorDistribucion("La muestra no tiene variabilidad.")
    z = (m - m.mean()) / sd
    return z[rng.integers(0, z.size, size=forma)]


GENERADORES: Final[dict[str, Callable[..., Matriz]]] = {
    "normal": normal,
    "lognormal": lognormal,
    "uniforme": uniforme,
    "triangular": triangular,
    "t_student": t_student,
    "poisson": poisson,
    "exponencial": exponencial,
    "beta": beta,
    "gamma": gamma,
    "personalizada": personalizada,
}


def generar(nombre: str, rng: np.random.Generator, forma: Forma, **parametros: object) -> Matriz:
    """Punto de entrada único: genera shocks estandarizados de la distribución elegida.

    Args:
        nombre: Clave de `GENERADORES`.
        rng: Generador aleatorio (reproducible con semilla).
        forma: Tamaño de salida (entero o tupla).
        **parametros: Parámetros específicos (p. ej. `gl=4`, `muestra=...`).

    Raises:
        ErrorDistribucion: Si la distribución no existe o los parámetros son inválidos.
    """
    if nombre not in GENERADORES:
        raise ErrorDistribucion(f"Distribución desconocida: '{nombre}'.")
    try:
        return GENERADORES[nombre](rng, forma, **parametros)
    except TypeError as exc:
        raise ErrorDistribucion(f"Parámetros inválidos para '{nombre}': {exc}") from exc


def momentos(z: Matriz) -> dict[str, float]:
    """Media, desviación, asimetría y curtosis en exceso (Fisher) de una muestra."""
    z = np.asarray(z, dtype=np.float64).ravel()
    return {
        "media": float(z.mean()),
        "desv": float(z.std(ddof=1)),
        "asimetria": float(stats.skew(z)),
        "curtosis_exc": float(stats.kurtosis(z)),
    }


if __name__ == "__main__":
    # Verificación de momentos: python distribuciones.py
    rng = np.random.default_rng(42)
    muestra_hist = rng.standard_t(4, 1_000) * 0.01      # simula retornos con colas pesadas

    filas = []
    for nombre in GENERADORES:
        extra = {"muestra": muestra_hist} if nombre == "personalizada" else {}
        z = generar(nombre, rng, 300_000, **extra)
        filas.append({"distribucion": nombre, **momentos(z)})
    print(pd.DataFrame(filas).round(3).to_string(index=False))