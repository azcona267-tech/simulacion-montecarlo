"""
estadisticas.py
===============
Estadística descriptiva de los resultados simulados: tendencia central,
dispersión y forma, más utilidades para los gráficos (densidad KDE y
distribución acumulada empírica) y una prueba de normalidad.

Lógica pura NumPy/SciPy: no depende de la UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Vector = NDArray[np.float64]


class ErrorEstadistica(ValueError):
    """Datos insuficientes o inválidos para el cálculo estadístico."""


def _vector(valores: object) -> Vector:
    """Convierte a vector float64 y valida tamaño y finitud.

    Raises:
        ErrorEstadistica: Si hay menos de 3 datos o valores no finitos.
    """
    v = np.asarray(valores, dtype=np.float64).ravel()
    if v.size < 3:
        raise ErrorEstadistica("Se requieren al menos 3 datos.")
    if not np.all(np.isfinite(v)):
        raise ErrorEstadistica("Los datos contienen valores no finitos.")
    return v


def media(valores: object) -> float:
    """Media aritmética:  x̄ = (1/n)·Σ xᵢ."""
    return float(np.mean(_vector(valores)))


def mediana(valores: object) -> float:
    """Mediana: valor central; el 50% de los datos queda por debajo."""
    return float(np.median(_vector(valores)))


def varianza(valores: object) -> float:
    """Varianza muestral:  s² = Σ(xᵢ − x̄)² / (n − 1)."""
    return float(np.var(_vector(valores), ddof=1))


def desviacion(valores: object) -> float:
    """Desviación estándar muestral:  s = √s²."""
    return float(np.std(_vector(valores), ddof=1))


def asimetria(valores: object) -> float:
    """Asimetría de Fisher-Pearson:  g₁ = E[(X − μ)³] / σ³.

    > 0 cola derecha larga (típico del precio lognormal); < 0 cola izquierda.
    """
    return float(stats.skew(_vector(valores)))


def curtosis(valores: object) -> float:
    """Curtosis EN EXCESO (Fisher):  g₂ = E[(X − μ)⁴] / σ⁴ − 3.

    0 = colas como la normal; > 0 colas pesadas; < 0 colas livianas.
    """
    return float(stats.kurtosis(_vector(valores)))


def curva_densidad(
    valores: object, puntos: int = 400, max_muestra: int = 20_000, semilla: int = 0
) -> tuple[Vector, Vector]:
    """Estima la función de densidad con KDE gaussiano (kernel density).

        f̂(x) = (1/(n·h)) · Σ K((x − xᵢ)/h),   h por la regla de Scott: n^(−1/5)·σ

    Para rendimiento, sobre muestras grandes usa un submuestreo reproducible.

    Args:
        valores: Datos a suavizar.
        puntos: Resolución de la curva.
        max_muestra: Tamaño máximo de muestra para el KDE.
        semilla: Semilla del submuestreo.

    Returns:
        Tupla (x, densidad) acotada entre los percentiles 0.1 y 99.9.
    """
    v = _vector(valores)
    if v.size > max_muestra:
        v = np.random.default_rng(semilla).choice(v, max_muestra, replace=False)
    if float(np.ptp(v)) == 0.0:
        raise ErrorEstadistica("Los datos no tienen variabilidad.")
    try:
        kde = stats.gaussian_kde(v)
    except np.linalg.LinAlgError as exc:
        raise ErrorEstadistica("No se pudo estimar la densidad.") from exc
    x = np.linspace(np.percentile(v, 0.1), np.percentile(v, 99.9), puntos)
    return x, kde(x)


def moda(valores: object) -> float:
    """Moda de una variable continua: el máximo de la densidad KDE."""
    x, y = curva_densidad(valores, puntos=512)
    return float(x[int(np.argmax(y))])


def cdf_empirica(valores: object, puntos: int = 500) -> tuple[Vector, Vector]:
    """Distribución acumulada empírica:  F̂(x) = P(X ≤ x).

    Devuelve (x, p) con x = cuantiles y p uniforme en [0, 1].
    """
    v = _vector(valores)
    p = np.linspace(0.0, 1.0, puntos)
    return np.quantile(v, p), p


def prueba_normalidad(valores: object) -> tuple[float, float]:
    """Prueba de Jarque-Bera, basada en asimetría (S) y curtosis en exceso (K):

        JB = (n/6)·(S² + K²/4)  ~  χ²(2) bajo normalidad

    Returns:
        Tupla (estadístico, p-valor). p < 0.05 rechaza la normalidad.
    """
    res = stats.jarque_bera(_vector(valores))
    return float(res.statistic), float(res.pvalue)


@dataclass(frozen=True)
class ResumenEstadistico:
    """Resumen descriptivo completo de una variable simulada."""
    n: int
    media: float
    mediana: float
    moda: float
    varianza: float
    desviacion: float
    coef_variacion: float
    asimetria: float
    curtosis: float
    minimo: float
    maximo: float
    p05: float
    p25: float
    p75: float
    p95: float
    p99: float
    jb_estadistico: float
    jb_pvalor: float


def resumir(valores: object) -> ResumenEstadistico:
    """Calcula el resumen estadístico completo de un vector de resultados."""
    v = _vector(valores)
    p05, p25, p75, p95, p99 = (float(q) for q in np.percentile(v, [5, 25, 75, 95, 99]))
    m = float(v.mean())
    s = float(v.std(ddof=1))
    jb, pv = prueba_normalidad(v)
    return ResumenEstadistico(
        n=int(v.size), media=m, mediana=float(np.median(v)), moda=moda(v),
        varianza=float(v.var(ddof=1)), desviacion=s,
        coef_variacion=s / abs(m) if m != 0 else float("nan"),
        asimetria=float(stats.skew(v)), curtosis=float(stats.kurtosis(v)),
        minimo=float(v.min()), maximo=float(v.max()),
        p05=p05, p25=p25, p75=p75, p95=p95, p99=p99,
        jb_estadistico=jb, jb_pvalor=pv,
    )


def a_diccionario(r: ResumenEstadistico) -> dict[str, float]:
    """Resumen como diccionario con etiquetas en español (tablas y exportación)."""
    return {
        "Observaciones": r.n, "Media": r.media, "Mediana": r.mediana, "Moda": r.moda,
        "Varianza": r.varianza, "Desviación estándar": r.desviacion,
        "Coef. de variación": r.coef_variacion, "Asimetría": r.asimetria,
        "Curtosis (exceso)": r.curtosis, "Mínimo": r.minimo, "Máximo": r.maximo,
        "Percentil 5": r.p05, "Percentil 25": r.p25, "Percentil 75": r.p75,
        "Percentil 95": r.p95, "Percentil 99": r.p99,
        "Jarque-Bera": r.jb_estadistico, "p-valor JB": r.jb_pvalor,
    }


if __name__ == "__main__":
    # Simulado vs. teoría lognormal: python estadisticas.py
    import pandas as pd

    from simulador import ParametrosSimulacion, generar_trayectorias

    CAP, MU, SIG, DIAS, N = 10_000.0, 0.12, 0.28, 365, 50_000
    res = generar_trayectorias(ParametrosSimulacion(CAP, MU, SIG, DIAS, N, "lognormal", semilla=42))
    r = resumir(res.finales)

    T = res.pasos / 252
    s, m = SIG * np.sqrt(T), (MU - 0.5 * SIG**2) * T
    teo = stats.lognorm(s=s, scale=CAP * np.exp(m))
    media_t, var_t, asim_t, curt_t = (float(x) for x in teo.stats(moments="mvsk"))

    tabla = pd.DataFrame(
        {
            "simulado": [r.media, r.mediana, r.moda, r.desviacion, r.asimetria, r.curtosis],
            "teorico": [media_t, float(teo.median()), CAP * np.exp(m - s**2),
                        np.sqrt(var_t), asim_t, curt_t],
        },
        index=["media", "mediana", "moda", "desviacion", "asimetria", "curtosis_exc"],
    )
    print(tabla.round(3).to_string())
    print(f"\nOrden esperado en asimetría positiva: moda < mediana < media "
          f"({r.moda:,.0f} < {r.mediana:,.0f} < {r.media:,.0f})")
    print(f"Jarque-Bera = {r.jb_estadistico:,.0f}, p-valor = {r.jb_pvalor:.3g} "
          f"-> {'se rechaza' if r.jb_pvalor < 0.05 else 'no se rechaza'} la normalidad del valor final")
