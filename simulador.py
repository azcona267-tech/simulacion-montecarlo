"""
simulador.py
============
Motor Monte Carlo: genera trayectorias de precios bajo dos modelos.

    - "lognormal": Movimiento Browniano Geométrico (GBM), esquema exacto.
    - "normal":    retornos simples normales (precios con riesgo de llegar a 0).

Lógica matemática pura: no depende de Tkinter, Matplotlib ni yfinance.
Todas las funciones devuelven matrices de forma (n_simulaciones, horizonte_dias + 1),
donde la columna 0 es el precio inicial S0.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

DIAS_BURSATILES: int = 252
MAX_CELDAS: int = 60_000_000   # tope de memoria (~480 MB en float64)

Modelo = Literal["lognormal", "normal"]
Matriz = NDArray[np.float64]


class ErrorSimulacion(ValueError):
    """Parámetros inválidos para la simulación (se muestra al usuario en la UI)."""


def _validar(
    s0: float, sigma: float, horizonte_dias: int, n_simulaciones: int
) -> None:
    """Valida los parámetros de entrada.

    Raises:
        ErrorSimulacion: Si algún parámetro es inválido o el tamaño excede la memoria.
    """
    if s0 <= 0:
        raise ErrorSimulacion("El precio inicial debe ser positivo.")
    if sigma <= 0:
        raise ErrorSimulacion("La volatilidad debe ser positiva.")
    if horizonte_dias < 1:
        raise ErrorSimulacion("El horizonte debe ser de al menos 1 día.")
    if n_simulaciones < 1_000:
        raise ErrorSimulacion("Usa al menos 1,000 simulaciones.")
    if n_simulaciones * (horizonte_dias + 1) > MAX_CELDAS:
        raise ErrorSimulacion("Combinación simulaciones x horizonte demasiado grande.")


def _generar_shocks(
    rng: np.random.Generator,
    n_simulaciones: int,
    n_pasos: int,
    antitetico: bool,
) -> Matriz:
    """Genera la matriz de shocks aleatorios Z ~ N(0, 1).

    Con variables antitéticas, por cada Z se incluye también -Z. Las dos
    trayectorias quedan negativamente correlacionadas, lo que reduce la
    varianza del estimador sin costo extra de convergencia.

    Args:
        rng: Generador de números aleatorios (reproducible con semilla).
        n_simulaciones: Número de trayectorias.
        n_pasos: Número de pasos de tiempo por trayectoria.
        antitetico: Si es True, usa pares antitéticos (Z, -Z).

    Returns:
        Matriz (n_simulaciones, n_pasos) de shocks.
    """
    if not antitetico:
        return rng.standard_normal((n_simulaciones, n_pasos))
    mitad = (n_simulaciones + 1) // 2
    z = rng.standard_normal((mitad, n_pasos))
    return np.vstack([z, -z])[:n_simulaciones]


def simular_lognormal(
    s0: float,
    mu: float,
    sigma: float,
    horizonte_dias: int,
    n_simulaciones: int = 5_000,
    semilla: int | None = None,
    antitetico: bool = True,
) -> Matriz:
    """Simula trayectorias con Movimiento Browniano Geométrico (GBM).

    Modelo:
        dS = μ·S·dt + σ·S·dW

    Solución exacta discretizada (lema de Itô), con Δt = 1/252:
        S_{t+Δt} = S_t · exp[ (μ − σ²/2)·Δt + σ·√Δt · Z ],   Z ~ N(0, 1)

    Como la solución es exacta, no hay error de discretización por el
    tamaño de Δt. Propiedades: S_t > 0 siempre; ln(S_T/S0) ~ Normal, por lo
    tanto S_T ~ Lognormal; E[S_T] = S0·exp(μT).

    Args:
        s0: Precio inicial.
        mu: Rentabilidad esperada anual (drift).
        sigma: Volatilidad anual.
        horizonte_dias: Días bursátiles a simular.
        n_simulaciones: Número de trayectorias.
        semilla: Semilla para reproducibilidad (None = aleatorio).
        antitetico: Usa variables antitéticas.

    Returns:
        Matriz (n_simulaciones, horizonte_dias + 1) de precios.
    """
    _validar(s0, sigma, horizonte_dias, n_simulaciones)
    rng = np.random.default_rng(semilla)
    dt = 1.0 / DIAS_BURSATILES

    z = _generar_shocks(rng, n_simulaciones, horizonte_dias, antitetico)
    log_incrementos = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    trayectorias = np.empty((n_simulaciones, horizonte_dias + 1), dtype=np.float64)
    trayectorias[:, 0] = s0
    trayectorias[:, 1:] = s0 * np.exp(np.cumsum(log_incrementos, axis=1))
    return trayectorias


def simular_normal(
    s0: float,
    mu: float,
    sigma: float,
    horizonte_dias: int,
    n_simulaciones: int = 5_000,
    semilla: int | None = None,
    antitetico: bool = True,
) -> Matriz:
    """Simula trayectorias con retornos simples normales.

    Modelo (retorno simple diario ~ Normal):
        r_t = μ·Δt + σ·√Δt · Z,        Z ~ N(0, 1)
        S_{t+Δt} = S_t · (1 + r_t)

    Diferencia clave con el GBM: aquí el RETORNO es normal, no el log-retorno,
    así que el precio final no es exactamente lognormal y, en teoría, un
    retorno < −100% llevaría el precio bajo cero. Se acota con max(1 + r, 0),
    es decir, el precio se queda en 0 (quiebra). Con σ realistas ocurre
    casi nunca, pero sirve para comparar ambos supuestos en la sustentación.

    Args:
        s0: Precio inicial.
        mu: Rentabilidad esperada anual.
        sigma: Volatilidad anual.
        horizonte_dias: Días bursátiles a simular.
        n_simulaciones: Número de trayectorias.
        semilla: Semilla para reproducibilidad.
        antitetico: Usa variables antitéticas.

    Returns:
        Matriz (n_simulaciones, horizonte_dias + 1) de precios.
    """
    _validar(s0, sigma, horizonte_dias, n_simulaciones)
    rng = np.random.default_rng(semilla)
    dt = 1.0 / DIAS_BURSATILES

    z = _generar_shocks(rng, n_simulaciones, horizonte_dias, antitetico)
    retornos = mu * dt + sigma * np.sqrt(dt) * z
    factores = np.maximum(1.0 + retornos, 0.0)

    trayectorias = np.empty((n_simulaciones, horizonte_dias + 1), dtype=np.float64)
    trayectorias[:, 0] = s0
    trayectorias[:, 1:] = s0 * np.cumprod(factores, axis=1)
    return trayectorias


def simular(
    modelo: Modelo,
    s0: float,
    mu: float,
    sigma: float,
    horizonte_dias: int,
    n_simulaciones: int = 5_000,
    semilla: int | None = None,
    antitetico: bool = True,
) -> Matriz:
    """Punto de entrada único: despacha al modelo elegido en la interfaz.

    Raises:
        ErrorSimulacion: Si el modelo no existe o los parámetros son inválidos.
    """
    if modelo == "lognormal":
        return simular_lognormal(s0, mu, sigma, horizonte_dias, n_simulaciones, semilla, antitetico)
    if modelo == "normal":
        return simular_normal(s0, mu, sigma, horizonte_dias, n_simulaciones, semilla, antitetico)
    raise ErrorSimulacion(f"Modelo desconocido: '{modelo}'.")


def precios_finales(trayectorias: Matriz) -> Matriz:
    """Devuelve el precio al final del horizonte de cada trayectoria (vector)."""
    return trayectorias[:, -1]


def percentiles_por_dia(
    trayectorias: Matriz,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> dict[float, Matriz]:
    """Calcula percentiles del precio en cada día (insumo del cono de incertidumbre).

    Args:
        trayectorias: Matriz (n_simulaciones, horizonte_dias + 1).
        percentiles: Percentiles a calcular, entre 0 y 100.

    Returns:
        Diccionario {percentil: vector de largo horizonte_dias + 1}.
    """
    valores = np.percentile(trayectorias, percentiles, axis=0)
    return {p: valores[i] for i, p in enumerate(percentiles)}


if __name__ == "__main__":
    # Prueba de consistencia: python simulador.py
    S0, MU, SIGMA, DIAS, N = 348.24, 0.2123, 0.2807, 252, 20_000
    T = DIAS / DIAS_BURSATILES

    for nombre in ("lognormal", "normal"):
        tray = simular(nombre, S0, MU, SIGMA, DIAS, N, semilla=42)
        fin = precios_finales(tray)
        print(f"\n[{nombre}] forma={tray.shape}")
        print(f"  Media simulada   : {fin.mean():8.2f}   (teórica E[S_T] = {S0 * np.exp(MU * T):8.2f})")
        print(f"  Mediana simulada : {np.median(fin):8.2f}   (teórica GBM    = {S0 * np.exp((MU - 0.5 * SIGMA**2) * T):8.2f})")
        print(f"  Mín / Máx        : {fin.min():.2f} / {fin.max():.2f}")