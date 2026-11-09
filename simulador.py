"""
simulador.py
============
Motor Monte Carlo del sistema. Simula la evolución de un CAPITAL invertido
bajo distintas distribuciones, con inflación y portafolios correlacionados.

Dos esquemas de simulación (Δt = 1/252):

    LOGNORMAL (GBM, solución exacta):
        V_{t+1} = V_t · exp[ (μ − σ²/2)·Δt + σ·√Δt · Z ],   Z ~ N(0, 1)

    RESTO (retornos simples con shocks estandarizados Z, E[Z]=0, Var[Z]=1):
        V_{t+1} = V_t · max(1 + μ·Δt + σ·√Δt · Z, 0)

Lógica pura NumPy: no depende de Tkinter ni Matplotlib. Admite barra de
progreso y cancelación (botón «Detener») procesando en lotes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from constantes import (
    DIAS_BURSATILES_ANIO,
    MAX_CELDAS,
    SIMULACIONES_DEFECTO,
    SIMULACIONES_MAX,
    SIMULACIONES_MIN,
)
from distribuciones import GENERADORES, ErrorDistribucion, generar
from utilidades import dias_a_pasos

Matriz = NDArray[np.float64]
Vector = NDArray[np.float64]
Progreso = Callable[[float], None]

_SIMETRICAS: frozenset[str] = frozenset({"normal", "lognormal", "uniforme", "t_student"})


class ErrorSimulacion(ValueError):
    """Parámetros inválidos para la simulación (se muestra al usuario en la UI)."""


class SimulacionCancelada(Exception):
    """El usuario presionó «Detener» durante la simulación."""


@dataclass(frozen=True, eq=False)
class ParametrosSimulacion:
    """Parámetros de entrada de una simulación (los porcentajes van en decimal).

    Atributos:
        capital: Capital inicial invertido.
        mu: Rentabilidad esperada anual (0.12 = 12%).
        sigma: Volatilidad anual (0.28 = 28%).
        horizonte_dias: Horizonte en días CALENDARIO (30/90/180/365).
        n_simulaciones: Número de trayectorias.
        distribucion: Clave de `distribuciones.GENERADORES`; "lognormal" = GBM.
        inflacion: Inflación anual (0.03 = 3%), para el valor real.
        semilla: Semilla de reproducibilidad (None = aleatorio).
        antitetico: Pide variables antitéticas (solo se aplican si la
            distribución es simétrica).
        parametros_dist: Parámetros extra de la distribución (p. ej. {"gl": 4}).
        muestra: Datos para la distribución "personalizada" (bootstrap).
    """
    capital: float
    mu: float
    sigma: float
    horizonte_dias: int
    n_simulaciones: int = SIMULACIONES_DEFECTO
    distribucion: str = "lognormal"
    inflacion: float = 0.0
    semilla: int | None = None
    antitetico: bool = True
    parametros_dist: dict[str, float] = field(default_factory=dict)
    muestra: Matriz | None = None


@dataclass(frozen=True, eq=False)
class ResultadoSimulacion:
    """Resultado de una simulación: trayectorias nominales y deflactor.

    `trayectorias` tiene forma (n_simulaciones, pasos + 1); la columna 0 es el
    capital inicial. Para portafolios, `finales_por_activo` guarda el valor
    final de cada activo.
    """
    params: ParametrosSimulacion
    pasos: int
    trayectorias: Matriz
    deflactor: Vector
    finales_por_activo: Matriz | None = None
    nombres: tuple[str, ...] = ()

    @property
    def reales(self) -> Matriz:
        """Trayectorias en términos reales: V_real(t) = V_nominal(t) · (1+i)^(−t)."""
        return self.trayectorias * self.deflactor

    @property
    def finales(self) -> Vector:
        """Valor final nominal de cada trayectoria."""
        return self.trayectorias[:, -1]

    @property
    def finales_reales(self) -> Vector:
        """Valor final real (descontada la inflación) de cada trayectoria."""
        return self.trayectorias[:, -1] * self.deflactor[-1]


# --------------------------------------------------------------------------- #
#  Validación y utilidades internas
# --------------------------------------------------------------------------- #
def _validar(p: ParametrosSimulacion, pasos: int) -> None:
    """Valida parámetros y tamaño de memoria.

    Raises:
        ErrorSimulacion: Si algún parámetro es inválido.
    """
    if p.capital <= 0:
        raise ErrorSimulacion("El capital inicial debe ser positivo.")
    if p.sigma <= 0:
        raise ErrorSimulacion("La volatilidad debe ser positiva.")
    if p.inflacion <= -1.0:
        raise ErrorSimulacion("La inflación debe ser mayor a -100%.")
    if pasos < 1:
        raise ErrorSimulacion("El horizonte debe ser de al menos 1 día.")
    if not SIMULACIONES_MIN <= p.n_simulaciones <= SIMULACIONES_MAX:
        raise ErrorSimulacion(
            f"El número de simulaciones debe estar entre {SIMULACIONES_MIN:,} y {SIMULACIONES_MAX:,}."
        )
    if p.n_simulaciones * (pasos + 1) > MAX_CELDAS:
        raise ErrorSimulacion("Combinación simulaciones × horizonte demasiado grande.")
    if p.distribucion != "lognormal" and p.distribucion not in GENERADORES:
        raise ErrorSimulacion(f"Distribución desconocida: '{p.distribucion}'.")


def calcular_deflactor(inflacion: float, pasos: int) -> Vector:
    """Factor que convierte valores nominales a reales en cada paso.

        deflactor_t = (1 + i)^(−t),   t = paso / 252  (años)
    """
    t = np.arange(pasos + 1, dtype=np.float64) / DIAS_BURSATILES_ANIO
    return (1.0 + inflacion) ** (-t)


def _admite_antitetico(p: ParametrosSimulacion) -> bool:
    """Las antitéticas solo son válidas si Z y −Z tienen la MISMA distribución."""
    if p.distribucion in _SIMETRICAS:
        return True
    if p.distribucion == "triangular":
        return float(p.parametros_dist.get("moda", 0.0)) == 0.0
    return False


def _shocks_base(rng: np.random.Generator, p: ParametrosSimulacion, forma: tuple[int, int]) -> Matriz:
    """Genera shocks estandarizados de la distribución elegida."""
    if p.distribucion == "lognormal":          # GBM: el shock es normal estándar
        return rng.standard_normal(forma)
    extra: dict[str, object] = dict(p.parametros_dist)
    if p.distribucion == "personalizada":
        extra["muestra"] = p.muestra
    return generar(p.distribucion, rng, forma, **extra)


def _shocks_lote(
    rng: np.random.Generator, p: ParametrosSimulacion, filas: int, pasos: int, antitetico: bool
) -> Matriz:
    """Shocks de un lote, con pares antitéticos (Z, −Z) si corresponde."""
    if not antitetico:
        return _shocks_base(rng, p, (filas, pasos))
    mitad = (filas + 1) // 2
    z = _shocks_base(rng, p, (mitad, pasos))
    return np.vstack([z, -z])[:filas]


def _simular_lote(
    rng: np.random.Generator, p: ParametrosSimulacion, filas: int, pasos: int, antitetico: bool
) -> Matriz:
    """Simula un lote de trayectorias; devuelve (filas, pasos + 1)."""
    dt = 1.0 / DIAS_BURSATILES_ANIO
    z = _shocks_lote(rng, p, filas, pasos, antitetico)

    if p.distribucion == "lognormal":
        incrementos = (p.mu - 0.5 * p.sigma**2) * dt + p.sigma * np.sqrt(dt) * z
        factor = np.exp(np.cumsum(incrementos, axis=1))
    else:
        retornos = p.mu * dt + p.sigma * np.sqrt(dt) * z
        factor = np.cumprod(np.maximum(1.0 + retornos, 0.0), axis=1)

    lote = np.empty((filas, pasos + 1), dtype=np.float64)
    lote[:, 0] = p.capital
    lote[:, 1:] = p.capital * factor
    return lote


def _revisar_cancelacion(cancelar: threading.Event | None) -> None:
    if cancelar is not None and cancelar.is_set():
        raise SimulacionCancelada("Simulación detenida por el usuario.")


def _ejecutar(
    p: ParametrosSimulacion,
    pasos: int,
    progreso: Progreso | None,
    cancelar: threading.Event | None,
    tam_lote: int,
) -> ResultadoSimulacion:
    """Bucle por lotes común a todas las distribuciones."""
    _validar(p, pasos)
    rng = np.random.default_rng(p.semilla)
    antitetico = p.antitetico and _admite_antitetico(p)
    n = p.n_simulaciones
    trayectorias = np.empty((n, pasos + 1), dtype=np.float64)

    hecho = 0
    try:
        while hecho < n:
            _revisar_cancelacion(cancelar)
            filas = min(tam_lote, n - hecho)
            trayectorias[hecho:hecho + filas] = _simular_lote(rng, p, filas, pasos, antitetico)
            hecho += filas
            if progreso is not None:
                progreso(hecho / n)
    except ErrorDistribucion as exc:
        raise ErrorSimulacion(str(exc)) from exc

    return ResultadoSimulacion(
        params=p, pasos=pasos, trayectorias=trayectorias,
        deflactor=calcular_deflactor(p.inflacion, pasos),
    )


# --------------------------------------------------------------------------- #
#  API pública (nombres pedidos en el enunciado)
# --------------------------------------------------------------------------- #
def generar_trayectorias(
    p: ParametrosSimulacion,
    progreso: Progreso | None = None,
    cancelar: threading.Event | None = None,
    tam_lote: int = 5_000,
) -> ResultadoSimulacion:
    """Punto de entrada principal: simula según `p.distribucion`.

    Args:
        p: Parámetros de la simulación.
        progreso: Callback opcional con la fracción completada (0 a 1).
        cancelar: `threading.Event`; si se activa, aborta con `SimulacionCancelada`.
        tam_lote: Trayectorias por lote (controla memoria y frecuencia del progreso).

    Raises:
        ErrorSimulacion: Parámetros inválidos.
        SimulacionCancelada: Si el usuario detuvo la simulación.
    """
    return _ejecutar(p, dias_a_pasos(p.horizonte_dias), progreso, cancelar, tam_lote)


def simular_normal(p: ParametrosSimulacion, **kw: object) -> ResultadoSimulacion:
    """Retornos simples con shocks Normal(0, 1)."""
    return generar_trayectorias(replace(p, distribucion="normal"), **kw)  # type: ignore[arg-type]


def simular_lognormal(p: ParametrosSimulacion, **kw: object) -> ResultadoSimulacion:
    """Movimiento Browniano Geométrico (precio final lognormal)."""
    return generar_trayectorias(replace(p, distribucion="lognormal"), **kw)  # type: ignore[arg-type]


def simular_uniforme(p: ParametrosSimulacion, **kw: object) -> ResultadoSimulacion:
    """Retornos simples con shocks uniformes (colas livianas)."""
    return generar_trayectorias(replace(p, distribucion="uniforme"), **kw)  # type: ignore[arg-type]


def simular_triangular(p: ParametrosSimulacion, **kw: object) -> ResultadoSimulacion:
    """Retornos simples con shocks triangulares (moda en `parametros_dist['moda']`)."""
    return generar_trayectorias(replace(p, distribucion="triangular"), **kw)  # type: ignore[arg-type]


def simular_portafolio(
    capital: float,
    pesos: Vector,
    mus: Vector,
    sigmas: Vector,
    correlacion: Matriz,
    horizonte_dias: int,
    n_simulaciones: int = SIMULACIONES_DEFECTO,
    inflacion: float = 0.0,
    semilla: int | None = None,
    nombres: tuple[str, ...] = (),
    progreso: Progreso | None = None,
    cancelar: threading.Event | None = None,
) -> ResultadoSimulacion:
    """Simula un portafolio de k activos correlacionados (buy-and-hold).

    Cada activo sigue un GBM; los shocks se correlacionan con Cholesky:

        ρ = L·Lᵀ            (descomposición de Cholesky)
        Z_corr = Z_indep · Lᵀ   ->   Cov(Z_corr) = L·Lᵀ = ρ

        V_i(t) = w_i · C · exp[ (μ_i − σ_i²/2)·t + σ_i·W_i(t) ]
        V_portafolio(t) = Σ_i V_i(t)

    Sin rebalanceo: los pesos derivan con el mercado. En `params` se guardan
    μ y σ del portafolio con los pesos iniciales (aproximación de referencia).

    Args:
        capital: Capital inicial total.
        pesos: Pesos iniciales (k,), no negativos, suman 1.
        mus: Rentabilidades esperadas anuales (k,).
        sigmas: Volatilidades anuales (k,).
        correlacion: Matriz de correlación (k, k), simétrica y definida positiva.
        horizonte_dias: Horizonte en días calendario.
        n_simulaciones: Número de trayectorias.
        inflacion: Inflación anual.
        semilla: Semilla de reproducibilidad.
        nombres: Nombres de los activos (para tablas y gráficos).
        progreso: Callback de progreso (0 a 1).
        cancelar: Evento para detener la simulación.

    Returns:
        `ResultadoSimulacion` con `finales_por_activo` y `nombres` completos.
    """
    w = np.asarray(pesos, dtype=np.float64)
    mu = np.asarray(mus, dtype=np.float64)
    sg = np.asarray(sigmas, dtype=np.float64)
    corr = np.asarray(correlacion, dtype=np.float64)
    k = w.size

    if k < 2:
        raise ErrorSimulacion("Un portafolio requiere al menos 2 activos.")
    if mu.shape != (k,) or sg.shape != (k,) or corr.shape != (k, k):
        raise ErrorSimulacion("Pesos, rentabilidades, volatilidades y correlaciones no coinciden en tamaño.")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ErrorSimulacion("Los pesos deben ser no negativos y sumar 100%.")
    if np.any(sg <= 0):
        raise ErrorSimulacion("Todas las volatilidades deben ser positivas.")
    if not (np.allclose(corr, corr.T) and np.allclose(np.diag(corr), 1.0) and np.all(np.abs(corr) <= 1.0)):
        raise ErrorSimulacion("La matriz de correlación debe ser simétrica, con diagonal 1 y valores en [-1, 1].")
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ErrorSimulacion("La matriz de correlación no es definida positiva.") from exc

    pasos = dias_a_pasos(horizonte_dias)
    vol_p = float(np.sqrt((w * sg) @ corr @ (w * sg)))
    params = ParametrosSimulacion(
        capital=capital, mu=float(w @ mu), sigma=vol_p, horizonte_dias=horizonte_dias,
        n_simulaciones=n_simulaciones, distribucion="lognormal",
        inflacion=inflacion, semilla=semilla,
    )
    _validar(params, pasos)

    dt = 1.0 / DIAS_BURSATILES_ANIO
    deriva = (mu - 0.5 * sg**2) * dt
    difusion = sg * np.sqrt(dt)
    inicial = w * capital

    rng = np.random.default_rng(semilla)
    trayectorias = np.empty((n_simulaciones, pasos + 1), dtype=np.float64)
    finales = np.empty((n_simulaciones, k), dtype=np.float64)

    hecho, tam_lote = 0, 2_000
    while hecho < n_simulaciones:
        _revisar_cancelacion(cancelar)
        filas = min(tam_lote, n_simulaciones - hecho)
        z = rng.standard_normal((filas, pasos, k)) @ chol.T          # (filas, pasos, k)
        valores = inicial * np.exp(np.cumsum(deriva + difusion * z, axis=1))
        trayectorias[hecho:hecho + filas, 0] = capital
        trayectorias[hecho:hecho + filas, 1:] = valores.sum(axis=2)
        finales[hecho:hecho + filas] = valores[:, -1, :]
        hecho += filas
        if progreso is not None:
            progreso(hecho / n_simulaciones)

    return ResultadoSimulacion(
        params=params, pasos=pasos, trayectorias=trayectorias,
        deflactor=calcular_deflactor(inflacion, pasos),
        finales_por_activo=finales,
        nombres=nombres or tuple(f"Activo {i + 1}" for i in range(k)),
    )


def calcular_precio_final(trayectorias: Matriz) -> Vector:
    """Valor (o precio) al final del horizonte de cada trayectoria."""
    return trayectorias[:, -1]


def percentiles_por_dia(
    trayectorias: Matriz,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> dict[float, Vector]:
    """Percentiles de la distribución cross-sectional en cada paso (cono)."""
    valores = np.percentile(trayectorias, percentiles, axis=0)
    return {p: valores[i] for i, p in enumerate(percentiles)}


def simular(
    modelo: str, s0: float, mu: float, sigma: float, pasos: int,
    n_simulaciones: int = SIMULACIONES_DEFECTO, semilla: int | None = None,
    antitetico: bool = True,
) -> Matriz:
    """Compatibilidad con los `__main__` de riesgo.py y graficos.py.

    OJO: aquí `pasos` son pasos BURSÁTILES (no días calendario) y se devuelve
    solo la matriz de trayectorias.
    """
    p = ParametrosSimulacion(
        capital=s0, mu=mu, sigma=sigma, horizonte_dias=pasos,
        n_simulaciones=n_simulaciones, distribucion=modelo,
        semilla=semilla, antitetico=antitetico,
    )
    return _ejecutar(p, pasos, None, None, 5_000).trayectorias


if __name__ == "__main__":
    # Validación contra teoría: python simulador.py
    import pandas as pd
    from scipy import stats

    CAP, MU, SIG, DIAS, N = 10_000.0, 0.12, 0.28, 365, 20_000
    pasos = dias_a_pasos(DIAS)
    T, dt = pasos / DIAS_BURSATILES_ANIO, 1.0 / DIAS_BURSATILES_ANIO
    muestra = np.random.default_rng(1).standard_t(4, 1_000) * 0.01

    filas = []
    for dist in ("lognormal", "normal", "uniforme", "triangular", "t_student", "personalizada"):
        p = ParametrosSimulacion(CAP, MU, SIG, DIAS, N, dist, semilla=42,
                                 muestra=muestra if dist == "personalizada" else None)
        fin = generar_trayectorias(p).finales
        teorica = CAP * np.exp(MU * T) if dist == "lognormal" else CAP * (1 + MU * dt) ** pasos
        filas.append({
            "distribucion": dist,
            "media_sim": fin.mean(), "media_teorica": teorica,
            "mediana": np.median(fin),
            "P(perdida)_%": 100 * np.mean(fin < CAP),
            "asimetria": stats.skew(fin),
        })
    print(f"Capital {CAP:,.0f} · {DIAS} días calendario = {pasos} pasos\n")
    print(pd.DataFrame(filas).round(2).to_string(index=False))

    print("\n--- Inflación (3%) ---")
    r = generar_trayectorias(ParametrosSimulacion(CAP, MU, SIG, DIAS, N, "lognormal", inflacion=0.03, semilla=42))
    print(f"Media nominal : {r.finales.mean():10.2f}")
    print(f"Media real    : {r.finales_reales.mean():10.2f}   (teórica = {r.finales.mean() * 1.03 ** -T:10.2f})")

    print("\n--- Portafolio 60/40 (correlación 0.3) ---")
    pt = simular_portafolio(
        CAP, np.array([0.6, 0.4]), np.array([0.12, 0.08]), np.array([0.28, 0.18]),
        np.array([[1.0, 0.3], [0.3, 1.0]]), DIAS, N, semilla=42, nombres=("AAPL", "Bonos"),
    )
    teo = float(np.sum(np.array([0.6, 0.4]) * CAP * np.exp(np.array([0.12, 0.08]) * T)))
    print(f"Media simulada: {pt.finales.mean():10.2f}   (teórica = {teo:10.2f})")
    print(f"Correlación de valores finales entre activos: {np.corrcoef(pt.finales_por_activo.T)[0, 1]:.3f}")