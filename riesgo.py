"""
riesgo.py
=========
Métricas de riesgo de mercado sobre los precios finales simulados:
Value at Risk (VaR), Conditional VaR (CVaR / Expected Shortfall) y
probabilidad de pérdida. Incluye la versión analítica del modelo
lognormal para validar la simulación.

Convención: la PÉRDIDA se mide como  L = S0 - S_T  (positiva = se perdió dinero).
Lógica pura: no depende de Tkinter, Matplotlib ni yfinance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

DIAS_BURSATILES: int = 252
Vector = NDArray[np.float64]


class ErrorRiesgo(ValueError):
    """Entrada inválida para el análisis de riesgo (se muestra en la UI)."""


@dataclass(frozen=True)
class ResultadoRiesgo:
    """Resumen de riesgo de un conjunto de precios finales simulados.

    VaR y CVaR están en unidades monetarias (por acción) y en % de S0.
    """
    modelo: str
    s0: float
    horizonte_dias: int
    n_simulaciones: int
    precio_esperado: float
    precio_mediano: float
    retorno_esperado: float      # E[S_T]/S0 - 1
    prob_perdida: float          # P(S_T < S0)
    var_95: float
    var_95_pct: float
    cvar_95: float
    cvar_95_pct: float
    var_99: float
    var_99_pct: float
    cvar_99: float
    cvar_99_pct: float
    peor_caso: float
    mejor_caso: float


def _validar(precios_finales: Vector, s0: float, horizonte_dias: int) -> Vector:
    """Valida y convierte la entrada a un vector float64 unidimensional.

    Raises:
        ErrorRiesgo: Si hay datos insuficientes, no finitos o parámetros inválidos.
    """
    p = np.asarray(precios_finales, dtype=np.float64)
    if p.ndim != 1 or p.size < 1_000:
        raise ErrorRiesgo("Se requiere un vector de al menos 1,000 precios finales.")
    if not np.all(np.isfinite(p)):
        raise ErrorRiesgo("Los precios finales contienen valores no finitos.")
    if s0 <= 0:
        raise ErrorRiesgo("El precio inicial debe ser positivo.")
    if horizonte_dias < 1:
        raise ErrorRiesgo("El horizonte debe ser de al menos 1 día.")
    return p


def calcular_perdidas(precios_finales: Vector, s0: float) -> Vector:
    """Pérdida por trayectoria: L = S0 - S_T (negativa si hubo ganancia)."""
    return s0 - np.asarray(precios_finales, dtype=np.float64)


def var_cvar(perdidas: Vector, confianza: float) -> tuple[float, float]:
    """Calcula VaR y CVaR históricos (sobre la distribución simulada).

    Definiciones, con nivel de confianza α:
        VaR_α  = cuantil α de la distribución de pérdidas L
                 -> con probabilidad α la pérdida NO supera el VaR.
        CVaR_α = E[ L | L >= VaR_α ]
                 -> pérdida promedio en el (1-α) peor de los escenarios.

    Si el cuantil implica ganancia (L < 0), se acota en 0: no hay pérdida
    en ese nivel de confianza.

    Args:
        perdidas: Vector de pérdidas L = S0 - S_T.
        confianza: Nivel α en (0, 1), ej. 0.95.

    Returns:
        Tupla (VaR, CVaR) en unidades monetarias, ambos >= 0.
    """
    if not 0.0 < confianza < 1.0:
        raise ErrorRiesgo("La confianza debe estar entre 0 y 1.")
    var_bruto = float(np.percentile(perdidas, confianza * 100.0))
    cola = perdidas[perdidas >= var_bruto]          # nunca vacía: incluye el máximo
    cvar_bruto = float(cola.mean())
    return max(var_bruto, 0.0), max(cvar_bruto, 0.0)


def probabilidad_perdida(precios_finales: Vector, s0: float) -> float:
    """Probabilidad empírica de terminar por debajo del precio inicial.

        P(pérdida) = P(S_T < S0) ≈ (# trayectorias con S_T < S0) / N
    """
    return float(np.mean(np.asarray(precios_finales) < s0))


def evaluar_riesgo(
    precios_finales: Vector,
    s0: float,
    horizonte_dias: int,
    modelo: str = "",
) -> ResultadoRiesgo:
    """Calcula todas las métricas de riesgo de una simulación.

    Args:
        precios_finales: Vector con S_T de cada trayectoria.
        s0: Precio inicial.
        horizonte_dias: Horizonte simulado, en días bursátiles.
        modelo: Etiqueta del modelo ("lognormal", "normal") para la UI.

    Returns:
        `ResultadoRiesgo` inmutable con VaR/CVaR al 95% y 99%.
    """
    p = _validar(precios_finales, s0, horizonte_dias)
    perdidas = calcular_perdidas(p, s0)

    var95, cvar95 = var_cvar(perdidas, 0.95)
    var99, cvar99 = var_cvar(perdidas, 0.99)

    return ResultadoRiesgo(
        modelo=modelo,
        s0=s0,
        horizonte_dias=horizonte_dias,
        n_simulaciones=int(p.size),
        precio_esperado=float(p.mean()),
        precio_mediano=float(np.median(p)),
        retorno_esperado=float(p.mean() / s0 - 1.0),
        prob_perdida=probabilidad_perdida(p, s0),
        var_95=var95, var_95_pct=var95 / s0,
        cvar_95=cvar95, cvar_95_pct=cvar95 / s0,
        var_99=var99, var_99_pct=var99 / s0,
        cvar_99=cvar99, cvar_99_pct=cvar99 / s0,
        peor_caso=float(p.min()),
        mejor_caso=float(p.max()),
    )


# --------------------------------------------------------------------------- #
#  Solución analítica (forma cerrada) del GBM: sirve para validar Monte Carlo
# --------------------------------------------------------------------------- #
def riesgo_analitico_lognormal(
    s0: float, mu: float, sigma: float, horizonte_dias: int, confianza: float
) -> tuple[float, float]:
    """VaR y CVaR exactos bajo GBM (precio final lognormal).

    Con T = días/252, m = (μ - σ²/2)·T, s = σ·√T y q = Φ⁻¹(1-α):
        S_q   = S0 · exp(m + s·q)                 -> cuantil (1-α) del precio
        VaR   = S0 - S_q
        CVaR  = S0 - S0·e^{μT} · Φ(q - s) / (1-α)

    El CVaR sale de la esperanza truncada de una lognormal:
        E[S_T ; Z <= q] = S0·e^{μT}·Φ(q - s).

    Returns:
        Tupla (VaR, CVaR) en unidades monetarias.
    """
    if not 0.0 < confianza < 1.0:
        raise ErrorRiesgo("La confianza debe estar entre 0 y 1.")
    t = horizonte_dias / DIAS_BURSATILES
    m = (mu - 0.5 * sigma**2) * t
    s = sigma * np.sqrt(t)
    q = norm.ppf(1.0 - confianza)

    s_q = s0 * np.exp(m + s * q)
    precio_cola = s0 * np.exp(mu * t) * norm.cdf(q - s) / (1.0 - confianza)
    return max(float(s0 - s_q), 0.0), max(float(s0 - precio_cola), 0.0)


def probabilidad_perdida_analitica(
    mu: float, sigma: float, horizonte_dias: int
) -> float:
    """P(S_T < S0) exacta bajo GBM.

        P = Φ( -m / s ),   m = (μ - σ²/2)·T,   s = σ·√T
    """
    t = horizonte_dias / DIAS_BURSATILES
    m = (mu - 0.5 * sigma**2) * t
    s = sigma * np.sqrt(t)
    return float(norm.cdf(-m / s))


def a_dataframe(resultados: dict[str, ResultadoRiesgo]) -> pd.DataFrame:
    """Arma una tabla comparativa (filas = métricas, columnas = modelos).

    Pensada para mostrarse en la UI (Treeview) o exportarse a Excel.
    """
    def columna(r: ResultadoRiesgo) -> dict[str, float]:
        return {
            "Precio inicial S0": r.s0,
            "Precio esperado": r.precio_esperado,
            "Precio mediano": r.precio_mediano,
            "Retorno esperado (%)": r.retorno_esperado * 100,
            "Prob. de pérdida (%)": r.prob_perdida * 100,
            "VaR 95% ($)": r.var_95,
            "VaR 95% (%)": r.var_95_pct * 100,
            "CVaR 95% ($)": r.cvar_95,
            "CVaR 95% (%)": r.cvar_95_pct * 100,
            "VaR 99% ($)": r.var_99,
            "VaR 99% (%)": r.var_99_pct * 100,
            "CVaR 99% ($)": r.cvar_99,
            "CVaR 99% (%)": r.cvar_99_pct * 100,
            "Peor caso": r.peor_caso,
            "Mejor caso": r.mejor_caso,
        }
    return pd.DataFrame({nombre: columna(r) for nombre, r in resultados.items()})


if __name__ == "__main__":
    # Validación cruzada simulación vs. fórmula cerrada: python riesgo.py
    from simulador import simular

    S0, MU, SIGMA, DIAS, N = 348.24, 0.2123, 0.2807, 252, 50_000

    resultados: dict[str, ResultadoRiesgo] = {}
    for nombre in ("lognormal", "normal"):
        tray = simular(nombre, S0, MU, SIGMA, DIAS, N, semilla=42)
        resultados[nombre] = evaluar_riesgo(tray[:, -1], S0, DIAS, nombre)

    print(a_dataframe(resultados).round(2).to_string())

    print("\n--- Validación lognormal: simulado vs. analítico ---")
    sim = resultados["lognormal"]
    for conf, v_sim, c_sim in ((0.95, sim.var_95, sim.cvar_95), (0.99, sim.var_99, sim.cvar_99)):
        v_an, c_an = riesgo_analitico_lognormal(S0, MU, SIGMA, DIAS, conf)
        print(f"{conf:.0%}  VaR  sim={v_sim:8.2f}  analítico={v_an:8.2f}")
        print(f"{conf:.0%}  CVaR sim={c_sim:8.2f}  analítico={c_an:8.2f}")
    print(f"P(pérdida) sim={sim.prob_perdida:.4f}  "
          f"analítica={probabilidad_perdida_analitica(MU, SIGMA, DIAS):.4f}")