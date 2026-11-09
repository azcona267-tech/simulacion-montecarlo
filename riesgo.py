"""
riesgo.py
=========
Métricas de riesgo de mercado sobre los resultados simulados:
    - Value at Risk (VaR) y Conditional VaR (CVaR / Expected Shortfall)
    - Probabilidad de pérdida (nominal y real, descontada la inflación)
    - Sharpe, Sortino y Downside Risk (anualizados)
    - Máximo Drawdown por trayectoria
Incluye la solución analítica del GBM para validar la simulación.

Convención: la PÉRDIDA se mide como  L = capital − V_T  (positiva = se perdió).
Lógica pura: no depende de Tkinter, Matplotlib ni yfinance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from constantes import DIAS_BURSATILES_ANIO, TASA_LIBRE_RIESGO_DEFECTO
from simulador import ResultadoSimulacion

Vector = NDArray[np.float64]
Matriz = NDArray[np.float64]
_NAN: float = float("nan")


class ErrorRiesgo(ValueError):
    """Entrada inválida para el análisis de riesgo (se muestra en la UI)."""


@dataclass(frozen=True)
class ResultadoRiesgo:
    """Resumen de riesgo de una simulación.

    VaR y CVaR en unidades monetarias y como fracción del capital inicial.
    `s0` es el capital inicial (o el precio inicial si se evalúa un activo).
    Las métricas ajustadas por riesgo quedan en NaN si no se calcularon.
    """
    modelo: str
    s0: float
    horizonte_dias: int
    n_simulaciones: int
    precio_esperado: float
    precio_mediano: float
    retorno_esperado: float
    prob_perdida: float
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
    # --- métricas ajustadas por riesgo (se llenan con evaluar_simulacion) ---
    valor_esperado_real: float = _NAN
    prob_perdida_real: float = _NAN
    sharpe: float = _NAN
    sortino: float = _NAN
    downside_risk: float = _NAN
    mdd_medio: float = _NAN
    mdd_mediano: float = _NAN
    mdd_peor: float = _NAN
    mdd_p95: float = _NAN
    tasa_libre: float = _NAN


def _validar(valores: Vector, capital: float, horizonte_dias: int) -> Vector:
    """Valida y convierte la entrada a un vector float64 unidimensional.

    Raises:
        ErrorRiesgo: Si hay datos insuficientes, no finitos o parámetros inválidos.
    """
    v = np.asarray(valores, dtype=np.float64)
    if v.ndim != 1 or v.size < 1_000:
        raise ErrorRiesgo("Se requiere un vector de al menos 1,000 valores finales.")
    if not np.all(np.isfinite(v)):
        raise ErrorRiesgo("Los valores finales contienen datos no finitos.")
    if capital <= 0:
        raise ErrorRiesgo("El capital inicial debe ser positivo.")
    if horizonte_dias < 1:
        raise ErrorRiesgo("El horizonte debe ser de al menos 1 día.")
    return v


def calcular_perdidas(valores_finales: Vector, capital: float) -> Vector:
    """Pérdida por trayectoria: L = capital − V_T (negativa si hubo ganancia)."""
    return capital - np.asarray(valores_finales, dtype=np.float64)


def var_cvar(perdidas: Vector, confianza: float) -> tuple[float, float]:
    """VaR y CVaR históricos sobre la distribución simulada de pérdidas.

        VaR_α  = cuantil α de L  -> con probabilidad α la pérdida NO lo supera.
        CVaR_α = E[ L | L ≥ VaR_α ]  -> pérdida media en el (1−α) peor de los casos.

    Si el cuantil implica ganancia (L < 0) se acota en 0.

    Returns:
        Tupla (VaR, CVaR) en unidades monetarias, ambos ≥ 0.
    """
    if not 0.0 < confianza < 1.0:
        raise ErrorRiesgo("La confianza debe estar entre 0 y 1.")
    var_bruto = float(np.percentile(perdidas, confianza * 100.0))
    cvar_bruto = float(perdidas[perdidas >= var_bruto].mean())
    return max(var_bruto, 0.0), max(cvar_bruto, 0.0)


def probabilidad_perdida(valores_finales: Vector, capital: float) -> float:
    """P(pérdida) = P(V_T < capital) ≈ (# trayectorias con pérdida) / N."""
    return float(np.mean(np.asarray(valores_finales) < capital))


def evaluar_riesgo(
    valores_finales: Vector, s0: float, horizonte_dias: int, modelo: str = ""
) -> ResultadoRiesgo:
    """Métricas básicas de riesgo (VaR/CVaR 95-99% y prob. de pérdida).

    Args:
        valores_finales: Vector con V_T de cada trayectoria.
        s0: Capital (o precio) inicial.
        horizonte_dias: Horizonte, solo informativo.
        modelo: Etiqueta del modelo para la UI.
    """
    p = _validar(valores_finales, s0, horizonte_dias)
    perdidas = calcular_perdidas(p, s0)
    var95, cvar95 = var_cvar(perdidas, 0.95)
    var99, cvar99 = var_cvar(perdidas, 0.99)
    return ResultadoRiesgo(
        modelo=modelo, s0=s0, horizonte_dias=horizonte_dias, n_simulaciones=int(p.size),
        precio_esperado=float(p.mean()), precio_mediano=float(np.median(p)),
        retorno_esperado=float(p.mean() / s0 - 1.0),
        prob_perdida=probabilidad_perdida(p, s0),
        var_95=var95, var_95_pct=var95 / s0, cvar_95=cvar95, cvar_95_pct=cvar95 / s0,
        var_99=var99, var_99_pct=var99 / s0, cvar_99=cvar99, cvar_99_pct=cvar99 / s0,
        peor_caso=float(p.min()), mejor_caso=float(p.max()),
    )


# --------------------------------------------------------------------------- #
#  Métricas ajustadas por riesgo (sobre las trayectorias completas)
# --------------------------------------------------------------------------- #
def razones_retorno_riesgo(
    trayectorias: Matriz,
    tasa_libre: float = TASA_LIBRE_RIESGO_DEFECTO,
    tam_lote: int = 2_000,
) -> tuple[float, float, float]:
    """Sharpe, Sortino y Downside Risk anualizados, con retornos diarios.

    Se juntan los retornos simples diarios de TODAS las trayectorias,
    r_t = V_t / V_{t−1} − 1, y se anualizan (252 días). Con rf diaria = rf/252:

        Sharpe        = (μ_anual − rf) / σ_anual            σ_anual = σ_d·√252
        Downside Risk = √E[ min(r − rf_d, 0)² ] · √252      (semidesviación)
        Sortino       = (μ_anual − rf) / Downside Risk

    Sharpe penaliza TODA la variabilidad; Sortino solo la que queda bajo la
    tasa libre de riesgo (la «mala»). Se procesa por lotes para no agotar memoria.

    Args:
        trayectorias: Matriz (n, pasos + 1) de valores nominales.
        tasa_libre: Tasa libre de riesgo anual (0.04 = 4%).
        tam_lote: Filas por lote.

    Returns:
        Tupla (sharpe, sortino, downside_risk); NaN si la dispersión es 0.
    """
    rf_d = tasa_libre / DIAS_BURSATILES_ANIO
    n = 0
    suma = suma2 = suma_neg2 = 0.0
    for ini in range(0, trayectorias.shape[0], tam_lote):
        b = trayectorias[ini:ini + tam_lote]
        previo = b[:, :-1]
        # Si el valor llegó a 0 la «rentabilidad» siguiente es 0, no NaN
        razon = np.divide(b[:, 1:], previo, out=np.ones_like(previo), where=previo > 0)
        r = razon - 1.0
        n += r.size
        suma += float(r.sum())
        suma2 += float(np.square(r).sum())
        suma_neg2 += float(np.square(np.minimum(r - rf_d, 0.0)).sum())

    media_d = suma / n
    sd_d = float(np.sqrt(max(suma2 / n - media_d**2, 0.0)))
    ret_anual = media_d * DIAS_BURSATILES_ANIO
    vol_anual = sd_d * np.sqrt(DIAS_BURSATILES_ANIO)
    downside = float(np.sqrt(suma_neg2 / n) * np.sqrt(DIAS_BURSATILES_ANIO))

    sharpe = (ret_anual - tasa_libre) / vol_anual if vol_anual > 0 else _NAN
    sortino = (ret_anual - tasa_libre) / downside if downside > 0 else _NAN
    return float(sharpe), float(sortino), downside


def maximo_drawdown(trayectorias: Matriz, tam_lote: int = 2_000) -> Vector:
    """Máximo Drawdown de cada trayectoria (valores ≤ 0).

    El drawdown en t es la caída desde el máximo previo:

        DD_t = V_t / max_{s≤t} V_s − 1        MDD = min_t DD_t

    Un MDD de −25% significa que en algún momento el capital cayó 25% desde
    su pico, aunque después se haya recuperado.

    Returns:
        Vector (n,) con el MDD de cada trayectoria.
    """
    salida = np.empty(trayectorias.shape[0], dtype=np.float64)
    for ini in range(0, trayectorias.shape[0], tam_lote):
        b = trayectorias[ini:ini + tam_lote]
        pico = np.maximum.accumulate(b, axis=1)
        salida[ini:ini + b.shape[0]] = (b / pico - 1.0).min(axis=1)
    return salida


def evaluar_simulacion(
    res: ResultadoSimulacion,
    tasa_libre: float = TASA_LIBRE_RIESGO_DEFECTO,
    etiqueta: str | None = None,
) -> ResultadoRiesgo:
    """Evaluación COMPLETA de riesgo de un `ResultadoSimulacion`.

    Combina VaR/CVaR/prob. de pérdida con Sharpe, Sortino, Downside Risk,
    Máximo Drawdown y las métricas en términos reales (descontada la inflación).

    Args:
        res: Resultado devuelto por `simulador.generar_trayectorias`.
        tasa_libre: Tasa libre de riesgo anual.
        etiqueta: Nombre del modelo para la UI (por defecto, la distribución).
    """
    p = res.params
    base = evaluar_riesgo(res.finales, p.capital, p.horizonte_dias, etiqueta or p.distribucion)
    sharpe, sortino, downside = razones_retorno_riesgo(res.trayectorias, tasa_libre)
    mdd = maximo_drawdown(res.trayectorias)
    return replace(
        base,
        valor_esperado_real=float(res.finales_reales.mean()),
        prob_perdida_real=float(np.mean(res.finales_reales < p.capital)),
        sharpe=sharpe, sortino=sortino, downside_risk=downside,
        mdd_medio=float(mdd.mean()), mdd_mediano=float(np.median(mdd)),
        mdd_peor=float(mdd.min()), mdd_p95=float(np.percentile(mdd, 5)),
        tasa_libre=tasa_libre,
    )


# --------------------------------------------------------------------------- #
#  Solución analítica (forma cerrada) del GBM: valida la simulación
# --------------------------------------------------------------------------- #
def riesgo_analitico_lognormal(
    capital: float, mu: float, sigma: float, pasos: int, confianza: float
) -> tuple[float, float]:
    """VaR y CVaR exactos bajo GBM (valor final lognormal).

    Con T = pasos/252, m = (μ − σ²/2)·T, s = σ·√T y q = Φ⁻¹(1−α):
        V_q  = C · exp(m + s·q)                    -> cuantil (1−α) del valor final
        VaR  = C − V_q
        CVaR = C − C·e^{μT} · Φ(q − s) / (1−α)     (esperanza truncada lognormal)

    Args:
        pasos: Pasos BURSÁTILES (252 = 1 año), no días calendario.

    Returns:
        Tupla (VaR, CVaR) en unidades monetarias.
    """
    if not 0.0 < confianza < 1.0:
        raise ErrorRiesgo("La confianza debe estar entre 0 y 1.")
    t = pasos / DIAS_BURSATILES_ANIO
    m = (mu - 0.5 * sigma**2) * t
    s = sigma * np.sqrt(t)
    q = norm.ppf(1.0 - confianza)
    v_q = capital * np.exp(m + s * q)
    v_cola = capital * np.exp(mu * t) * norm.cdf(q - s) / (1.0 - confianza)
    return max(float(capital - v_q), 0.0), max(float(capital - v_cola), 0.0)


def probabilidad_perdida_analitica(mu: float, sigma: float, pasos: int) -> float:
    """P(V_T < C) exacta bajo GBM:  Φ(−m/s),  m = (μ−σ²/2)·T,  s = σ·√T."""
    t = pasos / DIAS_BURSATILES_ANIO
    m = (mu - 0.5 * sigma**2) * t
    return float(norm.cdf(-m / (sigma * np.sqrt(t))))


def a_dataframe(resultados: dict[str, ResultadoRiesgo]) -> pd.DataFrame:
    """Tabla comparativa (filas = métricas, columnas = modelos) para la UI y Excel."""
    def columna(r: ResultadoRiesgo) -> dict[str, float]:
        return {
            "Capital inicial": r.s0,
            "Valor esperado": r.precio_esperado,
            "Valor esperado (real)": r.valor_esperado_real,
            "Valor mediano": r.precio_mediano,
            "Retorno esperado (%)": r.retorno_esperado * 100,
            "Prob. de pérdida (%)": r.prob_perdida * 100,
            "Prob. de pérdida real (%)": r.prob_perdida_real * 100,
            "VaR 95% ($)": r.var_95, "VaR 95% (%)": r.var_95_pct * 100,
            "CVaR 95% ($)": r.cvar_95, "CVaR 95% (%)": r.cvar_95_pct * 100,
            "VaR 99% ($)": r.var_99, "VaR 99% (%)": r.var_99_pct * 100,
            "CVaR 99% ($)": r.cvar_99, "CVaR 99% (%)": r.cvar_99_pct * 100,
            "Sharpe": r.sharpe, "Sortino": r.sortino,
            "Downside Risk (% anual)": r.downside_risk * 100,
            "Máx. Drawdown medio (%)": r.mdd_medio * 100,
            "Máx. Drawdown peor (%)": r.mdd_peor * 100,
            "Peor caso": r.peor_caso, "Mejor caso": r.mejor_caso,
        }
    return pd.DataFrame({nombre: columna(r) for nombre, r in resultados.items()})


if __name__ == "__main__":
    # Comparación entre distribuciones + validación analítica: python riesgo.py
    from simulador import ParametrosSimulacion, generar_trayectorias
    from utilidades import dias_a_pasos

    CAP, MU, SIG, DIAS, N, RF = 10_000.0, 0.12, 0.28, 365, 50_000, 0.04
    pasos = dias_a_pasos(DIAS)

    resultados: dict[str, ResultadoRiesgo] = {}
    for dist in ("lognormal", "normal", "uniforme", "t_student"):
        sim = generar_trayectorias(
            ParametrosSimulacion(CAP, MU, SIG, DIAS, N, dist, inflacion=0.03, semilla=42)
        )
        resultados[dist] = evaluar_simulacion(sim, RF)
    print(a_dataframe(resultados).round(2).to_string())

    print("\n--- Validación lognormal: simulado vs. analítico ---")
    ln = resultados["lognormal"]
    for conf, v_sim, c_sim in ((0.95, ln.var_95, ln.cvar_95), (0.99, ln.var_99, ln.cvar_99)):
        v_an, c_an = riesgo_analitico_lognormal(CAP, MU, SIG, pasos, conf)
        print(f"{conf:.0%}  VaR  sim={v_sim:9.2f}  analítico={v_an:9.2f}")
        print(f"{conf:.0%}  CVaR sim={c_sim:9.2f}  analítico={c_an:9.2f}")
    print(f"P(pérdida)  sim={ln.prob_perdida:.4f}  analítica={probabilidad_perdida_analitica(MU, SIG, pasos):.4f}")
    print(f"Sharpe aprox. teórico (μ−rf)/σ = {(MU - RF) / SIG:.3f}   simulado = {ln.sharpe:.3f}")