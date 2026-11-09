"""
indicadores.py
==============
Indicadores financieros de evaluación de inversiones sobre los resultados
simulados: ROI, VPN, TIR, rentabilidad acumulada y rentabilidad anualizada.

Dos niveles:
    - Funciones sobre vectores de valores finales (una por trayectoria).
    - Funciones sobre flujos de caja arbitrarios (`vpn_flujos`, `tir_flujos`).

Lógica pura NumPy/SciPy: no depende de la UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from constantes import DIAS_BURSATILES_ANIO, TASA_DESCUENTO_DEFECTO
from simulador import ResultadoSimulacion

Vector = NDArray[np.float64]
Matriz = NDArray[np.float64]


class ErrorIndicadores(ValueError):
    """Entrada inválida para el cálculo de indicadores (se muestra en la UI)."""


def _anios(pasos: int) -> float:
    """Convierte pasos bursátiles a años:  T = pasos / 252."""
    if pasos < 1:
        raise ErrorIndicadores("El horizonte debe ser de al menos 1 paso.")
    return pasos / DIAS_BURSATILES_ANIO


def _capital_valido(capital: float) -> float:
    if capital <= 0:
        raise ErrorIndicadores("El capital inicial debe ser positivo.")
    return float(capital)


# --------------------------------------------------------------------------- #
#  Rentabilidades
# --------------------------------------------------------------------------- #
def roi(valores_finales: Vector, capital: float) -> Vector:
    """Retorno sobre la inversión de cada trayectoria.

        ROI = (V_T − C) / C

    Args:
        valores_finales: Vector con V_T de cada trayectoria.
        capital: Capital inicial C.

    Returns:
        Vector de ROI (−1 = pérdida total, 0.10 = +10%).
    """
    c = _capital_valido(capital)
    return (np.asarray(valores_finales, dtype=np.float64) - c) / c


def rentabilidad_acumulada(trayectorias: Matriz) -> Matriz:
    """Rentabilidad acumulada en cada instante:  R_t = V_t / V_0 − 1.

    Su última columna coincide con el ROI. Devuelve una matriz del mismo
    tamaño que la de entrada (con 100,000 simulaciones ocupa ~200 MB).

    Args:
        trayectorias: Matriz (n, pasos + 1) de valores.
    """
    t = np.asarray(trayectorias, dtype=np.float64)
    if t.ndim != 2 or np.any(t[:, 0] <= 0):
        raise ErrorIndicadores("Las trayectorias deben ser una matriz con valor inicial positivo.")
    return t / t[:, [0]] - 1.0


def rentabilidad_anualizada(valores_finales: Vector, capital: float, pasos: int) -> Vector:
    """Rentabilidad anualizada compuesta (CAGR) de cada trayectoria.

        CAGR = (V_T / C)^(1/T) − 1,      T = pasos / 252 años

    CUIDADO con horizontes cortos: elevar a 1/T amplifica la dispersión
    (con T = 1/12, un +1% mensual se anualiza a +12.7%). Por eso la
    MEDIANA del CAGR es más representativa que la media.

    Args:
        valores_finales: Vector con V_T.
        capital: Capital inicial.
        pasos: Pasos bursátiles simulados.
    """
    c = _capital_valido(capital)
    razon = np.clip(np.asarray(valores_finales, dtype=np.float64) / c, 0.0, None)
    return razon ** (1.0 / _anios(pasos)) - 1.0


# --------------------------------------------------------------------------- #
#  VPN y TIR sobre flujos de caja arbitrarios
# --------------------------------------------------------------------------- #
def vpn_flujos(tasa: float, flujos: Vector) -> float:
    """Valor Presente Neto de una serie de flujos igualmente espaciados.

        VPN = Σ_{t=0}^{n} F_t / (1 + r)^t

    Args:
        tasa: Tasa de descuento POR PERIODO (> −100%).
        flujos: F_0 (normalmente negativo = inversión), F_1, ..., F_n.
    """
    if tasa <= -1.0:
        raise ErrorIndicadores("La tasa de descuento debe ser mayor a -100%.")
    f = np.asarray(flujos, dtype=np.float64)
    t = np.arange(f.size, dtype=np.float64)
    return float(np.sum(f / (1.0 + tasa) ** t))


def tir_flujos(flujos: Vector, periodos_por_anio: int = 1) -> float:
    """Tasa Interna de Retorno de una serie de flujos.

    Es la tasa r que hace VPN(r) = 0, y se halla numéricamente con el método
    de Brent (no tiene fórmula cerrada en general):

        Σ F_t / (1 + r)^t = 0

    Con varios cambios de signo puede haber más de una TIR; se devuelve la
    primera raíz del rango [−99.99%, 10,000%] por periodo.

    Args:
        flujos: Flujos F_0, ..., F_n (al menos uno negativo y uno positivo).
        periodos_por_anio: Para anualizar: (1 + r)^p − 1.

    Raises:
        ErrorIndicadores: Si no hay cambio de signo o no se halla raíz.
    """
    f = np.asarray(flujos, dtype=np.float64)
    if f.ndim != 1 or f.size < 2:
        raise ErrorIndicadores("Se requieren al menos 2 flujos.")
    if not (np.any(f < 0) and np.any(f > 0)):
        raise ErrorIndicadores("La TIR requiere al menos un flujo negativo y uno positivo.")
    try:
        r = brentq(lambda x: vpn_flujos(x, f), -0.9999, 100.0, xtol=1e-12)
    except ValueError as exc:
        raise ErrorIndicadores("No se encontró una TIR en el rango admitido.") from exc
    return float((1.0 + r) ** periodos_por_anio - 1.0)


# --------------------------------------------------------------------------- #
#  VPN y TIR sobre los resultados simulados (dos flujos: C hoy, V_T al final)
# --------------------------------------------------------------------------- #
def vpn_simulado(
    valores_finales: Vector, capital: float, pasos: int,
    tasa_descuento: float = TASA_DESCUENTO_DEFECTO,
) -> Vector:
    """VPN de cada trayectoria: inversión C hoy y valor V_T al final.

        VPN = −C + V_T / (1 + r)^T

    VPN > 0 significa que la inversión rindió MÁS que la tasa exigida r.

    Args:
        valores_finales: Vector con V_T.
        capital: Capital inicial.
        pasos: Pasos bursátiles simulados.
        tasa_descuento: Tasa anual exigida (costo de oportunidad), 0.08 = 8%.
    """
    if tasa_descuento <= -1.0:
        raise ErrorIndicadores("La tasa de descuento debe ser mayor a -100%.")
    c = _capital_valido(capital)
    v = np.asarray(valores_finales, dtype=np.float64)
    return -c + v / (1.0 + tasa_descuento) ** _anios(pasos)


def tir_simulada(valores_finales: Vector, capital: float, pasos: int) -> Vector:
    """TIR anual de cada trayectoria (vectorizada, forma cerrada).

    Con solo dos flujos, −C + V_T/(1+r)^T = 0 se despeja directo:

        r = (V_T / C)^(1/T) − 1

    que es EXACTAMENTE la rentabilidad anualizada. Con flujos intermedios
    habría que usar `tir_flujos` (método numérico).
    """
    return rentabilidad_anualizada(valores_finales, capital, pasos)


# --------------------------------------------------------------------------- #
#  Resumen para la UI y la exportación
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResumenIndicadores:
    """Resumen de indicadores de una simulación (fracciones, no porcentajes)."""
    tasa_descuento: float
    anios: float
    roi_medio: float
    roi_mediano: float
    roi_p05: float
    roi_p95: float
    roi_real_medio: float
    roi_real_mediano: float
    prob_roi_positivo: float
    prob_roi_real_positivo: float
    retorno_anualizado_esperado: float   # (E[V_T]/C)^(1/T) − 1
    anualizada_media: float
    anualizada_mediana: float
    anualizada_real_mediana: float
    vpn_medio: float
    vpn_mediano: float
    vpn_p05: float
    vpn_p95: float
    prob_vpn_positivo: float
    tir_mediana: float
    tir_p05: float
    tir_p95: float


def evaluar_indicadores(
    res: ResultadoSimulacion, tasa_descuento: float = TASA_DESCUENTO_DEFECTO
) -> ResumenIndicadores:
    """Calcula todos los indicadores de un `ResultadoSimulacion`.

    Args:
        res: Resultado de `simulador.generar_trayectorias`.
        tasa_descuento: Tasa anual exigida para el VPN.
    """
    cap, pasos = res.params.capital, res.pasos
    anios = _anios(pasos)
    nominal, real = res.finales, res.finales_reales

    roi_n, roi_r = roi(nominal, cap), roi(real, cap)
    cagr_n = rentabilidad_anualizada(nominal, cap, pasos)
    cagr_r = rentabilidad_anualizada(real, cap, pasos)
    vpn = vpn_simulado(nominal, cap, pasos, tasa_descuento)
    tir = tir_simulada(nominal, cap, pasos)

    roi_p05, roi_p95 = (float(x) for x in np.percentile(roi_n, [5, 95]))
    vpn_p05, vpn_p95 = (float(x) for x in np.percentile(vpn, [5, 95]))
    tir_p05, tir_p95 = (float(x) for x in np.percentile(tir, [5, 95]))

    return ResumenIndicadores(
        tasa_descuento=tasa_descuento, anios=anios,
        roi_medio=float(roi_n.mean()), roi_mediano=float(np.median(roi_n)),
        roi_p05=roi_p05, roi_p95=roi_p95,
        roi_real_medio=float(roi_r.mean()), roi_real_mediano=float(np.median(roi_r)),
        prob_roi_positivo=float(np.mean(roi_n > 0)),
        prob_roi_real_positivo=float(np.mean(roi_r > 0)),
        retorno_anualizado_esperado=float((nominal.mean() / cap) ** (1.0 / anios) - 1.0),
        anualizada_media=float(cagr_n.mean()), anualizada_mediana=float(np.median(cagr_n)),
        anualizada_real_mediana=float(np.median(cagr_r)),
        vpn_medio=float(vpn.mean()), vpn_mediano=float(np.median(vpn)),
        vpn_p05=vpn_p05, vpn_p95=vpn_p95, prob_vpn_positivo=float(np.mean(vpn > 0)),
        tir_mediana=float(np.median(tir)), tir_p05=tir_p05, tir_p95=tir_p95,
    )


def a_diccionario(r: ResumenIndicadores) -> dict[str, float]:
    """Resumen como diccionario con etiquetas en español (tablas y exportación)."""
    return {
        "Tasa de descuento (%)": r.tasa_descuento * 100,
        "Horizonte (años)": r.anios,
        "ROI medio (%)": r.roi_medio * 100,
        "ROI mediano (%)": r.roi_mediano * 100,
        "ROI percentil 5 (%)": r.roi_p05 * 100,
        "ROI percentil 95 (%)": r.roi_p95 * 100,
        "ROI real medio (%)": r.roi_real_medio * 100,
        "ROI real mediano (%)": r.roi_real_mediano * 100,
        "Prob. ROI > 0 (%)": r.prob_roi_positivo * 100,
        "Prob. ROI real > 0 (%)": r.prob_roi_real_positivo * 100,
        "Retorno anualizado del valor esperado (%)": r.retorno_anualizado_esperado * 100,
        "Rentabilidad anualizada media (%)": r.anualizada_media * 100,
        "Rentabilidad anualizada mediana (%)": r.anualizada_mediana * 100,
        "Rentabilidad anualizada real mediana (%)": r.anualizada_real_mediana * 100,
        "VPN medio ($)": r.vpn_medio,
        "VPN mediano ($)": r.vpn_mediano,
        "VPN percentil 5 ($)": r.vpn_p05,
        "VPN percentil 95 ($)": r.vpn_p95,
        "Prob. VPN > 0 (%)": r.prob_vpn_positivo * 100,
        "TIR mediana (%)": r.tir_mediana * 100,
        "TIR percentil 5 (%)": r.tir_p05 * 100,
        "TIR percentil 95 (%)": r.tir_p95 * 100,
    }


if __name__ == "__main__":
    # Validación contra teoría GBM: python indicadores.py
    from scipy.stats import norm

    from simulador import ParametrosSimulacion, generar_trayectorias

    CAP, MU, SIG, N, R = 10_000.0, 0.12, 0.28, 50_000, 0.08
    res = generar_trayectorias(
        ParametrosSimulacion(CAP, MU, SIG, 365, N, "lognormal", inflacion=0.03, semilla=42)
    )
    ind = evaluar_indicadores(res, R)
    T = ind.anios
    m = MU - 0.5 * SIG**2

    print(f"Horizonte: {T:.2f} años · tasa de descuento {R:.0%}\n")
    print(f"{'indicador':42s}{'simulado':>12s}{'teórico':>12s}")
    filas = [
        ("ROI medio (%)", ind.roi_medio * 100, (np.exp(MU * T) - 1) * 100),
        ("Retorno anualizado del valor esperado (%)", ind.retorno_anualizado_esperado * 100,
         (np.exp(MU) - 1) * 100),
        ("Rentabilidad anualizada MEDIANA (%)", ind.anualizada_mediana * 100,
         (np.exp(m) - 1) * 100),
        ("VPN medio ($)", ind.vpn_medio, -CAP + CAP * np.exp(MU * T) / (1 + R) ** T),
        ("P(VPN > 0) (%)", ind.prob_vpn_positivo * 100,
         100 * norm.cdf((m - np.log1p(R)) * T / (SIG * np.sqrt(T)))),
    ]
    for nombre, sim, teo in filas:
        print(f"{nombre:42s}{sim:12.2f}{teo:12.2f}")
    print(f"\nROI real medio: {ind.roi_real_medio:.2%} · P(ROI real > 0): {ind.prob_roi_real_positivo:.2%}")

    vpn = vpn_simulado(res.finales, CAP, res.pasos, R)
    tir = tir_simulada(res.finales, CAP, res.pasos)
    print(f"\nCoherencia VPN>0 <=> TIR>tasa en las {N:,} trayectorias: {bool(np.all((vpn > 0) == (tir > R)))}")

    v = float(res.finales[0])
    print(f"TIR numérica (1 año)  : {tir_flujos([-CAP, v]):.6f}  cerrada: {v / CAP - 1:.6f}")
    print(f"TIR numérica (2 años) : {tir_flujos([-CAP, 0.0, v]):.6f}  cerrada: {np.sqrt(v / CAP) - 1:.6f}")

    print("\n--- Por qué no conviene anualizar horizontes cortos ---")
    print(f"{'días':>6s}{'ROI mediano':>14s}{'CAGR mediano':>15s}{'CAGR medio':>13s}")
    for dias in (30, 90, 180, 365):
        r = generar_trayectorias(ParametrosSimulacion(CAP, MU, SIG, dias, 20_000, "lognormal", semilla=42))
        i = evaluar_indicadores(r, R)
        print(f"{dias:6d}{i.roi_mediano:14.2%}{i.anualizada_mediana:15.2%}{i.anualizada_media:13.2%}")