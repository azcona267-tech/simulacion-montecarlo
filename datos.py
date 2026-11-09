"""
datos.py
========
Capa de datos del sistema: descarga el histórico desde Yahoo Finance,
lo limpia y estima los parámetros (rentabilidad y volatilidad) que
alimentan el motor Monte Carlo.

Este módulo NO depende de Tkinter ni de Matplotlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

DIAS_BURSATILES: int = 252          # convención: días de negociación por año
MIN_OBSERVACIONES: int = 60         # mínimo de datos para estimar con sentido
PERIODOS_VALIDOS: tuple[str, ...] = ("1y", "2y", "5y", "10y", "max")
_PATRON_TICKER = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")


class ErrorDatos(Exception):
    """Error controlado de la capa de datos (se muestra al usuario en la UI)."""


@dataclass(frozen=True)
class ParametrosMercado:
    """Parámetros estimados a partir del histórico de un activo.

    Atributos:
        ticker: Símbolo del activo (ej. "AAPL").
        precio_inicial: Último precio de cierre ajustado (S0 de la simulación).
        mu_log_diario: Media de los retornos logarítmicos diarios.
        sigma_diario: Desviación estándar de los retornos logarítmicos diarios.
        mu_anual: Rentabilidad esperada anualizada del GBM (drift μ).
        sigma_anual: Volatilidad anualizada (σ).
        n_observaciones: Cantidad de retornos usados en la estimación.
        fecha_inicio: Primera fecha del histórico usado.
        fecha_fin: Última fecha del histórico usado.
    """
    ticker: str
    precio_inicial: float
    mu_log_diario: float
    sigma_diario: float
    mu_anual: float
    sigma_anual: float
    n_observaciones: int
    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp


def normalizar_ticker(ticker: str) -> str:
    """Limpia y valida el ticker ingresado por el usuario.

    Args:
        ticker: Texto crudo del campo de entrada.

    Returns:
        Ticker en mayúsculas y sin espacios.

    Raises:
        ErrorDatos: Si está vacío o tiene caracteres no permitidos.
    """
    limpio = (ticker or "").strip().upper()
    if not limpio:
        raise ErrorDatos("Ingresa un ticker (ej. AAPL, MSFT, TSLA).")
    if not _PATRON_TICKER.match(limpio):
        raise ErrorDatos(f"Ticker inválido: '{limpio}'.")
    return limpio


def descargar_historico(ticker: str, periodo: str = "5y") -> pd.DataFrame:
    """Descarga el histórico diario OHLCV desde Yahoo Finance.

    Usa precios ajustados por splits y dividendos (auto_adjust=True),
    necesarios para que los retornos reflejen el rendimiento total real.

    Args:
        ticker: Símbolo del activo.
        periodo: Ventana histórica ("1y", "2y", "5y", "10y", "max").

    Returns:
        DataFrame indexado por fecha (sin zona horaria).

    Raises:
        ErrorDatos: Si el periodo es inválido, falla la conexión o no hay datos.
    """
    ticker = normalizar_ticker(ticker)
    if periodo not in PERIODOS_VALIDOS:
        raise ErrorDatos(f"Periodo inválido. Usa uno de: {PERIODOS_VALIDOS}.")

    try:
        df = yf.Ticker(ticker).history(period=periodo, auto_adjust=True)
    except Exception as exc:  # yfinance puede lanzar errores de red/HTTP variados
        raise ErrorDatos(f"No se pudo conectar con Yahoo Finance: {exc}") from exc

    if df is None or df.empty:
        raise ErrorDatos(f"Sin datos para '{ticker}'. Verifica que el ticker exista.")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.Series:
    """Extrae y depura la serie de precios de cierre.

    Pasos: ordena por fecha, elimina duplicados, descarta nulos y
    precios no positivos (el logaritmo exige P > 0).

    Args:
        df: DataFrame crudo devuelto por `descargar_historico`.

    Returns:
        Serie `Close` limpia, indexada por fecha.

    Raises:
        ErrorDatos: Si faltan datos suficientes tras la limpieza.
    """
    if "Close" not in df.columns:
        raise ErrorDatos("El histórico descargado no contiene la columna 'Close'.")

    precios = (
        df["Close"]
        .sort_index()
        .loc[lambda s: ~s.index.duplicated(keep="last")]
        .dropna()
    )
    precios = precios[precios > 0].astype(float)
    precios.name = "Close"

    if len(precios) < MIN_OBSERVACIONES + 1:
        raise ErrorDatos(
            f"Histórico insuficiente: {len(precios)} datos "
            f"(mínimo {MIN_OBSERVACIONES + 1})."
        )
    return precios


def calcular_retornos_log(precios: pd.Series) -> pd.Series:
    """Calcula los retornos logarítmicos diarios.

    Fórmula:
        r_t = ln(P_t / P_{t-1})

    Args:
        precios: Serie de precios de cierre limpios.

    Returns:
        Serie de retornos logarítmicos (sin el primer NaN).
    """
    return np.log(precios / precios.shift(1)).dropna().rename("retorno_log")


def estimar_parametros(
    ticker: str, precios: pd.Series, retornos: pd.Series
) -> ParametrosMercado:
    """Estima μ y σ anualizados para el Movimiento Browniano Geométrico.

    Fórmulas:
        m = media(r_t)                 -> drift diario de los log-retornos
        s = desv.est.(r_t), ddof=1     -> volatilidad diaria muestral
        σ_anual = s · √252             -> la varianza escala con el tiempo
        μ_anual = 252·m + σ²/2         -> drift del GBM (corrección de Itô)

    Args:
        ticker: Símbolo del activo.
        precios: Serie de precios limpios.
        retornos: Serie de retornos logarítmicos.

    Returns:
        Objeto `ParametrosMercado` inmutable.
    """
    m = float(retornos.mean())
    s = float(retornos.std(ddof=1))
    sigma_anual = s * np.sqrt(DIAS_BURSATILES)
    mu_anual = m * DIAS_BURSATILES + 0.5 * sigma_anual**2

    return ParametrosMercado(
        ticker=ticker,
        precio_inicial=float(precios.iloc[-1]),
        mu_log_diario=m,
        sigma_diario=s,
        mu_anual=float(mu_anual),
        sigma_anual=float(sigma_anual),
        n_observaciones=int(len(retornos)),
        fecha_inicio=precios.index[0],
        fecha_fin=precios.index[-1],
    )


def cargar_mercado(
    ticker: str, periodo: str = "5y"
) -> tuple[pd.Series, pd.Series, ParametrosMercado]:
    """Pipeline completo: descarga -> limpieza -> retornos -> parámetros.

    Es el único punto de entrada que debería llamar la interfaz.

    Args:
        ticker: Símbolo del activo.
        periodo: Ventana histórica.

    Returns:
        Tupla (precios, retornos_log, parametros).
    """
    ticker = normalizar_ticker(ticker)
    precios = limpiar_datos(descargar_historico(ticker, periodo))
    retornos = calcular_retornos_log(precios)
    return precios, retornos, estimar_parametros(ticker, precios, retornos)


if __name__ == "__main__":
    # Prueba rápida por consola: python datos.py
    try:
        _, _, p = cargar_mercado("AAPL", "5y")
        print(f"Ticker           : {p.ticker}")
        print(f"Periodo          : {p.fecha_inicio:%Y-%m-%d} -> {p.fecha_fin:%Y-%m-%d}")
        print(f"Observaciones    : {p.n_observaciones}")
        print(f"Precio inicial S0: {p.precio_inicial:,.2f}")
        print(f"mu anual         : {p.mu_anual:.2%}")
        print(f"sigma anual      : {p.sigma_anual:.2%}")
    except ErrorDatos as e:
        print(f"[ErrorDatos] {e}")