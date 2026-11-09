"""
utilidades.py
=============
Funciones auxiliares sin dependencias de UI: conversión de horizontes,
formato de números y lectura segura de campos de texto.
"""
from __future__ import annotations

from datetime import datetime

from constantes import CARPETAS_PROYECTO, DIAS_BURSATILES_ANIO, DIAS_CALENDARIO_ANIO


def dias_a_pasos(dias_calendario: int) -> int:
    """Convierte días calendario a pasos bursátiles.

    La UI ofrece horizontes de 30/90/180/365 días CALENDARIO, pero el mercado
    solo opera ~252 días al año:

        pasos = round(días × 252 / 365)        (365 días -> 252 pasos)

    Args:
        dias_calendario: Horizonte en días calendario (>= 1).

    Returns:
        Número de pasos de simulación (>= 1).
    """
    if dias_calendario < 1:
        raise ValueError("El horizonte debe ser de al menos 1 día.")
    return max(1, round(dias_calendario * DIAS_BURSATILES_ANIO / DIAS_CALENDARIO_ANIO))


def formatear_moneda(valor: float, simbolo: str = "$") -> str:
    """Formatea un monto: 12345.6 -> '$12,345.60'."""
    return f"{simbolo}{valor:,.2f}"


def formatear_pct(valor: float, decimales: int = 2) -> str:
    """Formatea una fracción como porcentaje: 0.2123 -> '21.23%'."""
    return f"{valor * 100:.{decimales}f}%"


def a_float(texto: str, nombre: str = "valor") -> float:
    """Convierte el texto de un campo de entrada a float de forma tolerante.

    Acepta coma decimal, espacios, separadores de miles con espacio y el
    símbolo '%'.

    Raises:
        ValueError: Con un mensaje listo para mostrar en la UI.
    """
    limpio = (texto or "").strip().replace("%", "").replace(" ", "").replace(",", ".")
    if not limpio:
        raise ValueError(f"Ingresa un valor para «{nombre}».")
    try:
        return float(limpio)
    except ValueError as exc:
        raise ValueError(f"«{nombre}» debe ser numérico (recibido: '{texto}').") from exc


def asegurar_carpetas() -> None:
    """Crea las carpetas del proyecto (iconos, reportes, resultados...) si faltan."""
    for carpeta in CARPETAS_PROYECTO:
        carpeta.mkdir(parents=True, exist_ok=True)


def nombre_con_fecha(prefijo: str, extension: str) -> str:
    """Nombre de archivo único: ('reporte', 'xlsx') -> 'reporte_20261109_153045.xlsx'."""
    return f"{prefijo}_{datetime.now():%Y%m%d_%H%M%S}.{extension.lstrip('.')}"