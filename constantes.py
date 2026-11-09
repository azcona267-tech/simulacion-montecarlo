"""
constantes.py
=============
Constantes globales: identidad de la app, rutas, parámetros de mercado,
límites de la simulación, etiquetas de la interfaz y paleta de colores.

Regla: este módulo no importa nada del proyecto, así cualquier otro módulo
puede importarlo sin riesgo de importaciones circulares.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
#  Identidad
# --------------------------------------------------------------------------- #
APP_NOMBRE: Final[str] = "Sistema de Simulación Financiera · Monte Carlo"
APP_VERSION: Final[str] = "1.0.0"
APP_CURSO: Final[str] = "Inteligencia de Negocios"
APP_INSTITUCION: Final[str] = "UTP"

# --------------------------------------------------------------------------- #
#  Rutas
# --------------------------------------------------------------------------- #
RUTA_BASE: Final[Path] = Path(__file__).resolve().parent
RUTA_ICONOS: Final[Path] = RUTA_BASE / "iconos"
RUTA_IMAGENES: Final[Path] = RUTA_BASE / "imagenes"
RUTA_REPORTES: Final[Path] = RUTA_BASE / "reportes"
RUTA_RESULTADOS: Final[Path] = RUTA_BASE / "resultados"
RUTA_AYUDA: Final[Path] = RUTA_BASE / "ayuda"
ARCHIVO_CONFIGURACION: Final[Path] = RUTA_BASE / "configuracion.json"
CARPETAS_PROYECTO: Final[tuple[Path, ...]] = (
    RUTA_ICONOS, RUTA_IMAGENES, RUTA_REPORTES, RUTA_RESULTADOS, RUTA_AYUDA,
)

# --------------------------------------------------------------------------- #
#  Mercado
# --------------------------------------------------------------------------- #
DIAS_BURSATILES_ANIO: Final[int] = 252      # días de negociación por año
DIAS_CALENDARIO_ANIO: Final[int] = 365      # los horizontes de la UI son calendario
PERIODOS_HISTORICO: Final[tuple[str, ...]] = ("1y", "2y", "5y", "10y", "max")
PERIODO_HISTORICO_DEFECTO: Final[str] = "5y"
TASA_LIBRE_RIESGO_DEFECTO: Final[float] = 0.04   # editable; usada en Sharpe/Sortino

# --------------------------------------------------------------------------- #
#  Simulación
# --------------------------------------------------------------------------- #
HORIZONTES_DIAS: Final[tuple[int, ...]] = (30, 90, 180, 365)
HORIZONTE_DEFECTO: Final[int] = 365
SIMULACIONES_MIN: Final[int] = 1_000
SIMULACIONES_MAX: Final[int] = 100_000
SIMULACIONES_DEFECTO: Final[int] = 5_000
MAX_CELDAS: Final[int] = 60_000_000          # tope de memoria (~480 MB en float64)
NIVELES_CONFIANZA: Final[tuple[float, float]] = (0.95, 0.99)

# Distribuciones que pide la pantalla principal (clave interna -> etiqueta UI)
DISTRIBUCIONES_UI: Final[dict[str, str]] = {
    "normal": "Normal",
    "lognormal": "LogNormal",
    "uniforme": "Uniforme",
    "triangular": "Triangular",
    "personalizada": "Personalizada",
}
# Distribuciones adicionales de distribuciones.py (comparación y análisis)
DISTRIBUCIONES_EXTRA: Final[dict[str, str]] = {
    "t_student": "t de Student",
    "poisson": "Poisson",
    "exponencial": "Exponencial",
    "beta": "Beta",
    "gamma": "Gamma",
}

VALORES_DEFECTO: Final[dict[str, float | int | str]] = {
    "ticker": "AAPL",
    "capital": 10_000.0,
    "rentabilidad_pct": 10.0,
    "volatilidad_pct": 20.0,
    "inflacion_pct": 3.0,
    "n_simulaciones": SIMULACIONES_DEFECTO,
    "horizonte_dias": HORIZONTE_DEFECTO,
    "distribucion": "lognormal",
}

# --------------------------------------------------------------------------- #
#  Tema visual (compartido por la UI de CustomTkinter y Matplotlib)
# --------------------------------------------------------------------------- #
TEMA: Final[dict[str, str]] = {
    "fondo": "#242424",        # coincide con el modo oscuro de CustomTkinter
    "fondo_ejes": "#1c1c1c",
    "panel": "#2b2b2b",
    "panel_alt": "#333333",
    "texto": "#e8e8e8",
    "texto_suave": "#9a9a9a",
    "borde": "#444444",
    "grid": "#333333",
    "acento": "#4cc9f0",
    "acento_hover": "#3aa8cc",
    "ganancia": "#2ecc71",
    "perdida": "#ff4d6d",
    "media": "#ffd166",
    "mediana": "#c77dff",
    "referencia": "#ffffff",
}
CICLO_MODELOS: Final[tuple[str, ...]] = ("#4cc9f0", "#ff9f1c", "#2ecc71", "#c77dff")
FUENTE_UI: Final[str] = "Segoe UI"
FUENTE_MONO: Final[str] = "Consolas"