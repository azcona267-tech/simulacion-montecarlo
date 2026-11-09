"""
graficos.py
===========
Capa de visualización: cada función recibe datos (arrays / Series de Pandas /
resultados de riesgo) y DEVUELVE un `matplotlib.figure.Figure` listo para
incrustar en Tkinter con `FigureCanvasTkAgg`.

Decisión de diseño: se usa `Figure` directamente y NO `pyplot`. Pyplot mantiene
un registro global de figuras y abre ventanas propias, lo que en una app
Tkinter provoca ventanas fantasma y fugas de memoria.

Tema oscuro, pensado para convivir con CustomTkinter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["text.parse_math"] = False
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

if TYPE_CHECKING:  # solo para tipado; evita acoplar graficos con riesgo en tiempo de ejecución
    from riesgo import ResultadoRiesgo

Matriz = NDArray[np.float64]
Vector = NDArray[np.float64]

TEMA: dict[str, str] = {
    "fondo": "#242424",        # coincide con el modo oscuro de CustomTkinter
    "fondo_ejes": "#1c1c1c",
    "texto": "#e8e8e8",
    "texto_suave": "#9a9a9a",
    "borde": "#444444",
    "grid": "#333333",
    "acento": "#4cc9f0",
    "ganancia": "#2ecc71",
    "perdida": "#ff4d6d",
    "media": "#ffd166",
    "mediana": "#c77dff",
    "referencia": "#ffffff",
}
_CICLO_MODELOS: tuple[str, ...] = ("#4cc9f0", "#ff9f1c", "#2ecc71", "#c77dff")
_PERCENTILES_CONO: tuple[float, ...] = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------- #
#  Utilidades internas
# --------------------------------------------------------------------------- #
def _moneda(valor: float, _pos: int | None = None) -> str:
    """Formatea un número como moneda para los ejes."""
    return f"${valor:,.0f}"


def _nueva_figura(
    titulo: str, subtitulo: str = "", tamano: tuple[float, float] = (9.0, 5.2)
) -> tuple[Figure, "object"]:
    """Crea figura + ejes con el tema oscuro aplicado."""
    fig = Figure(figsize=tamano, dpi=100, facecolor=TEMA["fondo"], layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_facecolor(TEMA["fondo_ejes"])
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TEMA["borde"])
    ax.tick_params(colors=TEMA["texto_suave"], labelsize=9)
    ax.grid(True, color=TEMA["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(titulo, loc="left", color=TEMA["texto"], fontsize=13,
                 fontweight="bold", pad=24)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes,
                color=TEMA["texto_suave"], fontsize=9, va="bottom")
    return fig, ax


def _leyenda(ax: "object", handles: list[Line2D]) -> None:
    """Leyenda con el estilo del tema."""
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=True,
              facecolor=TEMA["fondo"], edgecolor=TEMA["borde"],
              labelcolor=TEMA["texto"], framealpha=0.9)


# --------------------------------------------------------------------------- #
#  Gráfico 0: estado vacío (para el arranque de la UI)
# --------------------------------------------------------------------------- #
def figura_vacia(mensaje: str = "Ingresa un ticker y presiona «Simular»") -> Figure:
    """Figura placeholder para mostrar antes de la primera simulación."""
    fig = Figure(figsize=(9, 5.2), dpi=100, facecolor=TEMA["fondo"], layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_facecolor(TEMA["fondo"])
    ax.axis("off")
    ax.text(0.5, 0.5, mensaje, ha="center", va="center",
            color=TEMA["texto_suave"], fontsize=13, transform=ax.transAxes)
    return fig


# --------------------------------------------------------------------------- #
#  Gráfico 1: espagueti de trayectorias
# --------------------------------------------------------------------------- #
def figura_espagueti(
    trayectorias: Matriz,
    historico: pd.Series | None = None,
    n_mostrar: int = 150,
    ventana_historica: int = 120,
    ticker: str = "",
    semilla: int = 7,
) -> Figure:
    """Gráfico de espagueti: una muestra de las trayectorias simuladas.

    Por rendimiento solo se DIBUJA una muestra aleatoria (`n_mostrar`);
    todas las estadísticas (media, riesgo) usan las N trayectorias completas.
    Cada línea se colorea según termine por encima (verde) o por debajo (rojo)
    del precio inicial.

    Args:
        trayectorias: Matriz (n_simulaciones, horizonte_dias + 1).
        historico: Serie de precios pasados (opcional) para dar contexto.
        n_mostrar: Cantidad de trayectorias a dibujar.
        ventana_historica: Últimos N días del histórico a mostrar.
        ticker: Símbolo, solo para el título.
        semilla: Semilla del muestreo (la muestra es reproducible).

    Returns:
        Figure de Matplotlib.
    """
    n, columnas = trayectorias.shape
    x = np.arange(columnas)
    s0 = float(trayectorias[0, 0])

    rng = np.random.default_rng(semilla)
    idx = rng.choice(n, size=min(n_mostrar, n), replace=False)
    muestra = trayectorias[idx]

    colores = [TEMA["ganancia"] if fin >= s0 else TEMA["perdida"] for fin in muestra[:, -1]]
    segmentos = np.stack([np.broadcast_to(x, muestra.shape), muestra], axis=-1)

    titulo = f"Trayectorias simuladas{f' · {ticker}' if ticker else ''}"
    fig, ax = _nueva_figura(
        titulo, f"{n:,} simulaciones (se dibujan {len(idx)}) · horizonte {columnas - 1} días"
    )
    ax.add_collection(LineCollection(segmentos, colors=colores, linewidths=0.6, alpha=0.35))

    ax.plot(x, trayectorias.mean(axis=0), color=TEMA["media"], lw=2.0, zorder=5)
    ax.axhline(s0, color=TEMA["referencia"], lw=0.9, ls="--", alpha=0.6, zorder=4)

    x_min, y_min, y_max = 0, float(muestra.min()), float(muestra.max())
    handles = [
        Line2D([0], [0], color=TEMA["ganancia"], lw=2, label="Termina sobre S0"),
        Line2D([0], [0], color=TEMA["perdida"], lw=2, label="Termina bajo S0"),
        Line2D([0], [0], color=TEMA["media"], lw=2, label="Media de todas"),
    ]
    if historico is not None and len(historico) > 1:
        h = historico.iloc[-ventana_historica:].to_numpy(dtype=np.float64)
        xh = np.arange(-len(h) + 1, 1)
        ax.plot(xh, h, color=TEMA["texto"], lw=1.4, zorder=6)
        ax.axvline(0, color=TEMA["borde"], lw=1.0)
        x_min = int(xh[0])
        y_min, y_max = min(y_min, float(h.min())), max(y_max, float(h.max()))
        handles.append(Line2D([0], [0], color=TEMA["texto"], lw=2, label="Histórico real"))

    ax.set_xlim(x_min, columnas - 1)
    ax.set_ylim(y_min * 0.95, y_max * 1.05)
    ax.set_xlabel("Días bursátiles", color=TEMA["texto_suave"])
    ax.set_ylabel("Precio", color=TEMA["texto_suave"])
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
    _leyenda(ax, handles)
    return fig


# --------------------------------------------------------------------------- #
#  Gráfico 2: cono de incertidumbre (fan chart)
# --------------------------------------------------------------------------- #
def figura_cono(trayectorias: Matriz, ticker: str = "") -> Figure:
    """Cono de incertidumbre: bandas de percentiles del precio en cada día.

    Para cada día t se calculan los percentiles 5, 25, 50, 75 y 95 de la
    distribución del precio ENTRE las trayectorias. Las bandas se leen así:
    "con 90% de probabilidad el precio del día t estará entre P5 y P95".

    Tras incrustar la figura en Tkinter, llama a `activar_hover_cono(fig)`
    para habilitar el cursor interactivo.

    Args:
        trayectorias: Matriz (n_simulaciones, horizonte_dias + 1).
        ticker: Símbolo, solo para el título.

    Returns:
        Figure de Matplotlib.
    """
    n, columnas = trayectorias.shape
    x = np.arange(columnas)
    s0 = float(trayectorias[0, 0])
    perc = np.percentile(trayectorias, _PERCENTILES_CONO, axis=0)   # (5, columnas)
    media = trayectorias.mean(axis=0)

    titulo = f"Cono de incertidumbre{f' · {ticker}' if ticker else ''}"
    fig, ax = _nueva_figura(titulo, f"Percentiles diarios sobre {n:,} simulaciones")

    ax.fill_between(x, perc[0], perc[4], color=TEMA["acento"], alpha=0.18, lw=0)
    ax.fill_between(x, perc[1], perc[3], color=TEMA["acento"], alpha=0.35, lw=0)
    ax.plot(x, perc[2], color=TEMA["mediana"], lw=2.0)
    ax.plot(x, media, color=TEMA["media"], lw=1.6, ls="--")
    ax.axhline(s0, color=TEMA["referencia"], lw=0.9, ls=":", alpha=0.6)

    ax.set_xlim(0, columnas - 1)
    ax.set_xlabel("Días bursátiles", color=TEMA["texto_suave"])
    ax.set_ylabel("Precio", color=TEMA["texto_suave"])
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
    _leyenda(ax, [
        Line2D([0], [0], color=TEMA["acento"], lw=8, alpha=0.30, label="Rango 5%-95% (90%)"),
        Line2D([0], [0], color=TEMA["acento"], lw=8, alpha=0.55, label="Rango 25%-75% (50%)"),
        Line2D([0], [0], color=TEMA["mediana"], lw=2, label="Mediana"),
        Line2D([0], [0], color=TEMA["media"], lw=2, ls="--", label="Media"),
    ])
    fig.datos_cono = {"ax": ax, "perc": perc, "media": media}  # type: ignore[attr-defined]
    return fig


def activar_hover_cono(fig: Figure) -> None:
    """Activa el cursor interactivo del cono (línea vertical + tooltip).

    IMPORTANTE: llamar DESPUÉS de crear el `FigureCanvasTkAgg`, porque al
    incrustar la figura se reemplaza su canvas y los eventos conectados antes
    se pierden.

    Args:
        fig: Figura devuelta por `figura_cono`.
    """
    datos = getattr(fig, "datos_cono", None)
    if datos is None:
        return
    ax, perc, media = datos["ax"], datos["perc"], datos["media"]

    linea = ax.axvline(0, color=TEMA["texto_suave"], lw=0.8, ls=":", visible=False)
    nota = ax.annotate(
        "", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
        color=TEMA["texto"], fontsize=8.5, visible=False, zorder=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#111111", "ec": TEMA["borde"], "alpha": 0.95},
    )

    def al_mover(evento: "object") -> None:
        if evento.inaxes is not ax or evento.xdata is None:  # type: ignore[attr-defined]
            if nota.get_visible():
                nota.set_visible(False)
                linea.set_visible(False)
                fig.canvas.draw_idle()
            return
        i = int(np.clip(round(evento.xdata), 0, perc.shape[1] - 1))  # type: ignore[attr-defined]
        linea.set_xdata([i, i])
        nota.xy = (i, perc[2, i])
        nota.set_text(
            f"Día {i}\n"
            f"P95:     ${perc[4, i]:,.2f}\n"
            f"Media:   ${media[i]:,.2f}\n"
            f"Mediana: ${perc[2, i]:,.2f}\n"
            f"P5:      ${perc[0, i]:,.2f}"
        )
        nota.set_visible(True)
        linea.set_visible(True)
        fig.canvas.draw_idle()

    fig._hover_cono = al_mover  # type: ignore[attr-defined]  # referencia fuerte al callback
    fig.canvas.mpl_connect("motion_notify_event", al_mover)


# --------------------------------------------------------------------------- #
#  Gráfico 3: histograma de precios finales con VaR / CVaR
# --------------------------------------------------------------------------- #
def figura_histograma(precios_finales: Vector, riesgo: "ResultadoRiesgo") -> Figure:
    """Histograma de precios finales con las métricas de riesgo marcadas.

    La cola de pérdidas severas (por debajo del precio asociado al VaR 95%) se
    pinta en rojo. El eje X se recorta al percentil 99.5 para que la cola
    derecha de la lognormal no aplaste el gráfico.

    Precio asociado al VaR:  S0 - VaR  (nivel de precio que marca la pérdida).

    Args:
        precios_finales: Vector de S_T de cada trayectoria.
        riesgo: Resultado de `riesgo.evaluar_riesgo`.

    Returns:
        Figure de Matplotlib.
    """
    p = np.asarray(precios_finales, dtype=np.float64)
    s0 = riesgo.s0
    lim_inf, lim_sup = float(p.min()), float(np.percentile(p, 99.5))
    densidad, bordes = np.histogram(p, bins=70, range=(lim_inf, lim_sup), density=True)
    centros = (bordes[:-1] + bordes[1:]) / 2
    ancho = np.diff(bordes)

    umbral_95 = s0 - riesgo.var_95 if riesgo.var_95 > 0 else -np.inf
    colores = [TEMA["perdida"] if c <= umbral_95 else TEMA["acento"] for c in centros]

    fig, ax = _nueva_figura(
        f"Distribución de precios finales · modelo {riesgo.modelo}",
        f"{riesgo.n_simulaciones:,} simulaciones · horizonte {riesgo.horizonte_dias} días",
    )
    ax.bar(centros, densidad, width=ancho, color=colores, alpha=0.85, lw=0)

    ax.axvline(s0, color=TEMA["referencia"], lw=1.3, ls="--",
               label=f"Precio inicial: ${s0:,.2f}")
    ax.axvline(riesgo.precio_esperado, color=TEMA["media"], lw=1.6,
               label=f"Media: ${riesgo.precio_esperado:,.2f}")
    ax.axvline(riesgo.precio_mediano, color=TEMA["mediana"], lw=1.6,
               label=f"Mediana: ${riesgo.precio_mediano:,.2f}")
    if riesgo.var_95 > 0:
        ax.axvline(s0 - riesgo.var_95, color=TEMA["perdida"], lw=1.6,
                   label=f"VaR 95%: -${riesgo.var_95:,.2f} (precio ${s0 - riesgo.var_95:,.2f})")
    if riesgo.var_99 > 0:
        ax.axvline(s0 - riesgo.var_99, color="#b5179e", lw=1.6,
                   label=f"VaR 99%: -${riesgo.var_99:,.2f} (precio ${s0 - riesgo.var_99:,.2f})")

    resumen = (
        f"P(pérdida): {riesgo.prob_perdida:.1%}\n"
        f"CVaR 95%:  -${riesgo.cvar_95:,.2f}  ({riesgo.cvar_95_pct:.1%})\n"
        f"CVaR 99%:  -${riesgo.cvar_99:,.2f}  ({riesgo.cvar_99_pct:.1%})"
    )
    ax.text(0.98, 0.97, resumen, transform=ax.transAxes, ha="right", va="top",
            color=TEMA["texto"], fontsize=9, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "fc": TEMA["fondo"], "ec": TEMA["borde"]})

    ax.set_xlim(lim_inf, lim_sup)
    ax.set_xlabel("Precio final", color=TEMA["texto_suave"])
    ax.set_ylabel("Densidad", color=TEMA["texto_suave"])
    ax.xaxis.set_major_formatter(FuncFormatter(_moneda))
    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor=TEMA["fondo"],
              edgecolor=TEMA["borde"], labelcolor=TEMA["texto"], framealpha=0.9)
    return fig


# --------------------------------------------------------------------------- #
#  Gráfico 4: comparación entre modelos
# --------------------------------------------------------------------------- #
def figura_comparacion(finales_por_modelo: Mapping[str, Vector], s0: float) -> Figure:
    """Superpone las distribuciones de precio final de varios modelos.

    Sirve para mostrar visualmente en qué se parecen y en qué difieren los
    supuestos Normal y Lognormal (sobre todo en las colas).

    Args:
        finales_por_modelo: Diccionario {nombre_modelo: vector de S_T}.
        s0: Precio inicial (línea de referencia).

    Returns:
        Figure de Matplotlib.
    """
    todos = np.concatenate([np.asarray(v, dtype=np.float64) for v in finales_por_modelo.values()])
    lim_inf, lim_sup = float(todos.min()), float(np.percentile(todos, 99.5))
    bordes = np.linspace(lim_inf, lim_sup, 71)
    centros = (bordes[:-1] + bordes[1:]) / 2

    fig, ax = _nueva_figura(
        "Comparación de modelos", "Distribución del precio final bajo cada supuesto"
    )
    for i, (nombre, valores) in enumerate(finales_por_modelo.items()):
        color = _CICLO_MODELOS[i % len(_CICLO_MODELOS)]
        dens, _ = np.histogram(valores, bins=bordes, density=True)
        ax.fill_between(centros, dens, color=color, alpha=0.22, step="mid", lw=0)
        ax.step(centros, dens, where="mid", color=color, lw=1.8, label=nombre.capitalize())

    ax.axvline(s0, color=TEMA["referencia"], lw=1.2, ls="--", label=f"S0: ${s0:,.2f}")
    ax.set_xlim(lim_inf, lim_sup)
    ax.set_xlabel("Precio final", color=TEMA["texto_suave"])
    ax.set_ylabel("Densidad", color=TEMA["texto_suave"])
    ax.xaxis.set_major_formatter(FuncFormatter(_moneda))
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor=TEMA["fondo"],
              edgecolor=TEMA["borde"], labelcolor=TEMA["texto"], framealpha=0.9)
    return fig


if __name__ == "__main__":
    # Genera PNGs de prueba: python graficos.py
    from pathlib import Path

    from riesgo import evaluar_riesgo
    from simulador import simular

    S0, MU, SIGMA, DIAS, N = 348.24, 0.2123, 0.2807, 252, 5_000
    salida = Path("prueba_graficos")
    salida.mkdir(exist_ok=True)

    tray_ln = simular("lognormal", S0, MU, SIGMA, DIAS, N, semilla=42)
    tray_nr = simular("normal", S0, MU, SIGMA, DIAS, N, semilla=42)
    res = evaluar_riesgo(tray_ln[:, -1], S0, DIAS, "lognormal")

    figuras = {
        "espagueti": figura_espagueti(tray_ln, ticker="AAPL"),
        "cono": figura_cono(tray_ln, ticker="AAPL"),
        "histograma": figura_histograma(tray_ln[:, -1], res),
        "comparacion": figura_comparacion(
            {"lognormal": tray_ln[:, -1], "normal": tray_nr[:, -1]}, S0
        ),
    }
    for nombre, fig in figuras.items():
        ruta = salida / f"{nombre}.png"
        fig.savefig(ruta, dpi=110, facecolor=fig.get_facecolor())
        print(f"Guardado: {ruta}")