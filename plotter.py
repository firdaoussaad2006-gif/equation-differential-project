
import numpy as np
import sympy as sp
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_solution(parent, solution, x_min=-5, x_max=5):
    """
    Trace une solution symbolique dans un widget Tkinter.

    parent   : widget Tkinter qui accueillera le graphique
    solution : expression SymPy de la solution finale
    x_min    : début de l'intervalle
    x_max    : fin de l'intervalle

    Retourne le canvas Matplotlib.
    """

    solution = sp.sympify(solution)
    x = sp.Symbol("x", real=True)

    x_min = float(x_min)
    x_max = float(x_max)

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Les bornes doivent être finies.")

    if x_min >= x_max:
        raise ValueError(
            "La borne minimale doit être inférieure à la borne maximale."
        )

    # Générer les valeurs de x
    x_values = np.linspace(x_min, x_max, 500)

    # Transformer l'expression symbolique en fonction numérique
    function = sp.lambdify(x, solution, modules=["numpy"])

    try:
        with np.errstate(all="ignore"):
            y_values = np.asarray(
                function(x_values), dtype=complex
            )

        # Accepter une solution réelle, mais refuser
        # les valeurs significativement complexes.
        if np.any(np.abs(np.imag(y_values)) > 1e-9):
            raise ValueError(
                "La solution possède des valeurs complexes "
                "sur l'intervalle choisi."
            )

        y_values = np.real(y_values)

        # Ignorer les points non finis dans le graphique
        y_values[~np.isfinite(y_values)] = np.nan

        if not np.any(np.isfinite(y_values)):
            raise ValueError(
                "Aucune valeur réelle exploitable sur cet intervalle."
            )

    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Impossible de tracer la solution : {error}"
        ) from error

    # Créer la figure Matplotlib
    figure = Figure(figsize=(7, 4), dpi=100)
    axes = figure.add_subplot(111)

    axes.plot(
        x_values,
        y_values,
        label=f"y(x) = {sp.sstr(solution)}"
    )

    axes.axhline(0, color="black", linewidth=0.7)
    axes.axvline(0, color="black", linewidth=0.7)

    axes.set_title("Solution de l'équation différentielle")
    axes.set_xlabel("x")
    axes.set_ylabel("y(x)")
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()

    # Intégrer la figure dans Tkinter
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.draw()

    return canvas
