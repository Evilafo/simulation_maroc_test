"""Fonctions de visualisation partagees pour les deux ACP (Etape 1, UC-S1).

Toutes les fonctions sauvegardent a la fois la figure (PNG) et les donnees
sous-jacentes (CSV) dans les dossiers passes en argument, conformement a la
consigne de sauvegarder les donnees des graphiques.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 9


def _save(fig: plt.Figure, data: pd.DataFrame | None, fig_path: Path, data_path: Path | None = None) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    if data is not None and data_path is not None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False)


def plot_scree(explained_var_ratio: np.ndarray, fig_path: Path, data_path: Path, title: str) -> None:
    df = pd.DataFrame({
        "composante": [f"F{i+1}" for i in range(len(explained_var_ratio))],
        "valeur_propre": explained_var_ratio * len(explained_var_ratio),  # approx, recalcule si besoin ailleurs
        "variance_expliquee_pct": explained_var_ratio * 100,
        "variance_cumulee_pct": np.cumsum(explained_var_ratio) * 100,
    })
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(df["composante"], df["variance_expliquee_pct"], color="#4C72B0", label="Variance expliquee (%)")
    ax1.set_ylabel("Variance expliquee (%)")
    ax2 = ax1.twinx()
    ax2.plot(df["composante"], df["variance_cumulee_pct"], color="#C44E52", marker="o", label="Variance cumulee (%)")
    ax2.set_ylabel("Variance cumulee (%)")
    ax2.set_ylim(0, 105)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, df, fig_path, data_path)


def plot_individuals(
    coords: pd.DataFrame,
    fig_path: Path,
    data_path: Path,
    title: str,
    x_col: str = "F1",
    y_col: str = "F2",
    label_col: str | None = None,
    highlight: str | None = None,
    highlight_col: str | None = None,
    color_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if color_col is not None:
        cats = coords[color_col].astype(str)
        for cat in sorted(cats.unique()):
            sub = coords[cats == cat]
            ax.scatter(sub[x_col], sub[y_col], label=str(cat), s=40, alpha=0.8)
        ax.legend(fontsize=7, loc="best")
    else:
        ax.scatter(coords[x_col], coords[y_col], s=40, alpha=0.8, color="#4C72B0")

    if highlight is not None and highlight_col is not None:
        hl = coords[coords[highlight_col] == highlight]
        ax.scatter(hl[x_col], hl[y_col], s=90, facecolors="none", edgecolors="red", linewidths=1.5, label=highlight)

    if label_col is not None:
        for _, row in coords.iterrows():
            ax.annotate(str(row[label_col]), (row[x_col], row[y_col]), fontsize=7, alpha=0.8)

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, coords, fig_path, data_path)


def plot_correlation_circle(
    loadings: pd.DataFrame,
    fig_path: Path,
    data_path: Path,
    title: str,
    x_col: str = "F1",
    y_col: str = "F2",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, fill=False, color="grey", linestyle="--")
    ax.add_artist(circle)
    for var, row in loadings.iterrows():
        ax.arrow(0, 0, row[x_col], row[y_col], head_width=0.02, length_includes_head=True, color="#4C72B0", alpha=0.7)
        ax.annotate(var, (row[x_col], row[y_col]), fontsize=7)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, loadings.reset_index().rename(columns={"index": "variable"}), fig_path, data_path)


def plot_country_trajectory(
    coords: pd.DataFrame,
    country: str,
    fig_path: Path,
    data_path: Path,
    title: str,
    x_col: str = "F1",
    y_col: str = "F2",
    year_col: str = "year",
    country_col: str = "country_iso3",
) -> None:
    sub = coords[coords[country_col] == country].sort_values(year_col)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sub[x_col], sub[y_col], color="#4C72B0", alpha=0.6, linewidth=1.2, zorder=1)
    sc = ax.scatter(sub[x_col], sub[y_col], c=sub[year_col], cmap="viridis", s=35, zorder=2)
    fig.colorbar(sc, ax=ax, label="Annee")
    first_row = sub.iloc[0]
    last_row = sub.iloc[-1]
    ax.annotate(f"{int(first_row[year_col])}", (first_row[x_col], first_row[y_col]), fontsize=8, weight="bold")
    ax.annotate(f"{int(last_row[year_col])}", (last_row[x_col], last_row[y_col]), fontsize=8, weight="bold")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, sub, fig_path, data_path)
