"""Shared helpers for impact analysis tables and interpretation text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    annualize_return,
    annualize_vol,
    calmar,
    max_drawdown,
    sharpe,
    sortino,
)


MetricValues = Dict[str, float]


@dataclass
class MetricsComparison:
    table: pd.DataFrame
    current: MetricValues
    new: MetricValues


def _metric_block(ret: pd.Series, rf: float) -> MetricValues:
    if ret is None or ret.empty:
        return {
            "ret": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "dd": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
        }
    return {
        "ret": annualize_return(ret),
        "vol": annualize_vol(ret),
        "sharpe": sharpe(ret, rf),
        "dd": max_drawdown((1 + ret).cumprod()),
        "sortino": sortino(ret, rf),
        "calmar": calmar(ret),
    }


def _fmt_percent(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{v * 100:,.2f}%"


def _fmt_ratio(v: float) -> str:
    return "—" if v is None or np.isnan(v) else f"{v:,.2f}"


def _fmt_delta(v: float, kind: str) -> str:
    if v is None or np.isnan(v):
        return "—"
    suffix = " pp" if kind == "percent" else ""
    mult = 100 if kind == "percent" else 1
    return f"{v * mult:,.2f}{suffix}"


METRIC_ROWS = (
    ("ret", "Rend. anualizado", _fmt_percent, "percent"),
    ("vol", "Vol. anualizada", _fmt_percent, "percent"),
    ("sharpe", "Sharpe", _fmt_ratio, "ratio"),
    ("dd", "Max Drawdown", _fmt_percent, "percent"),
    ("sortino", "Sortino", _fmt_ratio, "ratio"),
    ("calmar", "Calmar", _fmt_ratio, "ratio"),
)


def build_metrics_comparison(
    current_ret: pd.Series, new_ret: pd.Series, rf: float
) -> MetricsComparison:
    """Return formatted metrics table plus the raw blocks for both cases."""
    current_block = _metric_block(current_ret, rf)
    new_block = _metric_block(new_ret, rf)
    rows = []
    for key, label, fmt_fn, delta_kind in METRIC_ROWS:
        rows.append(
            {
                "Métrica": label,
                "Actual": fmt_fn(current_block.get(key)),
                "Con candidato": fmt_fn(new_block.get(key)),
                "Δ": _fmt_delta(
                    (new_block.get(key, np.nan) or 0)
                    - (current_block.get(key, np.nan) or 0),
                    delta_kind,
                ),
            }
        )
    table = pd.DataFrame(rows)
    return MetricsComparison(table=table, current=current_block, new=new_block)


def interpret_candidate_effect(
    current: MetricValues, new: MetricValues, ticker: str, weight: float
) -> str:
    """Generate a short paragraph explaining the impact on risk-return ratios."""

    def _safe(v):
        return 0.0 if v is None or np.isnan(v) else float(v)

    d_sharpe = _safe(new.get("sharpe")) - _safe(current.get("sharpe"))
    verb_sharpe = "mejora" if d_sharpe >= 0 else "degrada"
    d_vol = _safe(new.get("vol")) - _safe(current.get("vol"))
    vol_dir = "sube" if d_vol > 0 else "baja"
    d_dd = _safe(new.get("dd")) - _safe(current.get("dd"))
    dd_dir = "aumenta" if d_dd > 0 else "disminuye"
    lines = [
        f"El Sharpe {verb_sharpe} {abs(d_sharpe):.2f} puntos al asignar {weight*100:,.1f}% a {ticker}.",
        f"La volatilidad {vol_dir} a {_safe(new.get('vol'))*100:,.2f}% y el max drawdown {dd_dir} a {_safe(new.get('dd'))*100:,.2f}%.",
    ]
    summary = (
        "Interpretación: para este peso, el candidato "
        f"{'incrementa' if d_sharpe >= 0 else 'reduce'} la eficiencia riesgo-retorno "
        "considerando los cambios en Sharpe, volatilidad y drawdown."
    )
    lines.append(summary)
    return "\n".join(lines)
