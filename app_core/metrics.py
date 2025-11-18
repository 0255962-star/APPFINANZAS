"""Performance metrics: annualize_return(), annualize_vol(), sharpe(), sortino(), max_drawdown(), calmar()."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualize_return(d, freq=252):
    if d.empty:
        return np.nan
    return float((1 + d).prod() ** (freq / max(len(d), 1)) - 1)


def annualize_vol(d, freq=252):
    if d.empty:
        return np.nan
    return float(d.std(ddof=0) * np.sqrt(freq))


def sharpe(d, rf=0.0, freq=252):
    if d.empty:
        return np.nan
    er = annualize_return(d, freq)
    ev = annualize_vol(d, freq)
    return (er - rf) / ev if ev and ev > 0 else np.nan


def sortino(d, rf=0.0, freq=252):
    if d.empty:
        return np.nan
    neg = d.copy()
    neg[neg > 0] = 0
    dd = np.sqrt((neg**2).mean()) * np.sqrt(freq)
    er = annualize_return(d, freq)
    return (er - rf) / dd if dd and dd > 0 else np.nan


def max_drawdown(cum):
    if cum.empty:
        return np.nan
    return float((cum / cum.cummax() - 1).min())


def calmar(d, freq=252):
    if d.empty:
        return np.nan
    er = annualize_return(d, freq)
    mdd = abs(max_drawdown((1 + d).cumprod()))
    return er / mdd if mdd and mdd > 0 else np.nan

