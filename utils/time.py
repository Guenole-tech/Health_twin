# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:45:18 2026

@author: gueno
"""

# ─────────────────────────────────────────────
# utils/time.py · Helpers datetime TWINOS
# ─────────────────────────────────────────────

from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd


def date_range(n_days: int, freq: str = "h") -> pd.DatetimeIndex:
    """Génère un index temporel sur n_days depuis aujourd'hui."""
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=n_days)
    return pd.date_range(start=start, end=end, freq=freq)


def hour_array(index: pd.DatetimeIndex) -> np.ndarray:
    """Retourne un array des heures de la journée (0–23)."""
    return np.array([d.hour for d in index])


def day_of_week_array(index: pd.DatetimeIndex) -> np.ndarray:
    """Retourne un array du jour de la semaine (0=lundi, 6=dimanche)."""
    return np.array([d.dayofweek for d in index])


def is_night(hour: np.ndarray, start: int = 23, end: int = 6) -> np.ndarray:
    """Masque booléen : True si l'heure est la nuit."""
    return (hour >= start) | (hour <= end)


def is_day(hour: np.ndarray, start: int = 7, end: int = 20) -> np.ndarray:
    """Masque booléen : True si l'heure est en journée."""
    return (hour >= start) & (hour <= end)


def circadian_wave(
    hour: np.ndarray,
    phase: float = 6.0,
    amplitude: float = 1.0,
    period: float = 24.0,
) -> np.ndarray:
    """
    Onde sinusoïdale circadienne.

    Parameters
    ----------
    phase     : heure du creux (pic = phase + period/2)
    amplitude : amplitude de l'onde
    period    : période en heures (défaut 24h)
    """
    return amplitude * np.sin(2 * np.pi * (hour - phase) / period)


def rolling_stats(
    series: pd.Series,
    window: int = 24,
    min_periods: int = 6,
) -> Tuple[pd.Series, pd.Series]:
    """
    Retourne (moyenne glissante, écart-type glissant) sur `window` points.
    Ces deux séries forment la *baseline personnelle* de la métrique.
    """
    mean = series.rolling(window, min_periods=min_periods).mean()
    std  = series.rolling(window, min_periods=min_periods).std()
    return mean, std


def z_score(series: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """Z-score par rapport à la baseline personnelle (evite division par 0)."""
    return (series - mean) / (std + 1e-6)


def format_duration(minutes: float) -> str:
    """Ex: 402 → '6h42'."""
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h{m:02d}"


def time_since(ts: pd.Timestamp) -> str:
    """Retourne une chaîne lisible 'il y a X min / heures'."""
    delta = datetime.now() - ts.to_pydatetime()
    total_minutes = int(delta.total_seconds() / 60)
    if total_minutes < 60:
        return f"il y a {total_minutes} min"
    return f"il y a {total_minutes // 60}h{total_minutes % 60:02d}"