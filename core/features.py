# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:01:04 2026

@author: gueno
"""

# ─────────────────────────────────────────────
# core/features.py · Feature engineering TWINOS
# Baseline personnalisée + features cross-modales
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
from utils.time import rolling_stats, z_score
from core.config import CONFIG, ModelConfig


# Métriques sur lesquelles on calcule la baseline personnelle
BASELINE_COLS = [
    "hr", "hrv", "sound_db", "light_lux",
    "skin_temp", "gsr", "voc_index",
    "steps_h", "blue_light_h",
]


def add_baseline_features(
    df: pd.DataFrame,
    window: int = CONFIG.model.rolling_window,
) -> pd.DataFrame:
    """
    Pour chaque métrique dans BASELINE_COLS, ajoute :
      - {col}_mean  : moyenne glissante personnelle
      - {col}_std   : écart-type glissant
      - {col}_z     : z-score par rapport à la baseline

    C'est le cœur de la différenciation TWINOS :
    on compare la personne à ELLE-MÊME, pas à une population.
    """
    df = df.copy()
    for col in BASELINE_COLS:
        if col not in df.columns:
            continue
        mean, std = rolling_stats(df[col], window=window)
        df[f"{col}_mean"] = mean
        df[f"{col}_std"]  = std
        df[f"{col}_z"]    = z_score(df[col], mean, std)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features temporelles utiles pour le modèle."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "dow" in df.columns:
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_crossmodal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features cross-modales : interactions entre capteurs.
    Ces corrélations sont la vraie valeur ajoutée d'un jumeau numérique.

    Exemples :
      - stress_index    : combinaison GSR + FC + sons + HRV
      - sleep_pressure  : facteurs qui dégradent le sommeil (son, lumière bleue…)
      - metabolic_risk  : signal olfactif + sédentarité + temp cutanée
      - recovery_score  : HRV haute + FC basse + peu de son = bonne récupération
    """
    df = df.copy()

    # ── Stress index composite (0 = calme, + = stressé)
    weights = CONFIG.model.stress_weights
    df["stress_index"] = sum(
        df.get(col, pd.Series(0, index=df.index)) * w
        for col, w in weights.items()
    )

    # ── Pression sommeil (prédit qualité nuit suivante)
    sound_roll  = df["sound_db"].rolling(3, min_periods=1).mean()
    blue_roll   = df["blue_light_h"].rolling(3, min_periods=1).mean()
    hrv_roll    = df["hrv"].rolling(3, min_periods=1).mean()
    light_roll  = df["light_lux"].rolling(3, min_periods=1).mean()

    df["sleep_pressure"] = (
        - sound_roll  * 0.012
        - blue_roll   * 0.060
        + hrv_roll    * 0.018
        + light_roll  * 0.000_05
    )

    # ── Risque métabolique (signal olfactif + sédentarité)
    sedentary = (df["steps_h"] < 100).astype(float)
    df["metabolic_risk"] = (
        df.get("voc_index", 0) * 0.50
        + df.get("skin_temp_z", 0) * 0.20
        + sedentary * 0.30
    ).clip(0, 3)

    # ── Score de récupération (+ = bien récupéré)
    df["recovery_score"] = (
        df.get("hrv_z", 0) * 0.50
        - df.get("hr_z", 0)  * 0.30
        - df.get("sound_db_z", 0) * 0.20
    )

    # ── Charge sensorielle journalière (cumul des stimuli)
    df["sensory_load"] = (
        df.get("sound_db_z", 0).clip(0) * 0.35
        + df.get("blue_light_h", 0) / 6 * 0.30
        + df.get("gsr_z", 0).clip(0)   * 0.35
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering."""
    df = add_baseline_features(df)
    df = add_temporal_features(df)
    df = add_crossmodal_features(df)
    df = df.dropna(subset=["hr_z"])   # supprime les premières lignes sans baseline
    return df


# Colonnes utilisées pour l'entraînement du modèle
MODEL_FEATURES = [
    "hr_z", "hrv_z", "sound_db_z", "light_lux_z",
    "skin_temp_z", "gsr_z", "voc_index_z",
    "steps_h_z", "blue_light_h_z",
    "stress_index", "metabolic_risk", "recovery_score",
    "sensory_load", "hour_sin", "hour_cos",
]