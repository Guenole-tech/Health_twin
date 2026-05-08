# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:00:09 2026

@author: gueno
"""

# ─────────────────────────────────────────────
# core/config.py · Paramètres globaux TWINOS
# ─────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulationConfig:
    """Paramètres de génération des données simulées."""
    n_days: int = 90
    freq: str = "h"                  # fréquence temporelle
    seed: int = 42
    anomaly_rate: float = 0.02       # proportion d'anomalies injectées


@dataclass
class SensorConfig:
    """Plages de référence pour chaque capteur."""

    # Fréquence cardiaque (bpm)
    hr_baseline: float = 63.0
    hr_circadian_amp: float = 8.0
    hr_noise_std: float = 3.0

    # HRV (ms)
    hrv_baseline: float = 55.0
    hrv_noise_std: float = 5.0

    # Sons (dB)
    sound_baseline: float = 45.0
    sound_amp: float = 20.0
    sound_clip_max: float = 100.0
    sound_safe_threshold: float = 80.0   # OMS : risque au-delà

    # Lumière (lux)
    light_day_base: float = 3000.0
    light_amp: float = 5000.0

    # Lumière bleue (heures/jour)
    blue_light_max_safe: float = 2.0

    # Olfactif (ppm)
    acetone_normal_max: float = 1.8      # seuil pré-diabétique
    isoprene_normal_max: float = 0.3
    h2s_normal_max_ppb: float = 1.5

    # Peau
    skin_temp_baseline: float = 36.2
    skin_temp_amp: float = 0.6
    gsr_baseline: float = 3.0           # μS

    # Mouvement
    steps_target: int = 8000


@dataclass
class ModelConfig:
    """Paramètres des modèles ML."""
    contamination: float = 0.02
    n_estimators: int = 300
    rolling_window: int = 24            # heures

    # Poids du stress index (doivent sommer à 1.0)
    stress_weights: dict = field(default_factory=lambda: {
        "gsr_z":      0.40,
        "hr_z":       0.30,
        "sound_db_z": 0.20,
        "hrv_z":     -0.10,   # HRV haute = calme
    })


@dataclass
class ScoringConfig:
    """Paramètres de scoring santé."""
    # Seuils risque (0–1)
    risk_low:  float = 0.20
    risk_med:  float = 0.50
    risk_high: float = 0.75

    # Poids score vitalité global
    vitality_weights: dict = field(default_factory=lambda: {
        "hr":      0.20,
        "hrv":     0.20,
        "sleep":   0.20,
        "stress":  0.15,
        "sound":   0.10,
        "olfact":  0.10,
        "skin":    0.05,
    })


@dataclass
class TwinosConfig:
    """Configuration maître."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sensors:    SensorConfig     = field(default_factory=SensorConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    scoring:    ScoringConfig    = field(default_factory=ScoringConfig)
    data_path:  str = "data/dashboard_data.json"


# Instance globale — importée par tous les modules
CONFIG = TwinosConfig()