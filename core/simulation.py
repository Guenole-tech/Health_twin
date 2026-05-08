# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:00:39 2026

@author: gueno
"""
# ─────────────────────────────────────────────
# core/simulation.py · Données simulées TWINOS
# Tous les capteurs : FC, HRV, sommeil, sons,
# lumière, olfactif, peau, mouvement
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
from utils.time import (
    date_range, hour_array, day_of_week_array,
    circadian_wave, is_night, is_day,
)
from core.config import CONFIG, SimulationConfig, SensorConfig


def generate(
    sim_cfg: SimulationConfig = CONFIG.simulation,
    sensor_cfg: SensorConfig  = CONFIG.sensors,
) -> pd.DataFrame:
    """
    Génère un DataFrame horaire complet simulant tous les capteurs
    du jumeau numérique sur sim_cfg.n_days jours.

    Colonnes produites
    ------------------
    ts, hour, dow,
    hr, hrv,                        — cardiovasculaire
    sleep_score, sleep_deep_h, sleep_rem_h, sleep_light_h, sleep_wake_h,
    sound_db,                       — environnement sonore
    light_lux, blue_light_h,        — lumière
    acetone_ppm, isoprene_ppm, h2s_ppb, voc_index,   — olfactif
    skin_temp, gsr, shower,         — peau / sensation
    steps_h, speed_ms,              — mouvement
    anomaly_injected                — label de vérité terrain (True = anomalie réelle)
    """
    np.random.seed(sim_cfg.seed)
    idx  = date_range(sim_cfg.n_days, sim_cfg.freq)
    n    = len(idx)
    hour = hour_array(idx)
    dow  = day_of_week_array(idx)
    c    = sensor_cfg

    df = pd.DataFrame({"ts": idx, "hour": hour, "dow": dow})

    # ── 1. Cardiovasculaire ──────────────────────────────────────────────
    df["hr"] = (
        c.hr_baseline
        + circadian_wave(hour, phase=6, amplitude=c.hr_circadian_amp)
        + np.random.normal(0, c.hr_noise_std, n)
    )
    # HRV : inverse de FC (plus haute la nuit, plus basse le jour)
    df["hrv"] = (
        c.hrv_baseline
        - circadian_wave(hour, phase=6, amplitude=12)
        + np.random.normal(0, c.hrv_noise_std, n)
    ).clip(10)

    # ── 2. Sommeil (données nocturnes uniquement) ────────────────────────
    night_mask = is_night(hour)
    sleep_base = np.where(night_mask,
                          72 + np.random.normal(0, 10, n), np.nan)
    df["sleep_score"]   = sleep_base
    df["sleep_deep_h"]  = np.where(night_mask, np.random.normal(1.2, 0.3, n).clip(0), np.nan)
    df["sleep_rem_h"]   = np.where(night_mask, np.random.normal(1.5, 0.4, n).clip(0), np.nan)
    df["sleep_light_h"] = np.where(night_mask, np.random.normal(3.5, 0.5, n).clip(0), np.nan)
    df["sleep_wake_h"]  = np.where(night_mask, np.random.exponential(0.3, n), np.nan)

    # ── 3. Sons environnementaux (dB) ────────────────────────────────────
    # Rythme réaliste : plus fort le matin/soir (transports, bureau)
    df["sound_db"] = np.clip(
        c.sound_baseline
        + circadian_wave(hour, phase=9, amplitude=c.sound_amp)
        + np.random.normal(0, 8, n),
        20, c.sound_clip_max,
    )

    # ── 4. Lumière (lux + lumière bleue) ────────────────────────────────
    day_mask = is_day(hour)
    df["light_lux"] = np.where(
        day_mask,
        np.clip(
            c.light_day_base
            + circadian_wave(hour, phase=6, amplitude=c.light_amp)
            + np.random.normal(0, 500, n),
            0, 100_000,
        ),
        0,
    )
    # Écran le soir
    df["blue_light_h"] = np.where(
        (hour >= 8) & (hour <= 23),
        np.clip(0.6 + np.random.normal(0, 0.3, n), 0, 6),
        0,
    )

    # ── 5. Olfactif — biomarqueurs volatils ─────────────────────────────
    # Acétone (marqueur pré-diabétique si > 1.8 ppm)
    df["acetone_ppm"]  = np.clip(0.4 + np.random.exponential(0.2, n), 0, 5)
    # Isoprène (marqueur cholestérol / stress oxydatif)
    df["isoprene_ppm"] = np.clip(0.1 + np.random.exponential(0.06, n), 0, 2)
    # H₂S en ppb (stress oxydatif, problèmes hépatiques)
    df["h2s_ppb"]      = np.clip(0.5 + np.random.exponential(0.25, n), 0, 10)
    # Index VOC composite (0–1, normalisé sur seuils cliniques)
    df["voc_index"] = (
        df["acetone_ppm"]  / c.acetone_normal_max  * 0.5
        + df["isoprene_ppm"] / c.isoprene_normal_max * 0.3
        + df["h2s_ppb"]      / c.h2s_normal_max_ppb  * 0.2
    ).clip(0, 3)

    # ── 6. Peau & sensation ──────────────────────────────────────────────
    df["skin_temp"] = (
        c.skin_temp_baseline
        + circadian_wave(hour, phase=14, amplitude=c.skin_temp_amp)
        + np.random.normal(0, 0.2, n)
    )
    # Conductance galvanique (μS) — proxy stress / arousal
    df["gsr"] = np.clip(
        c.gsr_baseline
        + circadian_wave(hour, phase=12, amplitude=2.0)
        + np.random.exponential(0.5, n),
        0.5, 30,
    )
    # Détection douche : pic temp + GSR autour de 7h et 19h
    shower_prob = np.where((hour == 7) | (hour == 8) | (hour == 19) | (hour == 20), 0.35, 0)
    df["shower"] = np.random.binomial(1, shower_prob)

    # ── 7. Mouvement ─────────────────────────────────────────────────────
    df["steps_h"] = np.clip(
        circadian_wave(hour, phase=8, amplitude=700)
        + 300
        + np.random.exponential(200, n),
        0, 4000,
    )
    df["speed_ms"] = (df["steps_h"] / 3600 * 0.8).clip(0)

    # ── 8. Injection d'anomalies contrôlées (vérité terrain) ─────────────
    df["anomaly_injected"] = False
    n_anomalies = int(n * sim_cfg.anomaly_rate)
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    for idx_a in anomaly_indices:
        atype = np.random.choice(["hr_spike", "hrv_drop", "sound_peak",
                                   "voc_surge", "gsr_spike", "sleep_crash"])
        if atype == "hr_spike":
            df.loc[idx_a, "hr"]      += np.random.uniform(25, 50)
        elif atype == "hrv_drop":
            df.loc[idx_a, "hrv"]     -= np.random.uniform(20, 40)
        elif atype == "sound_peak":
            df.loc[idx_a, "sound_db"] = np.random.uniform(88, 100)
        elif atype == "voc_surge":
            df.loc[idx_a, "acetone_ppm"] += np.random.uniform(1.5, 3.0)
            df.loc[idx_a, "voc_index"]   += 1.5
        elif atype == "gsr_spike":
            df.loc[idx_a, "gsr"]     += np.random.uniform(8, 20)
        elif atype == "sleep_crash":
            df.loc[idx_a, "sleep_score"] = np.random.uniform(20, 45)

        df.loc[idx_a, "anomaly_injected"] = True

    print(f"✓ Simulation : {len(df)} observations · {n_anomalies} anomalies injectées")
    return df