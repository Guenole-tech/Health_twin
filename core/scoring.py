# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:01:31 2026

@author: gueno
"""

# ─────────────────────────────────────────────
# core/scoring.py · Indices santé TWINOS
# Vitalité globale, risques anticipés, alertes
# ─────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd

from core.config import CONFIG, ScoringConfig, SensorConfig


# ── Structures de données ──────────────────────────────────────────────────

@dataclass
class HealthSnapshot:
    """Photo instantanée de l'état du jumeau à un moment donné."""
    ts:                str
    vitality_score:    float           # 0–100
    hr:                float
    hrv:               float
    sleep_score:       float
    stress_index:      float
    sound_db:          float
    voc_index:         float
    skin_temp:         float
    gsr:               float
    steps_today:       int
    anomalies_today:   int
    risks:             Dict[str, float] = field(default_factory=dict)
    alerts:            List[str]        = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ts":               self.ts,
            "vitality_score":   round(self.vitality_score, 1),
            "hr":               round(self.hr, 1),
            "hrv":              round(self.hrv, 1),
            "sleep_score":      round(self.sleep_score, 1),
            "stress_index":     round(self.stress_index, 3),
            "sound_db":         round(self.sound_db, 1),
            "voc_index":        round(self.voc_index, 3),
            "skin_temp":        round(self.skin_temp, 2),
            "gsr":              round(self.gsr, 2),
            "steps_today":      self.steps_today,
            "anomalies_today":  self.anomalies_today,
            "risks":            {k: round(v, 3) for k, v in self.risks.items()},
            "alerts":           self.alerts,
        }


# ── Fonctions de scoring ───────────────────────────────────────────────────

def compute_vitality(
    df: pd.DataFrame,
    cfg: ScoringConfig = CONFIG.scoring,
    sensor_cfg: SensorConfig = CONFIG.sensors,
) -> float:
    """
    Score de vitalité globale (0–100) basé sur les dernières 24h.

    Chaque composante est normalisée entre 0 et 100 puis pondérée.
    Le score reflète l'état actuel par rapport à la baseline personnelle.
    """
    last = df.tail(24)

    def safe_mean(col: str, default: float = 0.0) -> float:
        return float(last[col].mean()) if col in last.columns else default

    # ── FC : score max si FC ≈ baseline, pénalité si trop haute/basse
    hr_z = abs(safe_mean("hr_z"))
    hr_score = max(0, 100 - hr_z * 25)

    # ── HRV : plus c'est haut, mieux c'est
    hrv_z = safe_mean("hrv_z")
    hrv_score = min(100, max(0, 50 + hrv_z * 20))

    # ── Sommeil
    sleep_score = safe_mean("sleep_score", 70)

    # ── Stress (inverse)
    stress_raw = safe_mean("stress_index")
    stress_score = max(0, 100 - stress_raw * 30)

    # ── Sons : pénalité si exposition élevée
    sound_db = safe_mean("sound_db", 45)
    sound_score = max(0, 100 - max(0, sound_db - 50) * 1.5)

    # ── Olfactif : pénalité si VOC élevé
    voc = safe_mean("voc_index")
    voc_score = max(0, 100 - voc * 30)

    # ── Peau
    skin_z = abs(safe_mean("skin_temp_z"))
    skin_score = max(0, 100 - skin_z * 20)

    components = {
        "hr":     hr_score,
        "hrv":    hrv_score,
        "sleep":  sleep_score,
        "stress": stress_score,
        "sound":  sound_score,
        "olfact": voc_score,
        "skin":   skin_score,
    }

    w = cfg.vitality_weights
    total = sum(components[k] * w.get(k, 0) for k in components)
    return round(float(np.clip(total, 0, 100)), 1)


def compute_risks(
    df: pd.DataFrame,
    cfg: ScoringConfig = CONFIG.scoring,
    sensor_cfg: SensorConfig = CONFIG.sensors,
) -> Dict[str, float]:
    """
    Calcule les risques anticipés sur horizon 6 mois.

    Chaque risque est une probabilité (0–1) basée sur les tendances
    observées sur les 30 derniers jours.

    Retourne un dict {nom_risque: probabilité}.
    """
    last30 = df.tail(30 * 24)

    def safe_mean(col: str, default: float = 0.0) -> float:
        return float(last30[col].mean()) if col in last30.columns else default

    # ── Fatigue chronique (surexposition sonore)
    avg_db = safe_mean("sound_db", 45)
    sound_excess = max(0, avg_db - sensor_cfg.sound_safe_threshold)
    risk_fatigue = np.clip(sound_excess / 20, 0, 1) * 0.6 + \
                   max(0, -safe_mean("hrv_z")) * 0.15

    # ── Trouble du sommeil
    sleep_scores = last30["sleep_score"].dropna() if "sleep_score" in last30 else pd.Series([70])
    poor_nights  = (sleep_scores < 60).mean() if len(sleep_scores) > 0 else 0
    blue_excess  = max(0, safe_mean("blue_light_h") - sensor_cfg.blue_light_max_safe)
    risk_sleep   = np.clip(poor_nights * 0.6 + blue_excess * 0.1, 0, 1)

    # ── Risque cardiovasculaire
    hr_variability = abs(safe_mean("hr_z"))
    hrv_low        = max(0, -safe_mean("hrv_z"))
    stress_chronic = max(0, safe_mean("stress_index"))
    risk_cardio    = np.clip(
        hr_variability * 0.10 + hrv_low * 0.08 + stress_chronic * 0.06, 0, 1
    )

    # ── Résistance à l'insuline (signal olfactif acétone)
    acetone_excess = max(0, safe_mean("acetone_ppm") - sensor_cfg.acetone_normal_max)
    sedentary_ratio = (last30["steps_h"] < 100).mean() if "steps_h" in last30 else 0
    risk_insulin   = np.clip(acetone_excess * 0.25 + sedentary_ratio * 0.15, 0, 1)

    # ── Fatigue sensorielle (charge cumulée)
    risk_sensory = np.clip(safe_mean("sensory_load") * 0.30, 0, 1) \
        if "sensory_load" in last30 else 0.0

    # ── Dermatite / peau (conductance élevée + temp anormale)
    gsr_high  = max(0, safe_mean("gsr_z"))
    temp_high = abs(safe_mean("skin_temp_z"))
    risk_skin = np.clip(gsr_high * 0.05 + temp_high * 0.04, 0, 1)

    return {
        "fatigue_chronique":       round(float(risk_fatigue), 3),
        "trouble_sommeil":         round(float(risk_sleep), 3),
        "risque_cardiovasculaire": round(float(risk_cardio), 3),
        "resistance_insuline":     round(float(risk_insulin), 3),
        "fatigue_sensorielle":     round(float(risk_sensory), 3),
        "probleme_cutane":         round(float(risk_skin), 3),
    }


def generate_alerts(
    risks: Dict[str, float],
    df: pd.DataFrame,
    cfg: ScoringConfig = CONFIG.scoring,
    sensor_cfg: SensorConfig = CONFIG.sensors,
) -> List[str]:
    """
    Génère des alertes textuelles actionnables selon les risques et
    les données des dernières 24h.
    """
    alerts: List[str] = []
    last24 = df.tail(24)

    def safe_mean(col: str, default: float = 0.0) -> float:
        return float(last24[col].mean()) if col in last24.columns else default

    # Alertes risques
    for risk_name, prob in risks.items():
        if prob >= cfg.risk_high:
            alerts.append(f"🔴 Risque élevé · {risk_name.replace('_', ' ')} ({prob:.0%})")
        elif prob >= cfg.risk_med:
            alerts.append(f"🟡 Risque modéré · {risk_name.replace('_', ' ')} ({prob:.0%})")

    # Alertes capteurs immédiats
    if safe_mean("sound_db") > sensor_cfg.sound_safe_threshold:
        alerts.append(f"🔊 Exposition sonore élevée · {safe_mean('sound_db'):.0f} dB moy. aujourd'hui")

    if safe_mean("acetone_ppm") > sensor_cfg.acetone_normal_max:
        alerts.append(f"👃 Acétone élevée · {safe_mean('acetone_ppm'):.2f} ppm (seuil : {sensor_cfg.acetone_normal_max})")

    if safe_mean("gsr") > 12:
        alerts.append(f"⚡ Stress physiologique élevé · conductance {safe_mean('gsr'):.1f} μS")

    if not alerts:
        alerts.append("✅ Tous les indicateurs dans la norme personnelle")

    return alerts


def build_snapshot(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
) -> HealthSnapshot:
    """
    Construit un HealthSnapshot complet à partir des données enrichies
    et des anomalies détectées.
    """
    last = df.tail(1).iloc[0] if len(df) > 0 else {}
    today_anomalies = anomaly_df[
        anomaly_df["ts"].dt.date == df["ts"].iloc[-1].date()
    ] if "ts" in anomaly_df.columns else anomaly_df.tail(0)

    risks  = compute_risks(df)
    alerts = generate_alerts(risks, df)
    vitality = compute_vitality(df)

    return HealthSnapshot(
        ts               = str(df["ts"].iloc[-1]),
        vitality_score   = vitality,
        hr               = float(last.get("hr", 63)),
        hrv              = float(last.get("hrv", 50)),
        sleep_score      = float(df["sleep_score"].dropna().tail(8).mean()) if "sleep_score" in df else 70,
        stress_index     = float(last.get("stress_index", 0)),
        sound_db         = float(df["sound_db"].tail(24).mean()) if "sound_db" in df else 45,
        voc_index        = float(last.get("voc_index", 0)),
        skin_temp        = float(last.get("skin_temp", 36.2)),
        gsr              = float(last.get("gsr", 3)),
        steps_today      = int(df["steps_h"].tail(24).sum()) if "steps_h" in df else 0,
        anomalies_today  = len(today_anomalies),
        risks            = risks,
        alerts           = alerts,
    )