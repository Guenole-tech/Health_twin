# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:02:43 2026

@author: gueno
"""

# ─────────────────────────────────────────────
# utils/io.py · Lecture / écriture TWINOS
# ─────────────────────────────────────────────

import json
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


# ── JSON ──────────────────────────────────────

def save_json(data: Dict[str, Any], path: str) -> None:
    """Sauvegarde un dict Python en JSON indenté."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_serial)
    print(f"✓ JSON sauvegardé → {path}")


def load_json(path: str) -> Dict[str, Any]:
    """Charge un fichier JSON. Retourne {} si absent."""
    if not os.path.exists(path):
        print(f"⚠  Fichier introuvable : {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_serial(obj: Any) -> Any:
    """Sérialisation pour types non JSON natifs (numpy, datetime…)."""
    import numpy as np
    from datetime import datetime
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type non sérialisable : {type(obj)}")


# ── CSV / DataFrame ───────────────────────────

def save_df(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Sauvegarde un DataFrame en CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"✓ CSV sauvegardé → {path} ({len(df)} lignes)")


def load_df(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """Charge un CSV en DataFrame. Retourne DataFrame vide si absent."""
    if not os.path.exists(path):
        print(f"⚠  Fichier introuvable : {path}")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates or [])


# ── Apple Health XML ──────────────────────────

def parse_apple_health(xml_path: str) -> pd.DataFrame:
    """
    Parse l'export Apple Health (export.xml) et retourne un DataFrame
    avec les métriques clés : FC, HRV, sommeil, pas, O2 sanguin.

    Usage :
        df = parse_apple_health("~/Downloads/apple_health_export/export.xml")
    """
    import xml.etree.ElementTree as ET

    METRICS_MAP = {
        "HKQuantityTypeIdentifierHeartRate":                 "hr",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":  "hrv",
        "HKQuantityTypeIdentifierStepCount":                 "steps",
        "HKQuantityTypeIdentifierOxygenSaturation":          "spo2",
        "HKQuantityTypeIdentifierBodyTemperature":           "body_temp",
        "HKCategoryTypeIdentifierSleepAnalysis":             "sleep",
    }

    print(f"📂 Lecture Apple Health : {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for rec in root.iter("Record"):
        rtype = rec.get("type", "")
        if rtype not in METRICS_MAP:
            continue
        records.append({
            "metric":    METRICS_MAP[rtype],
            "value":     rec.get("value"),
            "startDate": rec.get("startDate"),
            "endDate":   rec.get("endDate"),
            "unit":      rec.get("unit", ""),
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠  Aucune donnée trouvée dans l'export Apple Health.")
        return df

    df["startDate"] = pd.to_datetime(df["startDate"])
    df["value"]     = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    print(f"✓ {len(df)} enregistrements chargés depuis Apple Health")
    print(df.groupby("metric")["value"].count().to_string())
    return df


def apple_health_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivote le DataFrame Apple Health en série horaire large
    (une colonne par métrique, un enregistrement par heure).
    """
    df = df.copy()
    df["hour"] = df["startDate"].dt.floor("h")
    pivot = (
        df.groupby(["hour", "metric"])["value"]
          .mean()
          .unstack("metric")
          .reset_index()
          .rename(columns={"hour": "ts"})
    )
    return pivot