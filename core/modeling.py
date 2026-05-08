# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:01:56 2026

@author: gueno
"""

# ─────────────────────────────────────────────
# core/modeling.py · Modèles ML TWINOS
# Détection d'anomalie multimodale
# ─────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.config import CONFIG, ModelConfig
from core.features import MODEL_FEATURES


@dataclass
class AnomalyResult:
    """Résultat d'une détection d'anomalie sur un DataFrame."""
    df:              pd.DataFrame           # données enrichies avec prédictions
    anomalies:       pd.DataFrame           # sous-ensemble des anomalies
    n_total:         int = 0
    n_anomalies:     int = 0
    precision:       Optional[float] = None # si vérité terrain disponible
    recall:          Optional[float] = None


class TwinosDetector:
    """
    Détecteur d'anomalies personnalisé basé sur Isolation Forest.

    Le modèle apprend la baseline INDIVIDUELLE de la personne :
    il n'y a pas de comparaison avec une population de référence.

    Usage
    -----
    detector = TwinosDetector()
    result = detector.fit_predict(df_with_features)
    result.anomalies  # DataFrame des anomalies
    """

    def __init__(self, cfg: ModelConfig = CONFIG.model):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            contamination=cfg.contamination,
            n_estimators=cfg.n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self._features_used: List[str] = []
        self.is_fitted = False

    # ── Entraînement + prédiction ─────────────────────────────────────────

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        truth_col: Optional[str] = "anomaly_injected",
    ) -> AnomalyResult:
        """
        Entraîne le modèle sur df et prédit les anomalies.

        Parameters
        ----------
        df           : DataFrame avec features engineering déjà appliqué
        feature_cols : liste des colonnes features (défaut = MODEL_FEATURES)
        truth_col    : colonne vérité terrain (pour calculer précision/rappel)
        """
        features = feature_cols or [c for c in MODEL_FEATURES if c in df.columns]
        self._features_used = features

        X_raw = df[features].values
        X     = self.scaler.fit_transform(X_raw)

        self.model.fit(X)
        self.is_fitted = True

        df = df.copy()
        df["anomaly_flag"]       = self.model.predict(X)          # -1 = anomalie
        df["anomaly_score"]      = -self.model.score_samples(X)   # + = suspect
        df["is_anomaly"]         = df["anomaly_flag"] == -1

        anomalies = df[df["is_anomaly"]].copy()

        result = AnomalyResult(
            df          = df,
            anomalies   = anomalies,
            n_total     = len(df),
            n_anomalies = len(anomalies),
        )

        if truth_col and truth_col in df.columns:
            result.precision, result.recall = self._evaluate(df, truth_col)

        self._print_summary(result)
        return result

    def predict_latest(self, df: pd.DataFrame, n_hours: int = 24) -> pd.DataFrame:
        """
        Prédit les anomalies sur les n_hours dernières observations.
        Utile pour le monitoring en temps réel.
        """
        if not self.is_fitted:
            raise RuntimeError("Appelle fit_predict() avant predict_latest().")

        recent = df.tail(n_hours).copy()
        features = self._features_used
        X = self.scaler.transform(recent[features].values)
        recent["anomaly_flag"]  = self.model.predict(X)
        recent["anomaly_score"] = -self.model.score_samples(X)
        recent["is_anomaly"]    = recent["anomaly_flag"] == -1
        return recent

    # ── Évaluation ────────────────────────────────────────────────────────

    def _evaluate(
        self,
        df: pd.DataFrame,
        truth_col: str,
    ) -> Tuple[float, float]:
        """Calcule précision et rappel vs vérité terrain."""
        y_true = df[truth_col].astype(bool)
        y_pred = df["is_anomaly"]

        tp = (y_true & y_pred).sum()
        fp = (~y_true & y_pred).sum()
        fn = (y_true & ~y_pred).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return round(precision, 3), round(recall, 3)

    # ── Affichage ─────────────────────────────────────────────────────────

    def _print_summary(self, r: AnomalyResult) -> None:
        print("\n── Résultats détection ────────────────────")
        print(f"  Observations analysées : {r.n_total:,}")
        print(f"  Anomalies détectées    : {r.n_anomalies:,} "
              f"({r.n_anomalies/r.n_total*100:.1f}%)")
        if r.precision is not None:
            print(f"  Précision              : {r.precision:.1%}")
            print(f"  Rappel                 : {r.recall:.1%}")
        if not r.anomalies.empty:
            top = r.anomalies.nlargest(3, "anomaly_score")[
                ["ts", "anomaly_score"] + [c for c in ["hr", "gsr", "sound_db"]
                                           if c in r.anomalies.columns]
            ]
            print("\n  Top 3 anomalies :")
            print(top.to_string(index=False))
        print("───────────────────────────────────────────\n")

    # ── Features importantes ──────────────────────────────────────────────

    def top_features(self, n: int = 5) -> pd.Series:
        """
        Retourne les features les plus importantes (moyenne depth Isolation Forest).
        Approximation : features avec variance la plus haute dans l'espace transformé.
        """
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné.")
        importances = np.std(
            [tree.feature_importances_
             for tree in self.model.estimators_], axis=0
        )
        return (
            pd.Series(importances, index=self._features_used)
              .sort_values(ascending=False)
              .head(n)
        )