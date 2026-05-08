# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:13:23 2026

@author: gueno
"""
# ─────────────────────────────────────────────
# main.py · Pipeline principal TWINOS
# Lance la simulation, le feature engineering,
# la détection d'anomalie et le scoring complet
# ─────────────────────────────────────────────

import sys
from pathlib import Path

# Assure que les modules locaux sont trouvables (Windows + Linux)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config    import CONFIG
from core.simulation import generate
from core.features  import build_features
from core.modeling  import TwinosDetector
from core.scoring   import build_snapshot
from utils.io       import save_json, save_df


def run(verbose: bool = True) -> None:
    sep = "─" * 44

    print(f"\n{'═'*44}")
    print("  TWINOS · Jumeau Numérique Médical v0.1")
    print(f"{'═'*44}\n")

    # ── 1. Simulation des capteurs ───────────────
    print(f"{sep}")
    print("  [1/4] Génération des données capteurs")
    print(sep)
    df_raw = generate(CONFIG.simulation, CONFIG.sensors)
    if verbose:
        print(f"  Colonnes : {list(df_raw.columns)}")
        print(f"  Période  : {df_raw['ts'].iloc[0].date()} → "
              f"{df_raw['ts'].iloc[-1].date()}")

    # ── 2. Feature engineering ───────────────────
    print(f"\n{sep}")
    print("  [2/4] Feature engineering · baseline perso.")
    print(sep)
    df_feat = build_features(df_raw)
    if verbose:
        new_cols = [c for c in df_feat.columns if c not in df_raw.columns]
        print(f"  {len(new_cols)} nouvelles features : {new_cols[:8]} …")

    # ── 3. Détection d'anomalies ─────────────────
    print(f"\n{sep}")
    print("  [3/4] Détection d'anomalies multimodale")
    print(sep)
    detector = TwinosDetector(CONFIG.model)
    result   = detector.fit_predict(df_feat, truth_col="anomaly_injected")

    if verbose:
        print(f"  Top features :")
        for feat, imp in detector.top_features(5).items():
            print(f"    {feat:<30} {imp:.4f}")

    # ── 4. Scoring & snapshot ────────────────────
    print(f"\n{sep}")
    print("  [4/4] Calcul du score de vitalité & risques")
    print(sep)
    snapshot = build_snapshot(result.df, result.anomalies)

    print(f"\n  ┌─ Snapshot santé ───────────────────────")
    print(f"  │  Vitalité       : {snapshot.vitality_score}/100")
    print(f"  │  FC / HRV       : {snapshot.hr:.0f} bpm / {snapshot.hrv:.0f} ms")
    print(f"  │  Sommeil        : {snapshot.sleep_score:.0f}/100")
    print(f"  │  Sons moy. 24h  : {snapshot.sound_db:.1f} dB")
    print(f"  │  VOC index      : {snapshot.voc_index:.3f}")
    print(f"  │  Stress index   : {snapshot.stress_index:.3f}")
    print(f"  │  Pas aujourd'hui: {snapshot.steps_today:,}")
    print(f"  │  Anomalies/jour : {snapshot.anomalies_today}")
    print(f"  └────────────────────────────────────────")

    print(f"\n  Risques anticipés (6 mois) :")
    for risk, prob in snapshot.risks.items():
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"    {risk:<28} {bar} {prob:.0%}")

    print(f"\n  Alertes :")
    for alert in snapshot.alerts:
        print(f"    {alert}")

    # ── 5. Export ────────────────────────────────
    print(f"\n{sep}")
    print("  Export des résultats")
    print(sep)
    Path("data").mkdir(exist_ok=True)

    save_json(snapshot.to_dict(), CONFIG.data_path)
    save_df(result.anomalies[["ts", "anomaly_score", "hr", "gsr", "sound_db",
                               "voc_index", "stress_index"]].head(50),
            "data/anomalies.csv")

    print(f"\n{'═'*44}")
    print(f"  ✓ Pipeline terminé — données prêtes pour le dashboard")
    print(f"  → streamlit run app/streamlit_app.py")
    print(f"{'═'*44}\n")


if __name__ == "__main__":
    run()