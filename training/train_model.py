"""
Train a log-linear OLS regression on DVF data for Paris 15e.

Model: log(valeur_fonciere) = b0 + b1*surface + b2*rooms + b3*is_appartement + b4*section_location
Where section_location is the smoothed mean log-price for each cadastral section (zone).

Cadastral sections are derived from id_parcelle by stripping the last 4 digits
(individual plot number), keeping only the zone-level identifier.
For example: "75115000CG0058" -> section "75115000CG".

Usage:
    python training/train_model.py                              # uses default paths
    python training/train_model.py --csv data/dvf.csv           # explicit CSV
    python training/train_model.py --out artifacts/model.json   # explicit output
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Allow imports when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.preprocessing import run_preprocessing


# Minimum number of transactions per section before we trust its mean.
# Sections with fewer observations get smoothed toward the global mean.
SECTION_SMOOTHING_MIN = 5

FEATURE_COLS = ["surface_reelle_bati", "nombre_pieces_principales", "is_appartement"]


def encode_sections(log_prices, section_ids, smoothing_min=SECTION_SMOOTHING_MIN):
    """
    Mean-target-encode cadastral section IDs with smoothing.

    For each section, compute the mean log-price of transactions in that section.
    Sections with fewer than `smoothing_min` observations are blended toward
    the global mean to avoid overfitting on rare sections.

    Returns:
        section_encoded: array of encoded values (one per row)
        section_means: dict mapping section_id -> smoothed mean log-price
        global_mean: the overall mean log-price (fallback for unseen sections)
    """
    global_mean = float(np.mean(log_prices))
    section_means = {}

    groups = defaultdict(list)
    for sid, lp in zip(section_ids, log_prices):
        groups[sid].append(lp)

    for sid, values in groups.items():
        n = len(values)
        local_mean = np.mean(values)
        # Smooth: weight local mean by n, global mean by smoothing_min
        weight = n / (n + smoothing_min)
        section_means[sid] = float(weight * local_mean + (1 - weight) * global_mean)

    section_encoded = np.array([
        section_means.get(sid, global_mean) for sid in section_ids
    ])
    return section_encoded, section_means, global_mean


def train(csv_path, output_path):
    """Train the model and save artifacts."""
    df = run_preprocessing(csv_path)

    # Target
    log_price = np.log(df["valeur_fonciere"].values)

    # Encode cadastral sections (all from Paris 15e after preprocessing filter)
    section_encoded, section_means, global_mean = encode_sections(
        log_price, df["section_id"].values
    )

    # Build feature matrix
    X = df[FEATURE_COLS].values.copy()
    X = np.column_stack([X, section_encoded])
    feature_names = FEATURE_COLS + ["section_location"]

    # Fit
    model = LinearRegression()
    model.fit(X, log_price)

    # Predictions & metrics
    log_pred = model.predict(X)
    residuals = log_price - log_pred
    sigma = float(np.std(residuals, ddof=1))
    r2 = float(r2_score(log_price, log_pred))

    # Back-transform for interpretable MAE
    pred_eur = np.exp(log_pred)
    actual_eur = df["valeur_fonciere"].values
    mae_eur = float(mean_absolute_error(actual_eur, pred_eur))

    # Build artifact
    coefficients = {"intercept": float(model.intercept_)}
    for name, coef in zip(feature_names, model.coef_):
        coefficients[name] = float(coef)

    artifact = {
        "coefficients": coefficients,
        "section_means": section_means,
        "global_mean_log_price": global_mean,
        "section_smoothing_min": SECTION_SMOOTHING_MIN,
        "sigma": sigma,
        "n_train": len(df),
        "r_squared": round(r2, 4),
        "mae_eur": round(mae_eur, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    # Report
    print("\n" + "=" * 60)
    print("TRAINING REPORT")
    print("=" * 60)
    print(f"  Samples:          {len(df)}")
    print(f"  Unique sections:  {len(section_means)}")
    print(f"  R-squared:        {r2:.4f}")
    print(f"  Residual sigma:   {sigma:.4f}")
    print(f"  MAE (EUR):        {mae_eur:,.0f}")
    print(f"\n  Coefficients:")
    for name, val in coefficients.items():
        print(f"    {name:30s} = {val:+.6f}")
    print(f"\n  Model saved to:   {output_path}")
    print("=" * 60)

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Paris 15e valuation model")
    parser.add_argument("--csv", default="data/dvf.csv", help="Path to DVF CSV")
    parser.add_argument("--out", default="artifacts/model.json", help="Output model path")
    args = parser.parse_args()

    train(args.csv, args.out)
