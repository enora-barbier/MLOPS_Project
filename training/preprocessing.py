"""
Preprocessing pipeline for DVF (Demandes de Valeurs Foncières) data.
"""
import pandas as pd
import numpy as np

ALLOWED_TYPES = ["Appartement", "Maison"]

PRICE_MIN_THRESHOLD = 1_000
PRICE_MAX_THRESHOLD = 50_000_000
SURFACE_MIN = 5
SURFACE_MAX = 1_000


def load_and_filter_paris15(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={"code_departement": str, "code_commune": str},
        low_memory=False,
    )

    df = df.dropna(
        subset=[
            "valeur_fonciere",
            "surface_reelle_bati",
            "nombre_pieces_principales",
            "code_departement",
            "type_local",
        ]
    )

    df = df[df["code_commune"] == "75115"].copy()
    df = df[df["type_local"].isin(ALLOWED_TYPES)]

    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df = df.dropna(subset=["valeur_fonciere"])
    df = df[
        (df["valeur_fonciere"] >= PRICE_MIN_THRESHOLD)
        & (df["valeur_fonciere"] <= PRICE_MAX_THRESHOLD)
    ]

    df = df.drop_duplicates(subset=["id_mutation", "id_parcelle"])
    df = df.reset_index(drop=True)
    print(f"[load_and_filter_paris15] {len(df)} rows after filtering.")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    df = df.dropna(subset=["surface_reelle_bati"])
    df = df[
        (df["surface_reelle_bati"] >= SURFACE_MIN)
        & (df["surface_reelle_bati"] <= SURFACE_MAX)
    ]

    df["nombre_pieces_principales"] = pd.to_numeric(
        df["nombre_pieces_principales"], errors="coerce"
    )
    median_rooms = df.groupby("type_local")["nombre_pieces_principales"].transform("median")
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(median_rooms)
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].clip(lower=0)

    df["is_appartement"] = (df["type_local"] == "Appartement").astype(int)

    df = df.reset_index(drop=True)
    print(f"[clean_features] {len(df)} rows after cleaning.")
    return df


def run_preprocessing(csv_path: str) -> pd.DataFrame:
    df = load_and_filter_paris15(csv_path)
    df = clean_features(df)

    final_features = [
        "valeur_fonciere",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "is_appartement",
        "id_parcelle",
    ]
    return df[final_features]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/dvf.csv"
    df_clean = run_preprocessing(path)
    print(df_clean.head())
    print("\nData types:\n", df_clean.dtypes)
    print("\nMissing values:\n", df_clean.isnull().sum())