"""
Preprocessing pipeline for DVF (Demandes de Valeurs Foncières) data.
"""
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

REQUIRED_COLUMNS = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "code_departement",
    "type_local",
    "valeur_fonciere",
]
 
USEFUL_COLUMNS = REQUIRED_COLUMNS + [
    "latitude",
    "longitude",
    "surface_terrain",
    "date_mutation",
]
 
ALLOWED_TYPES = ["Appartement", "Maison"]

# Paris 15e arrondissement center (approx.)
PARIS_15_CENTER_LAT = 48.8422
PARIS_15_CENTER_LON = 2.2999
 
# Paris city center (Notre-Dame)
PARIS_CENTER_LAT = 48.8530
PARIS_CENTER_LON = 2.3499
 
PRICE_MIN_THRESHOLD = 1_000       
PRICE_MAX_THRESHOLD = 50_000_000  
SURFACE_MIN = 5                   
SURFACE_MAX = 1_000   

def load_and_filter_paris15(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"code_departement": str, "code_commune": str})
 
    # Filter to Paris 15e 
    df = df[df["code_commune"] == "75115"].copy()
 
    # Keep only meaningful property types 
    df = df[df["type_local"].isin(ALLOWED_TYPES)]
 
    # Drop rows with no sale price 
    df = df.dropna(subset=["valeur_fonciere"])
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
    df = df.dropna(subset=["valeur_fonciere"])
 
    # Remove weird prices 
    df = df[
        (df["valeur_fonciere"] >= PRICE_MIN_THRESHOLD) &
        (df["valeur_fonciere"] <= PRICE_MAX_THRESHOLD)
    ]
 
    # Remove duplicate mutations (same id_mutation + same parcelle) 
    df = df.drop_duplicates(subset=["id_mutation", "id_parcelle"])
 
    df = df.reset_index(drop=True)
    print(f"[load_and_filter_paris15] {len(df)} rows after filtering.")
    return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Handle missing values for required features
    - Encode type_local as a binary flag
    - Remove surface outliers
    - Ensure numeric types
    """
    df = df.copy()
 
    # surface_reelle_bati
    df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
    # Drop rows where surface is missing or implausible
    df = df.dropna(subset=["surface_reelle_bati"])
    df = df[
        (df["surface_reelle_bati"] >= SURFACE_MIN) &
        (df["surface_reelle_bati"] <= SURFACE_MAX)
    ]
 
    # nombre_pieces_principales 
    df["nombre_pieces_principales"] = pd.to_numeric(
        df["nombre_pieces_principales"], errors="coerce"
    )
    # Fill missing room count with median (grouped by type_local)
    median_rooms = df.groupby("type_local")["nombre_pieces_principales"].transform("median")
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(median_rooms)
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].clip(lower=0)
 
    # surface_terrain
    df["surface_terrain"] = pd.to_numeric(df["surface_terrain"], errors="coerce").fillna(0)
 
    # latitude / longitude 
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # Fill missing coords with Paris 15e center
    df["latitude"] = df["latitude"].fillna(PARIS_15_CENTER_LAT)
    df["longitude"] = df["longitude"].fillna(PARIS_15_CENTER_LON)
 
    # code_departement 
    df["code_departement"] = df["code_departement"].astype(str).str.zfill(2)
 
    # Encode type_local 
    # 1 = Appartement, 0 = Maison 
    df["is_appartement"] = (df["type_local"] == "Appartement").astype(int)
 
    # date_mutation → year / month (useful as temporal features)
    if "date_mutation" in df.columns:
        df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
        df["mutation_year"] = df["date_mutation"].dt.year
        df["mutation_month"] = df["date_mutation"].dt.month
 
    df = df.reset_index(drop=True)
    print(f"[clean_features] {len(df)} rows after cleaning.")
    return df
 
 
######## Feature engineering
 
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
 
 
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive new features:
    - price_per_m2
    - dist_to_paris_center (km)
    """
    df = df.copy()
 
    # price_per_m2 
    df["price_per_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    # Cap extreme price_per_m2 (e.g. data entry errors)
    p1, p99 = df["price_per_m2"].quantile([0.01, 0.99])
    df = df[(df["price_per_m2"] >= p1) & (df["price_per_m2"] <= p99)]
 
    # Geographic distances 
    df["dist_to_paris_center_km"] = df.apply(
        lambda row: _haversine_km(
            row["latitude"], row["longitude"],
            PARIS_CENTER_LAT, PARIS_CENTER_LON
        ),
        axis=1,
    )
    df = df.reset_index(drop=True)
    return df
 
 
# ── Full pipeline ─────────────────────────────────────────────────────────────
 
def run_preprocessing(csv_path: str) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline and return a model-ready DataFrame.
    """
    df = load_and_filter_paris15(csv_path)
    df = clean_features(df)
    df = feature_engineering(df)
 
    final_features = [
        "valeur_fonciere",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "code_departement",
        "type_local",
        "is_appartement",
        "latitude",
        "longitude",
        "dist_to_paris_center_km",
        "surface_terrain",
        "price_per_m2",
        "mutation_year",
        "mutation_month",
    ]
    available = [c for c in final_features if c in df.columns]
    df_out = df[available]
    return df_out
 
 
if __name__ == "__main__":
    import sys
 
    path = sys.argv[1] if len(sys.argv) > 1 else "data/dvf.csv"
    df_clean = run_preprocessing(path)
    print(df_clean.head())
    print("\nData types:\n", df_clean.dtypes)
    print("\nMissing values:\n", df_clean.isnull().sum())
 
 