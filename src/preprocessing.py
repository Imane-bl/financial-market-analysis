

# src/preprocessing.py
 
import pandas as pd
import numpy as np
import os
from glob import glob
 
# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
 
RAW_DATA_DIR       = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
 
 
# ──────────────────────────────────────────────
# NETTOYAGE
# ──────────────────────────────────────────────
 
def clean_csv(file_path: str) -> None:
    file_name = os.path.basename(file_path)
    print(f"\n  Traitement : {file_name}")
 
    # ── Lecture robuste ──────────────────────────────────────────────────────
    # yfinance injecte parfois 1 ou 2 lignes parasites sous le header
    # (lignes contenant "Ticker" ou "Price" selon la version).
    # On lit tout, puis on filtre uniquement les lignes à date valide.
    df = pd.read_csv(file_path, header=0)
 
    # Rename première colonne → "Date"
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
 
    # Supprime les lignes parasites : garde seulement celles au format YYYY-MM-DD
    masque_date = df["Date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}")
    lignes_supprimees = (~masque_date).sum()
    if lignes_supprimees > 0:
        print(f"    {lignes_supprimees} ligne(s) parasite(s) yfinance supprimée(s)")
    df = df[masque_date].copy()
 
    # ── Index temporel ───────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
 
    # ── Colonnes OHLCV ───────────────────────────────────────────────────────
    # On ne garde que les 5 colonnes utiles ; on ignore Adj Close et les
    # éventuelles colonnes multi-niveaux créées par yfinance.
    colonnes_cibles = ["Open", "High", "Low", "Close", "Volume"]
 
    # Si yfinance a créé un MultiIndex de colonnes, on l'aplatit
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
 
    # Garde uniquement les colonnes disponibles parmi les cibles
    colonnes_presentes = [c for c in colonnes_cibles if c in df.columns]
    df = df[colonnes_presentes].copy()
 
    # Conversion numérique (au cas où certaines valeurs sont en str)
    df[colonnes_presentes] = df[colonnes_presentes].apply(pd.to_numeric, errors="coerce")
 
    # ── Nettoyage doublons / NaN ─────────────────────────────────────────────
    lignes_avant = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["Close"])          # on exige au moins un prix de clôture
    lignes_apres = len(df)
    print(f"    {lignes_avant - lignes_apres} ligne(s) supprimée(s) (doublons / NaN Close)")
 
    # ── Rendements ───────────────────────────────────────────────────────────
    # NaN conservé sur la première ligne : c'est normal (pas de jour J-1).
    # Les modules suivants (indicators.py, risk_analysis.py) gèrent le dropna.
    df["Daily Return"] = df["Close"].pct_change()
    df["Log Return"]   = np.log(df["Close"] / df["Close"].shift(1))
 
    # ── Sauvegarde ───────────────────────────────────────────────────────────
    out_name = file_name.replace(".csv", "_cleaned.csv")
    out_path = os.path.join(PROCESSED_DATA_DIR, out_name)
    df.to_csv(out_path, float_format="%.4f")
    print(f"    Sauvegardé → {out_path}  ({len(df)} lignes)")
 
 
# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────
 
def main():
    print("=" * 50)
    print("  PRÉPROCESSING DES DONNÉES BRUTES")
    print("=" * 50)
 
    fichiers = glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    print(f"\n  {len(fichiers)} fichier(s) trouvé(s) dans {RAW_DATA_DIR}")
 
    for file_path in fichiers:
        try:
            clean_csv(file_path)
        except Exception as e:
            print(f"  ERREUR sur {file_path} : {e}")
 
    print("\n  Préprocessing terminé. Fichiers dans data/processed/")
 
 
if __name__ == "__main__":
    main()
 