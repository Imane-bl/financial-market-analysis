
# src/indicators.py
 
import pandas as pd
import numpy as np
import os
from glob import glob
 
# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
 
PROCESSED_DATA_DIR = "data/processed/"
INDICATORS_DIR     = "data/indicators/"
os.makedirs(INDICATORS_DIR, exist_ok=True)
 
 
# ══════════════════════════════════════════════
# INDICATEURS TECHNIQUES DE BASE
# ══════════════════════════════════════════════
 
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index) — Wilder smoothing.
    Mesure la vitesse et l'amplitude des mouvements de prix.
        RSI < 30  → survente (signal d'achat potentiel)
        RSI > 70  → surachat (signal de vente potentiel)
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
 
 
def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    MA20  → tendance court terme  (1 mois de bourse)
    MA50  → tendance moyen terme  (~2 mois)
    MA200 → tendance long terme   (~1 an) — référence institutionnelle
    """
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df
 
 
def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Volatilité glissante sur 20 jours (écart-type des rendements journaliers)."""
    df["Volatility_20"] = df["Daily Return"].rolling(20).std()
    return df
 
 
def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum 10 jours = prix actuel − prix il y a 10 jours.
    Positif → tendance haussière à court terme.
    """
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    return df
 
 
# ══════════════════════════════════════════════
# INDICATEURS AVANCÉS
# ══════════════════════════════════════════════
 
def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    MACD     = EMA12 − EMA26
    Signal   = EMA9 du MACD
    Histogramme = MACD − Signal (ajouté en bonus)
 
    Croisement MACD > Signal → signal haussier.
    """
    df["EMA12"]      = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]      = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]       = df["EMA12"] - df["EMA26"]
    df["Signal"]     = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histo"] = df["MACD"] - df["Signal"]   # ← ajout vs version originale
    return df
 
 
def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bandes de Bollinger (±2 écarts-types autour de la MA20).
    Prix > UpperBand → surachat potentiel.
    Prix < LowerBand → survente potentielle.
    %B (ajouté) : position relative du prix dans les bandes [0, 1].
    """
    df["BB_MA20"]   = df["Close"].rolling(20).mean()
    df["BB_STD"]    = df["Close"].rolling(20).std()
    df["UpperBand"] = df["BB_MA20"] + (2 * df["BB_STD"])
    df["LowerBand"] = df["BB_MA20"] - (2 * df["BB_STD"])
 
    # %B = (Prix − LowerBand) / (UpperBand − LowerBand) : utile pour les signaux
    band_width       = df["UpperBand"] - df["LowerBand"]
    df["BB_pct"]     = (df["Close"] - df["LowerBand"]) / band_width.replace(0, np.nan)
    return df
 
 
# ══════════════════════════════════════════════
# GÉNÉRATION DES SIGNAUX DE TRADING
# ══════════════════════════════════════════════
 
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trois signaux indépendants agrégés en un signal global.
 
    Signal_MACD :  +1 si MACD > Signal  (haussier),  -1 sinon
    Signal_RSI  :  +1 si RSI < 30 (survente),  -1 si RSI > 70,  0 sinon
    Signal_MA   :  +1 si MA20 > MA50 (tendance haussière court terme),  -1 sinon
 
    Global_Signal ∈ {-3, -2, -1, 0, +1, +2, +3}
        +3 → tous les signaux haussiers (fort signal d'achat)
        -3 → tous les signaux baissiers (fort signal de vente)
    """
    df["Signal_MACD"] = np.where(df["MACD"] > df["Signal"], 1, -1)
 
    df["Signal_RSI"]  = np.where(df["RSI"] < 30,  1,
                         np.where(df["RSI"] > 70, -1, 0))
 
    df["Signal_MA"]   = np.where(df["MA20"] > df["MA50"], 1, -1)
 
    df["Global_Signal"] = (
        df["Signal_MACD"] +
        df["Signal_RSI"]  +
        df["Signal_MA"]
    )
    return df
 
 
# ══════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════
 
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = compute_moving_averages(df)
    df = compute_volatility(df)
    df = compute_momentum(df)
    df["RSI"] = compute_rsi(df["Close"])
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = generate_signals(df)
    return df
 
 
def process_file(file_path: str) -> None:
    df     = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df_ind = compute_indicators(df)
 
    # Supprime les lignes où les indicateurs longs (MA200, RSI…) sont encore NaN
    df_ind = df_ind.dropna(subset=["MA20", "MA50", "MA200", "RSI", "MACD", "UpperBand"])
 
    # Ordre des colonnes final
    cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Daily Return", "Log Return",
        # Tendance
        "MA20", "MA50", "MA200",
        # Risque & momentum
        "Volatility_20", "Momentum_10", "RSI",
        # MACD
        "EMA12", "EMA26", "MACD", "Signal", "MACD_Histo",
        # Bollinger
        "UpperBand", "LowerBand", "BB_pct",
        # Signaux
        "Signal_MACD", "Signal_RSI", "Signal_MA", "Global_Signal",
    ]
    # Ne garde que les colonnes effectivement présentes (sécurité)
    cols   = [c for c in cols if c in df_ind.columns]
    df_ind = df_ind[cols]
 
    out_name = os.path.basename(file_path).replace("_cleaned.csv", "_indicators.csv")
    out_path = os.path.join(INDICATORS_DIR, out_name)
    df_ind.to_csv(out_path, float_format="%.4f")
    print(f"  {out_name} → {len(df_ind)} lignes")
 
 
def main():
    print("=" * 50)
    print("  CALCUL DES INDICATEURS TECHNIQUES")
    print("=" * 50)
 
    fichiers = glob(os.path.join(PROCESSED_DATA_DIR, "*_cleaned.csv"))
    print(f"\n  {len(fichiers)} fichier(s) à traiter...\n")
 
    for file_path in fichiers:
        try:
            process_file(file_path)
        except Exception as e:
            print(f"  ERREUR sur {file_path} : {e}")
 
    print("\n  Indicateurs calculés. Fichiers dans data/indicators/")
 
 
if __name__ == "__main__":
    main()