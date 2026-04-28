# src/data_loader.py
 
import yfinance as yf
import pandas as pd
import os
 
# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
 
RAW_DATA_DIR = "data/raw/"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
 
TICKERS = {
    "LVMH":         "MC.PA",
    "TotalEnergies": "TTE.PA",
    "BNP_Paribas":  "BNP.PA",
}
 
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
 
 
# ──────────────────────────────────────────────
# FONCTIONS
# ──────────────────────────────────────────────
 
def download_stock_data(ticker: str,
                        start: str = START_DATE,
                        end:   str = END_DATE) -> pd.DataFrame:
    """
    Télécharge les données OHLCV via yfinance.
    auto_adjust=False : on garde les prix bruts + la colonne Adj Close séparée.
    Cela évite que yfinance écrase silencieusement les colonnes OHLC.
    """
    print(f"  Téléchargement : {ticker}...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
 
    if data.empty:
        raise ValueError(f"Aucune donnée reçue pour {ticker}. Vérifie le ticker ou ta connexion.")
 
    return data
 
 
def save_to_csv(data: pd.DataFrame, filename: str) -> None:
    path = os.path.join(RAW_DATA_DIR, filename)
    data.to_csv(path)
    print(f"  Sauvegardé → {path}  ({len(data)} lignes)")
 
 
# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────
 
def main():
    print("=" * 50)
    print("  TÉLÉCHARGEMENT DES DONNÉES BRUTES")
    print("=" * 50)
 
    for name, ticker in TICKERS.items():
        try:
            df = download_stock_data(ticker)
            save_to_csv(df, f"{name}.csv")
        except Exception as e:
            print(f"  ERREUR sur {name} ({ticker}) : {e}")
 
    print("\n  Téléchargement terminé. Fichiers dans data/raw/")
 
 
if __name__ == "__main__":
    main()