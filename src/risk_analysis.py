# src/risk_analysis.py
"""
risk_analysis.py
================
Calcul des indicateurs de risque avancés pour chaque action,
puis optimisation du portefeuille (théorie de Markowitz).

Métriques calculées :
    - Volatilité annualisée
    - Ratio de Sharpe
    - Ratio de Sortino
    - Maximum Drawdown
    - VaR historique et paramétrique (95% et 99%)
    - CVaR / Expected Shortfall
    - Bêta vs CAC 40
    - Alpha & Tracking Error & Information Ratio     ← NOUVEAU
    - Contribution au risque (Risk Budgeting)        ← NOUVEAU
    - Stress Testing sur crises historiques          ← NOUVEAU
    - Frontière efficiente de Markowitz
    - Simulation Monte Carlo (bug drift corrigé)     ← CORRIGÉ
"""

import os
import numpy as np
import pandas as pd
import scipy.optimize as sco
import yfinance as yf
from scipy import stats

# ──────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ──────────────────────────────────────────────

DOSSIER_DONNEES  = "data/processed/"
DOSSIER_RISQUE   = "data/risk/"
TAUX_SANS_RISQUE = 0.03          # OAT 10 ans France
JOURS_BOURSE     = 252
TICKER_MARCHE    = "^FCHI"       # CAC 40

ACTIONS = {
    "LVMH":          "MC.PA",
    "TotalEnergies": "TTE.PA",
    "BNP Paribas":   "BNP.PA",
}

# Scénarios de stress testing historiques
SCENARIOS_STRESS = {
    "Krach COVID (Feb–Mar 2020)":       ("2020-02-19", "2020-03-23"),
    "Rebond COVID (Mar–Août 2020)":     ("2020-03-23", "2020-08-18"),
    "Crise taux / inflation (2022)":    ("2022-01-04", "2022-10-13"),
    "Crise bancaire SVB (Mar 2023)":    ("2023-03-06", "2023-03-24"),
}

os.makedirs(DOSSIER_RISQUE, exist_ok=True)


# ══════════════════════════════════════════════
# PARTIE 1 — MÉTRIQUES INDIVIDUELLES PAR ACTION
# ══════════════════════════════════════════════

def calculer_sharpe(rendements: pd.Series) -> float:
    """
    Ratio de Sharpe = (Rendement annuel − Taux sans risque) / Volatilité annuelle.
    Mesure universelle de performance ajustée au risque.
        < 0   → sous-performe le taux sans risque
        0–1   → acceptable
        > 1   → bon  |  > 2 → excellent
    """
    rendement_annuel  = rendements.mean() * JOURS_BOURSE
    volatilite_annuel = rendements.std()  * np.sqrt(JOURS_BOURSE)
    return (rendement_annuel - TAUX_SANS_RISQUE) / volatilite_annuel


def calculer_sortino(rendements: pd.Series) -> float:
    """
    Ratio de Sortino : comme le Sharpe, mais on ne pénalise que la
    volatilité à la baisse. Les hausses ne sont pas du risque pour l'investisseur.
    """
    rendement_annuel  = rendements.mean() * JOURS_BOURSE
    baisses           = rendements[rendements < 0]
    volatilite_baisse = baisses.std() * np.sqrt(JOURS_BOURSE)
    return (rendement_annuel - TAUX_SANS_RISQUE) / volatilite_baisse


def calculer_max_drawdown(rendements_cumules: pd.Series) -> dict:
    """
    Maximum Drawdown = pire perte cumulée depuis un sommet jusqu'au creux suivant.
    Question clé : "Si j'avais investi au pire moment, combien aurais-je perdu ?"
    """
    sommet_glissant = rendements_cumules.cummax()
    drawdown        = rendements_cumules / sommet_glissant - 1
    pire_perte      = drawdown.min()
    date_creux      = drawdown.idxmin()
    date_sommet     = rendements_cumules[:date_creux].idxmax()
    duree_jours     = (date_creux - date_sommet).days
    return {
        "max_drawdown_%": round(pire_perte * 100, 2),
        "date_debut":     date_sommet,
        "date_fin":       date_creux,
        "duree_jours":    duree_jours,
    }


def calculer_var(rendements: pd.Series, confiance: float = 0.95) -> dict:
    """
    VaR = perte maximale journalière avec X% de probabilité.
    Deux méthodes :
        Historique   → percentile des vraies données (pas d'hypothèse)
        Paramétrique → hypothèse loi normale (standard bancaire)
    """
    alpha        = 1 - confiance
    var_hist     = rendements.quantile(alpha)
    mu, sigma    = rendements.mean(), rendements.std()
    var_normale  = mu + sigma * stats.norm.ppf(alpha)
    niveau       = int(confiance * 100)
    return {
        f"VaR_Historique_{niveau}%":   round(var_hist    * 100, 4),
        f"VaR_Parametrique_{niveau}%": round(var_normale * 100, 4),
    }


def calculer_cvar(rendements: pd.Series, confiance: float = 0.95) -> float:
    """
    CVaR (Expected Shortfall) = perte moyenne dans les pires cas (au-delà du VaR).
    Plus prudent que la VaR car il mesure la sévérité des scénarios extrêmes.
    Mesure imposée par Bâle III dans toutes les banques depuis 2016.
    """
    seuil = rendements.quantile(1 - confiance)
    cvar  = rendements[rendements <= seuil].mean()
    return round(cvar * 100, 4)


def calculer_beta(rendements_action: pd.Series,
                  rendements_marche: pd.Series) -> float:
    """
    Bêta = sensibilité de l'action au marché (CAC 40).
        > 1 → amplifie les mouvements du marché (cyclique)
        < 1 → amortit les mouvements (défensif)
        < 0 → se comporte en sens inverse (très rare)
    """
    donnees     = pd.concat([rendements_action, rendements_marche], axis=1).dropna()
    matrice_cov = np.cov(donnees.iloc[:, 0], donnees.iloc[:, 1])
    return round(matrice_cov[0, 1] / matrice_cov[1, 1], 4)


def calculer_tracking_error(rendements_action: pd.Series,
                             rendements_marche: pd.Series) -> dict:
    """
    Tracking Error & Information Ratio — LE langage des gestionnaires institutionnels.

    Tracking Error  = volatilité annualisée de l'écart vs benchmark (CAC 40).
    Alpha           = surperformance annualisée brute vs benchmark.
    Information Ratio = Alpha / Tracking Error.
        IR > 0.5 → surperformance régulière, considérée bonne par les institutionnels.
        IR > 1.0 → excellent (rare)

    La CDC utilise ces métriques pour évaluer ses mandats de gestion externe.
    """
    donnees       = pd.concat([rendements_action, rendements_marche], axis=1).dropna()
    donnees.columns = ["action", "marche"]

    active_return   = donnees["action"] - donnees["marche"]
    tracking_error  = active_return.std()  * np.sqrt(JOURS_BOURSE) * 100
    alpha           = active_return.mean() * JOURS_BOURSE * 100
    information_ratio = alpha / tracking_error if tracking_error != 0 else 0

    return {
        "Alpha_%":           round(alpha, 2),
        "Tracking_Error_%":  round(tracking_error, 2),
        "Information_Ratio": round(information_ratio, 3),
    }


# ══════════════════════════════════════════════
# PARTIE 2 — OPTIMISATION DE PORTEFEUILLE
# ══════════════════════════════════════════════

def performance_portefeuille(poids: np.ndarray,
                              rendements_moyens: pd.Series,
                              matrice_covariance: pd.DataFrame) -> tuple:
    """
    Calcule (rendement, volatilité, Sharpe) d'un portefeuille pondéré.
    La formule matricielle σ = √(w'Σw)×√252 capture l'effet de diversification :
    si les actions sont peu corrélées, σ_portef < moyenne des σ individuels.
    """
    rendement  = np.dot(poids, rendements_moyens) * JOURS_BOURSE
    volatilite = np.sqrt(poids @ matrice_covariance.values @ poids) * np.sqrt(JOURS_BOURSE)
    sharpe     = (rendement - TAUX_SANS_RISQUE) / volatilite
    return rendement, volatilite, sharpe


def portefeuille_max_sharpe(rendements_moyens: pd.Series,
                             matrice_covariance: pd.DataFrame) -> np.ndarray:
    """Portefeuille optimal de Markowitz : maximise le ratio de Sharpe."""
    n           = len(rendements_moyens)
    contrainte  = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bornes = tuple((0.05, 0.60) for _ in range(n))
    poids_init  = np.array([1 / n] * n)
    resultat    = sco.minimize(
        fun         = lambda w: -performance_portefeuille(w, rendements_moyens, matrice_covariance)[2],
        x0          = poids_init,
        method      = "SLSQP",
        bounds      = bornes,
        constraints = contrainte,
    )
    return resultat.x


def portefeuille_min_variance(rendements_moyens: pd.Series,
                               matrice_covariance: pd.DataFrame) -> np.ndarray:
    """Portefeuille le moins risqué possible (Minimum Variance)."""
    n           = len(rendements_moyens)
    contrainte  = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bornes = tuple((0.05, 0.60) for _ in range(n))
    poids_init  = np.array([1 / n] * n)
    resultat    = sco.minimize(
        fun         = lambda w: performance_portefeuille(w, rendements_moyens, matrice_covariance)[1],
        x0          = poids_init,
        method      = "SLSQP",
        bounds      = bornes,
        constraints = contrainte,
    )
    return resultat.x


def calculer_frontiere_efficiente(rendements_moyens: pd.Series,
                                   matrice_covariance: pd.DataFrame,
                                   nb_points: int = 100) -> pd.DataFrame:
    """
    Frontière Efficiente de Markowitz (Prix Nobel 1990).
    Pour chaque niveau de rendement cible, cherche le portefeuille
    qui l'atteint avec le minimum de risque.
    Tout portefeuille en dessous est sous-optimal.
    """
    n         = len(rendements_moyens)
    cible_min = rendements_moyens.min() * JOURS_BOURSE
    cible_max = rendements_moyens.max() * JOURS_BOURSE
    cibles    = np.linspace(cible_min, cible_max, nb_points)
    resultats = []

    for cible in cibles:
        contraintes = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, c=cible:
             performance_portefeuille(w, rendements_moyens, matrice_covariance)[0] - c},
        ]
        res = sco.minimize(
            fun         = lambda w: performance_portefeuille(w, rendements_moyens, matrice_covariance)[1],
            x0          = np.array([1 / n] * n),
            method      = "SLSQP",
            bounds      = tuple((0.0, 1.0) for _ in range(n)),
            constraints = contraintes,
        )
        if res.success:
            r, v, s = performance_portefeuille(res.x, rendements_moyens, matrice_covariance)
            ligne   = {"Rendement_%": round(r*100, 3),
                       "Volatilite_%": round(v*100, 3),
                       "Sharpe": round(s, 3)}
            for nom, p in zip(rendements_moyens.index, res.x):
                ligne[f"Poids_{nom}"] = round(p, 4)
            resultats.append(ligne)

    return pd.DataFrame(resultats)


def contribution_au_risque(poids: np.ndarray,
                            matrice_covariance: pd.DataFrame,
                            noms_actions: list) -> pd.DataFrame:
    """
    Risk Budgeting — décompose la volatilité totale par action.

    Question clé : LVMH a 50% de poids mais contribue à quelle part du risque ?
    Un portefeuille "équipondéré en risque" alloue le même budget de risque
    à chaque action, indépendamment des poids en capital.

    C'est ce qu'un Risk Manager CDC regarde en priorité :
    l'exposition au risque, pas l'exposition en capital.
    """
    poids = np.array(poids)
    sigma = np.sqrt(poids @ matrice_covariance.values @ poids) * np.sqrt(JOURS_BOURSE)

    # Dérivée partielle : sensibilité marginale de σ_portef au poids de chaque action
    risque_marginal  = (matrice_covariance.values @ poids) * np.sqrt(JOURS_BOURSE) / sigma
    contrib_absolue  = poids * risque_marginal
    contrib_relative = contrib_absolue / sigma * 100

    df = pd.DataFrame({
        "Action":                noms_actions,
        "Poids_%":               [round(p * 100, 1) for p in poids],
        "Contribution_Risque_%": [round(c, 1)       for c in contrib_relative],
        "Risque_Marginal":       [round(r, 4)        for r in risque_marginal],
    })
    return df.set_index("Action")


def stress_test(matrice_rendements: pd.DataFrame,
                poids: np.ndarray,
                scenarios: dict) -> pd.DataFrame:
    """
    Stress Testing sur crises historiques réelles.
    Simule la performance du portefeuille durant chaque crise.

    Beaucoup plus parlant en entretien que la VaR abstraite :
    "Mon portefeuille aurait perdu X% pendant le COVID."
    La CDC réalise des stress tests réglementaires similaires
    pour ses portefeuilles propres.
    """
    resultats = []
    for nom_scenario, (debut, fin) in scenarios.items():
        periode = matrice_rendements.loc[debut:fin]
        if len(periode) == 0:
            print(f"    Avertissement : aucune donnée pour '{nom_scenario}'")
            continue
        # Rendement cumulé du portefeuille sur la période
        rendement_portef = ((1 + periode @ poids).prod() - 1) * 100
        resultats.append({
            "Scénario":      nom_scenario,
            "Perte/Gain_%":  round(rendement_portef, 2),
            "Jours_crise":   len(periode),
            "Début":         debut,
            "Fin":           fin,
        })
    return pd.DataFrame(resultats).set_index("Scénario")


# ══════════════════════════════════════════════
# PARTIE 3 — SIMULATION MONTE CARLO  [BUG CORRIGÉ]
# ══════════════════════════════════════════════

def simulation_monte_carlo(rendements_moyens: pd.Series,
                            matrice_covariance: pd.DataFrame,
                            nb_simulations: int = 10_000,
                            nb_jours: int = 252,
                            valeur_initiale: float = 100_000) -> tuple:
    """
    Monte Carlo : 10 000 scénarios d'évolution du portefeuille sur 1 an.

    CORRECTION vs version originale :
    ──────────────────────────────────
    L'ancienne version divisait rendements_moyens par JOURS_BOURSE pour obtenir
    la moyenne journalière. Mais rendements_moyens = matrice.mean() est DÉJÀ
    une moyenne JOURNALIÈRE. Diviser à nouveau par 252 donnait un drift ≈ 0,
    ce qui expliquait la probabilité de perte absurde de 48.5%.

    Solution : utiliser rendements_moyens directement comme drift journalier.

    Méthode Cholesky :
    ──────────────────
    On génère des rendements corrélés via la décomposition de Cholesky
    de la matrice de covariance journalière. C'est la méthode standard
    en finance quantitative pour respecter la structure de dépendance
    entre actifs (capital pour les scénarios de crise corrélés).
    """
    n_actions = len(rendements_moyens)

    #  CORRIGÉ : rendements_moyens est déjà journalier (issu de matrice.mean())
    moy_journalier = rendements_moyens.values

    # Matrice de covariance journalière
    cov_journalier = matrice_covariance.values

    # Décomposition de Cholesky : injecte la structure de corrélation
    L     = np.linalg.cholesky(cov_journalier)
    poids = portefeuille_max_sharpe(rendements_moyens, matrice_covariance)

    valeurs_finales = []
    for _ in range(nb_simulations):
        # Chocs aléatoires indépendants (252 jours × 3 actions)
        Z = np.random.standard_normal((nb_jours, n_actions))

        # Injection de la corrélation via Cholesky
        rendements_journaliers  = moy_journalier + (L @ Z.T).T

        # Rendement journalier du portefeuille pondéré
        rendements_portefeuille = rendements_journaliers @ poids

        # Valeur finale = valeur initiale × ∏(1 + r_t)
        valeur_finale = valeur_initiale * np.prod(1 + rendements_portefeuille)
        valeurs_finales.append(valeur_finale)

    resultats = pd.Series(valeurs_finales, name="Valeur_Finale")
    var95     = np.percentile(resultats, 5)
    cvar95    = resultats[resultats <= var95].mean()

    print("\n📊 Résultats Monte Carlo (10 000 simulations, horizon 1 an)")
    print("=" * 52)
    print(f"  Valeur initiale         : {valeur_initiale:>12,.0f} €")
    print(f"  Valeur finale moyenne   : {resultats.mean():>12,.0f} €")
    print(f"  Médiane                 : {resultats.median():>12,.0f} €")
    print(f"  VaR 95%                 : {var95:>12,.0f} €")
    print(f"  CVaR 95%                : {cvar95:>12,.0f} €")
    print(f"  Meilleur scénario       : {resultats.max():>12,.0f} €")
    print(f"  Pire scénario           : {resultats.min():>12,.0f} €")
    print(f"  Probabilité de perte    : {(resultats < valeur_initiale).mean()*100:.1f} %")

    return resultats, poids


# ══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def charger_matrice_rendements() -> pd.DataFrame:
    """Charge les CSV nettoyés et construit la matrice de rendements journaliers."""
    FICHIERS = {
        "LVMH":          "LVMH_cleaned.csv",
        "TotalEnergies": "TotalEnergies_cleaned.csv",
        "BNP Paribas":   "BNP_Paribas_cleaned.csv",
    }
    colonnes = {}
    for nom, nom_fichier in FICHIERS.items():
        chemin = os.path.join(DOSSIER_DONNEES, nom_fichier)
        if not os.path.exists(chemin):
            print(f"  Fichier introuvable : {chemin}")
            continue
        df = pd.read_csv(chemin, index_col=0, parse_dates=True)
        if "Daily Return" not in df.columns:
            print(f"  Colonne 'Daily Return' absente dans {nom_fichier}")
            continue
        colonnes[nom] = df["Daily Return"].dropna()
        print(f"  {nom} chargé ({len(colonnes[nom])} jours)")

    return pd.DataFrame(colonnes).dropna()


def executer_analyse_individuelle(matrice: pd.DataFrame,
                                   rendements_marche: pd.Series) -> pd.DataFrame:
    """Calcule toutes les métriques de risque par action, sauvegarde le résumé."""
    lignes = []
    for action in matrice.columns:
        r    = matrice[action].dropna()
        cum  = (1 + r).cumprod()
        mdd  = calculer_max_drawdown(cum)
        var95 = calculer_var(r, 0.95)
        var99 = calculer_var(r, 0.99)
        te    = calculer_tracking_error(r, rendements_marche)

        lignes.append({
            "Action":                  action,
            "Rendement annualisé %":   round(r.mean() * JOURS_BOURSE * 100, 2),
            "Volatilité annualisée %": round(r.std()  * np.sqrt(JOURS_BOURSE) * 100, 2),
            "Ratio de Sharpe":         round(calculer_sharpe(r), 3),
            "Ratio de Sortino":        round(calculer_sortino(r), 3),
            "Max Drawdown %":          mdd["max_drawdown_%"],
            "Durée Drawdown (jours)":  mdd["duree_jours"],
            "VaR 95% Historique %":    var95["VaR_Historique_95%"],
            "VaR 99% Historique %":    var99["VaR_Historique_99%"],
            "CVaR 95% %":              calculer_cvar(r, 0.95),
            "Bêta vs CAC 40":          calculer_beta(r, rendements_marche),
            # ← NOUVEAU
            "Alpha %":                 te["Alpha_%"],
            "Tracking Error %":        te["Tracking_Error_%"],
            "Information Ratio":       te["Information_Ratio"],
        })

    resume = pd.DataFrame(lignes).set_index("Action")
    resume.to_csv(os.path.join(DOSSIER_RISQUE, "resume_risque.csv"))
    print("\n  Résumé sauvegardé → data/risk/resume_risque.csv")
    print(resume.T.to_string())
    return resume


def executer_analyse_portefeuille(matrice: pd.DataFrame) -> dict:
    """Optimisation Markowitz + Risk Budgeting + Stress Testing + Monte Carlo."""
    rendements_moyens  = matrice.mean()
    matrice_covariance = matrice.cov()

    matrice.corr().to_csv(os.path.join(DOSSIER_RISQUE, "matrice_correlation.csv"))

    # ── Portefeuilles optimaux ───────────────────────────────────────────────
    poids_ms = portefeuille_max_sharpe(rendements_moyens, matrice_covariance)
    poids_mv = portefeuille_min_variance(rendements_moyens, matrice_covariance)
    perf_ms  = performance_portefeuille(poids_ms, rendements_moyens, matrice_covariance)
    perf_mv  = performance_portefeuille(poids_mv, rendements_moyens, matrice_covariance)

    print("\n📈 Portefeuille Maximum Sharpe")
    for nom, p in zip(matrice.columns, poids_ms):
        print(f"  {nom:<20} {p*100:.1f} %")
    print(f"  Rendement : {perf_ms[0]*100:.2f}%  |  Volatilité : {perf_ms[1]*100:.2f}%  |  Sharpe : {perf_ms[2]:.3f}")

    print("\n📉 Portefeuille Minimum Variance")
    for nom, p in zip(matrice.columns, poids_mv):
        print(f"  {nom:<20} {p*100:.1f} %")
    print(f"  Rendement : {perf_mv[0]*100:.2f}%  |  Volatilité : {perf_mv[1]*100:.2f}%  |  Sharpe : {perf_mv[2]:.3f}")

    # ── Risk Budgeting ───────────────────────────────────────────────────────
    print("\n📊 Risk Budgeting — Contribution au risque (portefeuille Max Sharpe)")
    rb = contribution_au_risque(poids_ms, matrice_covariance, list(matrice.columns))
    print(rb.to_string())
    rb.to_csv(os.path.join(DOSSIER_RISQUE, "risk_budgeting.csv"))

    # ── Stress Testing ───────────────────────────────────────────────────────
    print("\n🔥 Stress Testing — Performance sur crises historiques")
    st = stress_test(matrice, poids_ms, SCENARIOS_STRESS)
    print(st.to_string())
    st.to_csv(os.path.join(DOSSIER_RISQUE, "stress_test.csv"))

    # ── Frontière Efficiente ─────────────────────────────────────────────────
    print("\n  Calcul de la frontière efficiente...")
    frontiere = calculer_frontiere_efficiente(rendements_moyens, matrice_covariance)
    frontiere.to_csv(os.path.join(DOSSIER_RISQUE, "frontiere_efficiente.csv"), index=False)

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    print("\n  Simulation Monte Carlo...")
    mc_resultats, poids_optimaux = simulation_monte_carlo(rendements_moyens, matrice_covariance)
    mc_resultats.to_csv(os.path.join(DOSSIER_RISQUE, "monte_carlo.csv"), index=False)

    return {
        "rendements_moyens":   rendements_moyens,
        "matrice_covariance":  matrice_covariance,
        "matrice_correlation": matrice.corr(),
        "poids_max_sharpe":    poids_ms,
        "poids_min_variance":  poids_mv,
        "risk_budgeting":      rb,
        "stress_test":         st,
        "frontiere":           frontiere,
        "monte_carlo":         mc_resultats,
        "poids_optimaux":      poids_optimaux,
    }


def main():
    print("=" * 55)
    print("  ANALYSE DES RISQUES — Projet Marchés Financiers")
    print("=" * 55)

    matrice = charger_matrice_rendements()
    print(f"\n  Actions chargées : {list(matrice.columns)}")
    print(f"  Période : {matrice.index[0].date()} → {matrice.index[-1].date()}")

    # Téléchargement CAC 40 pour Bêta, Alpha, Tracking Error
    print("\n  Téléchargement du CAC 40 (benchmark)...")
    cac40 = yf.download(TICKER_MARCHE,
                        start=matrice.index[0],
                        end=matrice.index[-1],
                        progress=False)
    rendements_marche = (cac40["Close"]
                         .pct_change()
                         .dropna()
                         .reindex(matrice.index)
                         .dropna())

    print("\n── Métriques individuelles ──────────────────────────")
    resume_risque = executer_analyse_individuelle(matrice, rendements_marche)

    print("\n── Optimisation du portefeuille ─────────────────────")
    resultats_portefeuille = executer_analyse_portefeuille(matrice)

    print("\n  Analyse complète. Tous les résultats dans data/risk/")
    return resume_risque, resultats_portefeuille


if __name__ == "__main__":
    main()