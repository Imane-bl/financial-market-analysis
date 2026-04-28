# src/visualization.py
"""
visualization.py
================
Génération de tous les graphiques professionnels du projet.

Graphiques produits :
    1.  Prix + Moyennes Mobiles (MA20 / MA50 / MA200)
    2.  Rendements cumulés normalisés base 100
    3.  Volatilité glissante annualisée
    4.  Distribution des rendements + VaR
    5.  Graphique des drawdowns
    6.  Heatmap de corrélation
    7.  Frontière Efficiente de Markowitz
    8.  Simulation Monte Carlo (histogramme + tableau)
    9.  Scatter Risque / Rendement
    10. Stress Testing — performance sur crises historiques  ← NOUVEAU
    11. Risk Budgeting — contribution au risque              ← NOUVEAU
    12. Dashboard exécutif (vue d'ensemble sur une page)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ───────────────────────────────────────────────

DOSSIER_DONNEES  = "data/indicators/"
DOSSIER_RISQUE   = "data/risk/"
DOSSIER_FIGURES  = "outputs/figures/"

os.makedirs(DOSSIER_FIGURES, exist_ok=True)

ACTIONS = {
    "LVMH":          "LVMH",
    "TotalEnergies": "TotalEnergies",
    "BNP Paribas":   "BNP_Paribas",
}

COULEURS = {
    "LVMH":          "#1A3C6E",
    "TotalEnergies": "#E85D04",
    "BNP Paribas":   "#2D6A4F",
    "positif":       "#2D6A4F",
    "negatif":       "#C1121F",
    "accent":        "#F4A261",
}


# ───────────────────────────────────────────────
# STYLE GLOBAL
# ───────────────────────────────────────────────

def appliquer_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F9FA",
        "axes.edgecolor":    "#DEE2E6",
        "axes.grid":         True,
        "grid.color":        "#E9ECEF",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

appliquer_style()


def ajouter_filigrane(ax, texte="Projet Analyse Financière"):
    ax.text(0.99, 0.01, texte, transform=ax.transAxes,
            fontsize=7, color="#ADB5BD", ha="right", va="bottom",
            style="italic", alpha=0.7)


def formater_axe_pct(ax, axe="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.1f}%")
    if axe == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ══════════════════════════════════════════════
# GRAPHIQUE 1 — PRIX + MOYENNES MOBILES
# ══════════════════════════════════════════════

def graphique_prix_ma(df: pd.DataFrame, nom_action: str, chemin_sortie: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    couleur = COULEURS.get(nom_action, "#1A3C6E")

    ax.plot(df.index, df["Close"], label="Prix clôture",
            color=couleur, linewidth=1.8, zorder=3)
    ax.plot(df.index, df["MA20"],  label="MA 20",  color="#F4A261",
            linewidth=1.2, linestyle="--")
    ax.plot(df.index, df["MA50"],  label="MA 50",  color="#E76F51",
            linewidth=1.2, linestyle="--")
    ax.plot(df.index, df["MA200"], label="MA 200", color="#264653",
            linewidth=1.5, linestyle=":")

    ax.fill_between(df.index, df["Close"], df["MA200"],
                    where=df["Close"] >= df["MA200"],
                    alpha=0.08, color=COULEURS["positif"], label="Au-dessus MA200")
    ax.fill_between(df.index, df["Close"], df["MA200"],
                    where=df["Close"] < df["MA200"],
                    alpha=0.08, color=COULEURS["negatif"])

    ax.set_title(f"{nom_action} — Prix de Clôture & Moyennes Mobiles (2019–2024)")
    ax.set_ylabel("Prix (€)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", ncol=5, fontsize=9)
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 2 — RENDEMENTS CUMULÉS BASE 100
# ══════════════════════════════════════════════

def graphique_rendements_cumules(toutes_donnees: dict, chemin_sortie: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    for nom, df in toutes_donnees.items():
        cumule = (1 + df["Daily Return"]).cumprod() * 100
        ax.plot(cumule.index, cumule, label=nom,
                color=COULEURS.get(nom, "#999"), linewidth=2.2)

    ax.axhline(100, color="#6C757D", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.annotate("Krach\nCOVID-19",
                xy=(pd.Timestamp("2020-03-16"), 60),
                xytext=(pd.Timestamp("2020-07-01"), 48),
                arrowprops=dict(arrowstyle="->", color=COULEURS["negatif"]),
                fontsize=8.5, color=COULEURS["negatif"])

    ax.set_title("Rendements Cumulés — Base 100 (Janvier 2019 – Décembre 2024)")
    ax.set_ylabel("Valeur du portefeuille (Base = 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=11)
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 3 — VOLATILITÉ GLISSANTE
# ══════════════════════════════════════════════

def graphique_volatilite(toutes_donnees: dict, chemin_sortie: str, fenetre: int = 60):
    fig, ax = plt.subplots(figsize=(14, 5))

    for nom, df in toutes_donnees.items():
        vol = df["Daily Return"].rolling(fenetre).std() * np.sqrt(252) * 100
        ax.plot(vol.index, vol, label=nom,
                color=COULEURS.get(nom, "#999"), linewidth=1.8)

    ax.set_title(f"Volatilité Annualisée Glissante (fenêtre {fenetre} jours)")
    ax.set_ylabel("Volatilité (%)")
    formater_axe_pct(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=10)
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 4 — DISTRIBUTION DES RENDEMENTS + VaR
# ══════════════════════════════════════════════

def graphique_distribution(toutes_donnees: dict, chemin_sortie: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Distribution des Rendements Journaliers & VaR",
                 fontsize=14, fontweight="bold")

    for ax, (nom, df) in zip(axes, toutes_donnees.items()):
        r       = df["Daily Return"].dropna()
        couleur = COULEURS.get(nom, "#1A3C6E")

        ax.hist(r * 100, bins=80, color=couleur, alpha=0.75,
                edgecolor="white", linewidth=0.3, density=True)

        mu, sigma = r.mean() * 100, r.std() * 100
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
        ax.plot(x, norm.pdf(x, mu, sigma), "k--", linewidth=1.5, label="Loi normale")

        var95 = r.quantile(0.05) * 100
        var99 = r.quantile(0.01) * 100
        ax.axvline(var95, color="#E76F51", linewidth=2, linestyle="--",
                   label=f"VaR 95% : {var95:.2f}%")
        ax.axvline(var99, color=COULEURS["negatif"], linewidth=2, linestyle=":",
                   label=f"VaR 99% : {var99:.2f}%")

        ax.set_title(nom)
        ax.set_xlabel("Rendement journalier (%)")
        ax.legend(fontsize=8)
        ajouter_filigrane(ax)

    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 5 — DRAWDOWN
# ══════════════════════════════════════════════

def graphique_drawdown(toutes_donnees: dict, chemin_sortie: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    for nom, df in toutes_donnees.items():
        r   = df["Daily Return"].dropna()
        cum = (1 + r).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100

        ax.plot(dd.index, dd, label=nom,
                color=COULEURS.get(nom, "#999"), linewidth=1.8)
        ax.fill_between(dd.index, dd, 0,
                        alpha=0.07, color=COULEURS.get(nom, "#999"))

    ax.set_title("Analyse des Drawdowns — Perte depuis le Sommet (2019–2024)")
    ax.set_ylabel("Drawdown (%)")
    formater_axe_pct(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=10)
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 6 — HEATMAP DE CORRÉLATION
# ══════════════════════════════════════════════

def graphique_correlation(matrice_corr: pd.DataFrame, chemin_sortie: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrice_corr,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1, vmax=1, center=0,
        square=True,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title("Matrice de Corrélation des Rendements Journaliers\n"
                 "(Valeurs proches de 0 = meilleure diversification)",
                 fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 7 — FRONTIÈRE EFFICIENTE
# ══════════════════════════════════════════════

def graphique_frontiere(frontiere_df: pd.DataFrame,
                         resume_risque: pd.DataFrame,
                         chemin_sortie: str):
    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        frontiere_df["Volatilite_%"],
        frontiere_df["Rendement_%"],
        c=frontiere_df["Sharpe"],
        cmap="YlOrRd",
        alpha=0.6, s=15, zorder=2,
        label="Portefeuilles simulés"
    )
    plt.colorbar(sc, ax=ax, label="Ratio de Sharpe")

    ax.plot(frontiere_df["Volatilite_%"], frontiere_df["Rendement_%"],
            color="#1A3C6E", linewidth=2.5, zorder=4, label="Frontière Efficiente")

    for i, (action, row) in enumerate(resume_risque.iterrows()):
        v = row["Volatilité annualisée %"]
        r = row["Rendement annualisé %"]
        c = list(COULEURS.values())[i]
        ax.scatter(v, r, color=c, s=250, zorder=5,
                   edgecolors="white", linewidths=2)
        ax.annotate(action, (v, r), textcoords="offset points",
                    xytext=(8, 5), fontsize=10, color=c, fontweight="bold")

    ax.set_title("Frontière Efficiente de Markowitz\n"
                 "Compromis Rendement / Risque pour l'optimisation du portefeuille")
    ax.set_xlabel("Volatilité Annualisée (%)")
    ax.set_ylabel("Rendement Annualisé (%)")
    formater_axe_pct(ax, "x")
    formater_axe_pct(ax, "y")
    ax.legend(fontsize=9, loc="upper left")
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 8 — MONTE CARLO
# ══════════════════════════════════════════════

def graphique_monte_carlo(resultats_mc: pd.Series,
                           valeur_initiale: float,
                           chemin_sortie: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Simulation Monte Carlo — 10 000 Scénarios sur 1 An",
                 fontsize=13, fontweight="bold")

    var95  = np.percentile(resultats_mc, 5)
    cvar95 = resultats_mc[resultats_mc <= var95].mean()

    ax1.hist(resultats_mc, bins=100, color=COULEURS["LVMH"],
             alpha=0.75, edgecolor="white", linewidth=0.3)
    ax1.axvline(valeur_initiale, color=COULEURS["positif"], linewidth=2,
                linestyle="--", label=f"Valeur initiale : {valeur_initiale:,.0f} €")
    ax1.axvline(var95,  color=COULEURS["accent"],  linewidth=2,
                linestyle=":", label=f"VaR 95%  : {var95:,.0f} €")
    ax1.axvline(cvar95, color=COULEURS["negatif"], linewidth=2,
                linestyle=":", label=f"CVaR 95% : {cvar95:,.0f} €")
    ax1.axvline(resultats_mc.mean(), color="#FFD700", linewidth=2,
                linestyle="-", label=f"Moyenne  : {resultats_mc.mean():,.0f} €")
    ax1.set_xlabel("Valeur Finale du Portefeuille (€)")
    ax1.set_ylabel("Fréquence")
    ax1.set_title("Distribution des Valeurs Finales")
    ax1.legend(fontsize=8.5)
    ajouter_filigrane(ax1)

    stats = {
        "Valeur initiale":       f"{valeur_initiale:,.0f} €",
        "Valeur finale moyenne": f"{resultats_mc.mean():,.0f} €",
        "Médiane":               f"{resultats_mc.median():,.0f} €",
        "Écart-type":            f"{resultats_mc.std():,.0f} €",
        "VaR 95%":               f"{var95:,.0f} €",
        "CVaR 95%":              f"{cvar95:,.0f} €",
        "Meilleur scénario":     f"{resultats_mc.max():,.0f} €",
        "Pire scénario":         f"{resultats_mc.min():,.0f} €",
        "Probabilité de perte":  f"{(resultats_mc < valeur_initiale).mean()*100:.1f} %",
    }

    ax2.axis("off")
    tableau = ax2.table(
        cellText  = [[k, v] for k, v in stats.items()],
        colLabels = ["Indicateur", "Valeur"],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0.05, 0.1, 0.9, 0.8],
    )
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(11)
    for (r, c), cellule in tableau.get_celld().items():
        if r == 0:
            cellule.set_facecolor("#1A3C6E")
            cellule.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cellule.set_facecolor("#EEF2FF")
        cellule.set_edgecolor("#DEE2E6")
    ax2.set_title("Résumé Statistique", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 9 — SCATTER RISQUE / RENDEMENT
# ══════════════════════════════════════════════

def graphique_risque_rendement(resume_risque: pd.DataFrame, chemin_sortie: str):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (action, row) in enumerate(resume_risque.iterrows()):
        x       = row["Volatilité annualisée %"]
        y       = row["Rendement annualisé %"]
        taille  = max(row["Ratio de Sharpe"] * 600, 100)
        couleur = list(COULEURS.values())[i]

        ax.scatter(x, y, s=taille, color=couleur, alpha=0.85,
                   edgecolors="white", linewidths=2, zorder=4)
        ax.annotate(action, (x, y), textcoords="offset points",
                    xytext=(10, 5), fontsize=11, fontweight="bold", color=couleur)
        ax.annotate(f"Sharpe: {row['Ratio de Sharpe']:.2f}", (x, y),
                    textcoords="offset points",
                    xytext=(10, -12), fontsize=8, color="#6C757D")

    ax.set_title("Profil Risque / Rendement par Action (2019–2024)\n"
                 "(Taille de la bulle = ratio de Sharpe)")
    ax.set_xlabel("Volatilité Annualisée (%)")
    ax.set_ylabel("Rendement Annualisé (%)")
    formater_axe_pct(ax, "x")
    formater_axe_pct(ax, "y")
    ax.axhline(0, color="#6C757D", linewidth=0.8, linestyle="--")
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 10 — STRESS TESTING   ← NOUVEAU
# ══════════════════════════════════════════════

def graphique_stress_test(stress_df: pd.DataFrame, chemin_sortie: str):
    """
    Représente la performance du portefeuille lors de chaque crise historique.

    Ce graphique répond à la question centrale des comités de risque :
    "Combien aurait-on perdu lors des crises passées ?"

    Barres rouges = pertes   |   Barres vertes = gains (rebonds)
    Les chiffres dans les barres = performance exacte sur la période.

    C'est un graphique de stress testing standard dans les rapports
    de risque des investisseurs institutionnels (CDC, fonds de pension).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    valeurs  = stress_df["Perte/Gain_%"].values
    labels   = stress_df.index.tolist()
    couleurs = [COULEURS["positif"] if v > 0 else COULEURS["negatif"] for v in valeurs]

    barres = ax.barh(labels, valeurs, color=couleurs, alpha=0.85,
                     edgecolor="white", linewidth=0.8, height=0.55)

    # Valeur affichée à l'intérieur / l'extérieur de chaque barre
    for barre, val in zip(barres, valeurs):
        offset  = 0.5 if val >= 0 else -0.5
        ha      = "left" if val >= 0 else "right"
        ax.text(val + offset, barre.get_y() + barre.get_height() / 2,
                f"{val:+.2f}%", va="center", ha=ha,
                fontsize=11, fontweight="bold",
                color="white" if abs(val) > 5 else "#333")

    ax.axvline(0, color="#6C757D", linewidth=1.2)
    ax.set_title("Stress Testing — Performance du Portefeuille sur Crises Historiques\n"
                 "(Portefeuille Max Sharpe : 60% LVMH / 5% TotalEnergies / 35% BNP Paribas)",
                 fontsize=12)
    ax.set_xlabel("Performance (%)")
    formater_axe_pct(ax, "x")
    ax.invert_yaxis()   # scénario le plus récent en bas
    ajouter_filigrane(ax)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 11 — RISK BUDGETING   ← NOUVEAU
# ══════════════════════════════════════════════

def graphique_risk_budgeting(risk_budget_df: pd.DataFrame, chemin_sortie: str):
    """
    Compare la répartition du CAPITAL (poids %) vs la répartition du RISQUE.

    Message clé : un portefeuille équipondéré en capital n'est PAS équipondéré
    en risque. LVMH peut représenter 60% du capital mais seulement 57% du risque
    si sa volatilité est plus faible que BNP.

    C'est exactement ce que regarde un Risk Manager CDC :
    "Mon budget de risque est-il bien alloué ?"

    Un portefeuille Risk Parity (allocation égale du risque) est une
    stratégie institutionnelle majeure utilisée par Bridgewater Associates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Risk Budgeting — Poids en Capital vs Contribution au Risque",
                 fontsize=13, fontweight="bold")

    actions = risk_budget_df.index.tolist()
    couleurs_liste = [COULEURS.get(a, "#999") for a in actions]

    # ── Gauche : poids en capital
    poids = risk_budget_df["Poids_%"].values
    wedges1, texts1, autotexts1 = axes[0].pie(
        poids, labels=actions, colors=couleurs_liste,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 11},
    )
    for at in autotexts1:
        at.set_fontweight("bold")
        at.set_fontsize(12)
    axes[0].set_title("Répartition du Capital", fontsize=11, pad=15)

    # ── Droite : contribution au risque
    contrib = risk_budget_df["Contribution_Risque_%"].values
    wedges2, texts2, autotexts2 = axes[1].pie(
        contrib, labels=actions, colors=couleurs_liste,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 11},
    )
    for at in autotexts2:
        at.set_fontweight("bold")
        at.set_fontsize(12)
    axes[1].set_title("Contribution au Risque Total", fontsize=11, pad=15)

    # ── Tableau comparatif en bas
    tableau_data = [
        [f"{p:.1f}%" for p in poids],
        [f"{c:.1f}%" for c in contrib],
        [f"{r:.4f}" for r in risk_budget_df["Risque_Marginal"].values],
    ]
    fig.text(0.5, 0.02,
             "  Action            " + "".join(f"{a:<22}" for a in actions) + "\n" +
             "  Poids capital     " + "".join(f"{p:<22}" for p in tableau_data[0]) + "\n" +
             "  Contrib. risque   " + "".join(f"{c:<22}" for c in tableau_data[1]),
             ha="center", fontsize=9, family="monospace",
             color="#495057",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA",
                       edgecolor="#DEE2E6", alpha=0.8))

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# GRAPHIQUE 12 — DASHBOARD EXÉCUTIF
# ══════════════════════════════════════════════

def graphique_dashboard(toutes_donnees: dict,
                         resume_risque: pd.DataFrame,
                         matrice_corr: pd.DataFrame,
                         chemin_sortie: str):
    """
    Dashboard exécutif — une seule page qui résume tout le projet.
    Format "one-pager" standard dans les pitchbooks des banques d'affaires.

    CORRECTION vs version originale :
    - Colonnes du tableau filtrées dynamiquement (évite KeyError si une
      colonne est absente du CSV chargé)
    - Ajout des nouvelles métriques Alpha & Information Ratio dans le tableau
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Tableau de Bord — Analyse des Marchés Financiers\n"
                 "LVMH  ·  TotalEnergies  ·  BNP Paribas  ·  Janvier 2019 – Décembre 2024",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Rendements cumulés ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for nom, df in toutes_donnees.items():
        cumule = (1 + df["Daily Return"]).cumprod() * 100
        ax1.plot(cumule.index, cumule, label=nom,
                 color=COULEURS.get(nom, "#999"), linewidth=2)
    ax1.axhline(100, color="#6C757D", linewidth=0.6, linestyle="--")
    ax1.set_title("Rendements Cumulés (Base = 100)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(fontsize=9)

    # ── Matrice de corrélation ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(matrice_corr, ax=ax2, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, cbar=False,
                annot_kws={"size": 11, "weight": "bold"})
    ax2.set_title("Corrélations")

    # ── Tableau des métriques de risque ─────────────────────────────────────
    # CORRECTION : filtrage dynamique — on n'affiche que les colonnes présentes
    colonnes_souhaitees = [
        "Rendement annualisé %", "Volatilité annualisée %",
        "Ratio de Sharpe", "Ratio de Sortino",
        "Max Drawdown %", "VaR 95% Historique %", "Bêta vs CAC 40",
        "Alpha %", "Information Ratio",          # ← nouvelles métriques
    ]
    colonnes_affichees = [c for c in colonnes_souhaitees if c in resume_risque.columns]
    df_affiche         = resume_risque[colonnes_affichees].copy()

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis("off")
    tableau = ax3.table(
        cellText  = df_affiche.round(3).values,
        rowLabels = df_affiche.index.tolist(),
        colLabels = [c.replace(" %", "\n%").replace(" Ratio", "\nRatio")
                     for c in colonnes_affichees],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1],
    )
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(8.5)
    for (r, c), cellule in tableau.get_celld().items():
        if r == 0 or c == -1:
            cellule.set_facecolor("#1A3C6E")
            cellule.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cellule.set_facecolor("#EEF2FF")
        cellule.set_edgecolor("#DEE2E6")
    ax3.set_title("Métriques de Risque", fontsize=11, fontweight="bold", pad=10)

    # ── Drawdowns ───────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    for nom, df in toutes_donnees.items():
        r   = df["Daily Return"].dropna()
        cum = (1 + r).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        ax4.plot(dd.index, dd, label=nom,
                 color=COULEURS.get(nom, "#999"), linewidth=1.4)
    ax4.set_title("Drawdowns")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
    ax4.set_ylabel("%")

    # ── Volatilité glissante ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    for nom, df in toutes_donnees.items():
        vol = df["Daily Return"].rolling(60).std() * np.sqrt(252) * 100
        ax5.plot(vol.index, vol, label=nom,
                 color=COULEURS.get(nom, "#999"), linewidth=1.6)
    ax5.set_title("Volatilité Annualisée Glissante (60 jours)")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax5.set_ylabel("%")
    ax5.legend(fontsize=9)

    # ── Scatter risque/rendement ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    for i, (action, row) in enumerate(resume_risque.iterrows()):
        c = list(COULEURS.values())[i]
        ax6.scatter(row["Volatilité annualisée %"], row["Rendement annualisé %"],
                    s=200, color=c, edgecolors="white", linewidths=1.5, zorder=4)
        ax6.annotate(action,
                     (row["Volatilité annualisée %"], row["Rendement annualisé %"]),
                     textcoords="offset points", xytext=(5, 4), fontsize=8, color=c)
    ax6.set_title("Risque / Rendement")
    ax6.set_xlabel("Volatilité (%)")
    ax6.set_ylabel("Rendement (%)")

    plt.savefig(chemin_sortie)
    plt.close()
    print(f"  ✔ Sauvegardé : {chemin_sortie}")


# ══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def charger_toutes_donnees() -> dict:
    donnees = {}
    for nom, ticker in ACTIONS.items():
        chemin = os.path.join(DOSSIER_DONNEES, f"{ticker}_indicators.csv")
        if os.path.exists(chemin):
            df = pd.read_csv(chemin, index_col=0, parse_dates=True)
            donnees[nom] = df
        else:
            print(f"  ⚠️  Fichier introuvable : {chemin}")
    return donnees


def main():
    print("=" * 55)
    print("  VISUALISATION — Projet Marchés Financiers")
    print("=" * 55)

    toutes_donnees = charger_toutes_donnees()
    if not toutes_donnees:
        print("  ❌ Aucune donnée trouvée. Lance d'abord indicators.py")
        return

    # Chargement des résultats de risk_analysis.py
    def charger(chemin):
        return pd.read_csv(chemin, index_col=0) if os.path.exists(chemin) else None

    resume_risque   = charger(os.path.join(DOSSIER_RISQUE, "resume_risque.csv"))
    matrice_corr    = charger(os.path.join(DOSSIER_RISQUE, "matrice_correlation.csv"))
    frontiere_df    = pd.read_csv(os.path.join(DOSSIER_RISQUE, "frontiere_efficiente.csv")) \
                      if os.path.exists(os.path.join(DOSSIER_RISQUE, "frontiere_efficiente.csv")) else None
    resultats_mc    = pd.read_csv(os.path.join(DOSSIER_RISQUE, "monte_carlo.csv"))["Valeur_Finale"] \
                      if os.path.exists(os.path.join(DOSSIER_RISQUE, "monte_carlo.csv")) else None
    stress_df       = charger(os.path.join(DOSSIER_RISQUE, "stress_test.csv"))
    risk_budget_df  = charger(os.path.join(DOSSIER_RISQUE, "risk_budgeting.csv"))

    # ── Graphiques individuels par action ────────────────────────────────────
    print("\n── Graphiques individuels ──")
    for nom, df in toutes_donnees.items():
        nom_fichier = nom.replace(" ", "_")
        graphique_prix_ma(df, nom,
            os.path.join(DOSSIER_FIGURES, f"{nom_fichier}_prix_ma.png"))

    graphique_rendements_cumules(toutes_donnees,
        os.path.join(DOSSIER_FIGURES, "rendements_cumules.png"))
    graphique_volatilite(toutes_donnees,
        os.path.join(DOSSIER_FIGURES, "volatilite_glissante.png"))
    graphique_distribution(toutes_donnees,
        os.path.join(DOSSIER_FIGURES, "distribution_rendements.png"))
    graphique_drawdown(toutes_donnees,
        os.path.join(DOSSIER_FIGURES, "drawdowns.png"))

    # ── Graphiques de risque de portefeuille ─────────────────────────────────
    print("\n── Graphiques de portefeuille ──")
    if matrice_corr is not None:
        graphique_correlation(matrice_corr,
            os.path.join(DOSSIER_FIGURES, "heatmap_correlation.png"))
    if resume_risque is not None:
        graphique_risque_rendement(resume_risque,
            os.path.join(DOSSIER_FIGURES, "scatter_risque_rendement.png"))
    if frontiere_df is not None and resume_risque is not None:
        graphique_frontiere(frontiere_df, resume_risque,
            os.path.join(DOSSIER_FIGURES, "frontiere_efficiente.png"))
    if resultats_mc is not None:
        graphique_monte_carlo(resultats_mc, 100_000,
            os.path.join(DOSSIER_FIGURES, "monte_carlo.png"))

    # ── Nouveaux graphiques institutionnels ──────────────────────────────────
    print("\n── Graphiques institutionnels (nouveaux) ──")
    if stress_df is not None:
        graphique_stress_test(stress_df,
            os.path.join(DOSSIER_FIGURES, "stress_test.png"))
    if risk_budget_df is not None:
        graphique_risk_budgeting(risk_budget_df,
            os.path.join(DOSSIER_FIGURES, "risk_budgeting.png"))

    # ── Dashboard exécutif ───────────────────────────────────────────────────
    print("\n── Dashboard exécutif ──")
    if resume_risque is not None and matrice_corr is not None:
        graphique_dashboard(toutes_donnees, resume_risque, matrice_corr,
            os.path.join(DOSSIER_FIGURES, "dashboard_executif.png"))

    print(f"\n  ✅ Tous les graphiques dans {DOSSIER_FIGURES}")


if __name__ == "__main__":
    main()