"""
app.py — Dashboard Streamlit
Financial Market Analytics — CAC 40
Imene Bellaghma · Master MIAGE · Paris Cité

Lancement : streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")

# ──────────────────────────────────────────────
# CONFIG PAGE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Market Analytics — CAC 40",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond principal */
    .stApp { background-color: #0a0f1e; color: #e2e8f0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1428; border-right: 1px solid #1e3a5f; }
    
    /* Métriques */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1f3c, #0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="metric-container"] label { color: #64b5f6 !important; font-size: 0.75rem; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color: #e2e8f0 !important; }
    
    /* Headers */
    h1, h2, h3 { color: #90caf9 !important; }
    
    /* Séparateur */
    hr { border-color: #1e3a5f; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .badge-green  { background: rgba(0,200,100,.15); color: #00e676; border: 1px solid rgba(0,200,100,.3); }
    .badge-red    { background: rgba(255,60,60,.12);  color: #ff5252; border: 1px solid rgba(255,60,60,.25);}
    .badge-blue   { background: rgba(30,150,255,.12); color: #64b5f6; border: 1px solid rgba(30,150,255,.25);}
    .badge-gold   { background: rgba(255,183,0,.12);  color: #fbbf24; border: 1px solid rgba(255,183,0,.25); }
    
    /* Card */
    .info-card {
        background: linear-gradient(135deg,#0d1f3c,#0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────
TICKERS = {
    "LVMH":          "MC.PA",
    "TotalEnergies": "TTE.PA",
    "BNP Paribas":   "BNP.PA",
}
COLORS = {
    "LVMH":          "#00e676",
    "TotalEnergies": "#fbbf24",
    "BNP Paribas":   "#64b5f6",
}
TAUX_SANS_RISQUE = 0.03
JOURS_BOURSE     = 252
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

SCENARIOS_STRESS = {
    "Krach COVID (Fév–Mar 2020)":    ("2020-02-19", "2020-03-23"),
    "Rebond COVID (Mar–Août 2020)":  ("2020-03-23", "2020-08-18"),
    "Crise taux / inflation (2022)": ("2022-01-04", "2022-10-13"),
    "Crise bancaire SVB (Mar 2023)": ("2023-03-06", "2023-03-24"),
}

# ──────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (cache 1h)
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Téléchargement des données de marché…")
def charger_donnees():
    """Télécharge OHLCV pour les 3 actions + CAC 40."""
    prix = {}
    for nom, ticker in TICKERS.items():
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            st.error(f"Impossible de télécharger {nom} ({ticker})")
            return None, None
        prix[nom] = df["Close"].squeeze()

    cac = yf.download("^FCHI", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)["Close"].squeeze()

    prix_df = pd.DataFrame(prix).dropna()
    rendements = prix_df.pct_change().dropna()
    rendements_cac = cac.pct_change().dropna().reindex(rendements.index).dropna()
    rendements = rendements.reindex(rendements_cac.index).dropna()

    return prix_df, rendements, rendements_cac


# ──────────────────────────────────────────────
# FONCTIONS FINANCIÈRES (réutilisées depuis risk_analysis.py)
# ──────────────────────────────────────────────
def sharpe(r):
    return (r.mean() * JOURS_BOURSE - TAUX_SANS_RISQUE) / (r.std() * np.sqrt(JOURS_BOURSE))

def sortino(r):
    baisses = r[r < 0]
    vol_down = baisses.std() * np.sqrt(JOURS_BOURSE)
    return (r.mean() * JOURS_BOURSE - TAUX_SANS_RISQUE) / vol_down

def max_drawdown(cum):
    dd = cum / cum.cummax() - 1
    return dd.min()

def var_hist(r, c=0.95):
    return r.quantile(1 - c)

def cvar(r, c=0.95):
    seuil = r.quantile(1 - c)
    return r[r <= seuil].mean()

def beta(r_action, r_marche):
    data = pd.concat([r_action, r_marche], axis=1).dropna()
    cov  = np.cov(data.iloc[:, 0], data.iloc[:, 1])
    return cov[0, 1] / cov[1, 1]

def alpha_ir(r_action, r_marche):
    data = pd.concat([r_action, r_marche], axis=1).dropna()
    data.columns = ["a", "m"]
    active = data["a"] - data["m"]
    te  = active.std() * np.sqrt(JOURS_BOURSE) * 100
    alp = active.mean() * JOURS_BOURSE * 100
    ir  = alp / te if te != 0 else 0
    return round(alp, 2), round(te, 2), round(ir, 3)

def perf_portefeuille(poids, rend_moy, cov):
    r = np.dot(poids, rend_moy) * JOURS_BOURSE
    v = np.sqrt(poids @ cov @ poids) * np.sqrt(JOURS_BOURSE)
    s = (r - TAUX_SANS_RISQUE) / v
    return r, v, s

def max_sharpe_poids(rend_moy, cov):
    n  = len(rend_moy)
    c  = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    b  = tuple((0.05, 0.60) for _ in range(n))
    r  = sco.minimize(lambda w: -perf_portefeuille(w, rend_moy, cov)[2],
                      np.ones(n) / n, method="SLSQP", bounds=b, constraints=c)
    return r.x

def min_var_poids(rend_moy, cov):
    n  = len(rend_moy)
    c  = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    b  = tuple((0.05, 0.60) for _ in range(n))
    r  = sco.minimize(lambda w: perf_portefeuille(w, rend_moy, cov)[1],
                      np.ones(n) / n, method="SLSQP", bounds=b, constraints=c)
    return r.x

def frontiere_efficiente(rend_moy, cov, n=80):
    pts = []
    targets = np.linspace(rend_moy.min() * JOURS_BOURSE,
                          rend_moy.max() * JOURS_BOURSE, n)
    for t in targets:
        cs = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
              {"type": "eq", "fun": lambda w, t=t: perf_portefeuille(w, rend_moy, cov)[0] - t}]
        r = sco.minimize(lambda w: perf_portefeuille(w, rend_moy, cov)[1],
                         np.ones(len(rend_moy)) / len(rend_moy),
                         method="SLSQP",
                         bounds=tuple((0.0, 1.0) for _ in range(len(rend_moy))),
                         constraints=cs)
        if r.success:
            rv, vv, sv = perf_portefeuille(r.x, rend_moy, cov)
            pts.append({"Rendement": rv * 100, "Volatilite": vv * 100, "Sharpe": sv})
    return pd.DataFrame(pts)

@st.cache_data(ttl=3600, show_spinner="Simulation Monte Carlo en cours…")
def monte_carlo(rend_moy_vals, cov_vals, noms, mise, n_sim=10_000, n_jours=252):
    poids = max_sharpe_poids(pd.Series(rend_moy_vals, index=noms),
                             pd.DataFrame(cov_vals, index=noms, columns=noms))
    L = np.linalg.cholesky(cov_vals)
    vals = []
    for _ in range(n_sim):
        Z  = np.random.standard_normal((n_jours, len(noms)))
        rd = rend_moy_vals + (L @ Z.T).T
        rp = rd @ poids
        vals.append(mise * np.prod(1 + rp))
    s    = pd.Series(vals)
    v95  = np.percentile(s, 5)
    cv95 = s[s <= v95].mean()
    return s, v95, cv95, poids


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
      <div style='font-size:2rem'>📈</div>
      <div style='font-size:1rem;font-weight:700;color:#90caf9'>Financial Analytics</div>
      <div style='font-size:0.7rem;color:#64748b;margin-top:.2rem'>CAC 40 · 2019–2024</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Vue d'ensemble", "📊 Actif par actif",
         "🏆 Portefeuille optimal", "🎲 Monte Carlo",
         "🔥 Stress Testing"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    <div style='font-size:0.65rem;color:#475569;line-height:1.8'>
    <b style='color:#64748b'>Stack technique</b><br>
    Python · NumPy · SciPy<br>
    Pandas · yfinance · Plotly<br>
    Streamlit · SLSQP optimizer<br><br>
    <b style='color:#64748b'>Auteure</b><br>
    Imene Bellaghma<br>
    Master MIAGE · Paris Cité<br>
    Valorisation des Données
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────
result = charger_donnees()
if result is None or result[0] is None:
    st.error("Erreur de connexion. Vérifiez votre accès internet.")
    st.stop()

prix_df, rendements, rendements_cac = result
actions = list(rendements.columns)
rend_moy  = rendements.mean()
cov_mat   = rendements.cov()


# ══════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════
if page == "🏠 Vue d'ensemble":
    st.markdown("## 📈 Financial Market Analytics — CAC 40")
    st.markdown(
        "<div style='color:#64748b;font-size:.85rem;margin-bottom:1.5rem'>"
        "Analyse quantitative · 2019–2024 · LVMH · TotalEnergies · BNP Paribas"
        "</div>", unsafe_allow_html=True
    )

    # KPIs résumés
    cols = st.columns(3)
    metriques = {
        "LVMH":          {"rend": 19.2, "vol": 29.0, "sharpe": 0.557, "mdd": -36.6},
        "TotalEnergies": {"rend":  6.5, "vol": 30.1, "sharpe": 0.116, "mdd": -58.3},
        "BNP Paribas":   {"rend": 12.6, "vol": 34.1, "sharpe": 0.282, "mdd": -54.5},
    }
    for i, (nom, m) in enumerate(metriques.items()):
        with cols[i]:
            couleur = COLORS[nom]
            st.markdown(f"""
            <div class='info-card'>
              <div style='color:{couleur};font-weight:700;font-size:1rem;margin-bottom:.6rem'>{nom}</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:.4rem'>
                <div><div style='font-size:.65rem;color:#64748b'>Rendement ann.</div>
                     <div style='font-size:1.1rem;color:#00e676;font-weight:600'>+{m["rend"]}%</div></div>
                <div><div style='font-size:.65rem;color:#64748b'>Volatilité ann.</div>
                     <div style='font-size:1.1rem;color:#fbbf24;font-weight:600'>{m["vol"]}%</div></div>
                <div><div style='font-size:.65rem;color:#64748b'>Ratio de Sharpe</div>
                     <div style='font-size:1.1rem;color:#e2e8f0;font-weight:600'>{m["sharpe"]}</div></div>
                <div><div style='font-size:.65rem;color:#64748b'>Max Drawdown</div>
                     <div style='font-size:1.1rem;color:#ff5252;font-weight:600'>{m["mdd"]}%</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Graphique rendements cumulés
    st.markdown("### Rendements cumulés (2019–2024)")
    cum = (1 + rendements).cumprod()
    fig = go.Figure()
    for nom in actions:
        fig.add_trace(go.Scatter(
            x=cum.index, y=(cum[nom] - 1) * 100,
            name=nom, line=dict(color=COLORS[nom], width=2),
            hovertemplate=f"<b>{nom}</b><br>%{{x|%d/%m/%Y}}<br>+%{{y:.1f}}%<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#1e3a5f", title=""),
        yaxis=dict(gridcolor="#1e3a5f", title="Rendement cumulé (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f"),
        hovermode="x unified", height=400, margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insight clé
    st.markdown("""
    <div class='info-card'>
    <b style='color:#fbbf24'>💡 Insight clé</b><br><br>
    <span style='color:#cbd5e1'>
    LVMH est le seul actif affichant un <b style='color:#00e676'>Alpha significatif (+10.4%)</b>
    et un <b style='color:#00e676'>Information Ratio > 0.5</b> — le seuil institutionnel de qualité.
    TotalEnergies génère un Alpha négatif : son seul intérêt est la
    <b style='color:#64b5f6'>diversification</b> (corrélation 0.41 avec LVMH).
    </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — ACTIF PAR ACTIF
# ══════════════════════════════════════════════
elif page == "📊 Actif par actif":
    st.markdown("## 📊 Analyse par actif")

    actif = st.selectbox("Sélectionner un actif", actions,
                         format_func=lambda x: f"{'🟢' if x=='LVMH' else '🟡' if x=='TotalEnergies' else '🔵'} {x}")
    r = rendements[actif]
    p = prix_df[actif]
    couleur = COLORS[actif]
    alp, te, ir = alpha_ir(r, rendements_cac)

    # Métriques
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rendement ann.", f"+{r.mean()*JOURS_BOURSE*100:.1f}%")
    c2.metric("Volatilité ann.", f"{r.std()*np.sqrt(JOURS_BOURSE)*100:.1f}%")
    c3.metric("Ratio de Sharpe", f"{sharpe(r):.3f}")
    cum_r = (1 + r).cumprod()
    c4.metric("Max Drawdown", f"{max_drawdown(cum_r)*100:.1f}%")
    c5.metric("Information Ratio", f"{ir:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        # Prix + moyennes mobiles
        st.markdown("##### Prix & Moyennes Mobiles")
        ma20  = p.rolling(20).mean()
        ma50  = p.rolling(50).mean()
        ma200 = p.rolling(200).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p.index, y=p, name="Prix",
                                  line=dict(color=couleur, width=1.5)))
        fig.add_trace(go.Scatter(x=ma20.index, y=ma20, name="MA 20",
                                  line=dict(color="#64b5f6", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA 50",
                                  line=dict(color="#fbbf24", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA 200",
                                  line=dict(color="#ff5252", width=1, dash="dash")))
        fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
                          font=dict(color="#94a3b8", size=10),
                          xaxis=dict(gridcolor="#1e3a5f"),
                          yaxis=dict(gridcolor="#1e3a5f", title="Prix (€)"),
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution des rendements
        st.markdown("##### Distribution des rendements journaliers")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=r * 100, nbinsx=60,
            marker_color=couleur, opacity=0.7, name="Rendements"
        ))
        # Courbe normale théorique
        x_range = np.linspace(r.min() * 100, r.max() * 100, 200)
        mu_, sigma_ = r.mean() * 100, r.std() * 100
        y_norm = stats.norm.pdf(x_range, mu_, sigma_) * len(r) * (r.max() - r.min()) * 100 / 60
        fig.add_trace(go.Scatter(x=x_range, y=y_norm, name="Loi normale",
                                  line=dict(color="#ff5252", width=2)))
        # VaR
        var95_ = var_hist(r) * 100
        fig.add_vline(x=var95_, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"VaR 95%: {var95_:.2f}%",
                      annotation_font_color="#fbbf24")
        fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
                          font=dict(color="#94a3b8", size=10),
                          xaxis=dict(gridcolor="#1e3a5f", title="Rendement (%)"),
                          yaxis=dict(gridcolor="#1e3a5f"),
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    st.markdown("##### Drawdowns")
    dd_series = cum_r / cum_r.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series * 100,
        fill="tozeroy", fillcolor=f"rgba(255,82,82,0.15)",
        line=dict(color="#ff5252", width=1.2), name="Drawdown"
    ))
    fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
                      font=dict(color="#94a3b8", size=10),
                      xaxis=dict(gridcolor="#1e3a5f"),
                      yaxis=dict(gridcolor="#1e3a5f", title="Drawdown (%)"),
                      height=260, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Tableau métriques
    st.markdown("##### Métriques complètes")
    beta_ = beta(r, rendements_cac)
    df_m = pd.DataFrame({
        "Métrique": ["Rendement annualisé", "Volatilité annualisée", "Ratio de Sharpe",
                     "Ratio de Sortino", "Max Drawdown", "VaR 95% (historique)",
                     "CVaR 95%", "Bêta vs CAC 40", "Alpha vs CAC 40",
                     "Tracking Error", "Information Ratio"],
        "Valeur": [
            f"+{r.mean()*JOURS_BOURSE*100:.2f}%",
            f"{r.std()*np.sqrt(JOURS_BOURSE)*100:.2f}%",
            f"{sharpe(r):.3f}",
            f"{sortino(r):.3f}",
            f"{max_drawdown(cum_r)*100:.2f}%",
            f"{var_hist(r)*100:.3f}%",
            f"{cvar(r)*100:.3f}%",
            f"{beta_:.3f}",
            f"{alp:+.2f}%",
            f"{te:.2f}%",
            f"{ir:.3f}",
        ]
    })
    st.dataframe(df_m, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PAGE 3 — PORTEFEUILLE OPTIMAL
# ══════════════════════════════════════════════
elif page == "🏆 Portefeuille optimal":
    st.markdown("## 🏆 Optimisation de Markowitz")
    st.markdown("<div style='color:#64748b;font-size:.85rem;margin-bottom:1rem'>Frontière efficiente · Max Sharpe · Min Variance · Risk Budgeting</div>", unsafe_allow_html=True)

    with st.spinner("Calcul de la frontière efficiente…"):
        poids_ms = max_sharpe_poids(rend_moy, cov_mat.values)
        poids_mv = min_var_poids(rend_moy, cov_mat.values)
        perf_ms  = perf_portefeuille(poids_ms, rend_moy.values, cov_mat.values)
        perf_mv  = perf_portefeuille(poids_mv, rend_moy.values, cov_mat.values)
        front    = frontiere_efficiente(rend_moy.values, cov_mat.values)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### Frontière Efficiente")
        fig = go.Figure()

        # Frontière
        if not front.empty:
            fig.add_trace(go.Scatter(
                x=front["Volatilite"], y=front["Rendement"],
                mode="lines",
                line=dict(color="#64b5f6", width=2),
                name="Frontière efficiente",
                hovertemplate="Vol: %{x:.1f}%<br>Rend: %{y:.1f}%<extra></extra>"
            ))

        # Actifs individuels
        for nom in actions:
            r_ind = rendements[nom]
            v_ind = r_ind.std() * np.sqrt(JOURS_BOURSE) * 100
            r_a   = r_ind.mean() * JOURS_BOURSE * 100
            fig.add_trace(go.Scatter(
                x=[v_ind], y=[r_a], mode="markers+text",
                name=nom,
                marker=dict(color=COLORS[nom], size=12, symbol="circle"),
                text=[nom], textposition="top center",
                textfont=dict(color=COLORS[nom], size=10),
            ))

        # Max Sharpe
        fig.add_trace(go.Scatter(
            x=[perf_ms[1] * 100], y=[perf_ms[0] * 100],
            mode="markers+text", name="Max Sharpe ⭐",
            marker=dict(color="#fbbf24", size=16, symbol="star"),
            text=["Max Sharpe"], textposition="top right",
            textfont=dict(color="#fbbf24", size=11),
        ))

        # Min Variance
        fig.add_trace(go.Scatter(
            x=[perf_mv[1] * 100], y=[perf_mv[0] * 100],
            mode="markers+text", name="Min Variance",
            marker=dict(color="#00e676", size=12, symbol="diamond"),
            text=["Min Variance"], textposition="bottom right",
            textfont=dict(color="#00e676", size=10),
        ))

        fig.update_layout(
            paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#1e3a5f", title="Volatilité annualisée (%)"),
            yaxis=dict(gridcolor="#1e3a5f", title="Rendement annualisé (%)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f"),
            height=430, margin=dict(l=20,r=20,t=20,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Portefeuille Max Sharpe ⭐")
        # Donut
        fig_pie = go.Figure(go.Pie(
            labels=actions, values=[p * 100 for p in poids_ms],
            hole=0.55,
            marker=dict(colors=[COLORS[a] for a in actions],
                        line=dict(color="#0a0f1e", width=2)),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0", size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0a0f1e", font=dict(color="#94a3b8"),
            showlegend=False, height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(text=f"Sharpe<br>{perf_ms[2]:.3f}",
                              x=0.5, y=0.5, font=dict(size=14, color="#fbbf24"),
                              showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Métriques
        st.metric("Rendement", f"{perf_ms[0]*100:.1f}%")
        st.metric("Volatilité", f"{perf_ms[1]*100:.1f}%")
        st.metric("Sharpe", f"{perf_ms[2]:.3f}")

    # Risk Budgeting
    st.divider()
    st.markdown("##### Risk Budgeting — Contribution au risque")
    sigma_total = np.sqrt(poids_ms @ cov_mat.values @ poids_ms) * np.sqrt(JOURS_BOURSE)
    risque_marg = (cov_mat.values @ poids_ms) * np.sqrt(JOURS_BOURSE) / sigma_total
    contrib_abs = poids_ms * risque_marg
    contrib_rel = contrib_abs / sigma_total * 100

    rb_df = pd.DataFrame({
        "Action":             actions,
        "Poids (%)" :         [f"{p*100:.1f}%" for p in poids_ms],
        "Contribution Risque (%)": [f"{c:.1f}%" for c in contrib_rel],
        "Risque Marginal":    [f"{r:.4f}" for r in risque_marg],
    })

    cols_rb = st.columns(3)
    for i, (action, poids_v, contrib_v) in enumerate(zip(actions, poids_ms, contrib_rel)):
        with cols_rb[i]:
            diff = contrib_v - poids_v * 100
            st.markdown(f"""
            <div class='info-card'>
              <div style='color:{COLORS[action]};font-weight:700'>{action}</div>
              <div style='margin-top:.5rem'>
                <div style='font-size:.7rem;color:#64748b'>Poids capital</div>
                <div style='font-size:1.3rem;font-weight:700'>{poids_v*100:.1f}%</div>
              </div>
              <div style='margin-top:.4rem'>
                <div style='font-size:.7rem;color:#64748b'>Contribution au risque</div>
                <div style='font-size:1.3rem;font-weight:700;color:{"#ff5252" if diff > 5 else "#00e676"}'>{contrib_v:.1f}%</div>
              </div>
              <div style='font-size:.68rem;color:#64748b;margin-top:.3rem'>
                Écart poids/risque : {diff:+.1f}pp
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Corrélations
    st.divider()
    st.markdown("##### Matrice de corrélation")
    corr = rendements.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=actions, y=actions,
        colorscale=[[0, "#0a1628"], [0.5, "#1e3a5f"], [1, "#00e676"]],
        text=[[f"{corr.iloc[i,j]:.3f}" for j in range(3)] for i in range(3)],
        texttemplate="%{text}",
        textfont=dict(size=14, color="#e2e8f0"),
        showscale=True,
        zmid=0,
    ))
    fig_corr.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(color="#94a3b8"), height=280,
        margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 4 — MONTE CARLO
# ══════════════════════════════════════════════
elif page == "🎲 Monte Carlo":
    st.markdown("## 🎲 Simulation Monte Carlo")
    st.markdown("<div style='color:#64748b;font-size:.85rem;margin-bottom:1rem'>10 000 scénarios · Horizon 1 an · Décomposition de Cholesky</div>", unsafe_allow_html=True)

    # Slider
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        mise = st.slider(
            "💰 Mise de départ (€)",
            min_value=10_000, max_value=1_000_000,
            value=100_000, step=10_000,
            format="%d €"
        )
    with col_s2:
        n_sim = st.select_slider("Simulations", [1000, 5000, 10000], value=10_000)

    with st.spinner("Simulation en cours…"):
        sim_vals, var95_, cvar95_, poids_opt = monte_carlo(
            rend_moy.values, cov_mat.values, actions, mise, n_sim
        )

    prob_perte = (sim_vals < mise).mean() * 100
    gain_moy   = sim_vals.mean() - mise

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mise initiale",     f"{mise:,.0f} €")
    c2.metric("Valeur moy. finale", f"{sim_vals.mean():,.0f} €",
              delta=f"+{gain_moy:,.0f} €")
    c3.metric("Médiane",           f"{sim_vals.median():,.0f} €")
    c4.metric("VaR 95%",           f"{var95_:,.0f} €",
              delta=f"{var95_ - mise:,.0f} €", delta_color="inverse")
    c5.metric("CVaR 95%",          f"{cvar95_:,.0f} €",
              delta=f"{cvar95_ - mise:,.0f} €", delta_color="inverse")

    st.divider()
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### Distribution des valeurs finales")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=sim_vals, nbinsx=80,
            marker=dict(color="#64b5f6", opacity=0.7),
            name="Scénarios"
        ))
        fig.add_vline(x=mise,     line_dash="dash", line_color="#e2e8f0",
                      annotation_text="Mise initiale", annotation_font_color="#e2e8f0")
        fig.add_vline(x=var95_,   line_dash="dot", line_color="#fbbf24",
                      annotation_text=f"VaR 95%", annotation_font_color="#fbbf24")
        fig.add_vline(x=cvar95_,  line_dash="dot", line_color="#ff5252",
                      annotation_text=f"CVaR 95%", annotation_font_color="#ff5252")
        fig.add_vline(x=sim_vals.mean(), line_dash="dash", line_color="#00e676",
                      annotation_text="Moyenne", annotation_font_color="#00e676")
        fig.update_layout(
            paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#1e3a5f", title="Valeur finale (€)"),
            yaxis=dict(gridcolor="#1e3a5f", title="Fréquence"),
            height=380, margin=dict(l=20,r=20,t=20,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Résumé de risque")
        perte_max = mise - sim_vals.min()
        gain_max  = sim_vals.max() - mise
        p10 = np.percentile(sim_vals, 10)
        p90 = np.percentile(sim_vals, 90)

        items = [
            ("Probabilité de perte", f"{prob_perte:.1f}%",
             "🔴" if prob_perte > 35 else "🟡" if prob_perte > 25 else "🟢"),
            ("Gain espéré", f"+{gain_moy:,.0f} €", "🟢"),
            ("Perte max simulée", f"-{perte_max:,.0f} €", "🔴"),
            ("Gain max simulé",   f"+{gain_max:,.0f} €",  "🟢"),
            ("Percentile 10%",    f"{p10:,.0f} €", "🟡"),
            ("Percentile 90%",    f"{p90:,.0f} €", "🟡"),
        ]
        for label, valeur, icone in items:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:.45rem .7rem;border-bottom:1px solid #1e3a5f;'>
              <span style='font-size:.78rem;color:#94a3b8'>{icone} {label}</span>
              <span style='font-size:.82rem;font-weight:700;color:#e2e8f0'>{valeur}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-card' style='margin-top:.8rem'>
          <div style='font-size:.7rem;color:#64748b;margin-bottom:.4rem'>
            INTERPRÉTATION BÂLE III
          </div>
          <div style='font-size:.78rem;color:#cbd5e1'>
            Sur {n_sim:,} scénarios, votre portefeuille perd en moyenne
            <b style='color:#ff5252'>{(mise - cvar95_):,.0f} €</b>
            dans les 5% de pires cas (CVaR).
            Ce montant représente votre exposition au risque de queue.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Allocation optimale utilisée
    st.divider()
    st.markdown("##### Allocation Max Sharpe utilisée pour la simulation")
    fig_bar = go.Figure(go.Bar(
        x=actions, y=[p * 100 for p in poids_opt],
        marker_color=[COLORS[a] for a in actions],
        text=[f"{p*100:.1f}%" for p in poids_opt],
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(color="#94a3b8"), yaxis=dict(title="Poids (%)", gridcolor="#1e3a5f"),
        height=240, margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 5 — STRESS TESTING
# ══════════════════════════════════════════════
elif page == "🔥 Stress Testing":
    st.markdown("## 🔥 Stress Testing — Crises historiques")
    st.markdown("<div style='color:#64748b;font-size:.85rem;margin-bottom:1rem'>Portefeuille Max Sharpe testé sur les 4 principales crises 2019–2024</div>", unsafe_allow_html=True)

    poids_ms = max_sharpe_poids(rend_moy, cov_mat.values)

    resultats_stress = []
    for nom_sc, (debut, fin) in SCENARIOS_STRESS.items():
        periode = rendements.loc[debut:fin]
        if len(periode) == 0:
            continue
        perf = ((1 + periode @ poids_ms).prod() - 1) * 100
        resultats_stress.append({
            "Scénario": nom_sc,
            "Performance (%)": round(perf, 2),
            "Durée (jours)": len(periode),
            "Début": debut,
            "Fin": fin,
        })
    df_stress = pd.DataFrame(resultats_stress)

    # Graphique barres
    couleurs_stress = ["#ff5252" if v < 0 else "#00e676"
                       for v in df_stress["Performance (%)"].tolist()]
    fig = go.Figure(go.Bar(
        x=df_stress["Scénario"],
        y=df_stress["Performance (%)"],
        marker_color=couleurs_stress,
        text=[f"{v:+.1f}%" for v in df_stress["Performance (%)"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.add_hline(y=0, line_color="#475569", line_width=1)
    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", title="Performance (%)"),
        height=380, margin=dict(l=20,r=40,t=20,b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cartes détail
    st.divider()
    cols_s = st.columns(2)
    interpretations = {
        "Krach COVID (Fév–Mar 2020)":    "Choc de liquidité mondial. Vente forcée de tous les actifs risqués.",
        "Rebond COVID (Mar–Août 2020)":  "Injection massive de liquidités (Fed, BCE). Rebond historiquement rapide.",
        "Crise taux / inflation (2022)": "Remontée des taux directeurs. Réévaluation de tous les actifs à duration longue.",
        "Crise bancaire SVB (Mar 2023)": "Risque systémique bancaire. Impact ciblé sur BNP Paribas.",
    }
    for i, row in df_stress.iterrows():
        col_idx = i % 2
        perf_v  = row["Performance (%)"]
        couleur = "#ff5252" if perf_v < 0 else "#00e676"
        with cols_s[col_idx]:
            st.markdown(f"""
            <div class='info-card'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div style='font-size:.82rem;font-weight:700;color:#e2e8f0'>{row["Scénario"]}</div>
                <div style='font-size:1.4rem;font-weight:800;color:{couleur}'>{perf_v:+.1f}%</div>
              </div>
              <div style='font-size:.72rem;color:#64748b;margin:.4rem 0'>
                {row["Début"]} → {row["Fin"]} · {row["Durée (jours)"]} jours de bourse
              </div>
              <div style='font-size:.75rem;color:#94a3b8'>
                {interpretations.get(row["Scénario"], "")}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Performance par actif sur chaque crise
    st.divider()
    st.markdown("##### Performance par actif sur chaque crise")
    perf_detail = []
    for nom_sc, (debut, fin) in SCENARIOS_STRESS.items():
        periode = rendements.loc[debut:fin]
        if len(periode) == 0:
            continue
        for action in actions:
            perf = ((1 + periode[action]).prod() - 1) * 100
            perf_detail.append({"Scénario": nom_sc.split("(")[0].strip(),
                                 "Action": action, "Performance (%)": round(perf, 1)})
    df_det = pd.DataFrame(perf_detail)

    fig2 = go.Figure()
    for action in actions:
        sub = df_det[df_det["Action"] == action]
        fig2.add_trace(go.Bar(
            x=sub["Scénario"], y=sub["Performance (%)"],
            name=action, marker_color=COLORS[action],
            text=[f"{v:+.1f}%" for v in sub["Performance (%)"]],
            textposition="outside",
        ))
    fig2.add_hline(y=0, line_color="#475569", line_width=1)
    fig2.update_layout(
        barmode="group",
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1428",
        font=dict(color="#94a3b8", size=10),
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", title="Performance (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=360, margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
