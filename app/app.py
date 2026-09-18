import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from textwrap import dedent
import numpy as np
import plotly.express as px


# ============================================================
# CONFIGURATION DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import predict_fraud, load_model
from src.explainability import explain_prediction


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="AI Insurance Risk",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background-color: #f5f7fb;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background-color: transparent;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #d1d5db !important;
}

.sidebar-logo {
    width: 58px;
    height: 58px;
    border-radius: 16px;

    background: linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 30px;

    margin-bottom: 16px;

    box-shadow:
        0 10px 25px rgba(37, 99, 235, 0.25);
}

.sidebar-title {
    color: #ffffff !important;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 3px;
}

.sidebar-subtitle {
    color: #9ca3af !important;
    font-size: 12px;
}

.sidebar-section {
    color: #f9fafb;
    font-size: 14px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 10px;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-box {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #172554 50%,
        #0f766e 100%
    );

    border-radius: 24px;

    padding: 38px 42px;

    margin-bottom: 32px;

    box-shadow:
        0 18px 45px rgba(15, 23, 42, 0.16);

    position: relative;

    overflow: hidden;
}

.hero-box::after {
    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    border-radius: 50%;

    right: -90px;
    top: -110px;

    background: rgba(255,255,255,0.06);
}

.hero-icon {
    font-size: 44px;
    margin-bottom: 12px;
}

.hero-title {
    color: #ffffff !important;

    font-size: 40px;

    font-weight: 800;

    letter-spacing: -1px;

    line-height: 1.15;

    margin-bottom: 10px;
}

.hero-description {
    color: #dbeafe !important;

    font-size: 15px;

    line-height: 1.7;

    max-width: 850px;
}

.status-badge {
    display: inline-block;

    margin-top: 20px;

    padding: 8px 15px;

    border-radius: 999px;

    background-color: rgba(16, 185, 129, 0.14);

    border: 1px solid rgba(110, 231, 183, 0.35);

    color: #6ee7b7 !important;

    font-size: 13px;

    font-weight: 700;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    color: #172033;

    font-size: 24px;

    font-weight: 800;

    margin-top: 25px;

    margin-bottom: 5px;
}

.section-description {
    color: #667085;

    font-size: 14px;

    line-height: 1.6;

    margin-bottom: 20px;
}


/* ============================================================
   KPI CARDS
   ============================================================ */

.kpi-card {
    background-color: #ffffff;

    border: 1px solid #e5e7eb;

    border-radius: 18px;

    padding: 22px;

    min-height: 130px;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.05);

    transition: transform 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
}

.kpi-icon {
    font-size: 23px;

    margin-bottom: 10px;
}

.kpi-label {
    color: #667085;

    font-size: 13px;

    font-weight: 500;
}

.kpi-value {
    color: #111827;

    font-size: 30px;

    font-weight: 800;

    margin-top: 5px;
}


/* ============================================================
   RISK CARDS
   ============================================================ */

.risk-card {
    background-color: #ffffff;

    border: 1px solid #e5e7eb;

    border-radius: 18px;

    padding: 22px;

    text-align: center;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.04);
}

.risk-number {
    color: #111827;

    font-size: 32px;

    font-weight: 800;
}

.risk-label {
    color: #667085;

    font-size: 13px;

    margin-top: 6px;
}


/* ============================================================
   INFO CARD
   ============================================================ */

.info-box {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #ecfeff
    );

    border: 1px solid #bae6fd;

    border-radius: 18px;

    padding: 24px;

    margin-top: 25px;

    box-shadow:
        0 5px 18px rgba(14, 116, 144, 0.04);
}

.info-title {
    color: #164e63;

    font-size: 18px;

    font-weight: 750;
}

.info-text {
    color: #475569;

    font-size: 14px;

    line-height: 1.7;

    margin-top: 8px;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background-color: #ffffff;

    border: 1px solid #e5e7eb;

    border-radius: 16px;

    padding: 8px;

    box-shadow:
        0 5px 18px rgba(15, 23, 42, 0.04);
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 11px;

    min-height: 45px;

    font-weight: 700;

    border: none;
}

.stDownloadButton > button {
    border-radius: 11px;

    min-height: 45px;

    font-weight: 700;
}


/* ============================================================
   DATAFRAME
   ============================================================ */

[data-testid="stDataFrame"] {
    border-radius: 14px;

    overflow: hidden;

    border: 1px solid #e5e7eb;
}


/* ============================================================
   METRICS
   ============================================================ */

[data-testid="stMetric"] {
    background-color: #ffffff;

    border: 1px solid #e5e7eb;

    border-radius: 14px;

    padding: 15px;
}


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {
    border-radius: 14px;

    border: 1px solid #e5e7eb;

    background-color: #ffffff;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    color: #98a2b3;

    font-size: 12px;

    margin-top: 50px;

    padding-top: 22px;

    border-top: 1px solid #e5e7eb;
}

/* ============================================================
   PROBABILITY CARDS
   ============================================================ */

.probability-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px 18px;
    min-height: 105px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.probability-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}

.probability-label {
    color: #475569;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 6px;
}

.probability-value {
    color: #111827;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.1;
}

.probability-description {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 6px;
    line-height: 1.4;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">🛡️</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-title">AI Insurance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Risk Intelligence Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section">⚙️ Configuration</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Plateforme intelligente dédiée à "
        "l'analyse des risques de fraude "
        "en assurance."
    )

    st.markdown(
        '<div class="sidebar-section">🤖 Modèle</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
**Random Forest**

• Classification binaire

• Probabilité de fraude

• Score de risque

• Décision prédictive
    """)

    st.markdown(
        '<div class="sidebar-section">📌 Fonctionnalités</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
✓ Analyse des dossiers

✓ Détection de fraude

✓ Score de risque

✓ Analyse statistique

✓ Priorisation des dossiers
    """)

    st.divider()

    st.caption("AI Insurance Risk Prediction")
    st.caption("Machine Learning • XAI • Insurance")


# ============================================================
# HERO
# ============================================================

st.markdown(
    dedent("""
    <div class="hero-box">
        <div class="hero-icon">🛡️</div>
        <div class="hero-title">AI Insurance Risk</div>
        <div class="hero-description">
            Plateforme intelligente d'analyse des dossiers
            d'assurance et de détection des risques potentiels
            de fraude grâce au Machine Learning.
        </div>
        <div class="status-badge">● Modèle opérationnel</div>
    </div>
    """),
    unsafe_allow_html=True
)


# ============================================================
# ANALYSE DES DOSSIERS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Analyse des dossiers'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    "Importez un fichier CSV contenant les dossiers "
    "d'assurance pour lancer automatiquement "
    "l'analyse du risque."
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Importer un fichier CSV",
    type=["csv"],
    help="Format accepté : CSV"
)


# ============================================================
# TRAITEMENT DU FICHIER
# ============================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(
            f"❌ Impossible de lire le fichier CSV : {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    st.success(
        f"✓ Fichier chargé avec succès — "
        f"**{len(df):,} dossiers**"
    )


    # --------------------------------------------------------
    # APERÇU
    # --------------------------------------------------------

    with st.expander(
        "👁️ Aperçu des données",
        expanded=False
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Dossiers",
                f"{len(df):,}"
            )

        with c2:

            st.metric(
                "Variables",
                f"{len(df.columns):,}"
            )

        with c3:

            st.metric(
                "Taille",
                f"{uploaded_file.size / 1024:.1f} KB"
            )


    st.markdown("")


    # --------------------------------------------------------
    # BOUTON ANALYSE
    # --------------------------------------------------------

    if st.button(
        "🔍  Lancer l'analyse de risque",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🤖 Analyse intelligente des dossiers en cours..."
        ):

            try:

                result = predict_fraud(df)

                st.session_state["analysis_result"] = result

                st.success(
                    "✓ Analyse terminée avec succès !"
                )

            except Exception as e:

                st.error(
                    f"❌ Erreur pendant l'analyse : {e}"
                )

                st.stop()


# ============================================================
# RÉSULTATS
# ============================================================

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]


    # ========================================================
    # KPI
    # ========================================================

    total = len(result)

    fraud_count = int(
        (result["prediction"] == 1).sum()
    )

    high_risk_count = int(
        (result["risk_level"] == "Élevé").sum()
    )

    moderate_risk_count = int(
        (result["risk_level"] == "Modéré").sum()
    )

    low_risk_count = int(
        (result["risk_level"] == "Faible").sum()
    )

    fraud_rate = (
        fraud_count / total * 100
        if total > 0
        else 0
    )

    avg_probability = (
        result["fraud_probability"].mean() * 100
        if total > 0
        else 0
    )


    # ========================================================
    # VUE D'ENSEMBLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Vue d’ensemble'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-icon">📁</div>'
            '<div class="kpi-label">'
            'Dossiers analysés'
            '</div>'
            f'<div class="kpi-value">{total:,}</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-icon">🚨</div>'
            '<div class="kpi-label">'
            'Risque élevé'
            '</div>'
            f'<div class="kpi-value">{high_risk_count:,}</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-icon">⚠️</div>'
            '<div class="kpi-label">'
            'Fraude prédite'
            '</div>'
            f'<div class="kpi-value">{fraud_rate:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-icon">🎯</div>'
            '<div class="kpi-label">'
            'Probabilité moyenne'
            '</div>'
            f'<div class="kpi-value">{avg_probability:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # RÉPARTITION DU RISQUE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Répartition du risque'
        '</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)


    with r1:

        st.markdown(
            '<div class="risk-card">'
            f'<div class="risk-number">'
            f'{low_risk_count:,}'
            '</div>'
            '<div class="risk-label">'
            '🟢 Risque faible'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with r2:

        st.markdown(
            '<div class="risk-card">'
            f'<div class="risk-number">'
            f'{moderate_risk_count:,}'
            '</div>'
            '<div class="risk-label">'
            '🟠 Risque modéré'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with r3:

        st.markdown(
            '<div class="risk-card">'
            f'<div class="risk-number">'
            f'{high_risk_count:,}'
            '</div>'
            '<div class="risk-label">'
            '🔴 Risque élevé'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # ANALYSE VISUELLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Analyse visuelle'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        "Les graphiques ci-dessous permettent d'identifier "
        "rapidement la répartition du risque et les dossiers "
        "nécessitant une attention particulière."
        '</div>',
        unsafe_allow_html=True
    )

    probabilities = result["fraud_probability"] * 100

    chart1, chart2 = st.columns(2)


    # ========================================================
    # GRAPHIQUE 1 — RÉPARTITION DU RISQUE
    # ========================================================

    with chart1:

        st.markdown(
            "#### 🎯 Répartition des niveaux de risque"
        )

        # Calcul des pourcentages
        risk_counts = pd.Series({
            "Risque faible": low_risk_count,
            "Risque modéré": moderate_risk_count,
            "Risque élevé": high_risk_count
        })

        risk_percentages = (
            risk_counts / total * 100
            if total > 0
            else risk_counts * 0
        )

        risk_data = pd.DataFrame({
            "Nombre de dossiers": risk_counts.astype(int),
            "Pourcentage": risk_percentages.round(1)
        })

        # Graphique
        risk_chart_df = pd.DataFrame({
            "Niveau de risque": [
                "Risque faible",
                "Risque modéré",
                "Risque élevé"
            ],
            "Nombre de dossiers": [
                low_risk_count,
                moderate_risk_count,
                high_risk_count
            ]
        })

        fig_risk = px.bar(
            risk_chart_df,
            x="Niveau de risque",
            y="Nombre de dossiers",
            color="Niveau de risque",
            color_discrete_map={
                "Risque faible": "#22C55E",
                "Risque modéré": "#F59E0B",
                "Risque élevé": "#EF4444"
            },
            text="Nombre de dossiers"
        )

        fig_risk.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Dossiers : %{y}<extra></extra>"
        )

        fig_risk.update_layout(
            showlegend=False,
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title=None,
            yaxis_title="Nombre de dossiers",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )


        # ========================================================
        # DISTRIBUTION DES PROBABILITÉS DE FRAUDE
        # ========================================================

        st.markdown("")

        st.markdown(
            "#### 📊 Distribution des probabilités de fraude"
        )

        st.caption(
            "Répartition des dossiers selon leur probabilité estimée de fraude."
        )

        # --------------------------------------------------------
        # PRÉPARATION DES PROBABILITÉS
        # --------------------------------------------------------

        probabilities = result["fraud_probability"] * 100

        # 5 catégories simples et faciles à interpréter
        bins = [
            0,
            20,
            40,
            60,
            80,
            100.0001
        ]

        labels = [
            "0–20%",
            "20–40%",
            "40–60%",
            "60–80%",
            "80–100%"
        ]

        distribution = pd.cut(
            probabilities,
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        )

        distribution_counts = (
            distribution
            .value_counts()
            .sort_index()
        )

        distribution_df = pd.DataFrame({
            "Probabilité": distribution_counts.index.astype(str),
            "Nombre de dossiers": distribution_counts.values
        })

        # --------------------------------------------------------
        # GRAPHIQUE
        # --------------------------------------------------------

        fig_distribution = px.bar(
            distribution_df,
            x="Probabilité",
            y="Nombre de dossiers",
            color="Probabilité",
            color_discrete_map={
                "0–20%": "#22C55E",
                "20–40%": "#EAB308",
                "40–60%": "#F97316",
                "60–80%": "#F43F5E",
                "80–100%": "#DC2626"
            },
            text="Nombre de dossiers"
        )

        fig_distribution.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b> %{x}</b><br>"
                "Dossiers : %{y}"
                "<extra></extra>"
            )
        )

        fig_distribution.update_layout(
            showlegend=False,
            height=330,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title=None,
            yaxis_title="Nombre de dossiers",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )

        # Résumé sous le graphique
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #e5e7eb;
                border-radius:14px;
                padding:14px 18px;
                margin-top:10px;
            ">

            <div style="
                display:flex;
                justify-content:space-between;
                margin-bottom:8px;
            ">
                <span>🟢 Risque faible</span>
                <strong>{low_risk_count:,} ({risk_percentages['Risque faible']:.1f}%)</strong>
            </div>

            <div style="
                display:flex;
                justify-content:space-between;
                margin-bottom:8px;
            ">
                <span>🟠 Risque modéré</span>
                <strong>{moderate_risk_count:,} ({risk_percentages['Risque modéré']:.1f}%)</strong>
            </div>

            <div style="
                display:flex;
                justify-content:space-between;
            ">
                <span>🔴 Risque élevé</span>
                <strong>{high_risk_count:,} ({risk_percentages['Risque élevé']:.1f}%)</strong>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )        



    # ========================================================
    # GRAPHIQUE 2 — DOSSIERS À EXAMINER
    # ========================================================

    with chart2:

        # --------------------------------------------------------
        # SÉLECTION DU NOMBRE DE DOSSIERS
        # --------------------------------------------------------

        st.markdown(
            "#### 🚨 Dossiers à examiner"
        )

        selection = st.selectbox(
            "Afficher",
            options=["Tous les dossiers", "Top 5", "Top 10", "Top 15"],
            index=0,
            key="risk_top_selection"
        )


        # --------------------------------------------------------
        # PRÉPARATION DES DONNÉES
        # --------------------------------------------------------

        top_risk = result.copy()

        # Identifiant lisible du dossier
        top_risk["original_index"] = top_risk.index
        top_risk["Dossier"] = [
            f"Dossier {i + 1}"
            for i in range(len(top_risk))
        ]

        # Probabilité en pourcentage
        top_risk["Probabilité fraude (%)"] = (
            top_risk["fraud_probability"] * 100
        )

        # Tri décroissant
        top_risk = top_risk.sort_values(
            "Probabilité fraude (%)",
            ascending=False
        )


        # --------------------------------------------------------
        # APPLICATION DU FILTRE
        # --------------------------------------------------------

        if selection == "Top 5":

            top_risk = top_risk.head(5)
            chart_title = "🚨 Top 5 des dossiers à examiner"

        elif selection == "Top 10":

            top_risk = top_risk.head(10)
            chart_title = "🚨 Top 10 des dossiers à examiner"

        elif selection == "Top 15":

            top_risk = top_risk.head(15)
            chart_title = "🚨 Top 15 des dossiers à examiner"

        else:

            chart_title = "📊 Probabilité de fraude — Tous les dossiers"


        # --------------------------------------------------------
        # ARRONDIR
        # --------------------------------------------------------

        top_risk["Probabilité fraude (%)"] = (
            top_risk["Probabilité fraude (%)"]
            .round(1)
        )


        # --------------------------------------------------------
        # TITRE DYNAMIQUE
        # --------------------------------------------------------

        st.markdown(
            f"**{chart_title}**"
        )


        # --------------------------------------------------------
        # DONNÉES DU GRAPHIQUE
        # --------------------------------------------------------

        top_chart = (
            top_risk[
                ["Dossier", "Probabilité fraude (%)"]
            ]
            .set_index("Dossier")
        )


        # --------------------------------------------------------
        # AFFICHAGE DU GRAPHIQUE
        # --------------------------------------------------------

        if selection == "Tous les dossiers":

            # ========================================================
            # TOUS LES DOSSIERS → COURBE
            # ========================================================

            fig_all = px.line(
                top_risk,
                x="Dossier",
                y="Probabilité fraude (%)",
                markers=True
            )

            fig_all.update_traces(
                line=dict(
                    color="#6366F1",
                    width=2
                ),
                marker=dict(
                    size=5,
                    color="#6366F1"
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Probabilité : %{y:.1f}%"
                    "<extra></extra>"
                )
            )

            fig_all.update_layout(
                height=380,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title=None,
                yaxis_title="Probabilité de fraude (%)",
                yaxis=dict(
                    range=[0, 105]
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False
            )

            st.plotly_chart(
                fig_all,
                use_container_width=True
            )


        else:

            # ========================================================
            # TOP 5 / TOP 10 / TOP 15 → BARRES
            # ========================================================

            fig_top = px.bar(
                top_risk,
                x="Dossier",
                y="Probabilité fraude (%)",
                color="Probabilité fraude (%)",
                color_continuous_scale=[
                    "#FDE68A",
                    "#F97316",
                    "#EF4444",
                    "#B91C1C"
                ],
                text="Probabilité fraude (%)"
            )

            fig_top.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Probabilité de fraude : %{y:.1f}%"
                    "<extra></extra>"
                )
            )

            fig_top.update_layout(
                height=380,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title=None,
                yaxis_title="Probabilité de fraude (%)",
                yaxis=dict(
                    range=[0, 105]
                ),
                coloraxis_showscale=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False
            )

            st.plotly_chart(
                fig_top,
                use_container_width=True
            )



        # --------------------------------------------------------
        # INFORMATIONS SOUS LE GRAPHIQUE
        # --------------------------------------------------------

        if len(top_risk) > 0:

            max_probability = top_risk[
                "Probabilité fraude (%)"
            ].max()

            min_probability = top_risk[
                "Probabilité fraude (%)"
            ].min()

        else:

            max_probability = 0
            min_probability = 0


        if selection == "Tous les dossiers":

            description = (
                f"Les **{len(top_risk):,} dossiers** sont affichés "
                "selon leur probabilité de fraude."
            )

        else:

            description = (
                f"Les **{len(top_risk)} dossiers** présentant "
                "les probabilités de fraude les plus élevées "
                "sont affichés en priorité."
            )


        # --------------------------------------------------------
        # INFORMATIONS SOUS LE GRAPHIQUE
        # --------------------------------------------------------

        st.markdown("")

        if selection == "Tous les dossiers":

            st.info(
                f"📊 **Analyse globale**\n\n"
                f"Les **{len(top_risk):,} dossiers** sont affichés "
                f"selon leur probabilité de fraude."
            )

        else:

            st.warning(
                f"🚨 **Priorité d'analyse**\n\n"
                f"Les **{len(top_risk)} dossiers** présentant "
                f"les probabilités de fraude les plus élevées "
                f"sont affichés en priorité."
            )


        # --------------------------------------------------------
        # INDICATEURS DU GRAPHIQUE
        # --------------------------------------------------------

        info_col1, info_col2 = st.columns(2, gap="small")

        with info_col1:

            st.markdown(
                f"""
                <div class="probability-card">
                    <div class="probability-label">
                        🔴 Probabilité la plus élevée
                    </div>
                    <div class="probability-value">
                        {max_probability:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with info_col2:

            st.markdown(
                f"""
                <div class="probability-card">
                    <div class="probability-label">
                        🟢 Probabilité la plus faible
                    </div>
                    <div class="probability-value">
                        {min_probability:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ========================================================
        # PROFIL DES SCORES
        # ========================================================

        st.markdown("")

        st.markdown(
            "#### 📊 Concentration des scores de risque"
        )

        st.caption(
            "Position et dispersion des probabilités estimées."
        )

        # --------------------------------------------------------
        # STATISTIQUES
        # --------------------------------------------------------

        q1 = probabilities.quantile(0.25)
        median = probabilities.quantile(0.50)
        q3 = probabilities.quantile(0.75)

        # --------------------------------------------------------
        # BOXPLOT
        # --------------------------------------------------------

        score_df = pd.DataFrame({
            "Probabilité de fraude (%)": probabilities
        })

        fig_box = px.box(
            score_df,
            x="Probabilité de fraude (%)",
            points=False
        )

        fig_box.update_traces(
            line=dict(
                color="#6366F1",
                width=2
            ),
            fillcolor="rgba(99,102,241,0.18)"
        )

        # --------------------------------------------------------
        # LIGNES Q1 / MÉDIANE / Q3
        # --------------------------------------------------------

        fig_box.add_vline(
            x=q1,
            line_dash="dash",
            line_color="#22C55E",
            line_width=2
        )

        fig_box.add_vline(
            x=median,
            line_dash="solid",
            line_color="#F59E0B",
            line_width=3
        )

        fig_box.add_vline(
            x=q3,
            line_dash="dash",
            line_color="#EF4444",
            line_width=2
        )

        # --------------------------------------------------------
        # ANNOTATIONS
        # --------------------------------------------------------

        fig_box.add_annotation(
            x=q1,
            y=0.85,
            text=f"<b>Q1</b><br>{q1:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=-25,
            ay=-50,
            font=dict(
                size=12,
                color="#22C55E"
            ),
            bgcolor="white",
            bordercolor="#22C55E",
            borderwidth=1,
            borderpad=5
        )

        fig_box.add_annotation(
            x=median,
            y=0.85,
            text=f"<b>Médiane</b><br>{median:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-65,
            font=dict(
                size=12,
                color="#F59E0B"
            ),
            bgcolor="white",
            bordercolor="#F59E0B",
            borderwidth=1,
            borderpad=5
        )

        fig_box.add_annotation(
            x=q3,
            y=0.85,
            text=f"<b>Q3</b><br>{q3:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-50,
            font=dict(
                size=12,
                color="#EF4444"
            ),
            bgcolor="white",
            bordercolor="#EF4444",
            borderwidth=1,
            borderpad=5
        )

        # --------------------------------------------------------
        # DESIGN
        # --------------------------------------------------------

        fig_box.update_layout(
            height=230,

            margin=dict(
                l=10,
                r=10,
                t=65,
                b=10
            ),

            xaxis=dict(
                title="Probabilité de fraude (%)",
                range=[0, 100],
                tickmode="linear",
                tick0=0,
                dtick=20
            ),

            yaxis=dict(
                showticklabels=False,
                title=None,
                range=[-0.5, 1.5]
            ),

            plot_bgcolor="white",
            paper_bgcolor="white",

            showlegend=False
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )



    # ========================================================
    # EXPLAINABLE AI — SHAP
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🔍 Explainable AI — Explication du risque'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        "Cette analyse explique les principales variables qui "
        "contribuent à augmenter ou à diminuer le risque de fraude "
        "pour un dossier sélectionné."
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SÉLECTION DU DOSSIER
    # --------------------------------------------------------

    shap_candidates = top_risk.copy()

    # Limiter aux 15 dossiers prioritaires
    shap_candidates = shap_candidates.head(15)

    if len(shap_candidates) > 0:

        dossier_options = {}

        for _, row in shap_candidates.iterrows():

            dossier_label = (
                f"{row['Dossier']} — "
                f"{row['Probabilité fraude (%)']:.1f}%"
            )

            dossier_options[dossier_label] = row["original_index"]

        selected_label = st.selectbox(
            "📁 Sélectionner un dossier à expliquer",
            options=list(dossier_options.keys()),
            key="shap_dossier_selection"
        )

        selected_index = dossier_options[selected_label]

        # ----------------------------------------------------
        # DOSSIER ORIGINAL
        # ----------------------------------------------------

        dossier = df.iloc[[selected_index]]

        try:

            # ------------------------------------------------
            # CHARGEMENT DU MODÈLE
            # ------------------------------------------------

            model, threshold = load_model()

            # ------------------------------------------------
            # EXPLICATION SHAP
            # ------------------------------------------------

            explanation = explain_prediction(
                model,
                dossier,
                top_n=10
            )

            # =================================================
            # NETTOYAGE DES NOMS DES VARIABLES POUR L'AFFICHAGE
            # =================================================

            def clean_shap_name(name):

                name = str(name)

                # Supprimer les préfixes du preprocessing
                name = name.replace("cat__", "", 1)
                name = name.replace("num__", "", 1)

                # Remplacer les underscores par des espaces
                name = name.replace("_", " ")

                # Nettoyer les espaces multiples
                name = " ".join(name.split())

                return name


            explanation["Variable propre"] = (
                explanation["Variable"]
                .apply(clean_shap_name)
            )

            # =================================================
            # VALEURS RÉELLES DU DOSSIER
            # =================================================

            def get_original_feature_value(variable, dossier):

                variable = str(variable)

                # Retirer les préfixes du preprocessing
                clean_variable = variable

                if clean_variable.startswith("cat__"):
                    clean_variable = clean_variable.replace("cat__", "", 1)

                elif clean_variable.startswith("num__"):
                    clean_variable = clean_variable.replace("num__", "", 1)

                # Variables catégorielles
                categorical_columns = [
                    "insured_hobbies",
                    "authorities_contacted",
                    "incident_type",
                    "collision_type",
                    "incident_severity",
                    "police_report_available",
                    "witnesses",
                    "property_damage",
                    "vehicle_make",
                    "vehicle_category",
                    "policy_state",
                    "policy_csl",
                    "insured_relationship",
                    "insured_occupation",
                    "insured_education_level",
                    "insured_marital_status",
                    "insured_sex",
                    "incident_period"
                ]

                for col in categorical_columns:

                    prefix = f"{col}_"

                    if clean_variable.startswith(prefix):

                        if col in dossier.columns:
                            return str(dossier.iloc[0][col])

                # Variables numériques
                if clean_variable in dossier.columns:

                    value = dossier.iloc[0][clean_variable]

                    if pd.isna(value):
                        return "Manquante"

                    if isinstance(
                        value,
                        (int, float, np.integer, np.floating)
                    ):

                        if float(value).is_integer():
                            return str(int(value))

                        return f"{float(value):.2f}"

                    return str(value)

                return "—"


            # Ajouter la valeur réelle
            explanation["Valeur"] = explanation["Variable"].apply(
                lambda x: get_original_feature_value(
                    x,
                    dossier
                )
            )

            # ------------------------------------------------
            # INFORMATIONS DU DOSSIER
            # ------------------------------------------------

            selected_probability = (
                result.iloc[selected_index]["fraud_probability"] * 100
            )

            selected_risk = result.iloc[
                selected_index
            ]["risk_level"]

            selected_decision = result.iloc[
                selected_index
            ]["decision"]

            # =================================================
            # RÉSUMÉ DU DOSSIER — PLEINE LARGEUR
            # =================================================

            st.markdown("### 📌 Résumé du dossier analysé")

            info1, info2, info3 = st.columns(3)

            with info1:

                st.html(f"""
                <div class="probability-card">
                    <div class="probability-label">
                        🎯 Probabilité de fraude
                    </div>

                    <div class="probability-value">
                        {selected_probability:.1f}%
                    </div>

                    <div class="probability-description">
                        Score estimé par le modèle
                    </div>
                </div>
                """)

            with info2:

                st.html(f"""
                <div class="probability-card">
                    <div class="probability-label">
                        ⚠️ Niveau de risque
                    </div>

                    <div class="probability-value">
                        {selected_risk}
                    </div>

                    <div class="probability-description">
                        Classification du niveau de risque
                    </div>
                </div>
                """)

            with info3:

                st.html(f"""
                <div class="probability-card">
                    <div class="probability-label">
                        📌 Décision prédictive
                    </div>

                    <div class="probability-value">
                        {selected_decision}
                    </div>

                    <div class="probability-description">
                        Résultat de la prédiction
                    </div>
                </div>
                """)

            st.markdown("")

            # =================================================
            # FACTEURS AUGMENTANT LE RISQUE
            # =================================================

            risk_factors = (
                explanation[
                    explanation["SHAP"] > 0
                ]
                .sort_values(
                    "SHAP",
                    ascending=False
                )
                .head(5)
            )

            st.markdown(
                "### 🚨 Facteurs augmentant le risque"
            )

            if len(risk_factors) > 0:

                # Une seule ligne contenant les 5 facteurs
                cards_html = ""

                for rank, (_, row) in enumerate(
                    risk_factors.iterrows(),
                    start=1
                ):

                    cards_html += f"""
                    <div style="
                        flex:1;
                        min-width:0;
                        background:#fff7ed;
                        border:1px solid #fed7aa;
                        border-top:5px solid #f97316;
                        border-radius:12px;
                        padding:16px 14px;
                        min-height:150px;
                        box-sizing:border-box;
                    ">

                        <div style="
                            color:#9a3412;
                            font-size:11px;
                            font-weight:700;
                            margin-bottom:12px;
                        ">
                            FACTEUR #{rank}
                        </div>

                        <div style="
                            color:#7c2d12;
                            font-size:14px;
                            font-weight:700;
                            line-height:1.4;
                            word-break:break-word;
                            margin-bottom:15px;
                        ">
                            {row['Variable propre']}
                        </div>

                        <div style="
                            color:#9a3412;
                            font-size:10px;
                            margin-bottom:3px;
                        ">
                            Contribution SHAP
                        </div>

                        <div style="
                            color:#ea580c;
                            font-size:18px;
                            font-weight:800;
                        ">
                            +{row['SHAP']:.4f}
                        </div>

                    </div>
                    """

                st.html(f"""
                <div style="
                    display:flex;
                    gap:12px;
                    width:100%;
                    align-items:stretch;
                ">
                    {cards_html}
                </div>
                """)

            else:

                st.info(
                    "Aucun facteur principal n'augmente le risque "
                    "pour ce dossier."
                )

            # =================================================
            # FACTEURS RÉDUISANT LE RISQUE
            # =================================================

            st.markdown(
                "### 🛡️ Facteurs réduisant le risque"
            )

            protective_factors = (
                explanation[
                    explanation["SHAP"] < 0
                ]
                .sort_values(
                    "SHAP",
                    ascending=True
                )
                .head(5)
            )

            if len(protective_factors) > 0:

                # Une seule ligne contenant les 5 facteurs
                cards_html = ""

                for rank, (_, row) in enumerate(
                    protective_factors.iterrows(),
                    start=1
                ):

                    cards_html += f"""
                    <div style="
                        flex:1;
                        min-width:0;
                        background:#ecfdf5;
                        border:1px solid #a7f3d0;
                        border-top:5px solid #10b981;
                        border-radius:12px;
                        padding:16px 14px;
                        min-height:150px;
                        box-sizing:border-box;
                    ">

                        <div style="
                            color:#047857;
                            font-size:11px;
                            font-weight:700;
                            margin-bottom:12px;
                        ">
                            FACTEUR #{rank}
                        </div>

                        <div style="
                            color:#065f46;
                            font-size:14px;
                            font-weight:700;
                            line-height:1.4;
                            word-break:break-word;
                            margin-bottom:15px;
                        ">
                            {row['Variable propre']}
                        </div>

                        <div style="
                            color:#047857;
                            font-size:10px;
                            margin-bottom:3px;
                        ">
                            Contribution SHAP
                        </div>

                        <div style="
                            color:#059669;
                            font-size:18px;
                            font-weight:800;
                        ">
                            {row['SHAP']:.4f}
                        </div>

                    </div>
                    """

                st.html(f"""
                <div style="
                    display:flex;
                    gap:12px;
                    width:100%;
                    align-items:stretch;
                ">
                    {cards_html}
                </div>
                """)

            else:

                st.info(
                    "Aucun facteur principal ne réduit le risque "
                    "pour ce dossier."
                )


            # =================================================
            # GRAPHIQUE SHAP — PLEINE LARGEUR
            # =================================================

            st.markdown("")
            st.markdown(
                "### 📊 Contribution des variables au risque"
            )

            st.caption(
                "Chaque barre représente l'influence d'une variable "
                "sur la prédiction du modèle pour le dossier sélectionné."
            )

            # -------------------------------------------------
            # PRÉPARATION DES DONNÉES SHAP
            # -------------------------------------------------

            shap_chart = explanation.copy()


            # -------------------------------------------------
            # NETTOYAGE DES NOMS DES VARIABLES
            # -------------------------------------------------

            def clean_shap_name(name):

                name = str(name)

                # Supprimer les préfixes du preprocessing
                name = name.replace("cat__", "", 1)
                name = name.replace("num__", "", 1)

                # Remplacer les underscores par des espaces
                name = name.replace("_", " ")

                # Nettoyer les espaces multiples
                name = " ".join(name.split())

                return name


            shap_chart["Variable propre"] = (
                shap_chart["Variable"]
                .apply(clean_shap_name)
            )


            # -------------------------------------------------
            # DIRECTION DU RISQUE
            # -------------------------------------------------

            shap_chart["Direction"] = np.where(
                shap_chart["SHAP"] > 0,
                "Augmente le risque",
                "Diminue le risque"
            )


            # -------------------------------------------------
            # TRI
            # -------------------------------------------------

            shap_chart = shap_chart.sort_values(
                "SHAP",
                ascending=True
            )


            # -------------------------------------------------
            # VALEUR MAXIMALE POUR L'ÉCHELLE
            # -------------------------------------------------

            max_abs_shap = shap_chart["SHAP"].abs().max()

            # Ajouter de l'espace autour des valeurs
            x_padding = max_abs_shap * 0.18

            x_min = -max_abs_shap - x_padding
            x_max = max_abs_shap + x_padding


            # -------------------------------------------------
            # GRAPHIQUE
            # -------------------------------------------------

            fig_shap = px.bar(
                shap_chart,
                x="SHAP",
                y="Variable propre",
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "Augmente le risque": "#EF4444",
                    "Diminue le risque": "#22C55E"
                },
                text="SHAP"
            )


            # -------------------------------------------------
            # VALEURS SHAP SUR LES BARRES
            # -------------------------------------------------

            fig_shap.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                cliponaxis=False,

                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Contribution SHAP : %{x:.4f}"
                    "<extra></extra>"
                )
            )


            # -------------------------------------------------
            # DESIGN
            # -------------------------------------------------

            fig_shap.update_layout(

                height=550,

                margin=dict(
                    l=300,
                    r=120,
                    t=55,
                    b=60
                ),

                xaxis=dict(
                    title="Contribution SHAP",
                    range=[x_min, x_max],
                    zeroline=True,
                    zerolinewidth=2,
                    tickformat=".2f"
                ),

                yaxis=dict(
                    title=None
                ),

                plot_bgcolor="white",
                paper_bgcolor="white",

                legend=dict(
                    title="Direction",
                    orientation="v",
                    x=1.0,
                    xanchor="right",
                    y=1.05,
                    yanchor="bottom"
                )
            )


            # -------------------------------------------------
            # AFFICHAGE
            # -------------------------------------------------

            st.plotly_chart(
                fig_shap,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )


            # =================================================
            # PRINCIPAL FACTEUR
            # =================================================

            if len(risk_factors) > 0:

                main_factor = risk_factors.iloc[0]["Variable propre"]
                main_shap = risk_factors.iloc[0]["SHAP"]

                st.html(f"""
                <div style="
                    background:linear-gradient(
                        135deg,
                        #eff6ff,
                        #ecfeff
                    );
                    border:1px solid #bae6fd;
                    border-radius:16px;
                    padding:20px 24px;
                    margin-top:15px;
                    width:100%;
                    box-sizing:border-box;
                ">

                    <div style="
                        color:#0369a1;
                        font-size:13px;
                        font-weight:700;
                        margin-bottom:6px;
                    ">
                        🔎 PRINCIPAL FACTEUR DE RISQUE
                    </div>

                    <div style="
                        color:#0f172a;
                        font-size:20px;
                        font-weight:800;
                        line-height:1.4;
                        word-break:break-word;
                    ">
                        {main_factor}
                    </div>

                    <div style="
                        color:#475569;
                        font-size:14px;
                        margin-top:8px;
                    ">
                        Contribution SHAP :
                        <strong style="color:#dc2626;">
                            +{main_shap:.4f}
                        </strong>
                    </div>

                </div>
                """)

            # =================================================
            # INTERPRÉTATION
            # =================================================

            st.markdown("")
            st.markdown(
                "### 🧠 Interprétation de la prédiction"
            )

            if len(risk_factors) > 0:

                main_factor = risk_factors.iloc[0]["Variable propre"]

                st.info(
                    f"Pour ce dossier, la variable **{main_factor}** "
                    f"fait partie des principaux facteurs contribuant "
                    f"à l'augmentation du score de fraude."
                )

            else:

                st.info(
                    "Aucun facteur fortement contributeur au risque "
                    "n'a été identifié parmi les variables principales."
                )

            # =================================================
            # EXPLICATION SHAP
            # =================================================

            st.html("""
            <div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:14px;
                padding:18px 22px;
                margin-top:15px;
                width:100%;
                box-sizing:border-box;
            ">

                <strong style="color:#334155;">
                    ℹ️ Comment lire cette analyse ?
                </strong>

                <p style="
                    color:#64748b;
                    font-size:13px;
                    line-height:1.7;
                    margin-top:8px;
                ">

                    Une contribution <strong>SHAP positive</strong>
                    indique que la variable contribue à augmenter
                    la sortie du modèle vers la classe fraude.

                    <br><br>

                    Une contribution <strong>SHAP négative</strong>
                    indique que la variable contribue à diminuer
                    cette sortie.

                    <br><br>

                    L'explication est spécifique au dossier
                    sélectionné et permet de mieux comprendre
                    pourquoi le modèle attribue ce niveau de risque.

                </p>

            </div>
            """)

        except Exception as e:

            st.error(
                f"❌ Impossible de générer l'explication SHAP : {e}"
            )

    else:

        st.info(
            "Aucun dossier disponible pour l'explication SHAP."
        )

    # ========================================================
    # TABLEAU DES RÉSULTATS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Résultats de l’analyse'
        '</div>',
        unsafe_allow_html=True
    )

    display_result = result.copy()


    display_result["fraud_probability"] = (
        display_result["fraud_probability"] * 100
    ).round(2)


    display_result = display_result.rename(
        columns={
            "fraud_probability":
                "Probabilité fraude (%)",

            "prediction":
                "Prédiction",

            "risk_level":
                "Niveau de risque",

            "decision":
                "Décision"
        }
    )


    st.dataframe(
        display_result,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # TÉLÉCHARGEMENT
    # ========================================================

    csv_result = display_result.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Télécharger les résultats CSV",
        data=csv_result,
        file_name="insurance_risk_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # DOSSIERS À RISQUE ÉLEVÉ
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🚨 Dossiers à risque élevé'
        '</div>',
        unsafe_allow_html=True
    )


    high_risk = result[
        result["risk_level"] == "Élevé"
    ].copy()


    high_risk["fraud_probability"] = (
        high_risk["fraud_probability"] * 100
    ).round(2)


    high_risk = high_risk.rename(
        columns={
            "fraud_probability":
                "Probabilité fraude (%)",

            "prediction":
                "Prédiction",

            "risk_level":
                "Niveau de risque",

            "decision":
                "Décision"
        }
    )


    if len(high_risk) > 0:

        st.warning(
            f"⚠️ **{len(high_risk):,}** dossier(s) "
            "présentent un niveau de risque élevé."
        )

        st.dataframe(
            high_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "✓ Aucun dossier à risque élevé détecté."
        )


    # ========================================================
    # INFORMATIONS DU MODÈLE
    # ========================================================

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">'
        '🧠 À propos du modèle'
        '</div>'
        '<div class="info-text">'
        'Le système utilise un modèle '
        '<strong>Random Forest</strong> entraîné sur '
        "des données historiques d'assurance."
        '<br><br>'
        'Pour chaque dossier, le modèle calcule une '
        'probabilité de fraude puis attribue un niveau '
        'de risque : <strong>Faible</strong>, '
        '<strong>Modéré</strong> ou '
        '<strong>Élevé</strong>.'
        '<br><br>'
        "Les résultats constituent une aide à l'analyse "
        'et ne remplacent pas la validation par un expert '
        'en assurance.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE D'ACCUEIL — AUCUN FICHIER
# ============================================================

else:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">'
        '🚀 Prêt à analyser vos dossiers'
        '</div>'
        '<div class="info-text">'
        "Importez un fichier CSV contenant les dossiers "
        "d'assurance pour commencer."
        '<br><br>'
        '<strong>Workflow :</strong> '
        'Importation → Préparation des données → '
        'Prédiction → Score de risque → '
        'Identification des dossiers prioritaires.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🛡️ <strong>AI Insurance Risk Prediction</strong>'
    ' &nbsp;•&nbsp; Machine Learning'
    ' &nbsp;•&nbsp; Explainable AI'
    ' &nbsp;•&nbsp; Insurance Analytics'
    '</div>',
    unsafe_allow_html=True
)