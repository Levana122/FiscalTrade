import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import os
import time
import threading
import ssl
import plotly.graph_objects as go
from datetime import datetime, date
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.message import EmailMessage
import smtplib
from taux_fiscal import get_taux_imposition, calcul_impot_usa, calcul_impot_uk
from test_yfinance import get_price_history, get_stock_price, analyze_trend_custom, get_stock_price_value
import ia_predict
from ia_predict import predire_tendance
import openpyxl
from openpyxl.styles import Font

# Initialisation de l'historique des transactions
if "transactions" not in st.session_state:
    st.session_state.transactions = []

# Configuration de la page
st.set_page_config(
    page_title="FiscalTrade - Application de Gestion Financière",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💼"
)

# Style CSS pour une apparence professionnelle
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #34495E;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .sub-header {
        font-size: 1.4em;
        font-weight: bold;
        color: #5D6D7E;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .success-message {
        background-color: #D4EDDA;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .warning-message {
        background-color: #FFF3CD;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .error-message {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .info-message {
        background-color: #D1ECF1;
        color: #0C5460;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #BDC3C7;
        padding: 8px;
    }
    .stDataFrame {
        border-radius: 5px;
        border: 1px solid #BDC3C7;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour afficher des messages stylisés
def display_message(type, message):
    if type == "success":
        st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
    elif type == "warning":
        st.markdown(f'<div class="warning-message">{message}</div>', unsafe_allow_html=True)
    elif type == "error":
        st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
    elif type == "info":
        st.markdown(f'<div class="info-message">{message}</div>', unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">FiscalTrade - Application de Gestion Financière</div>', unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("Navigation")
sections = [
    "Watchlist",
    "Analyse du Marché",
    "Calcul de l'Impôt",
    "Gestion des Transactions",
    "Analyse Avancée",
    "Simulation de Vente",
    "Rapport et Envoi",
    "Actualités Financières"
]
selected_section = st.sidebar.radio("Sélectionnez une section", sections)

# --- CSS pour style Google Finance-like ---
st.markdown("""
<style>
.watchlist-container {
    max-width: 900px;
    margin: 0 auto 40px auto;
    font-family: "Roboto", sans-serif;
    font-size: 14px;
}
.watchlist-row {
    display: flex;
    align-items: center;
    border-bottom: 1px solid #eee;
    padding: 15px 0;
}
.ticker-badge {
    font-weight: 700;
    font-size: 13px;
    color: white;
    border-radius: 4px;
    padding: 4px 10px;
    margin-right: 12px;
    width: 70px;
    text-align: center;
    background-color: #202124;
}
.company-name {
    flex: 1 1 auto;
    color: #3c4043;
    font-weight: 500;
}
.price {
    width: 90px;
    text-align: right;
    font-weight: 600;
    color: #202124;
}
.abs-change {
    width: 90px;
    text-align: right;
    font-weight: 600;
}
.pct-change {
    width: 70px;
    text-align: center;
    font-weight: 600;
    border-radius: 100px;
    padding: 5px 0;
    margin-left: 15px;
    font-size: 13px;
}
.positive {
    color: #137333;
    background-color: #d7f4d7;
}
.negative {
    color: #a50e0e;
    background-color: #fbd7d7;
}
.add-button {
    margin-left: 20px;
    background: none;
    border: none;
    cursor: pointer;
    color: #1a73e8;
    font-weight: 700;
    font-size: 20px;
    padding: 0 12px;
    transition: color 0.3s ease;
}
.add-button:hover {
    color: #1558b0;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_company_name(sym):
    try:
        # fast_info n'a pas le nom donc on garde .info
        # MAIS on met ttl=3600 donc appelé 1 seule fois par heure
        # C'est acceptable car le nom change jamais
        info = yf.Ticker(sym).info
        return info.get("shortName") or sym
    except:
        return sym

@st.cache_data(ttl=60)
def get_ticker_data(sym):
    try:
        fi = yf.Ticker(sym).fast_info
        price = fi.last_price
        prev_close = fi.previous_close
        return price, prev_close
    except:
        return None, None

# --- Initialisation watchlists dans session state ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist1 = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


# --- Fonction pour afficher une watchlist (affichage propre) ---
def display_watchlist(watchlist_name):
    st.markdown(f"## Watchlist: {watchlist_name}")

    # Input + bouton d'ajout dans une ligne
    col1, col2 = st.columns([4,1])
    with col1:
        new_ticker = st.text_input(f"Ajouter un ticker à {watchlist_name}", key=f"input_{watchlist_name}")
    with col2:
        if st.button(f"Ajouter", key=f"btn_add_{watchlist_name}") and new_ticker:
            new_ticker = new_ticker.strip().upper()
            if new_ticker not in st.session_state[watchlist_name]:
                st.session_state[watchlist_name].append(new_ticker)
                st.rerun()
            else:
                st.warning(f"{new_ticker} est déjà dans {watchlist_name}.")

    st.markdown('<div class="watchlist-container">', unsafe_allow_html=True)

    for i, sym in enumerate(st.session_state[watchlist_name]):
        try:
            price, prev_close = get_ticker_data(sym)
            company_name = get_company_name(sym)
            
        if price is None or prev_close is None:
            price_text = "N/A"
            abs_change_text = "N/A"
            pct_change_text = "N/A"
            pct_class = ""
        else:
            price_text = f"{price:.2f} $"
            abs_change = price - prev_close
            abs_change_text = f"{abs_change:+.2f} $"
            pct_change = (abs_change / prev_close) * 100
            pct_change_text = f"{pct_change:+.2f} %"
            pct_class = "positive" if abs_change >= 0 else "negative"

        st.markdown(f'''
        <div class="watchlist-row">
            <div class="ticker-badge">{sym}</div>
            <div class="company-name">{company_name}</div>
            <div class="price">{price_text}</div>
            <div class="abs-change">{abs_change_text}</div>
            <div class="pct-change {pct_class}">{pct_change_text}</div>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Erreur pour {sym}: {e}")
            
            # Rendu html par ligne
            st.markdown(f'''
            <div class="watchlist-row">
                <div class="ticker-badge">{sym}</div>
                <div class="company-name" title="{company_name}">{company_name}</div>
                <div class="price">{price_text}</div>
                <div class="abs-change">{abs_change_text}</div>
                <div class="pct-change {pct_class}">{pct_change_text}</div>
                <button class="add-button" title="Supprimer" onclick="window.location.href=window.location.href + '?del={watchlist_name}_{i}'">×</button>
            </div>
            ''', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Erreur pour {sym}: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Gestion suppression via query params pour simuler suppression bouton ---
# Simple gestion pour demo: si utilisateur clique sur "x", on supprime le ticker
query_params = st.query_params
del_param = query_params.get("del")
if del_param:
    try:
        watchlist_id, index_str = del_param[0].split("_")
        index = int(index_str)
        if watchlist_id in st.session_state and 0 <= index < len(st.session_state[watchlist_id]):
            st.session_state[watchlist_id].pop(index)
            st.query_params.clear()  # reset URL pour éviter suppressions multiples
            st.rerun()
    except Exception:
        pass

# Gestion des sections
if selected_section == "Watchlist1":
    st.title("Watchlist - Style Google Finance")
    display_watchlist("watchlist1")

elif selected_section == "Watchlist2":
    st.title("Watchlist - Style Google Finance")
    display_watchlist("watchlist2")

elif selected_section == "Analyse du Marché":
    st.markdown('<div class="section-header">Analyse du Marché</div>', unsafe_allow_html=True)
    
    ticker = st.text_input("Ticker", value="AAPL")
    date_debut = st.date_input("Date d'achat", value=date(2024, 1, 1))
    date_fin = st.date_input("Date de vente", value=date.today())
    
    if st.button("Analyser"):
        try:
            data = yf.download(ticker, start=str(date_debut), end=str(date_fin), auto_adjust=False)
            if not data.empty:
                prix_debut = float(data['Close'].dropna().iloc[0].item())
                prix_fin = float(data['Close'].dropna().iloc[-1].item())
                variation = ((prix_fin - prix_debut) / prix_debut) * 100
                tendance = "Hausse" if variation > 0 else ("Baisse" if variation < 0 else "Stable")
                display_message("success", f"Analyse de {ticker} de {date_debut} à {date_fin}")
                display_message("info", f"Prix initial: {prix_debut:.2f} $ | Final: {prix_fin:.2f} $ | Variation: {variation:.2f}% | {tendance}")
                st.line_chart(data['Close'])
            else:
                display_message("warning", "Aucune donnée disponible.")
        except Exception as e:
            display_message("error", f"Erreur lors de l'analyse: {e}")

elif selected_section == "Calcul de l'Impôt":
    st.markdown('<div class="section-header">Calcul de l\'Impôt</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pays = st.selectbox("Pays", ["France", "USA", "UK", "Autre"])
        achat = st.number_input("Prix d'achat", step=0.01)
        vente = st.number_input("Prix de vente", step=0.01)
    with col2:
        quantite = st.number_input("Quantité", step=1)
        duree = 0
        if pays == "USA":
            duree = st.number_input("Durée de détention (années)", step=0.1)
    
    if st.button("Calculer l'impôt"):
        try:
            plus_value = (vente - achat) * quantite
            if pays == "France":
                taux = 0.3
            elif pays == "USA":
                taux = 0.15 if duree >= 1 else 0.3
            elif pays == "UK":
                taux = 0.1
            else:
                taux = 0.25
            impot = plus_value * taux
            display_message("success", f"Plus-value: {plus_value:.2f} € | Impôt: {impot:.2f} €")
            st.session_state.transactions.append({
                "Symbole": ticker if 'ticker' in locals() else "N/A",
                "Pays": pays,
                "Prix Achat": achat,
                "Prix Vente": vente,
                "Quantité": quantite,
                "Plus-value": plus_value,
                "Impôt": impot
            })
        except Exception as e:
            display_message("error", f"Erreur de calcul: {e}")

elif selected_section == "Gestion des Transactions":
    st.markdown('<div class="section-header">Gestion des Transactions</div>', unsafe_allow_html=True)
    
    # Export Excel
    st.markdown('<div class="sub-header">Export Excel</div>', unsafe_allow_html=True)
    if st.button("Exporter Excel"):
        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)
            df.to_excel("transactions.xlsx", index=False)
            with open("transactions.xlsx", "rb") as f:
                st.download_button("Télécharger Excel", f, file_name="transactions.xlsx")
        else:
            display_message("warning", "Aucune donnée à exporter.")
    
    # Import CSV
    st.markdown('<div class="sub-header">Importer un fichier CSV de transactions</div>', unsafe_allow_html=True)
    fichier_csv = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if fichier_csv is not None:
        try:
            df = pd.read_csv(fichier_csv)
            colonnes_attendues = ["Symbole", "Prix Achat", "Prix Vente", "Quantité", "Pays"]
            if not all(col in df.columns for col in colonnes_attendues):
                display_message("error", f"Le fichier doit contenir les colonnes suivantes: {', '.join(colonnes_attendues)}")
            else:
                display_message("success", "Fichier CSV chargé avec succès!")
                st.dataframe(df)
                
                taux_par_pays = {"France": 0.3, "USA": 0.2, "UK": 0.19, "Allemagne": 0.25, "Canada": 0.15, "Autre": 0.3}
                
                def calculer_impot(row):
                    taux = taux_par_pays.get(row["Pays"], 0.3)
                    plus_value = (row["Prix Vente"] - row["Prix Achat"]) * row["Quantité"]
                    return round(plus_value * taux, 2)
                
                df["Impôt"] = df.apply(calculer_impot, axis=1)
                
                for _, ligne in df.iterrows():
                    st.session_state.transactions.append({
                        "Symbole": ligne["Symbole"],
                        "Prix Achat": ligne["Prix Achat"],
                        "Prix Vente": ligne["Prix Vente"],
                        "Quantité": ligne["Quantité"],
                        "Pays": ligne["Pays"],
                        "Impôt": ligne["Impôt"]
                    })
                
                display_message("success", "Transactions ajoutées à l’historique avec succès!")
        except Exception as e:
            display_message("error", f"Erreur lors de l’import: {e}")
    
    # Graphique des plus-values par pays
    st.markdown('<div class="sub-header">Plus-values par Pays</div>', unsafe_allow_html=True)
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        if all(col in df.columns for col in ["Prix Vente", "Prix Achat", "Quantité", "Pays"]):
            df["Plus-Value"] = (df["Prix Vente"] - df["Prix Achat"]) * df["Quantité"]
            pv_par_pays = df.groupby("Pays")["Plus-Value"].sum()
            st.bar_chart(pv_par_pays)
        else:
            display_message("warning", "Données incomplètes pour générer le graphique.")
    else:
        display_message("info", "Aucune transaction enregistrée.")
    
   
    
    # Bilan Fiscal Global
    st.markdown('<div class="sub-header">Bilan Fiscal Global</div>', unsafe_allow_html=True)
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        total_pv = ((df["Prix Vente"] - df["Prix Achat"]) * df["Quantité"]).sum()
        total_impot = df["Impôt"].sum()
        nb_transactions = len(df)
        pays_plus_rentable = df.groupby("Pays").apply(lambda d: ((d["Prix Vente"] - d["Prix Achat"]) * d["Quantité"]).sum()).idxmax()
        
        display_message("success", f"""
        Plus-value totale: {total_pv:.2f} EUR
        Impôt total estimé: {total_impot:.2f} EUR
        Nombre de transactions: {nb_transactions}
        Pays le plus rentable: {pays_plus_rentable}
        """)
    else:
        display_message("info", "Aucune transaction enregistrée.")

# Section 5: Analyse Avancée
elif selected_section == "Analyse Avancée":
    st.markdown('<div class="section-header">Analyse Avancée</div>', unsafe_allow_html=True)
    
    # RSI / MACD
    st.markdown('<div class="sub-header">RSI / MACD</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker pour l'analyse", value="AAPL")
    if st.button("Afficher RSI / MACD"):
        try:
            df = yf.download(ticker, start="2023-01-01", end=str(date.today()))
            if df.empty or 'Close' not in df.columns:
                display_message("warning", "Données indisponibles ou colonne 'Close' manquante.")
            else:
                # Calcul MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                
                # Calcul RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Graphique
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=macd.squeeze(), name="MACD"))
                fig.add_trace(go.Scatter(x=df.index, y=signal.squeeze(), name="Signal MACD"))
                fig.add_trace(go.Scatter(x=df.index, y=rsi.squeeze(), name="RSI", yaxis="y2"))
                
                fig.update_layout(
                    title="RSI et MACD",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="MACD"),
                    yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
                    legend=dict(x=0, y=1.15, orientation="h")
                )
                
                st.plotly_chart(fig)
        except Exception as e:
            display_message("error", f"Erreur lors du graphique: {e}")
    
        # Conseil IA Global
    st.markdown('<div class="sub-header">Conseil IA Global</div>', unsafe_allow_html=True)
    if st.button("Analyser mes positions avec IA"):
        if not st.session_state.transactions:
            display_message("warning", "Aucune transaction enregistrée pour l'analyse.")
        else:
            conseils = []
            for t in st.session_state.transactions:
                try:
                    symbole = t.get("Symbole", "")
                    prix_achat = t.get("Prix Achat", 0)
                    quantite = t.get("Quantité", 0)
                    prix_actuel = get_stock_price_value(symbole)
                    tendance = analyze_trend_custom(symbole, "1mo")
                    plus_value = (prix_actuel - prix_achat) * quantite
                    
                    if plus_value < -500:
                        conseil = "Vendre rapidement - perte importante"
                    elif plus_value > 1000:
                        conseil = "Considérer la vente - forte plus-value"
                    elif "BAISSIÈRE" in tendance.upper():
                        conseil = "Attention à la tendance baissière"
                    else:
                        conseil = "Conserver - situation stable"
                    
                    conseils.append({
                        "Symbole": symbole,
                        "Prix Actuel": prix_actuel,
                        "Plus-Value": plus_value,
                        "Tendance": tendance,
                        "Conseil": conseil
                    })
                except Exception as e:
                    display_message("warning", f"Erreur pour {t.get('Symbole', '?')}: {e}")
            
            if not conseils:
                display_message("warning", "Aucune donnée exploitable pour l'analyse IA.")
            else:
                display_message("success", "Analyse IA terminée")
                for conseil in conseils:
                    st.markdown(f"""
                    ---
                    **{conseil['Symbole']}**
                    - Prix actuel: {conseil['Prix Actuel']:.2f} €
                    - Plus-value estimée: {conseil['Plus-Value']:.2f} €
                    - Tendance: {conseil['Tendance']}
                    - Conseil IA: {conseil['Conseil']}
                    """)

# Section 6: Simulation de Vente
elif selected_section == "Simulation de Vente":
    st.markdown('<div class="section-header">Simulation de Vente</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        ticker_sim = st.text_input("Ticker", value="AAPL", key="sim_ticker")
        prix_achat_sim = st.number_input("Prix d'achat", step=0.01, key="sim_achat")
        quantite_sim = st.number_input("Quantité", step=1, min_value=1, key="sim_qte")
    with col2:
        pays_sim = st.selectbox("Pays", ["France", "USA", "UK", "Allemagne", "Canada", "Autre"], key="sim_pays")
        duree_sim = 0
        if pays_sim == "USA":
            duree_sim = st.number_input("Durée de détention (années)", step=0.1, min_value=0.0, key="sim_duree")
    
    if st.button("Simuler la vente"):
        try:
            info = yf.Ticker(ticker_sim).history(period="1d")
            if info.empty:
                display_message("warning", "Impossible de récupérer le prix actuel.")
            else:
                prix_actuel = float(info["Close"].iloc[-1])
                plus_value = (prix_actuel - prix_achat_sim) * quantite_sim
                
                if pays_sim == "France":
                    taux = 0.3
                elif pays_sim == "USA":
                    taux = 0.15 if duree_sim >= 1 else 0.3
                elif pays_sim == "UK":
                    taux = 0.1
                elif pays_sim == "Allemagne":
                    taux = 0.25
                elif pays_sim == "Canada":
                    taux = 0.15
                else:
                    taux = 0.3
                
                impot = plus_value * taux
                net = plus_value - impot
                
                display_message("success", f"""
                {ticker_sim}
                Prix actuel: {prix_actuel:.2f} EUR
                Plus-value estimée: {plus_value:.2f} EUR
                Taux d'imposition: {taux*100:.1f}%
                Impôt estimé: {impot:.2f} EUR
                Net après impôt: {net:.2f} EUR
                """)
        except Exception as e:
            display_message("error", f"Erreur lors de la simulation: {e}")

# Section 7: Rapport et Envoi
elif selected_section == "Rapport et Envoi":
    st.markdown('<div class="section-header">Rapport et Envoi</div>', unsafe_allow_html=True)
    
    # Envoi du rapport fiscal par e-mail
    st.markdown('<div class="sub-header">Envoi du rapport fiscal par e-mail</div>', unsafe_allow_html=True)
    email_client = st.text_input("Adresse e-mail du client", key="email_client")
    
    if st.button("Envoyer le rapport par e-mail"):
        if not email_client:
            display_message("warning", "Veuillez saisir l’adresse e-mail du client.")
        else:
            try:
                transactions = st.session_state.get("transactions", [
                    {"Symbole": "AAPL", "Prix Achat": 120.0, "Prix Vente": 150.0, "Quantité": 10, "Impôt": 90.0},
                    {"Symbole": "TSLA", "Prix Achat": 200.0, "Prix Vente": 180.0, "Quantité": 5, "Impôt": 0.0},
                ])
                
                contenu = "Résumé FiscalTrade\n\n"
                contenu += f"{'Symbole':<10} {'Achat (€)':<12} {'Vente (€)':<12} {'Qté':<6} {'Impôt (€)':<10}\n"
                contenu += "-" * 55 + "\n"
                
                total_impot = 0
                for t in transactions:
                    contenu += f"{t.get('Symbole',''):<10} {t.get('Prix Achat', 0):<12.2f} {t.get('Prix Vente', 0):<12.2f} {t.get('Quantité', 0):<6} {t.get('Impôt', 0):<10.2f}\n"
                    total_impot += t.get("Impôt", 0)
                
                contenu += "-" * 55 + "\n"
                contenu += f"{'Total Impôt estimé :':<40} {total_impot:.2f} €\n\n"
                contenu += "Merci d’avoir utilisé FiscalTrade.\nL’équipe FiscalTrade"
                
                msg = MIMEMultipart()
                msg["From"] = "levanaestla@gmail.com"  # Remplacer par votre e-mail
                msg["To"] = email_client
                msg["Subject"] = "Votre rapport fiscal - FiscalTrade"
                msg.attach(MIMEText(contenu, "plain", _charset="utf-8"))
                
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login("levanaestla@gmail.com", "tfegcpqgkzhpuahy")  # Remplacer par vos credentials
                    server.send_message(msg)
                
                display_message("success", "Rapport fiscal envoyé avec succès!")
            except Exception as e:
                display_message("error", f"Erreur lors de l’envoi: {e}")

# Section 8: Actualités Financières
elif selected_section == "Actualités Financières":
    st.markdown('<div class="section-header">Actualités Financières Récentes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        symbole = st.text_input("Ticker (ex: AAPL, BTC, TSLA)", value="AAPL")
    with col2:
        mot_cle = st.text_input("Mot-clé (optionnel)", value="")
    
    secteur = st.selectbox("Secteur", ["Aucun", "Crypto", "Technologie", "Énergie", "Banques", "Santé", "Automobile", "Luxe"])
    date_min = st.date_input("À partir du:", value=date(2024, 1, 1))
    
    def get_news(symbole=None, mot_cle=None, secteur=None, date_min=None, language="fr", limit=5):
        api_key = st.secrets["marketaux"]["api_key"]  # Assurez-vous que la clé API est configurée
        params = {
            "language": language,
            "api_token": api_key,
            "limit": limit,
            "published_after": date_min.strftime("%Y-%m-%d")
        }
        
        secteurs_query = {
            "Crypto": "bitcoin OR crypto OR ethereum",
            "Technologie": "Google OR Apple OR Microsoft OR IA OR Nvidia",
            "Énergie": "pétrole OR gaz OR énergie OR Total",
            "Banques": "banques OR taux OR obligations OR BNP",
            "Santé": "pharma OR santé OR biotech",
            "Automobile": "Tesla OR voitures OR batteries OR Ford",
            "Luxe": "LVMH OR Kering OR Hermès"
        }
        
        if secteur and secteur != "Aucun":
            params["query"] = secteurs_query.get(secteur, "")
        elif mot_cle:
            params["query"] = mot_cle
        elif symbole:
            params["symbols"] = symbole
        
        try:
            response = requests.get("https://api.marketaux.com/v1/news/all", params=params)
            data = response.json()
            if "data" in data and data["data"]:
                return data["data"]
            else:
                return []
        except Exception as e:
            return [{
                "title": "Erreur de récupération",
                "description": str(e),
                "url": "#",
                "published_at": ""
            }]
    
    articles = get_news(symbole.upper(), mot_cle, secteur, date_min)
    
    if articles:
        for article in articles:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(f"Publié le: {article['published_at'][:10]}")
            st.write(article['description'][:300] + "...")
            st.markdown("---")
    else:
        display_message("info", "Aucune actualité trouvée pour ce filtre.")

# Pied de page
st.markdown("""
    <hr>
    <div style='text-align: center; color: #7D7D7D; font-size: 0.9em;'>
        FiscalTrade - Application de Gestion Financière | Version 1.0 | © 2024
    </div>
""", unsafe_allow_html=True)
