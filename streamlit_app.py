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
from datetime import datetime, date
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.message import EmailMessage
import smtplib
import plotly.graph_objects as go
import joblib  # Pour charger le modèle IA
import warnings
warnings.filterwarnings('ignore')

# Imports personnalisés (assumés présents ou à adapter)
from taux_fiscal import get_taux_imposition, calcul_impot_usa, calcul_impot_uk
from test_yfinance import get_price_history, get_stock_price, analyze_trend_custom, get_stock_price_value
from ia_predict import predire_tendance  # Utilise la fonction améliorée
from news import get_news

import openpyxl
from openpyxl.styles import Font

# 📌 Initialisation de l'historique des transactions
if "transactions" not in st.session_state:
    st.session_state.transactions = []

# 📌 Charger le modèle IA pré-entraîné (si disponible)
@st.cache_resource
def charger_modele_ia():
    try:
        model = joblib.load("modele_bourse.pkl")
        return model
    except FileNotFoundError:
        st.warning("Modèle IA non trouvé. Entraînez-le d'abord via le script séparé.")
        return None

modele_ia = charger_modele_ia()

# ✅ Configuration de la page
st.set_page_config(page_title="FiscalTrade", layout="wide", initial_sidebar_state="expanded")

# === Sidebar pour navigation ===
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Aller à :", ["Watchlist", "Analyse Marché", "Calcul Impôt", "Export/Import", "Simulations", "Rapports", "IA & Actualités"])

# === Fonctions utilitaires améliorées ===
def plot_sparkline(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        line=dict(color='crimson', width=2),
        fill='tozeroy',
        fillcolor='rgba(220, 20, 60, 0.1)',
        mode='lines',
        showlegend=False
    ))
    fig.update_layout(
        height=40,
        width=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def calculer_indicateurs_avances(df):
    """Calcule RSI, MACD, etc., pour les graphiques."""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# === Page Watchlist ===
if page == "Watchlist":
    st.header("📋 Watchlist - Style MSN Finance")
    
    # === Initialisation de la watchlist ===
    defaut_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = [{"ticker": t} for t in defaut_tickers]
    
    # === Ajouter ou rechercher un ticker ===
    col1, col2 = st.columns([4, 1])
    with col1:
        new_ticker = st.text_input("🔍 Rechercher ou ajouter un ticker (ex: AAPL, BTC-USD)", key="watch_add")
    with col2:
        if st.button("Ajouter à la watchlist") and new_ticker:
            new_ticker = new_ticker.upper()
            if new_ticker not in [t['ticker'] for t in st.session_state.watchlist]:
                st.session_state.watchlist.append({"ticker": new_ticker})
                st.success(f"{new_ticker} ajouté à la watchlist")
            else:
                st.warning("Déjà présent dans la watchlist")
    
    # === Style CSS compact ===
    st.markdown("""<style>
        .element-container:nth-child(n) > div > div {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem;
        }
    </style>""", unsafe_allow_html=True)
    
    # === Affichage de la watchlist ===
    if st.session_state.watchlist:
        for i, item in enumerate(st.session_state.watchlist):
            ticker = item['ticker']
            try:
                data = yf.Ticker(ticker).history(period="7d")
                if data.empty:
                    st.warning(f"Aucune donnée pour {ticker}")
                    continue
                
                prix = data['Close'].iloc[-1]
                prix_prec = data['Close'].iloc[-2] if len(data['Close']) > 1 else prix
                variation = ((prix - prix_prec) / prix_prec) * 100 if prix_prec else 0
                variation_txt = f"{variation:+.2f} %"
                couleur = "green" if variation >= 0 else "red"
                symbole = "🔺" if variation >= 0 else "🔻"
                
                col1, col2, col3, col4, col5 = st.columns([1, 2.5, 1, 1, 0.5])
                col1.markdown(f"**{ticker}**")
                col2.plotly_chart(plot_sparkline(data), use_container_width=True)
                col3.markdown(f"<span style='font-size: 14px;'>{prix:.2f} $</span>", unsafe_allow_html=True)
                col4.markdown(f"<span style='color:{couleur}; font-size: 14px;'>{symbole} {variation_txt}</span>", unsafe_allow_html=True)
                if col5.button("❌", key=f"del_{i}"):
                    st.session_state.watchlist.pop(i)
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Erreur {ticker} : {e}")
    else:
        st.info("Aucun actif surveillé.")

# === Page Analyse Marché ===
elif page == "Analyse Marché":
    st.header("📈 Analyse du Marché")
    ticker = st.text_input("Ticker", value="AAPL")
    date_debut = st.date_input("Date d'achat", value=date(2024, 1, 1))
    date_fin = st.date_input("Date de vente", value=date.today())
    periode = st.selectbox("Période pour indicateurs", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if st.button("Analyser"):
        try:
            data = yf.download(ticker, start=str(date_debut), end=str(date_fin), auto_adjust=False)
            if not data.empty:
                prix_debut = float(data['Close'].dropna().iloc[0].item())
                prix_fin = float(data['Close'].dropna().iloc[-1].item())
                variation = ((prix_fin - prix_debut) / prix_debut) * 100
                tendance = "🔼 Hausse" if variation > 0 else ("🔽 Baisse" if variation < 0 else "Stable")
                st.success(f"{ticker} de {date_debut} à {date_fin}")
                st.info("Prix initial : {:.2f} $ | Final : {:.2f} $ | Variation : {:.2f}% | {}".format(
                    prix_debut, prix_fin, variation, tendance))
                st.line_chart(data['Close'])
                
                # Prédiction IA intégrée
                if modele_ia:
                    pred = predire_tendance(ticker, modele_ia, periode=periode)
                    st.info(f"🧠 Prédiction IA : {pred}")
                else:
                    st.warning("Modèle IA non chargé.")
            else:
                st.warning("Aucune donnée.")
        except Exception as e:
            st.error(f"Erreur analyse : {e}")

# === Page Calcul Impôt ===
elif page == "Calcul Impôt":
    st.header("💰 Calcul de l'Impôt")
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
            st.success(f"Plus-value : {plus_value:.2f} € | Impôt : {impot:.2f} €")
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
            st.error(f"Erreur de calcul : {e}")

# === Page Export/Import ===
elif page == "Export/Import":
    st.header("📤 Export Excel")
    if st.button("Exporter Excel"):
        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)
            df.to_excel("transactions.xlsx", index=False)
            with open("transactions.xlsx", "rb") as f:
                st.download_button("⬇️ Télécharger Excel", f, file_name="transactions.xlsx")
        else:
            st.warning("Aucune donnée.")
    
    st.header("📚 Importer un fichier CSV de transactions")
    fichier_csv = st.file_uploader("📂 Choisis un fichier CSV", type="csv")
    if fichier_csv is not None:
        try:
            df = pd.read_csv(fichier_csv)
            colonnes_attendues = ["Symbole", "Prix Achat", "Prix Vente", "Quantité", "Pays"]
            if not all(col in df.columns for col in colonnes_attendues):
                st.error(f"❌ Le fichier doit contenir les colonnes suivantes : {', '.join(colonnes_attendues)}")
            else:
                st.success("✅ Fichier CSV chargé avec succès !")
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
                st.success("📥 Transactions ajoutées à l’historique avec succès !")
        except Exception as e:
            st.error(f"❌ Erreur lors de l’import : {e}")

# === Page Simulations ===
elif page == "Simulations":
    st.header("🔮 Simulation de Vente")
    col1, col2 = st.columns(2)
    with col1:
        ticker_sim = st.text_input("🔍 Ticker", value="AAPL", key="sim_ticker")
        prix_achat_sim = st.number_input("💶 Prix d'achat", step=0.01, key="sim_achat")
        quantite_sim = st.number_input("🔢 Quantité", step=1, min_value=1, key="sim_qte")
    with col2:
        pays_sim = st.selectbox("🌍 Pays", ["France", "USA", "UK", "Allemagne", "Canada", "Autre"], key="sim_pays")
        duree_sim = 0
        if pays_sim == "USA":
            duree_sim = st.number_input("⏳ Durée de détention (années)", step=0.1, min_value=0.0, key="sim_duree")
    
    if st.button("🧮 Simuler la vente"):
        try:
            info = yf.Ticker(ticker_sim).history(period="1d")
            if info.empty:
                st.warning("⚠️ Impossible de récupérer le prix actuel.")
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
                st.success(f"""
                🔍 **{ticker_sim}**
                💶 Prix actuel : {prix_actuel:.2f} EUR
                📈 Plus-value estimée : {plus_value:.2f} EUR
                📌 Taux d'imposition : {taux*100:.1f}%
                💰 Impôt estimé : {impot:.2f} EUR
                ✅ Net après impôt : {net:.2f} EUR
                """)
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# === Page Rapports ===
elif page == "Rapports":
    st.header("📈 Plus-values par Pays")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        if "Prix Vente" in df.columns and "Prix Achat" in df.columns and "Quantité" in df.columns and "Pays" in df.columns:
            df["Plus-Value"] = (df["Prix Vente"] - df["Prix Achat"]) * df["Quantité"]
            pv_par_pays = df.groupby("Pays")["Plus-Value"].sum()
            st.bar_chart(pv_par_pays)
        else:
            st.warning("❌ Données incomplètes pour générer le graphique.")
    else:
        st.info("Aucune transaction enregistrée.")
    
    st.header("📋 Bilan Fiscal Global")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        total_pv = ((df["Prix Vente"] - df["Prix Achat"]) * df["Quantité"]).sum()
        total_impot = df["Impôt"].sum()
        nb_transactions = len(df)
        pays_plus_rentable = df.groupby("Pays").apply(lambda d: ((d["Prix Vente"] - d["Prix Achat"]) * d["Quantité"]).sum()).idxmax()
        st.success(f"""
        💰 **Plus-value totale** : {total_pv:.2f} EUR
        📌 **Impôt total estimé** : {total_impot:.2f} EUR
        📦 **Nombre de transactions** : {nb_transactions}
        🥇 **Pays le plus rentable** : {pays_plus_rentable}
        """)
    else:
        st.info("Aucune transaction enregistrée.")
    
    st.header("📊 RSI / MACD")
    ticker_chart = st.text_input("Ticker pour indicateurs", value="AAPL")
    if st.button("Afficher RSI / MACD"):
        try:
            df = yf.download(ticker_chart, start="2023-01-01", end=str(date.today()))
            if df.empty or 'Close' not in df.columns:
                st.warning("⚠️ Données indisponibles.")
            else:
                df = calculer_indicateurs_avances(df)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
                fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal MACD"))
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", yaxis="y2"))
                fig.update_layout(
                    title="RSI et MACD",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="MACD"),
                    yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
                    legend=dict(x=0, y=1.15, orientation="h")
                )
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"❌ Erreur graphique : {e}")
    
    st.header("📬 Envoi du rapport fiscal par e-mail")
    email_client = st.text_input("Adresse e-mail du client", key="email_client")
    if st.button("📧 Envoyer le rapport par e-mail"):
        if not email_client:
            st.warning("⚠️ Merci de saisir l’adresse e-mail du client.")
        else:
            try:
                transactions = st.session_state.get("transactions", [])
                contenu = "📊 *Résumé FiscalTrade
