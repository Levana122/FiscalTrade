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
import joblib
import warnings
warnings.filterwarnings('ignore')

from taux_fiscal import get_taux_imposition, calcul_impot_usa, calcul_impot_uk
from test_yfinance import get_price_history, get_stock_price, analyze_trend_custom, get_stock_price_value
from ia_predict import predire_tendance
from news import get_news

import openpyxl
from openpyxl.styles import Font

if "transactions" not in st.session_state:
    st.session_state.transactions = []

@st.cache_resource
def charger_modele_ia():
    try:
        model = joblib.load("modele_bourse.pkl")
        return model
    except FileNotFoundError:
        st.warning("Modele IA non trouve. Entrainez-le d'abord via le script separe.")
        return None

modele_ia = charger_modele_ia()

st.set_page_config(page_title="FiscalTrade", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller a :", ["Watchlist", "Analyse Marche", "Calcul Impot", "Export/Import", "Simulations", "Rapports", "IA & Actualites", "Gestion Portefeuille"])

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
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def calculer_impot_detaille(pays, plus_value, duree=0):
    if pays == "France":
        taux = get_taux_imposition(plus_value)  # Utilise la fonction importee
    elif pays == "USA":
        taux = calcul_impot_usa(plus_value, duree)
    elif pays == "UK":
        taux = calcul_impot_uk(plus_value)
    else:
        taux = 0.25
    return plus_value * taux

def exporter_pdf(df, filename="rapport_fiscal.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport Fiscal FiscalTrade", ln=True, align='C')
    pdf.ln(10)
    for index, row in df.iterrows():
        pdf.cell(200, 10, txt=f"Symbole: {row['Symbole']}, Plus-value: {row['Plus-value']:.2f}, Impot: {row['Impot']:.2f}", ln=True)
    pdf.output(filename)

if page == "Watchlist":
    st.header("Watchlist - Style MSN Finance")
    
    defaut_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = [{"ticker": t} for t in defaut_tickers]
    
    col1, col2 = st.columns([4, 1])
    with col1:
        new_ticker = st.text_input("Rechercher ou ajouter un ticker (ex: AAPL, BTC-USD)", key="watch_add")
    with col2:
        if st.button("Ajouter a la watchlist") and new_ticker:
            new_ticker = new_ticker.upper()
            if new_ticker not in [t['ticker'] for t in st.session_state.watchlist]:
                st.session_state.watchlist.append({"ticker": new_ticker})
                st.success(f"{new_ticker} ajoute a la watchlist")
            else:
                st.warning("Deja present dans la watchlist")
    
    st.markdown("""<style>
        .element-container:nth-child(n) > div > div {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem;
        }
    </style>""", unsafe_allow_html=True)
    
    if st.session_state.watchlist:
        for i, item in enumerate(st.session_state.watchlist):
            ticker = item['ticker']
            try:
                data = yf.Ticker(ticker).history(period="7d")
                if data.empty:
                    st.warning(f"Aucune donnee pour {ticker}")
                    continue
                
                prix = data['Close'].iloc[-1]
                prix_prec = data['Close'].iloc[-2] if len(data['Close']) > 1 else prix
                variation = ((prix - prix_prec) / prix_prec) * 100 if prix_prec else 0
                variation_txt = f"{variation:+.2f} %"
                couleur = "green" if variation >= 0 else "red"
                symbole = "Triangle up" if variation >= 0 else "Triangle down"
                
                col1, col2, col3, col4, col5 = st.columns([1, 2.5, 1, 1, 0.5])
                col1.markdown(f"**{ticker}**")
                col2.plotly_chart(plot_sparkline(data), use_container_width=True)
                col3.markdown(f"<span style='font-size: 14px;'>{prix:.2f} $</span>", unsafe_allow_html=True)
                col4.markdown(f"<span style='color:{couleur}; font-size: 14px;'>{symbole} {variation_txt}</span>", unsafe_allow_html=True)
                if col5.button("Supprimer", key=f"del_{i}"):
                    st.session_state.watchlist.pop(i)
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Erreur {ticker} : {e}")
    else:
        st.info("Aucun actif surveille.")

elif page == "Analyse Marche":
    st.header("Analyse du Marche")
    ticker = st.text_input("Ticker", value="AAPL")
    date_debut = st.date_input("Date d'achat", value=date(2024, 1, 1))
    date_fin = st.date_input("Date de vente", value=date.today())
    periode = st.selectbox("Periode pour indicateurs", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if st.button("Analyser"):
        try:
            data = yf.download(ticker, start=str(date_debut), end=str(date_fin), auto_adjust=False)
            if not data.empty:
                prix_debut = float(data['Close'].dropna().iloc[0].item())
                prix_fin = float(data['Close'].dropna().iloc[-1].item())
                variation = ((prix_fin - prix_debut) / prix_debut) * 100
                tendance = "Hausse" if variation > 0 else ("Baisse" if variation < 0 else "Stable")
                st.success(f"{ticker} de {date_debut} a {date_fin}")
                st.info("Prix initial : {:.2f} $ | Final : {:.2f} $ | Variation : {:.2f}% | {}".format(
                    prix_debut, prix_fin, variation, tendance))
                st.line_chart(data['Close'])
                
                if modele_ia:
                    pred = predire_tendance(ticker, modele_ia, periode=periode)
                    st.info(f"Prediction IA : {pred}")
                else:
                    st.warning("Modele IA non charge.")
                    
                # Graphique avec indicateurs
                data_ind = calculer_indicateurs_avances(data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind['Close'], name="Prix de cloture"))
                fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind['SMA_20'], name="SMA 20"))
                fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind['Bollinger_Upper'], name="Bollinger Upper", line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=data_ind.index, y=data_ind['Bollinger_Lower'], name="Bollinger Lower", line=dict(dash='dash')))
                st.plotly_chart(fig)
            else:
                st.warning("Aucune donnee.")
        except Exception as e:
            st.error(f"Erreur analyse : {e}")

elif page == "Calcul Impot":
    st.header("Calcul de l'Impot")
    col1, col2 = st.columns(2)
    with col1:
        pays = st.selectbox("Pays", ["France", "USA", "UK", "Allemagne", "Canada", "Autre"])
        achat = st.number_input("Prix d'achat", step=0.01)
        vente = st.number_input("Prix de vente", step=0.01)
    with col2:
        quantite = st.number_input("Quantite", step=1)
        duree = 0
        if pays == "USA":
            duree = st.number_input("Duree de detention (annees)", step=0.1)
    
    if st.button("Calculer l'impot"):
        try:
            plus_value = (vente - achat) * quantite
            impot = calculer_impot_detaille(pays, plus_value, duree)
            st.success(f"Plus-value : {plus_value:.2f} EUR | Impot : {impot:.2f} EUR")
            st.session_state.transactions.append({
                "Symbole": ticker if 'ticker' in locals() else "N/A",
                "Pays": pays,
                "Prix Achat": achat,
                "Prix Vente": vente,
                "Quantite": quantite,
                "Plus-value": plus_value,
                "Impot": impot
            })
        except Exception as e:
            st.error(f"Erreur de calcul : {e}")

elif page == "Export/Import":
    st.header("Export Excel")
    if st.button("Exporter Excel"):
        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)
            df.to_excel("transactions.xlsx", index=False)
            with open("transactions.xlsx", "rb") as f:
                st.download_button("Telecharger Excel", f, file_name="transactions.xlsx")
        else:
            st.warning("Aucune donnee.")
    
    st.header("Exporter PDF")
    if st.button("Exporter PDF"):
        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)
            exporter_pdf(df)
            with open("rapport_fiscal.pdf", "rb") as f:
                st.download_button("Telecharger PDF", f, file_name="rapport_fiscal.pdf")
        else:
            st.warning("Aucune donnee.")
    
    st.header("Importer un fichier CSV de transactions")
    fichier_csv = st.file_uploader("Choisis un fichier CSV", type="csv")
    if fichier_csv is not None:
        try:
            df = pd.read_csv(fichier_csv)
            colonnes_attendues = ["Symbole", "Prix Achat", "Prix Vente", "Quantite", "Pays"]
            if not all(col in df.columns for col in colonnes_attendues):
                st.error(f"Le fichier doit contenir les colonnes suivantes : {', '.join(colonnes_attendues)}")
            else:
                st.success("Fichier CSV charge avec succes !")
                st.dataframe(df)
                taux_par_pays = {"France": 0.3, "USA": 0.2, "UK": 0.19, "Allemagne": 0.25, "Canada": 0.15, "Autre": 0.3}
                def calculer_impot(row):
                    taux = taux_par_pays.get(row["Pays"], 0.3)
                    plus_value = (row["Prix Vente"] - row["Prix Achat"]) * row["Quantite"]
                    return round(plus_value * taux, 2)
                df["Impot"] = df.apply(calculer_impot, axis=1)
                for _, ligne in df.iterrows():
                    st.session_state.transactions.append({
                        "Symbole": ligne["Symbole"],
                        "Prix Achat": ligne["Prix Achat"],
                        "Prix Vente": ligne["Prix Vente"],
                        "Quantite": ligne["Quantite"],
                        "Pays": ligne["Pays"],
                        "Impot": ligne["Impot"]
                    })
                st.success("Transactions ajoutees a l'historique avec succes !")
        except Exception as e:
            st.error(f"Erreur lors de l'import : {e}")

elif page == "Simulations":
    st.header("Simulation de Vente")
    col1, col2 = st.columns(2)
    with col1:
        ticker_sim = st.text_input("Ticker", value="AAPL", key="sim_ticker")
        prix_achat_sim = st.number_input("Prix d'achat", step=0.01, key="sim_achat")
        quantite_sim = st.number_input("Quantite", step=1, min_value=1, key="sim_qte")
    with col2:
        pays_sim = st.selectbox("Pays", ["France", "USA", "UK", "Allemagne", "Canada", "Autre"], key="sim_pays")
        duree_sim = 0
        if pays_sim == "USA":
            duree_sim = st.number_input("Duree de detention (annees)", step=0.1, min_value=0.0, key="sim_duree")
    
    if st.button("Simuler la vente"):
        try:
            info = yf.Ticker(ticker_sim).history(period="1d")
            if info.empty:
                st.warning("Impossible de recuperer le prix actuel.")
            else:
                prix_actuel = float(info["Close"].iloc[-1])
                plus_value = (prix_actuel - prix_achat_sim) * quantite_sim
                impot = calculer_impot_detaille(pays_sim, plus_value, duree_sim)
                net = plus_value - impot
                st.success(f"""
                {ticker_sim}
                Prix actuel : {prix_actuel:.2f} EUR
                Plus-value estimee : {plus_value:.2f} EUR
                Taux d'imposition : {(impot / plus_value * 100 if plus_value != 0 else 0):.1f}%
                Impot estime : {impot:.2f} EUR
                Net apres impot : {net:.2f} EUR
                """)
        except Exception as e:
            st.error(f"Erreur : {e}")

elif page == "Rapports":
    st.header("Plus-values par Pays")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        if "Prix Vente" in df.columns and "Prix Achat" in df.columns and "Quantite" in df.columns and "Pays" in df.columns:
            df["Plus-Value"] = (df["Prix Vente"] - df["Prix Achat"]) * df["Quantite"]
            pv_par_pays = df.groupby("Pays")["Plus-Value"].sum()
            st.bar_chart(pv_par_pays)
        else:
            st.warning("Donnees incompletes pour generer le graphique.")
    else:
        st.info("Aucune transaction enregistree.")
    
    st.header("Bilan Fiscal Global")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        total_pv = ((df["Prix Vente"] - df["Prix Achat"]) * df["Quantite"]).sum()
        total_impot = df["Impot"].sum()
        nb_transactions = len(df)
        pays_plus_rentable = df.groupby("Pays").apply(lambda d: ((d["Prix Vente"] - d["Prix Achat"]) * d["Quantite"]).sum()).idxmax()
        st.success(f"""
        Plus-value totale : {total_pv:.2f} EUR
        Impot total estime : {total_impot:.2f} EUR
        Nombre de transactions : {nb_transactions}
        Pays le plus rentable : {pays_plus_rentable}
        """)
    else:
        st.info("Aucune transaction enregistree.")
    
    st.header("Indicateurs Techniques")
    ticker_chart = st.text_input("Ticker pour indicateurs", value="AAPL")
    indicateur = st.selectbox("Choisir indicateur", ["RSI", "MACD", "Bollinger Bands"])
    if st.button("Afficher Indicateur"):
        try:
            df = yf.download(ticker_chart, start="2023-01-01", end=str(date.today()))
            if df.empty or 'Close' not in df.columns:
                st.warning("Donnees indisponibles.")
            else:
                df = calculer_indicateurs_avances(df)
                fig = go.Figure()
                if indicateur == "RSI":
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
                    fig.update_yaxes(range=[0, 100])
                elif indicateur == "MACD":
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal MACD"))
                elif indicateur == "Bollinger Bands":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], name="Upper Band"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], name="Lower Band"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20"))
                    st.plotly_chart(fig)
        "published_after": start_date.strftime("%Y-%m-%d")
    }
    # Priority search: sector > keyword > symbol
    sector_queries = {
        "Crypto": "bitcoin OR crypto OR ethereum",
        "Technology": "Google OR Apple OR Microsoft OR AI OR Nvidia",
        "Energy": "oil OR gas OR energy OR Total",
        "Banks": "banks OR rates OR bonds OR BNP",
        "Health": "pharma OR health OR biotech",
        "Automotive": "Tesla OR cars OR batteries OR Ford",
        "Luxury": "LVMH OR Kering OR Hermès"
    }
    if sector and sector != "None":
        params["query"] = sector_queries.get(sector, "")
    elif keyword:
        params["query"] = keyword
    elif symbol:
        params["symbols"] = symbol
    try:
        response = requests.get("https://api.marketaux.com/v1/news/all", params=params)
        data = response.json()
        if "data" in data and data["data"]:
            return data["data"]
        else:
            return []
    except Exception as e:
        return [{
            "title": "Retrieval Error",
            "description": str(e),
            "url": "#",
            "published_at": ""
        }]
# Call
articles = fetch_news(symbol.upper(), keyword, sector, start_date)
# Display
if articles:
    for article in articles:
        st.markdown(f"### [{article['title']}]({article['url']})")
        st.write(f"Published: {article['published_at'][:10]}")
        st.write(article['description'][:300] + "...")
        st.markdown("---")
else:
    st.info("No news found for this filter.")
