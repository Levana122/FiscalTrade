
import streamlit as st
import pandas as pd
import yfinance as yf
from fpdf import FPDF
import plotly.graph_objects as go
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from datetime import date
import os
import numpy as np
import pandas as pd
import requests
import threading
import os
import time
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from test_yfinance import get_price_history
from test_yfinance import get_stock_price, analyze_trend_custom, get_stock_price_value
from taux_fiscal import get_taux_imposition, calcul_impot_usa, calcul_impot_uk
from fpdf import FPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.message import EmailMessage
import ssl
import ia_predict
from ia_predict import predire_tendance
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
npNaN = np.nan
from datetime import datetime
import json
import openpyxl
from openpyxl.styles import Font
# 📌 Initialisation de l'historique des transactions
historique = []

st.set_page_config(page_title="FiscalTrade", layout="wide")
st.title("💼 FiscalTrade - App Complète")

if "transactions" not in st.session_state:
    st.session_state.transactions = []

# Analyse du Marché
st.header("📈 Analyse du Marché")
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
            tendance = "🔼 Hausse" if variation > 0 else ("🔽 Baisse" if variation < 0 else "Stable")
            st.success(f"{ticker} de {date_debut} à {date_fin}")
            st.info("Prix initial : {:.2f} $ | Final : {:.2f} $ | Variation : {:.2f}% | {}".format(
                prix_debut, prix_fin, variation, tendance))
            st.line_chart(data['Close'])
        else:
            st.warning("Aucune donnée.")
    except Exception as e:
        st.error(f"Erreur analyse : {e}")

# Calcul d'impôt
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
            "Symbole": ticker,  # 🟢 C’est ça qui manquait
            "Pays": pays,
            "Prix Achat": achat,
            "Prix Vente": vente,
            "Quantité": quantite,
            "Plus-value": plus_value,
            "Impôt": impot
        })
        ticker = st.text_input("Symbole (Ticker)", value="AAPL")


    except Exception as e:
        st.error(f"Erreur de calcul : {e}")

# Export Excel
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

        # Vérification des colonnes minimales
        colonnes_attendues = ["Symbole", "Prix Achat", "Prix Vente", "Quantité", "Pays"]
        if not all(col in df.columns for col in colonnes_attendues):
            st.error(f"❌ Le fichier doit contenir les colonnes suivantes : {', '.join(colonnes_attendues)}")
        else:
            st.success("✅ Fichier CSV chargé avec succès !")
            st.dataframe(df)

            # Calcul de l'impôt pour chaque ligne
            taux_par_pays = {"France": 0.3, "USA": 0.2, "UK": 0.19, "Allemagne": 0.25, "Canada": 0.15, "Autre": 0.3}

            def calculer_impot(row):
                taux = taux_par_pays.get(row["Pays"], 0.3)
                plus_value = (row["Prix Vente"] - row["Prix Achat"]) * row["Quantité"]
                return round(plus_value * taux, 2)

            df["Impôt"] = df.apply(calculer_impot, axis=1)

            # Stocker dans la session
            if "transactions" not in st.session_state:
                st.session_state["transactions"] = []

            for _, ligne in df.iterrows():
                st.session_state["transactions"].append({
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

st.header("📈 Plus-values par Pays")

if "transactions" in st.session_state and st.session_state["transactions"]:
    df = pd.DataFrame(st.session_state["transactions"])
    if "Prix Vente" in df.columns and "Prix Achat" in df.columns and "Quantité" in df.columns and "Pays" in df.columns:
        df["Plus-Value"] = (df["Prix Vente"] - df["Prix Achat"]) * df["Quantité"]
        pv_par_pays = df.groupby("Pays")["Plus-Value"].sum()

        st.bar_chart(pv_par_pays)
    else:
        st.warning("❌ Données incomplètes pour générer le graphique.")
else:
    st.info("Aucune transaction enregistrée.")
st.header("📋 Bilan Fiscal Global")

if "transactions" in st.session_state and st.session_state["transactions"]:
    df = pd.DataFrame(st.session_state["transactions"])
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
        # 📈 Récupération du prix actuel
        info = yf.Ticker(ticker_sim).history(period="1d")
        if info.empty:
            st.warning("⚠️ Impossible de récupérer le prix actuel.")
        else:
            prix_actuel = float(info["Close"].iloc[-1])
            plus_value = (prix_actuel - prix_achat_sim) * quantite_sim

            # 💰 Calcul de l'impôt selon le pays
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




import streamlit as st
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === CONFIGURATION ===
EMAIL_EXPEDITEUR = "levanaestla@gmail.com"
MOT_DE_PASSE_APP = "tfegcpqgkzhpuahy"  # Ton mot de passe d'application Gmail

st.header("📬 Envoi du rapport fiscal par e-mail")

# === Champ utilisateur ===
email_client = st.text_input("Adresse e-mail du client", key="email_client")

# === Bouton d'envoi ===
if st.button("📧 Envoyer le rapport par e-mail"):
    if not email_client:
        st.warning("⚠️ Merci de saisir l’adresse e-mail du client.")
    else:
        try:
            # Transactions simulées si absentes
            transactions = st.session_state.get("transactions", [
                {"Symbole": "AAPL", "Prix Achat": 120.0, "Prix Vente": 150.0, "Quantité": 10, "Impôt": 90.0},
                {"Symbole": "TSLA", "Prix Achat": 200.0, "Prix Vente": 180.0, "Quantité": 5, "Impôt": 0.0},
            ])

            # === Préparer le corps de l’e-mail ===
            contenu = "📊 *Résumé FiscalTrade*\n\n"
            contenu += f"{'Symbole':<10} {'Achat (€)':<12} {'Vente (€)':<12} {'Qté':<6} {'Impôt (€)':<10}\n"
            contenu += "-" * 55 + "\n"

            total_impot = 0
            for t in transactions:
                contenu += f"{t.get('Symbole',''):<10} {t.get('Prix Achat', 0):<12.2f} {t.get('Prix Vente', 0):<12.2f} {t.get('Quantité', 0):<6} {t.get('Impôt', 0):<10.2f}\n"
                total_impot += t.get("Impôt", 0)

            contenu += "-" * 55 + "\n"
            contenu += f"{'Total Impôt estimé :':<40} {total_impot:.2f} €\n\n"
            contenu += "Merci d’avoir utilisé FiscalTrade.\nL’équipe FiscalTrade 📩"

            # === Création e-mail ===
            msg = MIMEMultipart()
            msg["From"] = EMAIL_EXPEDITEUR
            msg["To"] = email_client
            msg["Subject"] = "📄 Votre rapport fiscal - FiscalTrade"
            msg.attach(MIMEText(contenu, "plain", _charset="utf-8"))

            # === Envoi ===
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(EMAIL_EXPEDITEUR, MOT_DE_PASSE_APP)
                server.send_message(msg)

            st.success("✅ Rapport fiscal envoyé avec succès !")

        except Exception as e:
            st.error(f"❌ Erreur lors de l’envoi : {e}")



import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

st.header("📊 RSI / MACD")

if st.button("Afficher RSI / MACD"):
    try:
        df = yf.download(ticker, start="2023-01-01", end=str(date.today()))

        if df.empty or 'Close' not in df.columns:
            st.warning("⚠️ Données indisponibles ou colonne 'Close' manquante.")
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

            # Construction du graphique
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
        st.error(f"❌ Erreur graphique : {e}")

st.header("🧠 Conseil IA Global")

if st.button("🔍 Analyser mes positions avec IA"):
    if "transactions" not in st.session_state or not st.session_state["transactions"]:
        st.warning("Aucune donnée exploitable.")
    else:
        meilleurs_conseils = []
        for t in st.session_state["transactions"]:
            try:
                symbole = t["Symbole"]
                prix_achat = t["Prix Achat"]
                quantite = t["Quantité"]
                prix_actuel = get_stock_price_value(symbole)
                tendance = analyze_trend_custom(symbole, "1mo")
                plus_value = (prix_actuel - prix_achat) * quantite

                if plus_value < -500:
                    conseil = "🚨 Vendre rapidement - perte importante"
                elif plus_value > 1000:
                    conseil = "🌟 Considérer la vente - forte plus-value"
                elif "BAISSIÈRE" in tendance.upper():
                    conseil = "🔽 Attention à la tendance baissière"
                else:
                    conseil = "📊 Conserver - situation stable"

                meilleurs_conseils.append({
                    "Symbole": symbole,
                    "Prix Actuel": prix_actuel,
                    "Plus-Value": plus_value,
                    "Tendance": tendance,
                    "Conseil": conseil
                })

            except Exception as e:
                st.warning(f"Erreur pour {t.get('Symbole', '?')} : {e}")

        if not meilleurs_conseils:
            st.warning("Aucune donnée exploitable.")
        else:
            st.success("✅ Analyse IA terminée")
            for conseil in meilleurs_conseils:
                st.markdown(f"""
                ---
                **🔹 {conseil['Symbole']}**
                - Prix actuel : **{conseil['Prix Actuel']:.2f} €**
                - Plus-value estimée : **{conseil['Plus-Value']:.2f} €**
                - Tendance : **{conseil['Tendance']}**
                - 🧠 **Conseil IA** : {conseil['Conseil']}
                """)
