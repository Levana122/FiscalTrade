# news.py

import requests
import streamlit as st

def get_news(ticker, language="fr", limit=5):
    api_key = st.secrets["marketaux"]["api_key"]  # 🔐 Récupération sécurisée
    url = f"https://api.marketaux.com/v1/news/all?symbols={ticker}&language={language}&api_token={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if "data" in data and data["data"]:
            return data["data"][:limit]
        else:
            return []
    except Exception as e:
        return [{
            "title": "Erreur de récupération",
            "description": str(e),
            "url": "#",
            "published_at": ""
        }]
