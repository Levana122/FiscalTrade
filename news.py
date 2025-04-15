# news.py

import requests

API_KEY = "wAh7fdgSKR1GqtOrn10PX3LMGU9GNEYvXgdujskD"

def get_news(ticker, language="fr", limit=5):
    url = f"https://api.marketaux.com/v1/news/all?symbols={ticker}&language={language}&api_token={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if "data" in data and data["data"]:
            return data["data"][:limit]
        else:
            return []
    except Exception as e:
        return [{"title": "Erreur de récupération", "description": str(e), "url": "#", "published_at": ""}]
