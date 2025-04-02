import yfinance as yf
import pandas as pd

def get_stock_price(ticker):
    """Récupère le prix actuel d'une action."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="1d")

    if df.empty:
        return f"❌ Erreur : Impossible de récupérer le prix pour {ticker}. Vérifie le symbole."
    
    price = df["Close"].iloc[-1]
    return f"Prix actuel de {ticker} : {price:.2f} €"

def get_stock_price_value(ticker):
    """Renvoie juste le prix actuel brut (float) sans texte."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="1d")
    if df.empty:
        return 0.0
    return df["Close"].iloc[-1]
  
  
def analyze_trend_custom(ticker,periode):
    """Analyse la tendance en fonction de la periode choisie."""
    configs = {
       "1 jour" : {"period" : "1d", "sma1" :5 , "sma2" :20}, 
       "1 mois" : {"period" : "1mo", "sma1" :5 , "sma2" :20},
       "6 mois" : {"period" : "6mo", "sma1" :20 , "sma2" :50}, 
       "1 an" : {"period" : "1y", "sma1" :50 , "sma2" :200},
       "5 ans" : {"period" : "5y", "sma1" :100 , "sma2" :200},
    }

    config = configs.get(periode)
    if not config : 
        return "❌ Période invalide."

    df= yf.Ticker(ticker).history(period=config["period"])
    if df.empty: 
      return f"❌ Pas de données disponibles pour {ticker} sur {periode}."

    sma1 = config["sma1"]
    sma2 = config["sma2"]

    df[f"SMA_{sma1}"] = df["Close"].rolling(window=sma1).mean()
    df[f"SMA_{sma2}"] = df["Close"].rolling(window=sma2).mean()

    s1=df[f"SMA_{sma1}"].iloc[-1]
    s2=df[f"SMA_{sma2}"].iloc[-1]
    
    if pd.isna(s1) or pd.isna(s2): 
       return"❓ Données insuffisantes pour l’analyse."

    if s1 > s2: 
     return f"📈 Tendance HAUSSIÈRE ({sma1} > {sma2}) sur {periode}"
    else:
     return f"📉 Tendance BAISSIÈRE ({sma1} < {sma2}) sur {periode}" 
import yfinance as yf

def get_price_history(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df
 

if __name__ =="__main__":
    print(get_stock_price("AAPL"))
    print(analyze_trend_custom("AAPL", "1 jour"))
