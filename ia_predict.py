import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def predire_tendance(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    if len(data) < 10:
        return "Pas assez de données pour prédire."

    data["variation"] = data["Close"].pct_change()
    data["target"] = (data["variation"] > 0).astype(int)
    data = data.dropna()

    if len(data) < 10:
        return "Pas assez de données après nettoyage."

    X = data[["Open", "High", "Low", "Volume"]]
    y = data["target"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X[:-1], y[:-1])
    prediction = model.predict([X.iloc[-1]])[0]

    return "📈 Prédiction : Tendance Haussière" if prediction == 1 else "📉 Prédiction : Tendance Baissière"

def charger_donnees(ticker="AAPL", periode="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=periode)

    # Calculer les indicateurs simples
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["Volatilité"] = df["Close"].rolling(window=5).std()

    # Variable cible : est-ce que ça monte demain ?
    df["Target"] = df["Close"].shift(-1) > df["Close"]
    df["Target"] = df["Target"].astype(int)

    # Supprimer les lignes incomplètes
    df.dropna(inplace=True)

    return df

def entrainer_modele(df):
    features = ["Return", "SMA_5", "SMA_10", "Volatilité"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    return model, precision

if __name__ == "__main__":
    df = charger_donnees("AAPL")
    modele, precision = entrainer_modele(df)
    print(f"Modèle entraîné avec une précision de : {precision:.2%}")
