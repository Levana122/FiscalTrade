import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Pour sauvegarder/charger le modèle
import warnings
warnings.filterwarnings('ignore')  # Supprimer les avertissements inutiles

def calculer_indicateurs(df):
    """
    Calcule des indicateurs techniques supplémentaires pour améliorer les features.
    """
    # Indicateurs existants
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["Volatilité"] = df["Close"].rolling(window=5).std()
    
    # Nouveaux indicateurs
    df["EMA_12"] = df["Close"].ewm(span=12).mean()  # Moyenne exponentielle
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]  # MACD
    df["Signal_Line"] = df["MACD"].ewm(span=9).mean()  # Ligne de signal
    df["RSI"] = calculer_rsi(df["Close"], window=14)  # RSI
    df["Bollinger_Upper"] = df["SMA_20"] + 2 * df["Close"].rolling(window=20).std()
    df["Bollinger_Lower"] = df["SMA_20"] - 2 * df["Close"].rolling(window=20).std()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()  # Pour Bollinger
    
    return df

def calculer_rsi(series, window=14):
    """
    Calcule l'Indice de Force Relative (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def charger_donnees(ticker="AAPL", periode="1y", interval="1d"):
    """
    Charge les données historiques et calcule les indicateurs.
    Améliorations : Gestion d'erreurs, plus de données, indicateurs avancés.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=periode, interval=interval)
        if df.empty or len(df) < 50:  # Augmenter le seuil pour plus de robustesse
            return None, "Pas assez de données pour le ticker spécifié."
        
        df = calculer_indicateurs(df)
        
        # Variable cible : est-ce que ça monte demain ? (avec une légère modification pour éviter le lookahead)
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        
        # Supprimer les lignes incomplètes
        df.dropna(inplace=True)
        
        if len(df) < 30:
            return None, "Pas assez de données après calcul des indicateurs."
        
        return df, None
    except Exception as e:
        return None, f"Erreur lors du chargement des données : {str(e)}"

def entrainer_modele(df, optimiser=False):
    """
    Entraîne le modèle avec validation croisée temporelle et optimisation optionnelle.
    Améliorations : Validation croisée, GridSearch pour hyperparamètres, métriques supplémentaires.
    """
    features = ["Return", "SMA_5", "SMA_10", "Volatilité", "EMA_12", "EMA_26", "MACD", "Signal_Line", "RSI", "Bollinger_Upper", "Bollinger_Lower"]
    X = df[features]
    y = df["Target"]
    
    # Utiliser TimeSeriesSplit pour respecter l'ordre temporel
    tscv = TimeSeriesSplit(n_splits=5)
    
    if optimiser:
        # Optimisation des hyperparamètres
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=tscv, scoring='accuracy')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Validation croisée pour évaluer la précision
    accuracies = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    precision_moyenne = np.mean(accuracies)
    
    # Entraîner sur tout l'ensemble pour le modèle final
    model.fit(X, y)
    
    return model, precision_moyenne

def predire_tendance(ticker, model, periode="6mo"):
    """
    Prédit la tendance pour le prochain jour basé sur les données récentes.
    Améliorations : Utilise un modèle pré-entraîné, gestion d'erreurs.
    """
    df, erreur = charger_donnees(ticker, periode=periode)
    if erreur:
        return erreur
    
    features = ["Return", "SMA_5", "SMA_10", "Volatilité", "EMA_12", "EMA_26", "MACD", "Signal_Line", "RSI", "Bollinger_Upper", "Bollinger_Lower"]
    X_latest = df[features].iloc[-1:].values  # Dernière ligne pour prédiction
    
    try:
        prediction = model.predict(X_latest)[0]
        proba = model.predict_proba(X_latest)[0]
        tendance = "📈 Prédiction : Tendance Haussière" if prediction == 1 else "📉 Prédiction : Tendance Baissière"
        confiance = f" (Confiance : {max(proba)*100:.1f}%)"
        return tendance + confiance
    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"

def visualiser_predictions(df, model):
    """
    Visualise les prédictions vs réalité.
    """
    features = ["Return", "SMA_5", "SMA_10", "Volatilité", "EMA_12", "EMA_26", "MACD", "Signal_Line", "RSI", "Bollinger_Upper", "Bollinger_Lower"]
    X = df[features]
    y_pred = model.predict(X)
    
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Close"], label="Prix de clôture")
    plt.scatter(df.index[y_pred == 1], df["Close"][y_pred == 1], color='green', label='Prédiction Haussière', marker='^')
    plt.scatter(df.index[y_pred == 0], df["Close"][y_pred == 0], color='red', label='Prédiction Baissière', marker='v')
    plt.title("Prédictions de tendance vs Prix réel")
    plt.legend()
    plt.show()

def sauvegarder_modele(model, filename="modele_bourse.pkl"):
    """
    Sauvegarde le modèle entraîné.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")

def charger_modele(filename="modele_bourse.pkl"):
    """
    Charge un modèle sauvegardé.
    """
    try:
        model = joblib.load(filename)
        print(f"Modèle chargé depuis {filename}")
        return model
    except FileNotFoundError:
        print("Modèle non trouvé, entraînez-le d'abord.")
        return None

if __name__ == "__main__":
    ticker = "AAPL"
    
    # Charger et entraîner le modèle
    df, erreur = charger_donnees(ticker, periode="2y")  # Plus de données pour un meilleur entraînement
    if erreur:
        print(erreur)
        exit()
    
    modele, precision = entrainer_modele(df, optimiser=True)  # Activer l'optimisation
    print(f"Modèle entraîné avec une précision moyenne de : {precision:.2%}")
    
    # Sauvegarder le modèle
    sauvegarder_modele(modele)
    
    # Prédire la tendance
    prediction = predire_tendance(ticker, modele)
    print(prediction)
    
    # Visualiser (optionnel)
    # visualiser_predictions(df, modele)


