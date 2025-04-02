import json
import requests
import datetime
# 📌 Charger les taux locaux depuis le fichier JSON
with open("taux_fiscaux.json", "r", encoding="utf-8") as file:
    TAUX_FISCAUX = json.load(file)

def fetch_taux_fiscaux_api():
    """
    Récupère les taux d’imposition depuis une API externe.
    Retourne un dictionnaire vide si l’API échoue.
    """
    try:
        api_url = "https://api.taxdata.com/rates"  # Remplacer par la vraie URL
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}

import datetime

def get_taux_imposition(pays):
    """
    Renvoie le taux d’imposition selon l’API ou le fichier local,
    et vérifie la validité temporelle si disponible.
    """
    annee_actuelle = datetime.datetime.now().year

    # ✅ 1. API
    data_api = fetch_taux_fiscaux_api()
    if pays in data_api:
        pays_data_api = data_api[pays]
        if "taux" in pays_data_api:
            return pays_data_api["taux"]

    # ✅ 2. JSON local
    pays_data = TAUX_FISCAUX.get(pays)
    if pays_data:
        annee_valide = pays_data.get("annee_valide", annee_actuelle)
        if annee_valide >= annee_actuelle:
            return pays_data.get("taux_base", 0.30)
        else:
            print(f"📌 Taux fiscal pour {pays} non mis à jour (valide jusqu'à {annee_valide}).")
            return 0.30

    # ❌ 3. Aucun taux trouvé
    return 0.30


def calcul_impot_usa(plus_value, duree_detention):
    """
    Calcule l’impôt aux USA :
    - < 1 an → taux_max
    - ≥ 1 an → taux_base
    """
    usa_data = TAUX_FISCAUX.get("USA", {})
    
    if duree_detention < 1:
        taux = usa_data.get("taux_max", 0.37)
    else:
        taux = usa_data.get("taux_base", 0.20)

    impot = plus_value * taux
    return impot, taux

def calcul_impot_uk(plus_value):
    """
    Calcule l’impôt au Royaume-Uni en fonction d’un seuil :
    - ≤ seuil : taux_base
    - > seuil : taux_max
    """
    uk_data = TAUX_FISCAUX.get("UK", {})
    seuil = uk_data.get("seuil", 50000)
    taux_base = uk_data.get("taux_base", 0.10)
    taux_max = uk_data.get("taux_max", 0.20)

    if plus_value <= seuil:
        taux = taux_base
    else:
        taux = taux_max

    impot = plus_value * taux
    return impot, taux

# 📌 Bloc de test pour le terminal
if __name__ == "__main__":
    pays = input("Entrez votre pays : ")
    plus_value = float(input("Entrez votre plus-value (€) : "))

    if pays == "USA":
        duree = float(input("Durée de détention (en années) : "))
        impot, taux = calcul_impot_usa(plus_value, duree)
    elif pays == "UK":
        impot, taux = calcul_impot_uk(plus_value)
    else:
        taux = get_taux_imposition(pays)
        impot = plus_value * taux

    print(f"📌 Impôt à payer pour {pays} : {impot:.2f} € (taux appliqué : {taux*100:.0f}%)")
