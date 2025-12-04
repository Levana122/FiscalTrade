#  FiscalTrade - Analyse & Fiscalité des Investissements

FiscalTrade est une application web intuitive permettant d’analyser des actions financières, d’évaluer les plus-values réalisées, et de calculer automatiquement l’impôt dû selon le pays et la durée de détention.  
Le tout, avec des graphiques, une alerte de prix, des outils d’IA et des exports PDF/Excel intégrés.

---

##  Fonctionnalités

-  **Analyse de performance** d’un titre boursier sur différentes périodes
-  **Calcul automatique de l’impôt** sur les plus-values selon le pays (France, USA, etc.)
-  **Visualisation graphique** des évolutions de prix, RSI/MACD
-  **Alerte de prix** : notification visuelle quand un seuil est dépassé
-  **Export PDF et Excel** des résultats
-  **Prédiction IA** de tendance
-  **Envoi de rapports par email** (en option)
-  **Conseil intelligent** automatisé

---

##  Technologies utilisées

- Python 3.12
- [Streamlit](https://streamlit.io) (interface web)
- Pandas, Matplotlib, yFinance
- fpdf (PDF), openpyxl (Excel)
- pandas-ta (indicateurs techniques)
- Optionnel : scikit-learn pour les modèles IA

---

##  Installation locale

```bash
git clone https://github.com/TonNom/FiscalTrade_Web.git
cd FiscalTrade_Web
pip install -r requirements.txt
streamlit run streamlit_app.py
