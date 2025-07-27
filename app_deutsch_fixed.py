import streamlit as st
import joblib
import pandas as pd
from sklearn.base import BaseEstimator

# Dummy-Klasse, um das Ensemble-Modell beim Laden verfügbar zu machen
class SimpleAveragingEnsemble(BaseEstimator):
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return sum(preds) / len(preds)

st.title("🔬 Vorhersage der Mohs-Härte (deutsch)")
st.write("Dieses Tool nutzt ein Ensemble-Modell (CatBoost, LightGBM, XGBoost), um die Härte von Materialien vorherzusagen.")

# Eingabefelder in exakt der Reihenfolge, wie sie das Modell erwartet
data = {
    "allelectrons_Total": st.number_input("Gesamtelektronen (allelectrons_Total)", value=5.0, step=0.01),
    "density_Total": st.number_input("Dichte Gesamt (density_Total)", value=2.0, step=0.01),
    "allelectrons_Average": st.number_input("Durchschnittliche Elektronen (allelectrons_Average)", value=0.0, step=0.01),
    "val_e_Average": st.number_input("Valenzelektronen (val_e_Average)", value=0.0, step=0.01),
    "atomicweight_Average": st.number_input("Atomgewicht (atomicweight_Average)", value=0.0, step=0.01),
    "ionenergy_Average": st.number_input("Ionisierungsenergie (ionenergy_Average)", value=3.0, step=0.01),
    "el_neg_chi_Average": st.number_input("Elektronegativität (el_neg_chi_Average)", value=0.03, step=0.01),
    "R_vdw_element_Average": st.number_input("Van-der-Waals-Radius (R_vdw_element_Average)", value=0.09, step=0.01),
    "R_cov_element_Average": st.number_input("Kovalenzradius (R_cov_element_Average)", value=0.06, step=0.01),
    "zaratio_Average": st.number_input("Z-Verhältnis (zaratio_Average)", value=0.07, step=0.01),
    "density_Average": st.number_input("Durchschnittliche Dichte (density_Average)", value=0.05, step=0.01)
}

if st.button("📊 Vorhersage berechnen"):
    try:
        model = joblib.load("ensemble_model.pkl")
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        st.success(f"📌 Vorhergesagte Härte: {prediction:.2f} Mohs")
    except Exception as e:
        st.error(f"Fehler beim Vorhersagen: {e}")
