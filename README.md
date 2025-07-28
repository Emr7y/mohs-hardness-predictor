# mohs-hardness-predictor
# Projekt: Mohs-Härte Vorhersage mittels Ensemble-Modell (CatBoost, LightGBM, XGBoost)

## 📈 Projektbeschreibung

Dieses Projekt verwendet ein Ensemble-Modell zur Vorhersage der **Mohs-Härte** von Materialien basierend auf verschiedenen physikalisch-chemischen Eigenschaften. Das Modell kombiniert drei starke Regressoren (CatBoost, LightGBM, XGBoost), um eine robuste und generalisierende Vorhersage zu erzielen.

Das Projekt wurde im Rahmen eines Kaggle-Wettbewerbs (Playground Series S3E25) umgesetzt und als Streamlit-App lokal sowie auf Hugging Face deployt.

---

## 🎓 Verwendete Features (Eingabedaten)

* allelectrons\_Total
* density\_Total
* allelectrons\_Average
* val\_e\_Average
* atomicweight\_Average
* ionenergy\_Average
* el\_neg\_chi\_Average
* R\_vdw\_element\_Average
* R\_cov\_element\_Average
* zratio\_Average
* density\_Average

---

## 🚀 Technologien

* Python 3.x
* Scikit-Learn
* CatBoost
* XGBoost
* LightGBM
* Streamlit
* Hugging Face Spaces
* Pandas, NumPy

---

## 💡 Modell-Ansatz

Ein einfaches Ensemble-Modell, das die Durchschnittswerte der drei Regressoren berechnet:

```python
ensemble_preds = (catboost.predict(X) + lightgbm.predict(X) + xgboost.predict(X)) / 3
```

Das finale Modell wurde als `ensemble_model.pkl` gespeichert und in die App geladen.

---

## 💥 Projektergebnisse (Kaggle)

**Private Score:** 0.64522
**Public Score:** 0.63183

Die Bewertung erfolgt mit RMSLE (Root Mean Squared Logarithmic Error), je niedriger desto besser.

---

## 🏠 Lokale Ausführung

```bash
streamlit run app_deutsch.py
```

Alternativ:

```bash
streamlit run app.py
```

---

## 🔧 Anforderungen (requirements.txt)

```txt
streamlit
pandas
scikit-learn
catboost
xgboost
lightgbm
```

---

## 📁 Projektstruktur

```
Mohs-Hardness-Prediction/
├── app.py
├── ensemble_model.pkl
├── requirements.txt
├── submission_ensemble.csv
├── README.md
```

---

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht.

---

## 🙌 Credits

Erstellt von: **Emr7y**
Yedi-Meister-Projekt. Mit viel Humor und neuronaler Kraft.

---

## 🔗 Online App (Hugging Face)

[https://huggingface.co/spaces/Emr7y/Mohs_Hardness](https://huggingface.co/spaces/Emr7y/Mohs_Hardness)

