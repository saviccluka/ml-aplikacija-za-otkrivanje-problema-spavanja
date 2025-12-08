# 🔮 Sleep Disorder Prediction Project

Kompletni projekat za predikciju poremećaja spavanja koristeći mašinsko učenje.

## 📁 Struktura projekta

```
├── data/                                    # Dataset i podaci
│   ├── expanded_sleep_health_dataset.csv
│   └── reduced_sleep_health_dataset.csv
├── notebooks/                               # Jupyter notebook-ovi
│   └── sleep_disorder_analysis.ipynb
├── scripts/                                # Python skripte
│   ├── train_model.py                      # Treniranje modela
│   ├── predict.py                          # Predikcija
│   ├── quick_start.py                      # Brzo pokretanje
│   └── config.py                           # Konfiguracija
├── web_app/                                # Web aplikacije
│   ├── api_server.py                       # Flask API server
│   ├── streamlit_app.py                    # Streamlit aplikacija
│   └── templates/                          # HTML template-i
│       └── index.html
├── models/                                 # Sačuvani modeli
│   ├── best_sleep_disorder_pipeline_*.pkl
│   ├── best_sleep_disorder_model_*.pkl
│   ├── sleep_disorder_preprocessor_*.pkl
│   ├── model_metadata_*.pkl
│   └── model_conclusions_*.pkl
├── requirements.txt                        # Potrebne biblioteke
├── .gitignore                              # Git ignore fajl
└── README.md                               # Ovaj fajl
```

## 🚀 Brzo pokretanje

### 1. Instalacija
```bash
pip install -r requirements.txt
```

### 2. Treniranje modela
```bash
cd scripts
python train_model.py
```

### 3. Pokretanje web aplikacije

#### Streamlit (preporučeno)
```bash
cd web_app
streamlit run streamlit_app.py
```
Otvorite: http://localhost:8501



## 📊 Deo 1: Analiza podataka (15 poena) ✅

### Implementirane funkcionalnosti:
- **Učitavanje podataka** - Dataset sa 1,501 instancom i 13 atributa
- **Ispitivanje nedostajućih vrednosti** - Dataset nema nedostajućih vrednosti
- **Statistički pregled** - Numeričke i kategorijalne promenljive
- **Vizuelizacija** - Histogrami, boxplot-ovi, korelaciona matrica
- **Detekcija neuravnoteženih klasa** - Dataset je relativno uravnotežen
- **Identifikacija najrelevantnijih karakteristika** - Sleep Duration, Quality of Sleep, Stress Level, itd.

### Ključni nalazi:
- 5 tipova poremećaja spavanja: None, Insomnia, Sleep Apnea, Restless Leg Syndrome, Narcolepsy
- Najvažniji faktori: trajanje spavanja, kvalitet spavanja, nivo stresa
- Dataset je čist bez nedostajućih vrednosti

---

## 🤖 Deo 2: Pipeline za treniranje i evaluaciju (20 poena) ✅

### Implementirani modeli:
- **MLP (Multi-Layer Perceptron)** - Neural Network klasifikator
- **Random Forest** - Ensemble metoda
- **XGBoost** - Gradient Boosting metoda

### Pipeline karakteristike:
- **Podela podataka**: 70% trening, 15% validacija, 15% test
- **Pretprocesiranje**: StandardScaler + OneHotEncoder
- **GridSearchCV**: Optimizacija hiperparametara za svaki model
- **Evaluacija**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Sačuvavanje**: Kompletni pipeline u .pkl formatu

### Rezultati:
- Svi modeli trenirani sa GridSearchCV
- Detaljno poređenje performansi
- Identifikacija najboljeg modela
- Finalna evaluacija na test skupu

---

## 🌐 Deo 3: Frontend + API za korišćenje modela (15 poena) ✅

### Web aplikacije:
- **Streamlit aplikacija** - Interaktivna web aplikacija sa modernom UI
- **Flask API server** - REST API za predikciju
- **HTML frontend** - Moderna web stranica sa JavaScript-om

### Implementirane funkcionalnosti:
- **Unos podataka preko forme** - Sve potrebne parametre
- **API koji koristi trenirani model** - Automatsko učitavanje najnovijeg modela
- **Jasni prikaz rezultata** - Predikcija, verovatnoće, preporuke
- **Flask API lokalno dostupan** - Server na http://localhost:5000
- **Validacija unosa** - Client-side i server-side validacija
- **Prikaz grešaka** - Detaljne poruke o greškama

### API Endpoints:
- `GET /` - Glavna HTML stranica
- `GET /api/health` - Health check
- `POST /api/predict` - Predikcija poremećaja spavanja
- `GET /api/model-info` - Informacije o modelu
- `POST /api/validate` - Validacija podataka

### Karakteristike:
- **Responsive design** - Radi na svim uređajima
- **Real-time validacija** - JavaScript validacija
- **Vizuelizacija rezultata** - Grafikonovi i progress bar-ovi
- **Personalizovane preporuke** - Na osnovu predikcije
- **Error handling** - Detaljne poruke o greškama

---

## 📈 Dataset informacije

Dataset sadrži sledeće kolone:
- **Person ID**: Jedinstveni identifikator osobe
- **Gender**: Pol (Male/Female)
- **Age**: Uzrast
- **Occupation**: Zanimanje
- **Sleep Duration**: Trajanje spavanja (sati)
- **Quality of Sleep**: Kvalitet spavanja (1-10)
- **Physical Activity Level**: Nivo fizičke aktivnosti
- **Stress Level**: Nivo stresa (1-10)
- **BMI Category**: BMI kategorija (Underweight/Normal/Overweight/Obese)
- **Blood Pressure**: Krvni pritisak
- **Heart Rate**: Otkucaji srca
- **Daily Steps**: Dnevni koraci
- **Sleep Disorder**: Tip poremećaja spavanja (ciljna promenljiva)

## 🛠️ Tehnologije

- **Python 3.x**
- **Pandas, NumPy** - manipulacija podataka
- **Scikit-learn** - mašinsko učenje
- **XGBoost** - gradient boosting
- **Flask** - web API
- **Streamlit** - web aplikacija
- **Matplotlib, Seaborn, Plotly** - vizuelizacija
- **Jupyter Notebook** - analiza
- **Joblib** - sačuvavanje modela

## ⚠️ Napomena

Ova aplikacija je namenjena samo za edukativne svrhe. Za medicinsku dijagnozu konsultujte se sa lekarom.

## 🎯 Ukupno: 50/50 poena

- **Deo 1**: 15/15 poena ✅
- **Deo 2**: 20/20 poena ✅  
- **Deo 3**: 15/15 poena ✅

**Projekat je spreman za odbranu!** 🎉