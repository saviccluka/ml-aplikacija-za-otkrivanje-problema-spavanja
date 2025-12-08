# Predikcija poremećaja spavanja na osnovu životnih navika

Ovaj projekat koristi mašinsko učenje za predikciju poremećaja spavanja na osnovu različitih životnih navika i zdravstvenih parametara.

## Opis projekta

Projekat analizira dataset sa 1,501 instancom koje sadrže informacije o:
- Demografskim karakteristikama (pol, uzrast, zanimanje)
- Parametrima spavanja (trajanje, kvalitet)
- Fizičkim aktivnostima i zdravstvenim pokazateljima
- Nivoima stresa i drugim faktorima

Cilj je trenirati modele koji mogu predvideti tip poremećaja spavanja (None, Insomnia, Sleep Apnea, Restless Leg Syndrome, Narcolepsy).

## Struktura projekta

```
├── expanded_sleep_health_dataset.csv                    # Dataset sa podacima
├── sleep_disorder_analysis.ipynb                       # Jupyter notebook za analizu i ML
├── train_model.py                                      # Python skripta za treniranje
├── predict.py                                          # Python skripta za predikciju
├── quick_start.py                                      # Brzo pokretanje
├── config.py                                           # Konfiguracioni fajl
├── api_server.py                                       # Flask API server
├── streamlit_app.py                                    # Streamlit web aplikacija
├── templates/                                          # HTML template-i za Flask
│   └── index.html                                      # Glavna HTML stranica
├── requirements.txt                                     # Potrebne biblioteke
├── best_sleep_disorder_pipeline_*.pkl                  # Sačuvani pipeline-i
├── best_sleep_disorder_model_*.pkl                     # Sačuvani modeli
├── sleep_disorder_preprocessor.pkl                      # Preprocessor
├── model_metadata.pkl                                   # Metadati modela
├── model_conclusions.pkl                               # Zaključci analize
└── README.md                                           # Ovaj fajl
```

## Dataset

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

## Analiza podataka (Deo 1)

Prvi deo projekta fokusira se na detaljnu analizu podataka:

### Ključni nalazi:
- Dataset sadrži 1,501 instanci bez nedostajućih vrednosti
- 5 različitih tipova poremećaja spavanja
- Dataset je relativno uravnotežen
- Najvažniji faktori: trajanje spavanja, kvalitet spavanja, nivo stresa

### Vizuelizacije:
- Histogrami numeričkih promenljivih
- Boxplot-ovi za detekciju autlajera
- Korelaciona matrica
- Analiza ciljne promenljive
- Uticaj faktora na poremećaje spavanja

## Tehnologije

- **Python 3.x**
- **Pandas** - manipulacija podataka
- **NumPy** - numeričke operacije
- **Scikit-learn** - mašinsko učenje
- **XGBoost** - gradient boosting
- **Matplotlib/Seaborn** - vizuelizacija
- **Plotly** - interaktivne vizuelizacije
- **Jupyter Notebook** - interaktivna analiza
- **Joblib** - sačuvavanje modela

## Instalacija

### Automatska instalacija
```bash
pip install -r requirements.txt
```

### Ručna instalacija
```bash
pip install pandas numpy matplotlib seaborn plotly jupyter scikit-learn xgboost joblib flask flask-cors streamlit
```

## Pokretanje

### Opcija 1: Web aplikacije (preporučeno)

#### Streamlit aplikacija (lakša za korišćenje)
```bash
# Pokretanje Streamlit aplikacije
streamlit run streamlit_app.py
```
Otvorite: http://localhost:8501

#### Flask API + HTML frontend
```bash
# Pokretanje Flask API servera
python api_server.py
```
Otvorite: http://localhost:5000

### Opcija 2: Brzo pokretanje (terminal)
```bash
# Provera biblioteka
python quick_start.py --check

# Treniranje modela
python quick_start.py --train

# Interaktivna predikcija
python quick_start.py --predict
```

### Opcija 3: Direktno pokretanje
```bash
# Treniranje modela
python train_model.py

# Predikcija
python predict.py --interactive

# Batch predikcija iz CSV
python predict.py --csv your_data.csv --output results.csv
```

### Opcija 4: Jupyter Notebook
```bash
jupyter notebook
# Otvorite sleep_disorder_analysis.ipynb
```

## Implementirani modeli

### Deo 1: Analiza podataka ✅
- Detaljna eksplorativna analiza podataka
- Vizuelizacija ključnih karakteristika
- Identifikacija najvažnijih faktora

### Deo 2: Pipeline za treniranje i evaluaciju ✅
- **MLP (Multi-Layer Perceptron)** - Neural Network klasifikator
- **Random Forest** - Ensemble metoda  
- **XGBoost** - Gradient Boosting metoda

#### Pipeline karakteristike:
- Podela podataka: 70% trening, 15% validacija, 15% test
- Pretprocesiranje: StandardScaler + OneHotEncoder
- GridSearchCV za optimizaciju hiperparametara
- Evaluacija sa metrikama: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Sačuvavanje kompletnog pipeline-a u .pkl formatu

### Deo 3: Frontend + API ✅
- **Streamlit aplikacija** - Interaktivna web aplikacija
- **Flask API server** - REST API za predikciju
- **HTML frontend** - Moderna web stranica
- **Validacija podataka** - Client-side i server-side
- **Error handling** - Detaljne poruke o greškama
- **Responsive design** - Radi na svim uređajima

#### Frontend karakteristike:
- Unos podataka preko forme
- Real-time validacija
- Vizuelizacija rezultata
- Preporuke na osnovu predikcije
- Loading indikatori
- Error handling

## API Endpoints

### Flask API dostupan na http://localhost:5000

- `GET /` - Glavna HTML stranica
- `GET /api/health` - Health check
- `POST /api/predict` - Predikcija poremećaja spavanja
- `GET /api/model-info` - Informacije o modelu
- `POST /api/validate` - Validacija podataka

## Autor

Projekat je kreiran za analizu poremećaja spavanja korišćenjem mašinskog učenja.
