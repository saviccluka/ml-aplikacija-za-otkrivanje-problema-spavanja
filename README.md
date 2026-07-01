Sleep Disorder Prediction Project

A complete project for predicting sleep disorders using machine learning.

About

This project analyzes sleep health data and trains machine learning models to predict sleep disorders (Insomnia, Sleep Apnea, Restless Leg Syndrome, Narcolepsy, or none), based on factors such as sleep duration, sleep quality, stress level, physical activity, and more. It includes a full pipeline — from data analysis and model training to a web app and REST API for making predictions.


Note: This application is intended for educational purposes only. For a medical diagnosis, please consult a doctor.



Project Structure

ml-aplikacija-za-otkrivanje-problema-spavanja/
└── Aplikacija za otkrivanje problema sa spavanjem/
    ├── data/                                  # Dataset
    │   ├── expanded_sleep_health_dataset.csv
    │   └── reduced_sleep_health_dataset.csv
    ├── notebooks/                             # Jupyter notebooks
    │   └── sleep_disorder_analysis.ipynb
    ├── scripts/                               # Python scripts
    │   ├── train_model.py                     # Model training
    │   ├── predict.py                         # Prediction
    │   ├── quick_start.py                     # Quick start
    │   └── config.py                          # Configuration
    ├── web_app/                               # Web applications
    │   ├── api_server.py                      # Flask API server
    │   ├── streamlit_app.py                   # Streamlit app
    │   └── templates/
    │       └── index.html                     # HTML frontend
    ├── models/                                # Saved models
    │   ├── best_sleep_disorder_pipeline_*.pkl
    │   ├── best_sleep_disorder_model_*.pkl
    │   ├── sleep_disorder_preprocessor_*.pkl
    │   └── model_metadata_*.pkl
    ├── docs/
    │   └── README.md
    └── requirements.txt

Getting Started

1. Installation

bashpip install -r requirements.txt

2. Train the model

bashcd scripts
python train_model.py

3. Run the web application

Streamlit (recommended)

bashcd web_app
streamlit run streamlit_app.py

Open: http://localhost:8501

Part 1: Data Analysis

Implemented functionality:


Data loading — dataset with 1,501 instances and 13 attributes
Missing value inspection — the dataset has no missing values
Statistical overview — numerical and categorical variables
Visualization — histograms, boxplots, correlation matrix
Class imbalance detection — the dataset is relatively balanced
Identification of the most relevant features


Key findings:


5 types of sleep disorders: None, Insomnia, Sleep Apnea, Restless Leg Syndrome, Narcolepsy
Most important factors: sleep duration, sleep quality, stress level
The dataset is clean with no missing values


Part 2: Training and Evaluation Pipeline

Implemented models:


MLP (Multi-Layer Perceptron) — neural network classifier
Random Forest — ensemble method
XGBoost — gradient boosting method


Pipeline characteristics:


Data split: 70% training, 15% validation, 15% test
Preprocessing: StandardScaler + OneHotEncoder
GridSearchCV: hyperparameter optimization for each model
Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Saved as a complete pipeline in .pkl format


Part 3: Frontend + API for Model Usage

Web applications:


Streamlit app — interactive web app with a modern UI
Flask API server — REST API for predictions
HTML frontend — modern web page with JavaScript


Implemented functionality:


Data input via a form covering all required parameters
API that uses the trained model, automatically loading the latest one
Clear display of results — prediction, probabilities, recommendations
Flask API available locally at http://localhost:5000
Client-side and server-side input validation
Detailed error messages


API Endpoints:

Method & RouteDescriptionGET /Main HTML pageGET /api/healthHealth checkPOST /api/predictSleep disorder predictionGET /api/model-infoModel informationPOST /api/validateData validation

Features:


Responsive design — works on all devices
Real-time JavaScript validation
Result visualization — charts and progress bars
Personalized recommendations based on the prediction
Detailed error handling


Dataset

The dataset contains the following columns:

ColumnDescriptionPerson IDUnique person identifierGenderMale/FemaleAgeAgeOccupationOccupationSleep DurationSleep duration (hours)Quality of SleepSleep quality (1–10)Physical Activity LevelPhysical activity levelStress LevelStress level (1–10)BMI CategoryUnderweight/Normal/Overweight/ObeseBlood PressureBlood pressureHeart RateHeart rateDaily StepsDaily stepsSleep DisorderSleep disorder type (target variable)

Tech Stack


Python 3.x
Pandas, NumPy — data manipulation
Scikit-learn — machine learning
XGBoost — gradient boosting
Flask — web API
Streamlit — web application
Matplotlib, Seaborn, Plotly — visualization
Jupyter Notebook — analysis
Joblib — model persistence


Author

Luka Savić
