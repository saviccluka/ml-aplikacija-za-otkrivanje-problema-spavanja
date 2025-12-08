#!/usr/bin/env python3
"""
Sleep Disorder Prediction - Configuration File

Konfiguracioni fajl sa parametrima za treniranje i predikciju modela.
"""

# Dataset konfiguracija
DATASET_CONFIG = {
    'data_path': 'expanded_sleep_health_dataset.csv',
    'target_column': 'Sleep Disorder',
    'id_column': 'Person ID',
    'test_size': 0.15,
    'val_size': 0.176,  # 15% od preostalih 85%
    'random_state': 42
}

# Model konfiguracija
MODEL_CONFIG = {
    'models': ['MLP', 'Random Forest', 'XGBoost'],
    'cv_folds': 5,
    'scoring': 'f1_weighted',
    'n_jobs': -1,
    'verbose': 1
}

# MLP hiperparametri
MLP_PARAMS = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000],
    'early_stopping': True,
    'validation_fraction': 0.1
}

# Random Forest hiperparametri
RF_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# XGBoost hiperparametri
XGB_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'eval_metric': 'mlogloss',
    'use_label_encoder': False
}

# Preprocessing konfiguracija
PREPROCESSING_CONFIG = {
    'numerical_scaler': 'StandardScaler',
    'categorical_encoder': 'OneHotEncoder',
    'drop_first': True,
    'sparse_output': False
}

# Evaluacija konfiguracija
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
    'average': 'weighted',
    'save_plots': True,
    'plot_size': (12, 8)
}

# Sačuvavanje konfiguracija
SAVE_CONFIG = {
    'save_pipeline': True,
    'save_model': True,
    'save_preprocessor': True,
    'save_metadata': True,
    'save_conclusions': True,
    'timestamp_format': '%Y%m%d_%H%M%S'
}

# Feature importance
FEATURE_NAMES = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category', 
    'Blood Pressure', 'Heart Rate', 'Daily Steps'
]

# Klasne nazive
CLASS_NAMES = [
    'None', 'Insomnia', 'Sleep Apnea', 
    'Restless Leg Syndrome', 'Narcolepsy'
]

# Preporuke za svaku klasu
RECOMMENDATIONS = {
    'None': [
        "✅ Nemate poremećaje spavanja",
        "Nastavite sa zdravim navikama",
        "Redovno vreme za spavanje",
        "Balansirana ishrana"
    ],
    'Insomnia': [
        "😴 Preporučuje se konsultacija sa lekarom",
        "Redovno vreme za spavanje",
        "Izbegavanje kofeina pre spavanja",
        "Tehnike relaksacije",
        "Izbegavanje ekrana pre spavanja"
    ],
    'Sleep Apnea': [
        "🫁 Hitna konsultacija sa lekarom",
        "Kontrola težine",
        "Izbegavanje alkohola",
        "Spavanje na boku",
        "CPAP terapija (ako je potrebno)"
    ],
    'Restless Leg Syndrome': [
        "🦵 Konsultacija sa neurologom",
        "Redovna fizička aktivnost",
        "Masiranje nogu",
        "Izbegavanje kofeina",
        "Topli tuševi"
    ],
    'Narcolepsy': [
        "😵 Hitna konsultacija sa neurologom",
        "Redovni raspored spavanja",
        "Izbegavanje vožnje",
        "Kratki odmorci tokom dana",
        "Medicinski tretman"
    ]
}

# Logging konfiguracija
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'sleep_disorder_training.log'
}

# Performance konfiguracija
PERFORMANCE_CONFIG = {
    'memory_limit': '4GB',
    'max_workers': -1,
    'chunk_size': 1000,
    'cache_size': 100
}
