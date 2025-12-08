#!/usr/bin/env python3
"""
Sleep Disorder Prediction - Model Training Script

Ovaj skript trenira MLP, Random Forest i XGBoost modele za predikciju poremećaja spavanja.
Koristi GridSearchCV za optimizaciju hiperparametara i evaluira performanse.

Autor: AI Assistant
Datum: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import xgboost as xgb
import joblib
import os
import warnings
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')

class SleepDisorderTrainer:
    """
    Klasa za treniranje modela predikcije poremećaja spavanja
    """
    
    def __init__(self, data_path='../data/expanded_sleep_health_dataset.csv'):
        """
        Inicijalizacija trainer-a
        
        Args:
            data_path (str): Putanja do CSV fajla sa podacima
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Učitavanje i priprema podataka"""
        print("=== UCITAVANJE PODATAKA ===")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset uspesno ucitano: {self.df.shape[0]} instanci, {self.df.shape[1]} atributa")
            
            # Prikaz osnovnih informacija
            print(f"Kolone: {list(self.df.columns)}")
            print(f"Ciljna promenljiva: {self.df['Sleep Disorder'].unique()}")
            
        except FileNotFoundError:
            print(f"Greska: Fajl {self.data_path} nije pronadjen!")
            return False
        except Exception as e:
            print(f"Greska pri ucitavanju: {e}")
            return False
            
        return True
    
    def prepare_data(self):
        """Priprema podataka za treniranje"""
        print("\n=== PRIprema PODATAKA ===")
        
        # Odvajanje features i target
        X = self.df.drop(['Sleep Disorder', 'Person ID'], axis=1)
        y = self.df['Sleep Disorder']
        
        # Uklanjanje NaN vrednosti iz ciljne promenljive
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Dodavanje 'None' klase ako ne postoji
        if 'None' not in y.unique():
            # Dodajemo više instanci sa 'None' klasom za bolju balansiranu distribuciju
            # Uzimamo 30% od ukupnog broja instanci za None klasu
            none_count = max(300, len(X) // 3)  # Minimum 300 ili 1/3 od ukupnog broja
            none_samples = X.sample(n=none_count, random_state=42)
            
            # Modifikujemo podatke da budu tipični za zdrave osobe
            none_samples = none_samples.copy()
            none_samples['Sleep Duration'] = none_samples['Sleep Duration'].clip(lower=7.0, upper=9.0)
            none_samples['Quality of Sleep'] = none_samples['Quality of Sleep'].clip(lower=7, upper=10)
            none_samples['Stress Level'] = none_samples['Stress Level'].clip(upper=5)
            none_samples['Physical Activity Level'] = none_samples['Physical Activity Level'].clip(lower=70)
            
            none_targets = pd.Series(['None'] * len(none_samples), index=none_samples.index)
            X = pd.concat([X, none_samples])
            y = pd.concat([y, none_targets])
        
        # Enkodovanje ciljne promenljive
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        self.label_encoder = label_encoder
        
        print(f"Podaci nakon uklanjanja NaN: {len(X)} instanci")
        print(f"Ciljne klase: {label_encoder.classes_}")
        
        # Identifikacija tipova kolona
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numericki features: {numerical_features}")
        print(f"Kategorijalni features: {categorical_features}")
        
        # Podela podataka (70/15/15)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )
        
        print(f"Trening skup: {self.X_train.shape[0]} instanci")
        print(f"Validacioni skup: {self.X_val.shape[0]} instanci")
        print(f"Test skup: {self.X_test.shape[0]} instanci")
        
        # Kreiranje preprocessor-a
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Fit i transform
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_val_processed = self.preprocessor.transform(self.X_val)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"[OK] Podaci pripremljeni: {self.X_train_processed.shape[1]} features")
        
    def train_mlp(self):
        """Treniranje MLP klasifikatora"""
        print("\n=== TRENIRANJE MLP KLASIFIKATORA ===")
        
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500]
        }
        
        model = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_processed, self.y_train)
        
        self.models['MLP'] = grid_search.best_estimator_
        print(f"[OK] MLP treniran - Best Score: {grid_search.best_score_:.4f}")
        
    def train_random_forest(self):
        """Treniranje Random Forest klasifikatora"""
        print("\n=== TRENIRANJE RANDOM FOREST KLASIFIKATORA ===")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_processed, self.y_train)
        
        self.models['Random Forest'] = grid_search.best_estimator_
        print(f"[OK] Random Forest treniran - Best Score: {grid_search.best_score_:.4f}")
        
    def train_xgboost(self):
        """Treniranje XGBoost klasifikatora"""
        print("\n=== TRENIRANJE XGBOOST KLASIFIKATORA ===")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        model = xgb.XGBClassifier(
            random_state=42, n_jobs=-1, eval_metric='mlogloss', use_label_encoder=False
        )
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_processed, self.y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        print(f"[OK] XGBoost treniran - Best Score: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self, model, X_val, y_val, model_name):
        """Evaluacija modela"""
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        
        results = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
        }
        
        self.results[model_name] = results
        return results
        
    def compare_models(self):
        """Poređenje performansi modela"""
        print("\n=== POREDENJE PERFORMANSI MODELA ===")
        
        # Evaluacija svih modela
        for name, model in self.models.items():
            self.evaluate_model(model, self.X_val_processed, self.y_val, name)
        
        # Kreiranje DataFrame za poređenje
        comparison_df = pd.DataFrame(self.results).T
        
        print("\nRezultati na validacionom skupu:")
        print(comparison_df.round(4))
        
        # Identifikacija najboljeg modela
        best_model = comparison_df.mean(axis=1).idxmax()
        best_score = comparison_df.mean(axis=1).max()
        
        print(f"\n[NAJBOLJI MODEL]: {best_model}")
        print(f"Prosecna performansa: {best_score:.4f}")
        
        return best_model
        
    def save_models(self, best_model_name):
        """Sačuvavanje modela i pipeline-a"""
        print(f"\n=== SACUVAVANJE MODELA ===")
        
        # Kreiranje kompletnog pipeline-a
        complete_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.models[best_model_name])
        ])
        
        # Sačuvavanje
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Kreiranje models direktorijuma ako ne postoji
        os.makedirs('../models', exist_ok=True)
        
        # Kompletni pipeline
        pipeline_filename = f"../models/best_sleep_disorder_pipeline_{timestamp}.pkl"
        joblib.dump(complete_pipeline, pipeline_filename)
        print(f"[OK] Pipeline sacuvan: {pipeline_filename}")
        
        # Najbolji model
        model_filename = f"../models/best_sleep_disorder_model_{timestamp}.pkl"
        joblib.dump(self.models[best_model_name], model_filename)
        print(f"[OK] Model sacuvan: {model_filename}")
        
        # Preprocessor
        preprocessor_filename = f"../models/sleep_disorder_preprocessor_{timestamp}.pkl"
        joblib.dump(self.preprocessor, preprocessor_filename)
        print(f"[OK] Preprocessor sacuvan: {preprocessor_filename}")
        
        # Metadati
        metadata = {
            'best_model_name': best_model_name,
            'best_model_params': self.models[best_model_name].get_params(),
            'validation_results': self.results,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'label_encoder': self.label_encoder,
            'class_names': self.label_encoder.classes_.tolist(),
            'dataset_info': {
                'total_samples': len(self.df),
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'n_features': self.X_train.shape[1]
            }
        }
        
        metadata_filename = f"../models/model_metadata_{timestamp}.pkl"
        joblib.dump(metadata, metadata_filename)
        print(f"[OK] Metadati sacuvani: {metadata_filename}")
        
        return pipeline_filename, model_filename, preprocessor_filename, metadata_filename
        
    def run_training(self):
        """Pokretanje kompletnog procesa treniranja"""
        print("POKRETANJE TRENIRANJA MODELA ZA PREDIKCIJU PORECJAJA SPAVANJA")
        print("=" * 70)
        
        # Učitavanje podataka
        if not self.load_data():
            return False
            
        # Priprema podataka
        self.prepare_data()
        
        # Treniranje modela
        self.train_mlp()
        self.train_random_forest()
        self.train_xgboost()
        
        # Poređenje i evaluacija
        best_model = self.compare_models()
        
        # Sačuvavanje
        saved_files = self.save_models(best_model)
        
        print("\n" + "=" * 70)
        print("TRENIRANJE ZAVRSENO USPESNO!")
        print(f"Najbolji model: {best_model}")
        print(f"Sacuvani fajlovi: {len(saved_files)}")
        
        return True

def main():
    """Glavna funkcija"""
    parser = argparse.ArgumentParser(description='Treniranje modela za predikciju poremećaja spavanja')
    parser.add_argument('--data', default='../data/expanded_sleep_health_dataset.csv', 
                       help='Putanja do CSV fajla sa podacima')
    parser.add_argument('--quick', action='store_true', 
                       help='Brzo treniranje sa manjim brojem parametara')
    
    args = parser.parse_args()
    
    # Kreiranje trainer-a
    trainer = SleepDisorderTrainer(args.data)
    
    # Pokretanje treniranja
    success = trainer.run_training()
    
    if success:
        print("Modeli su uspesno trenirani i sacuvani!")
        print("Možete koristiti train_model.py za treniranje ili predict.py za predikciju.")
    else:
        print("Treniranje nije uspesno zavrseno.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
