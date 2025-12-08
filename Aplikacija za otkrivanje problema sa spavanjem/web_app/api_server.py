#!/usr/bin/env python3
"""
Sleep Disorder Prediction - Flask API Server

Flask API server za predikciju poremećaja spavanja.
Učitava trenirani model i pruža REST API endpoint-e.

Autor: AI Assistant
Datum: 2024
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import traceback

# Konfiguracija logging-a
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Omogućava CORS za frontend

class SleepDisorderAPI:
    """
    API klasa za predikciju poremećaja spavanja
    """
    
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        self.model_info = {}
        self.label_encoder = None
        self.class_names = []
        self.load_model()
    
    def load_model(self):
        """Učitavanje treniranog modela"""
        try:
            # Pokušaj pronalaženja najnovijeg pipeline-a
            models_dir = '../models' if os.path.exists('../models') else '.'
            pipeline_files = [f for f in os.listdir(models_dir) 
                            if f.startswith('best_sleep_disorder_pipeline_') and f.endswith('.pkl')]
            
            if not pipeline_files:
                logger.error("Nije pronađen nijedan pipeline fajl!")
                return False
            
            # Sortiranje po datumu kreiranja
            pipeline_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
            latest_pipeline = os.path.join(models_dir, pipeline_files[0])
            
            # Učitavanje pipeline-a
            self.pipeline = joblib.load(latest_pipeline)
            logger.info(f"Model uspešno učitan: {latest_pipeline}")
            
            # Učitavanje metadataka
            metadata_files = [f for f in os.listdir(models_dir) 
                            if f.startswith('model_metadata_') and f.endswith('.pkl')]
            if metadata_files:
                metadata_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                metadata = joblib.load(os.path.join(models_dir, metadata_files[0]))
                self.model_info = metadata
                
                # Učitavanje label_encoder-a i naziva klasa
                if 'label_encoder' in metadata:
                    self.label_encoder = metadata['label_encoder']
                    self.class_names = metadata.get('class_names', [])
                elif 'class_names' in metadata:
                    self.class_names = metadata['class_names']
                
                logger.info(f"Metadati učitani: {metadata.get('best_model_name', 'Unknown')}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Greška pri učitavanju modela: {e}")
            self.model_loaded = False
            return False
    
    def validate_input(self, data):
        """
        Validacija unetih podataka
        
        Args:
            data (dict): Podaci za validaciju
            
        Returns:
            tuple: (is_valid, errors)
        """
        errors = []
        
        # Obavezna polja
        required_fields = [
            'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
            'Physical Activity Level', 'Stress Level', 'BMI Category', 
            'Blood Pressure', 'Heart Rate', 'Daily Steps'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                errors.append(f"Polje '{field}' je obavezno")
        
        # Validacija tipova i opsega
        if 'Age' in data:
            try:
                age = int(data['Age'])
                if age < 0 or age > 120:
                    errors.append("Uzrast mora biti između 0 i 120 godina")
            except (ValueError, TypeError):
                errors.append("Uzrast mora biti broj")
        
        if 'Sleep Duration' in data:
            try:
                duration = float(data['Sleep Duration'])
                if duration < 0 or duration > 24:
                    errors.append("Trajanje spavanja mora biti između 0 i 24 sata")
            except (ValueError, TypeError):
                errors.append("Trajanje spavanja mora biti broj")
        
        if 'Quality of Sleep' in data:
            try:
                quality = int(data['Quality of Sleep'])
                if quality < 1 or quality > 10:
                    errors.append("Kvalitet spavanja mora biti između 1 i 10")
            except (ValueError, TypeError):
                errors.append("Kvalitet spavanja mora biti broj")
        
        if 'Stress Level' in data:
            try:
                stress = int(data['Stress Level'])
                if stress < 1 or stress > 10:
                    errors.append("Nivo stresa mora biti između 1 i 10")
            except (ValueError, TypeError):
                errors.append("Nivo stresa mora biti broj")
        
        if 'Physical Activity Level' in data:
            try:
                activity = int(data['Physical Activity Level'])
                if activity < 0:
                    errors.append("Nivo fizičke aktivnosti ne može biti negativan")
            except (ValueError, TypeError):
                errors.append("Nivo fizičke aktivnosti mora biti broj")
        
        if 'Heart Rate' in data:
            try:
                hr = int(data['Heart Rate'])
                if hr < 30 or hr > 200:
                    errors.append("Otkucaji srca moraju biti između 30 i 200")
            except (ValueError, TypeError):
                errors.append("Otkucaji srca moraju biti broj")
        
        if 'Daily Steps' in data:
            try:
                steps = int(data['Daily Steps'])
                if steps < 0:
                    errors.append("Dnevni koraci ne mogu biti negativni")
            except (ValueError, TypeError):
                errors.append("Dnevni koraci moraju biti broj")
        
        # Validacija kategorijalnih polja
        if 'Gender' in data and data['Gender'] not in ['Male', 'Female']:
            errors.append("Pol mora biti 'Male' ili 'Female'")
        
        if 'BMI Category' in data and data['BMI Category'] not in ['Underweight', 'Normal', 'Overweight', 'Obese']:
            errors.append("BMI kategorija mora biti 'Underweight', 'Normal', 'Overweight' ili 'Obese'")
        
        return len(errors) == 0, errors
    
    def predict(self, data):
        """
        Predikcija poremećaja spavanja
        
        Args:
            data (dict): Podaci za predikciju
            
        Returns:
            dict: Rezultat predikcije
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': 'Model nije učitan'
            }
        
        try:
            # Konvertovanje u DataFrame
            df = pd.DataFrame([data])
            
            # Predikcija
            prediction_numeric = self.pipeline.predict(df)[0]
            probabilities = self.pipeline.predict_proba(df)[0]
            
            # Konvertovanje numeričke predikcije u naziv klase
            if self.label_encoder:
                prediction = self.label_encoder.inverse_transform([prediction_numeric])[0]
                class_names = self.class_names
            else:
                prediction = str(prediction_numeric)
                class_names = [str(i) for i in range(len(probabilities))]
            
            # Kreiranje rezultata
            prob_dict = dict(zip(class_names, probabilities))
            
            # Sortiranje po verovatnoći
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Preporuke
            recommendations = self.get_recommendations(prediction)
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': max(probabilities),
                'probabilities': prob_dict,
                'sorted_probabilities': sorted_probs,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Greška pri predikciji: {e}")
            return {
                'success': False,
                'error': f'Greška pri predikciji: {str(e)}'
            }
    
    def get_recommendations(self, prediction):
        """Preporuke na osnovu predikcije"""
        recommendations = {
            'None': [
                "🎉 Odlično! Nemate poremećaje spavanja",
                "Nastavite sa trenutnim zdravim navikama",
                "Održavajte redovno vreme za spavanje (7-9 sati)",
                "Balansirana ishrana i redovna fizička aktivnost",
                "Izbegavajte stres i održavajte pozitivnu atmosferu"
            ],
            'Insomnia': [
                "😴 Konsultujte se sa lekarom ili spavaćim specijalistom",
                "Uspostavite redovni raspored spavanja (isti čas svaki dan)",
                "Izbegavajte kofein, alkohol i tešku hranu 4-6 sati pre spavanja",
                "Kreirajte opuštenu rutinu pre spavanja (čitanje, topla kupka)",
                "Izbegavajte ekrane (telefon, TV) 1 sat pre spavanja",
                "Koristite tamnu, hladnu i tišu spavaću sobu",
                "Probajte tehnike relaksacije (duboko disanje, meditacija)"
            ],
            'Sleep Apnea': [
                "🫁 HITNO konsultujte se sa lekarom - moguće ozbiljno stanje",
                "Kontrolišite težinu - gubitak kilograma može značajno pomoći",
                "Izbegavajte alkohol i sedative koji pogoršavaju simptome",
                "Spavajte na boku umesto na leđima",
                "Razmislite o CPAP terapiji (kontinuirani pozitivni pritisak)",
                "Izbegavajte pušenje koje pogoršava respiratorne probleme",
                "Redovno vežbajte ali ne previše blizu vremena spavanja"
            ],
            'Restless Leg Syndrome': [
                "🦵 Konsultujte se sa neurologom za dijagnozu i tretman",
                "Redovna umerena fizička aktivnost (šetnja, plivanje)",
                "Masirajte noge pre spavanja ili koristite toplu kupku",
                "Izbegavajte kofein, alkohol i duvan koji pogoršavaju simptome",
                "Probajte topao ili hladan kompres na noge",
                "Održavajte redovni raspored spavanja",
                "Razmislite o suplementima železa (ako je potrebno)"
            ],
            'Narcolepsy': [
                "😵 HITNO konsultujte se sa neurologom - potreban je specijalizovani tretman",
                "Uspostavite strogo redovni raspored spavanja i buđenja",
                "Planirajte kratke odmorce tokom dana (15-20 minuta)",
                "IZBEGAVAJTE vožnju dok se stanje ne stabilizuje",
                "Informišite porodicu i prijatelje o vašem stanju",
                "Koristite alarme i podsjetnike za važne aktivnosti",
                "Razmislite o medicinskom tretmanu (stimulansi, antidepresivi)"
            ]
        }
        
        # Ako je predikcija numerička, pokušaj da je konvertuješ
        if isinstance(prediction, (int, str)) and str(prediction).isdigit():
            prediction = int(prediction)
            if self.class_names and prediction < len(self.class_names):
                prediction = self.class_names[prediction]
        
        # Ako je predikcija string koji sadrži naziv klase
        if isinstance(prediction, str):
            # Pokušaj da pronađeš odgovarajuću preporuku
            for key in recommendations.keys():
                if key.lower() in prediction.lower() or prediction.lower() in key.lower():
                    return recommendations[key]
        
        return recommendations.get(prediction, ["Konsultujte se sa lekarom za detaljnu dijagnozu"])

# Kreiranje API instance
api = SleepDisorderAPI()

@app.route('/')
def index():
    """Glavna stranica"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model_loaded,
        'model_info': api.model_info.get('best_model_name', 'Unknown'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predikcija endpoint"""
    try:
        # Dobijanje podataka
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Nisu poslati podaci'
            }), 400
        
        # Validacija
        is_valid, errors = api.validate_input(data)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Neispravni podaci',
                'validation_errors': errors
            }), 400
        
        # Predikcija
        result = api.predict(data)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"API greška: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Server greška: {str(e)}'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Informacije o modelu"""
    return jsonify({
        'model_loaded': api.model_loaded,
        'model_info': api.model_info,
        'available_classes': api.class_names if api.class_names else []
    })

@app.route('/api/validate', methods=['POST'])
def validate_data():
    """Validacija podataka"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'valid': False,
                'errors': ['Nisu poslati podaci']
            }), 400
        
        is_valid, errors = api.validate_input(data)
        
        return jsonify({
            'valid': is_valid,
            'errors': errors
        })
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'errors': [f'Server greška: {str(e)}']
        }), 500

if __name__ == '__main__':
    print("🚀 Pokretanje Flask API servera...")
    print("📊 Model status:", "✅ Učitan" if api.model_loaded else "❌ Nije učitan")
    
    if api.model_loaded:
        print(f"🤖 Model: {api.model_info.get('best_model_name', 'Unknown')}")
        print(f"📈 Klase: {list(api.pipeline.classes_)}")
    
    print("🌐 Server dostupan na: http://localhost:5000")
    print("📋 API dokumentacija: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
