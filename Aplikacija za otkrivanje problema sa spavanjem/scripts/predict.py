#!/usr/bin/env python3
"""
Sleep Disorder Prediction - Prediction Script

Ovaj skript koristi trenirani model za predikciju poremećaja spavanja
na osnovu unetih parametara.

Autor: AI Assistant
Datum: 2024
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class SleepDisorderPredictor:
    """
    Klasa za predikciju poremećaja spavanja koristeći trenirani model
    """
    
    def __init__(self, model_path=None):
        """
        Inicijalizacija predictor-a
        
        Args:
            model_path (str): Putanja do sačuvanog pipeline-a
        """
        self.model_path = model_path
        self.pipeline = None
        self.feature_names = None
        self.class_names = None
        
    def load_model(self, model_path=None):
        """
        Učitavanje treniranog modela
        
        Args:
            model_path (str): Putanja do pipeline fajla
        """
        if model_path:
            self.model_path = model_path
            
        if not self.model_path:
            # Pokušaj pronalaženja najnovijeg modela
            models_dir = '../models' if os.path.exists('../models') else '.'
            pipeline_files = [f for f in os.listdir(models_dir) if f.startswith('best_sleep_disorder_pipeline_') and f.endswith('.pkl')]
            if pipeline_files:
                # Sortiranje po datumu kreiranja
                pipeline_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                self.model_path = os.path.join(models_dir, pipeline_files[0])
                print(f"🔍 Automatski pronađen model: {self.model_path}")
            else:
                print("❌ Nije pronađen nijedan pipeline fajl!")
                return False
        
        try:
            self.pipeline = joblib.load(self.model_path)
            print(f"✅ Model uspešno učitan: {self.model_path}")
            
            # Pokušaj učitavanja metadataka
            metadata_files = [f for f in os.listdir(models_dir) if f.startswith('model_metadata_') and f.endswith('.pkl')]
            if metadata_files:
                metadata_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                metadata = joblib.load(os.path.join(models_dir, metadata_files[0]))
                self.class_names = metadata.get('class_names', None)
                print(f"📊 Metadati učitani: {len(self.class_names) if self.class_names else 0} klasa")
            
            return True
            
        except Exception as e:
            print(f"❌ Greška pri učitavanju modela: {e}")
            return False
    
    def predict_single(self, data_dict):
        """
        Predikcija za jedan primer
        
        Args:
            data_dict (dict): Dictionary sa podacima za predikciju
            
        Returns:
            tuple: (predikcija, verovatnoće)
        """
        if self.pipeline is None:
            print("❌ Model nije učitan!")
            return None, None
        
        try:
            # Konvertovanje u DataFrame
            df = pd.DataFrame([data_dict])
            
            # Predikcija
            prediction = self.pipeline.predict(df)[0]
            probabilities = self.pipeline.predict_proba(df)[0]
            
            # Kreiranje dictionary-a sa verovatnoćama
            if self.class_names:
                prob_dict = dict(zip(self.class_names, probabilities))
            else:
                prob_dict = dict(zip(self.pipeline.classes_, probabilities))
            
            return prediction, prob_dict
            
        except Exception as e:
            print(f"❌ Greška pri predikciji: {e}")
            return None, None
    
    def predict_batch(self, data_list):
        """
        Predikcija za više primera
        
        Args:
            data_list (list): Lista dictionary-a sa podacima
            
        Returns:
            list: Lista predikcija sa verovatnoćama
        """
        if self.pipeline is None:
            print("❌ Model nije učitan!")
            return None
        
        try:
            # Konvertovanje u DataFrame
            df = pd.DataFrame(data_list)
            
            # Predikcije
            predictions = self.pipeline.predict(df)
            probabilities = self.pipeline.predict_proba(df)
            
            # Kreiranje rezultata
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                if self.class_names:
                    prob_dict = dict(zip(self.class_names, probs))
                else:
                    prob_dict = dict(zip(self.pipeline.classes_, probs))
                
                results.append({
                    'prediction': pred,
                    'probabilities': prob_dict,
                    'confidence': max(probs)
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Greška pri batch predikciji: {e}")
            return None
    
    def interactive_prediction(self):
        """Interaktivna predikcija"""
        print("\n🔮 INTERAKTIVNA PREDIKCIJA POREĆAJA SPAVANJA")
        print("=" * 50)
        
        # Unos podataka
        print("Molimo unesite sledeće podatke:")
        
        data = {}
        
        # Demografski podaci
        data['Gender'] = input("Pol (Male/Female): ").strip()
        data['Age'] = int(input("Uzrast: "))
        data['Occupation'] = input("Zanimanje: ").strip()
        
        # Parametri spavanja
        data['Sleep Duration'] = float(input("Trajanje spavanja (sati): "))
        data['Quality of Sleep'] = int(input("Kvalitet spavanja (1-10): "))
        
        # Fizički parametri
        data['Physical Activity Level'] = int(input("Nivo fizičke aktivnosti: "))
        data['Stress Level'] = int(input("Nivo stresa (1-10): "))
        
        # Zdravstveni parametri
        data['BMI Category'] = input("BMI kategorija (Underweight/Normal/Overweight/Obese): ").strip()
        data['Blood Pressure'] = input("Krvni pritisak (npr. 120/80): ").strip()
        data['Heart Rate'] = int(input("Otkucaji srca: "))
        data['Daily Steps'] = int(input("Dnevni koraci: "))
        
        print("\n" + "=" * 50)
        
        # Predikcija
        prediction, probabilities = self.predict_single(data)
        
        if prediction is not None:
            print(f"🎯 PREDIKCIJA: {prediction}")
            print(f"📊 VEROVATNOĆE:")
            for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"   {class_name}: {prob:.3f} ({prob*100:.1f}%)")
            
            # Preporuke
            self.print_recommendations(prediction, data)
        else:
            print("❌ Predikcija nije uspešna!")
    
    def print_recommendations(self, prediction, data):
        """Ispis preporuka na osnovu predikcije"""
        print(f"\n💡 PREPORUKE:")
        
        if prediction == "None":
            print("✅ Nemate poremećaje spavanja. Nastavite sa zdravim navikama!")
        elif prediction == "Insomnia":
            print("😴 Preporučuje se:")
            print("   - Redovno vreme za spavanje")
            print("   - Izbegavanje kofeina pre spavanja")
            print("   - Tehnike relaksacije")
        elif prediction == "Sleep Apnea":
            print("🫁 Preporučuje se:")
            print("   - Konsultacija sa lekarom")
            print("   - Kontrola težine")
            print("   - Izbegavanje alkohola")
        elif prediction == "Restless Leg Syndrome":
            print("🦵 Preporučuje se:")
            print("   - Redovna fizička aktivnost")
            print("   - Masiranje nogu")
            print("   - Izbegavanje kofeina")
        elif prediction == "Narcolepsy":
            print("😵 Preporučuje se:")
            print("   - Hitna konsultacija sa neurologom")
            print("   - Redovni raspored spavanja")
            print("   - Izbegavanje vožnje")
    
    def predict_from_csv(self, csv_path, output_path=None):
        """
        Predikcija za podatke iz CSV fajla
        
        Args:
            csv_path (str): Putanja do CSV fajla
            output_path (str): Putanja za čuvanje rezultata
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"📁 Učitano {len(df)} instanci iz {csv_path}")
            
            # Predikcije
            predictions = self.pipeline.predict(df)
            probabilities = self.pipeline.predict_proba(df)
            
            # Dodavanje rezultata u DataFrame
            df['Predicted_Sleep_Disorder'] = predictions
            df['Prediction_Confidence'] = np.max(probabilities, axis=1)
            
            # Dodavanje verovatnoća za svaku klasu
            if self.class_names:
                for i, class_name in enumerate(self.class_names):
                    df[f'Prob_{class_name}'] = probabilities[:, i]
            
            # Čuvanje rezultata
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"predictions_{timestamp}.csv"
            
            df.to_csv(output_path, index=False)
            print(f"✅ Rezultati sačuvani u {output_path}")
            
            # Statistike
            print(f"\n📊 STATISTIKE PREDIKCIJA:")
            print(df['Predicted_Sleep_Disorder'].value_counts())
            
        except Exception as e:
            print(f"❌ Greška pri predikciji iz CSV: {e}")

def main():
    """Glavna funkcija"""
    parser = argparse.ArgumentParser(description='Predikcija poremećaja spavanja')
    parser.add_argument('--model', help='Putanja do pipeline fajla')
    parser.add_argument('--csv', help='Putanja do CSV fajla za batch predikciju')
    parser.add_argument('--output', help='Putanja za čuvanje rezultata')
    parser.add_argument('--interactive', action='store_true', help='Interaktivna predikcija')
    
    args = parser.parse_args()
    
    # Kreiranje predictor-a
    predictor = SleepDisorderPredictor(args.model)
    
    # Učitavanje modela
    if not predictor.load_model():
        return 1
    
    # Pokretanje predikcije
    if args.interactive:
        predictor.interactive_prediction()
    elif args.csv:
        predictor.predict_from_csv(args.csv, args.output)
    else:
        print("🔮 PREDIKCIJA POREĆAJA SPAVANJA")
        print("Koristite --interactive za interaktivnu predikciju")
        print("ili --csv <fajl> za batch predikciju iz CSV fajla")
        
        # Jednostavan primer
        sample_data = {
            'Gender': 'Male',
            'Age': 35,
            'Occupation': 'Software Engineer',
            'Sleep Duration': 7.5,
            'Quality of Sleep': 6,
            'Physical Activity Level': 80,
            'Stress Level': 5,
            'BMI Category': 'Normal',
            'Blood Pressure': '120/80',
            'Heart Rate': 70,
            'Daily Steps': 8000
        }
        
        print(f"\n📝 PRIMER PREDIKCIJE:")
        prediction, probabilities = predictor.predict_single(sample_data)
        
        if prediction:
            print(f"Predikcija: {prediction}")
            print("Verovatnoće:")
            for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.3f}")
    
    return 0

if __name__ == "__main__":
    exit(main())
