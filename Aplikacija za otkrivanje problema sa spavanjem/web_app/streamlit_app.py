#!/usr/bin/env python3
"""
Sleep Disorder Prediction - Streamlit Web App

Streamlit aplikacija za predikciju poremećaja spavanja.
Kombinuje frontend i API funkcionalnost u jedan fajl.

Autor: AI Assistant
Datum: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Konfiguracija stranice
st.set_page_config(
    page_title="🔮 Predikcija poremećaja spavanja",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilovi
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4facfe;
        margin: 0.5rem 0;
    }
    
    .recommendation-item {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitSleepPredictor:
    """Streamlit aplikacija za predikciju poremećaja spavanja"""
    
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
            models_dir = 'models' if os.path.exists('models') else ('../models' if os.path.exists('../models') else '.')
            pipeline_files = [f for f in os.listdir(models_dir) 
                            if f.startswith('best_sleep_disorder_pipeline_') and f.endswith('.pkl')]
            
            if not pipeline_files:
                st.error("❌ Nije pronađen nijedan pipeline fajl!")
                st.info("Molimo pokrenite treniranje modela prvo sa: `python train_model.py`")
                return False
            
            # Sortiranje po datumu kreiranja
            pipeline_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
            latest_pipeline = os.path.join(models_dir, pipeline_files[0])
            
            # Učitavanje pipeline-a
            self.pipeline = joblib.load(latest_pipeline)
            
            # Učitavanje metadataka
            metadata_files = [f for f in os.listdir(models_dir) 
                            if f.startswith('model_metadata_') and f.endswith('.pkl')]
            if metadata_files:
                metadata_files.sort(key=lambda x: os.path.getctime(os.path.join(models_dir, x)), reverse=True)
                metadata = joblib.load(os.path.join(models_dir, metadata_files[0]))
                self.model_info = metadata
                
                # Učitavanje label_encoder-a i naziva klasa
                if 'label_encoder' in metadata and metadata['label_encoder'] is not None:
                    self.label_encoder = metadata['label_encoder']
                    self.class_names = metadata.get('class_names', [])
                    print(f"Label encoder loaded with classes: {self.label_encoder.classes_}")
                elif 'class_names' in metadata:
                    self.class_names = metadata['class_names']
                    print(f"Class names loaded: {self.class_names}")
                else:
                    print("Warning: No label encoder or class names found in metadata")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Greška pri učitavanju modela: {e}")
            self.model_loaded = False
            return False
    
    def validate_input(self, data):
        """Validacija unetih podataka"""
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
        
        # Validacija opsega
        if 'Age' in data and (data['Age'] < 0 or data['Age'] > 120):
            errors.append("Uzrast mora biti između 0 i 120 godina")
        
        if 'Sleep Duration' in data and (data['Sleep Duration'] < 0 or data['Sleep Duration'] > 24):
            errors.append("Trajanje spavanja mora biti između 0 i 24 sata")
        
        if 'Quality of Sleep' in data and (data['Quality of Sleep'] < 1 or data['Quality of Sleep'] > 10):
            errors.append("Kvalitet spavanja mora biti između 1 i 10")
        
        if 'Stress Level' in data and (data['Stress Level'] < 1 or data['Stress Level'] > 10):
            errors.append("Nivo stresa mora biti između 1 i 10")
        
        if 'Heart Rate' in data and (data['Heart Rate'] < 30 or data['Heart Rate'] > 200):
            errors.append("Otkucaji srca moraju biti između 30 i 200")
        
        return len(errors) == 0, errors
    
    def predict(self, data):
        """Predikcija poremećaja spavanja"""
        if not self.model_loaded:
            return None, {"error": "Model not loaded"}
        
        try:
            # Konvertovanje u DataFrame
            df = pd.DataFrame([data])
            
            # Predikcija
            prediction_numeric = self.pipeline.predict(df)[0]
            probabilities = self.pipeline.predict_proba(df)[0]
            
            # Konvertovanje numeričke predikcije u naziv klase
            if self.label_encoder is not None:
                prediction = self.label_encoder.inverse_transform([prediction_numeric])[0]
                class_names = self.class_names if self.class_names else self.label_encoder.classes_.tolist()
            else:
                # Fallback - koristimo pipeline classes direktno
                prediction = str(prediction_numeric)
                class_names = [str(i) for i in range(len(probabilities))]
            
            # Kreiranje rezultata - osiguravamo da imamo ispravnu mapiranje
            if len(class_names) == len(probabilities):
                prob_dict = dict(zip(class_names, probabilities))
            else:
                # Fallback mapping ako se brojevi ne poklapaju
                prob_dict = {}
                for i, prob in enumerate(probabilities):
                    if i < len(class_names):
                        prob_dict[class_names[i]] = prob
                    else:
                        prob_dict[f"Class_{i}"] = prob
            
            return prediction, prob_dict
            
        except Exception as e:
            print(f"Error in predict method: {e}")
            return None, {"error": str(e)}
    
    def get_recommendations(self, prediction):
        """Preporuke na osnovu predikcije - 3 glavne preporuke"""
        recommendations = {
            'None': [
                "🎉 Odlično! Nemate poremećaje spavanja",
                "Nastavite sa trenutnim zdravim navikama spavanja",
                "Održavajte redovno vreme za spavanje (7-9 sati)"
            ],
            'Insomnia': [
                "😴 Konsultujte se sa lekarom ili spavaćim specijalistom",
                "Uspostavite redovni raspored spavanja (isti čas svaki dan)",
                "Izbegavajte kofein i ekrane 1 sat pre spavanja"
            ],
            'Sleep Apnea': [
                "🫁 HITNO konsultujte se sa lekarom - moguće ozbiljno stanje",
                "Kontrolišite težinu - gubitak kilograma može značajno pomoći",
                "Spavajte na boku umesto na leđima"
            ],
            'Restless Leg Syndrome': [
                "🦵 Konsultujte se sa neurologom za dijagnozu i tretman",
                "Redovna umerena fizička aktivnost (šetnja, plivanje)",
                "Masirajte noge pre spavanja ili koristite toplu kupku"
            ],
            'Narcolepsy': [
                "😵 HITNO konsultujte se sa neurologom - potreban je specijalizovani tretman",
                "Uspostavite strogo redovni raspored spavanja i buđenja",
                "IZBEGAVAJTE vožnju dok se stanje ne stabilizuje"
            ]
        }
        
        # Direktno mapiranje
        if prediction in recommendations:
            return recommendations[prediction]
        
        # Fallback
        return ["Konsultujte se sa lekarom za detaljnu dijagnozu"]
    
    def create_input_form(self):
        """Kreiranje forme za unos podataka"""
        st.markdown('<div class="main-header"><h1>🔮 Predikcija poremećaja spavanja</h1><p>Unesite svoje podatke i saznajte kakvi su rizici za poremećaje spavanja</p></div>', 
                   unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Pol *", ["", "Male", "Female"], index=0)
                age = st.number_input("Uzrast *", min_value=0, max_value=120, value=30)
                occupation = st.text_input("Zanimanje *", placeholder="npr. Software Engineer")
                sleep_duration = st.number_input("Trajanje spavanja (sati) *", min_value=0.0, max_value=24.0, value=7.5, step=0.1)
                quality_sleep = st.slider("Kvalitet spavanja (1-10) *", min_value=1, max_value=10, value=6)
            
            with col2:
                physical_activity = st.number_input("Nivo fizičke aktivnosti *", min_value=0, value=80)
                stress_level = st.slider("Nivo stresa (1-10) *", min_value=1, max_value=10, value=5)
                bmi_category = st.selectbox("BMI kategorija *", ["", "Underweight", "Normal", "Overweight", "Obese"], index=0)
                blood_pressure = st.text_input("Krvni pritisak *", placeholder="npr. 120/80")
                heart_rate = st.number_input("Otkucaji srca *", min_value=30, max_value=200, value=70)
            
            daily_steps = st.number_input("Dnevni koraci *", min_value=0, value=8000)
            
            submitted = st.form_submit_button("🔮 Napravi predikciju", use_container_width=True)
            
            if submitted:
                # Prikupljanje podataka
                data = {
                    'Gender': gender,
                    'Age': age,
                    'Occupation': occupation,
                    'Sleep Duration': sleep_duration,
                    'Quality of Sleep': quality_sleep,
                    'Physical Activity Level': physical_activity,
                    'Stress Level': stress_level,
                    'BMI Category': bmi_category,
                    'Blood Pressure': blood_pressure,
                    'Heart Rate': heart_rate,
                    'Daily Steps': daily_steps
                }
                
                return data
        
        return None
    
    def display_results(self, prediction, probabilities, recommendations):
        """Prikaz rezultata"""
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        # Glavna predikcija
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🎯 Predikcija: **{prediction}**")
        
        with col2:
            confidence = max(probabilities.values())
            st.metric("Sigurnost", f"{confidence*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Verovatnoće
        st.markdown("### 📊 Verovatnoće po klasama")
        
        # Sortiranje po verovatnoći
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Kreiranje DataFrame za vizuelizaciju
        prob_df = pd.DataFrame(sorted_probs, columns=['Klasa', 'Verovatnoća'])
        prob_df['Verovatnoća %'] = prob_df['Verovatnoća'] * 100
        
        # Bar chart
        fig = px.bar(prob_df, x='Klasa', y='Verovatnoća %', 
                    title="Verovatnoće predikcije",
                    color='Verovatnoća %',
                    color_continuous_scale='Blues')
        
        fig.update_layout(
            xaxis_title="Tip poremećaja spavanja",
            yaxis_title="Verovatnoća (%)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela verovatnoća
        st.markdown("### 📋 Detaljne verovatnoće")
        
        for i, (class_name, prob) in enumerate(sorted_probs):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"{i+1}. **{class_name}**")
            
            with col2:
                st.progress(float(prob))
            
            with col3:
                st.write(f"{prob*100:.1f}%")
        
        # Preporuke
        st.markdown("### 💡 Preporuke")
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>{i+1}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)
        
        # Dodatne informacije
        st.markdown("### ℹ️ Dodatne informacije")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ukupno klasa", len(probabilities))
        
        with col2:
            st.metric("Najveća verovatnoća", f"{max(probabilities.values())*100:.1f}%")
        
        with col3:
            st.metric("Najmanja verovatnoća", f"{min(probabilities.values())*100:.1f}%")
    
    def display_model_info(self):
        """Prikaz informacija o modelu"""
        with st.sidebar:
            st.markdown("### 🤖 Informacije o modelu")
            
            if self.model_loaded:
                st.success("✅ Model je učitan")
                
                if self.model_info:
                    st.write(f"**Model:** {self.model_info.get('best_model_name', 'Unknown')}")
                    st.write(f"**Datum treniranja:** {self.model_info.get('training_date', 'Unknown')}")
                    
                    dataset_info = self.model_info.get('dataset_info', {})
                    st.write(f"**Trening instanci:** {dataset_info.get('train_samples', 'Unknown')}")
                    st.write(f"**Test instanci:** {dataset_info.get('test_samples', 'Unknown')}")
                    
                    # Prikazujemo broj features samo ako je dostupan
                    n_features = dataset_info.get('n_features')
                    if n_features is not None and n_features != 'Unknown':
                        st.write(f"**Ukupno features:** {n_features}")
            else:
                st.error("❌ Model nije učitan")
                st.info("Pokrenite treniranje sa: `python train_model.py`")
    
    def run(self):
        """Pokretanje aplikacije"""
        # Sidebar sa informacijama o modelu
        self.display_model_info()
        
        # Glavna forma
        data = self.create_input_form()
        
        if data is not None:
            # Validacija
            is_valid, errors = self.validate_input(data)
            
            if not is_valid:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ Neispravni podaci:")
                for error in errors:
                    st.write(f"• {error}")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Predikcija
            with st.spinner("🔮 Analiziram podatke..."):
                prediction, probabilities = self.predict(data)
            
            if prediction is None or (isinstance(probabilities, dict) and "error" in probabilities):
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                error_msg = "Greška pri predikciji"
                if isinstance(probabilities, dict) and "error" in probabilities:
                    error_msg = f"Greška pri predikciji: {probabilities['error']}"
                st.error(f"❌ {error_msg}")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Prikaz rezultata
            recommendations = self.get_recommendations(prediction)
            self.display_results(prediction, probabilities, recommendations)
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666; padding: 2rem;'>
                <p>⚠️ <strong>Napomena:</strong> Ova aplikacija je namenjena samo za edukativne svrhe.</p>
                <p>Za medicinsku dijagnozu konsultujte se sa lekarom.</p>
                <p>Generisano: {}</p>
            </div>
            """.format(datetime.now().strftime("%d.%m.%Y %H:%M")), unsafe_allow_html=True)

def main():
    """Glavna funkcija"""
    # Kreiranje aplikacije
    app = StreamlitSleepPredictor()
    
    # Pokretanje
    app.run()

if __name__ == "__main__":
    main()
