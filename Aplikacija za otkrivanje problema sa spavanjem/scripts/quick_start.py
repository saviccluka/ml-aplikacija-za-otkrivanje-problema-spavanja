#!/usr/bin/env python3
"""
Sleep Disorder Prediction - Quick Start Guide

Jednostavan skript za brzo pokretanje treniranja i predikcije.
"""

import os
import sys
import subprocess
import argparse

def check_requirements():
    """Provera da li su sve potrebne biblioteke instalirane"""
    print("🔍 Proveravam potrebne biblioteke...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'xgboost', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Nedostaju sledeće biblioteke: {', '.join(missing_packages)}")
        print("Instalirajte ih sa: pip install -r requirements.txt")
        return False
    
    print("✅ Sve potrebne biblioteke su instalirane!")
    return True

def run_training():
    """Pokretanje treniranja modela"""
    print("\n🚀 Pokretanje treniranja modela...")
    
    if not os.path.exists('expanded_sleep_health_dataset.csv'):
        print("❌ Dataset fajl 'expanded_sleep_health_dataset.csv' nije pronađen!")
        return False
    
    try:
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Treniranje uspešno završeno!")
            print(result.stdout)
            return True
        else:
            print("❌ Greška pri treniranju:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Greška: {e}")
        return False

def run_prediction():
    """Pokretanje interaktivne predikcije"""
    print("\n🔮 Pokretanje interaktivne predikcije...")
    
    try:
        result = subprocess.run([sys.executable, 'predict.py', '--interactive'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Predikcija uspešno završena!")
            print(result.stdout)
            return True
        else:
            print("❌ Greška pri predikciji:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Greška: {e}")
        return False

def show_help():
    """Prikaz pomoći"""
    print("""
🔮 SLEEP DISORDER PREDICTION - QUICK START

Dostupne komande:
  python quick_start.py --train     # Treniranje modela
  python quick_start.py --predict   # Interaktivna predikcija
  python quick_start.py --check     # Provera biblioteka
  python quick_start.py --help      # Ova pomoć

Primer korišćenja:
  1. Proverite biblioteke: python quick_start.py --check
  2. Trenirajte model: python quick_start.py --train
  3. Napravite predikciju: python quick_start.py --predict

Fajlovi:
  - expanded_sleep_health_dataset.csv  # Dataset
  - train_model.py                     # Treniranje
  - predict.py                         # Predikcija
  - requirements.txt                   # Potrebne biblioteke
  - config.py                          # Konfiguracija
""")

def main():
    """Glavna funkcija"""
    parser = argparse.ArgumentParser(description='Quick Start za Sleep Disorder Prediction')
    parser.add_argument('--train', action='store_true', help='Pokretanje treniranja')
    parser.add_argument('--predict', action='store_true', help='Pokretanje predikcije')
    parser.add_argument('--check', action='store_true', help='Provera biblioteka')
    parser.add_argument('--help-full', action='store_true', help='Detaljna pomoć')
    
    args = parser.parse_args()
    
    if args.help_full:
        show_help()
        return 0
    
    if args.check:
        if check_requirements():
            print("\n🎉 Sve je spremno za rad!")
        else:
            print("\n⚠️ Instalirajte nedostajuće biblioteke pre nastavka.")
        return 0
    
    if args.train:
        if check_requirements():
            run_training()
        else:
            print("❌ Instalirajte nedostajuće biblioteke pre treniranja.")
        return 0
    
    if args.predict:
        if check_requirements():
            run_prediction()
        else:
            print("❌ Instalirajte nedostajuće biblioteke pre predikcije.")
        return 0
    
    # Ako nema argumenata, prikaži pomoć
    show_help()
    return 0

if __name__ == "__main__":
    exit(main())
