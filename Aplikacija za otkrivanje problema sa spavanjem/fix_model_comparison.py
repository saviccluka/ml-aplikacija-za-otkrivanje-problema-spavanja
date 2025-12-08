# FIXED VERSION: Poređenje performansi svih modela
print("=== POREDENJE PERFORMANSI MODELA ===")

# Proverava koje rezultate modela imamo dostupne
available_results = []
model_names = []

# Proverava Random Forest rezultate
if 'rf_results' in locals():
    available_results.append(rf_results)
    model_names.append('Random Forest')
    print("✅ Random Forest rezultati dostupni")
else:
    print("❌ Random Forest rezultati nisu dostupni")

# Proverava XGBoost rezultate  
if 'xgb_results' in locals():
    available_results.append(xgb_results)
    model_names.append('XGBoost')
    print("✅ XGBoost rezultati dostupni")
else:
    print("❌ XGBoost rezultati nisu dostupni")

# Proverava MLP rezultate
if 'mlp_results' in locals():
    available_results.append(mlp_results)
    model_names.append('MLP')
    print("✅ MLP rezultati dostupni")
else:
    print("❌ MLP rezultati nisu dostupni")

# Prikupljanje svih dostupnih rezultata
all_results = available_results

print(f"\nDostupno {len(all_results)} modela za poređenje: {model_names}")

# Kreiranje DataFrame za poređenje (samo ako imamo rezultate)
if all_results:
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.set_index('model_name')
    
    print("\n=== TABELA POREDENJA ===")
    print(comparison_df.round(4))
    
    # Vizuelizacija poređenja (samo ako imamo rezultate)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(comparison_df.index, comparison_df['accuracy'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision comparison
    axes[0, 1].bar(comparison_df.index, comparison_df['precision'])
    axes[0, 1].set_title('Precision Comparison')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Recall comparison
    axes[0, 2].bar(comparison_df.index, comparison_df['recall'])
    axes[0, 2].set_title('Recall Comparison')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    axes[1, 0].bar(comparison_df.index, comparison_df['f1_score'])
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # ROC-AUC comparison
    axes[1, 1].bar(comparison_df.index, comparison_df['roc_auc'])
    axes[1, 1].set_title('ROC-AUC Comparison')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    axes[1, 2].bar(comparison_df.index, comparison_df['training_time'])
    axes[1, 2].set_title('Training Time Comparison')
    axes[1, 2].set_ylabel('Training Time (seconds)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Pronalaženje najboljeg modela
    best_model_idx = comparison_df.mean(axis=1).idxmax()
    best_model_score = comparison_df.mean(axis=1).max()
    
    print(f"\n=== NAJBOLJI MODEL ===")
    print(f"Model: {best_model_idx}")
    print(f"Prosečna performansa: {best_model_score:.4f}")
    
    # Detaljno poređenje
    print(f"\n=== DETALJNO POREDENJE ===")
    for metric in comparison_df.columns:
        best_for_metric = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"{metric.replace('_', ' ').title()}: {best_for_metric} ({best_score:.4f})")
        
else:
    print("\n❌ Nema dostupnih rezultata modela za poređenje!")
    print("Molimo pokrenite treniranje modela pre poređenja.")
    print("\nDa biste trenirali modele, pokrenite sledeće ćelije:")
    print("1. Random Forest treniranje")
    print("2. XGBoost treniranje") 
    print("3. MLP treniranje")
