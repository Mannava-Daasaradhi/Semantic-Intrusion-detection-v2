# src/layer_7_EVALUATION/generate_presentation_visuals.py
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def generate_training_plots():
    """
    Reads the XGBoost training history and generates separate plots for
    loss and accuracy, saving them with presentation-ready filenames.
    """
    print("--- Generating Training History Plots for Presentation ---")
    history_file = 'results/xgboost_training_history.json'
    loss_plot_file = 'results/plots/training_loss.png'
    accuracy_plot_file = 'results/plots/training_accuracy.png'

    if not os.path.exists(history_file):
        print(f"Error: Training history file not found at '{history_file}'")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)

    metric_sets = list(history.keys())
    if len(metric_sets) < 2 or 'mlogloss' not in history[metric_sets[0]]:
        print("Error: History file seems incomplete or malformed.")
        return
    
    train_key = metric_sets[0]
    eval_key = metric_sets[1]
    epochs = range(1, len(history[train_key]['mlogloss']) + 1)
    
    sns.set_style("whitegrid")

    # --- 1. Generate and Save Loss Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history[train_key]['mlogloss'], 'b-o', label=f'{train_key.capitalize()} Loss')
    plt.plot(epochs, history[eval_key]['mlogloss'], 'r-o', label=f'{eval_key.capitalize()} Loss')
    plt.title('XGBoost Model Training & Validation Loss', fontsize=16)
    plt.xlabel('Epochs (Boosting Rounds)', fontsize=12)
    plt.ylabel('LogLoss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(loss_plot_file)
    plt.close()
    print(f" - Saved training loss plot to '{loss_plot_file}'")

    # --- 2. Generate and Save Accuracy Plot ---
    if 'merror' in history[train_key] and 'merror' in history[eval_key]:
        train_acc = [1 - x for x in history[train_key]['merror']]
        eval_acc = [1 - x for x in history[eval_key]['merror']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, 'b-o', label=f'{train_key.capitalize()} Accuracy')
        plt.plot(epochs, eval_acc, 'r-o', label=f'{eval_key.capitalize()} Accuracy')
        plt.title('XGBoost Model Training & Validation Accuracy', fontsize=16)
        plt.xlabel('Epochs (Boosting Rounds)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(accuracy_plot_file)
        plt.close()
        print(f" - Saved training accuracy plot to '{accuracy_plot_file}'")
    else:
        print("\nWarning: 'merror' key not found in training history. Cannot generate accuracy plot.")
        print("Please re-run the training script with 'merror' in the 'eval_metric' list.")

def create_explanation_collage():
    """
    Finds individual LLM explanation images and combines them into a
    single HORIZONTAL collage for the presentation.
    """
    print("\n--- Creating LLM Explanation Collage ---")
    
    plot_dir = 'results/plots'
    output_file = os.path.join(plot_dir, 'llm_explanation_examples.png')
    
    source_files = [
        'explanation_brute_force.png',
        'explanation_web_attack.png',
        'explanation_unencrypted_login.png'
    ]
    
    images = [Image.open(os.path.join(plot_dir, f)) for f in source_files if os.path.exists(os.path.join(plot_dir, f))]
    
    if not images:
        print("Error: No source explanation images found. Cannot create collage.")
        return

    # --- MODIFIED: Create a horizontal collage ---
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    collage = Image.new('RGB', (total_width, max_height), color='white')
    
    current_x = 0
    for img in images:
        collage.paste(img, (current_x, 0))
        current_x += img.width
    # --- END MODIFICATION ---
        
    collage.save(output_file)
    print(f" - Saved HORIZONTAL explanation collage to '{output_file}'")


if __name__ == "__main__":
    os.makedirs('results/plots', exist_ok=True)
    
    generate_training_plots()
    create_explanation_collage()
    
    print("\n--- Presentation Visuals Generation Complete ---")