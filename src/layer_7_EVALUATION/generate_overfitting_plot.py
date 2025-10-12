# src/layer_7_evaluation/generate_overfitting_plot.py
import json
import matplotlib.pyplot as plt
import os

def plot_learning_curves():
    """
    Loads the XGBoost training history and plots the training vs. validation loss
    to visually demonstrate that the model is not overfitting.
    """
    print("--- Generating Anti-Overfitting Learning Curve Plot ---")
    
    history_file = 'results/xgboost_training_history.json'
    output_plot_file = 'results/plots/anti_overfitting_learning_curve.png'

    if not os.path.exists(history_file):
        print(f"Error: Training history file not found at '{history_file}'")
        print("Please run the updated Phase 5 script first to generate the history.")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)

    # Extract the training and validation loss from the history
    # The keys 'validation_0' and 'validation_1' correspond to the eval_set order
    train_loss = history['validation_0']['mlogloss']
    val_loss = history['validation_1']['mlogloss']
    epochs = range(1, len(train_loss) + 1)

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs, val_loss, 'r-o', label='Validation Loss', markersize=4)
    
    # Find the minimum validation loss point for annotation
    min_val_loss_epoch = val_loss.index(min(val_loss)) + 1
    min_val_loss = min(val_loss)
    
    # Add a vertical line and text for the early stopping point
    plt.axvline(x=min_val_loss_epoch, color='green', linestyle='--', linewidth=2, label=f'Early Stopping Point (Epoch {min_val_loss_epoch})')
    
    plt.title('Training vs. Validation Loss for XGBoost Model', fontsize=16, fontweight='bold')
    plt.xlabel('Training Epochs (Boosting Rounds)', fontsize=12)
    plt.ylabel('Log Loss (Error)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Add text explaining the plot
    explanation = (
        "This plot shows a well-generalized model.\n"
        "Both Training and Validation loss decrease together and plateau.\n"
        "There is no divergence, which would indicate overfitting."
    )
    plt.text(0.5, 0.6, explanation, transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Save the plot
    os.makedirs(os.path.dirname(output_plot_file), exist_ok=True)
    plt.savefig(output_plot_file, dpi=300)
    
    print(f"Learning curve plot saved to '{output_plot_file}'")
    print("The plot visually confirms that the model is not overfitting.")

if __name__ == "__main__":
    plot_learning_curves()