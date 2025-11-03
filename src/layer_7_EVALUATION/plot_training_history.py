# src/layer_7_EVALUATION/plot_training_history.py
import json
import matplotlib.pyplot as plt
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_history(history_file, output_plot_file):
    """
    Loads XGBoost training history from a JSON file and plots
    training vs. validation loss and error.
    """
    logging.info(f"Loading training history from: {history_file}")
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        logging.error(f"History file not found: {history_file}")
        logging.error("Please run the training script (Phase 5) first.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {history_file}")
        return

    evals_result = history

    # Check available metrics (mlogloss, merror)
    # The dictionary structure is like: {'validation_0': {'mlogloss': [...], 'merror': [...]}, 'validation_1': {...}}
    if 'validation_0' not in evals_result or 'validation_1' not in evals_result:
        logging.error("Evaluation results structure not as expected in JSON.")
        return

    train_logloss = evals_result['validation_0'].get('mlogloss', [])
    val_logloss = evals_result['validation_1'].get('mlogloss', [])
    train_merror = evals_result['validation_0'].get('merror', [])
    val_merror = evals_result['validation_1'].get('merror', [])

    epochs = range(1, len(train_logloss) + 1) if train_logloss else range(1, len(train_merror) + 1)

    if not epochs:
        logging.error("No training history data found in the file.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) # Create 2 side-by-side plots

    # Plot Loss (LogLoss)
    if train_logloss and val_logloss:
        ax1.plot(epochs, train_logloss, 'bo-', label='Training LogLoss', markersize=4)
        ax1.plot(epochs, val_logloss, 'ro-', label='Validation LogLoss', markersize=4)
        ax1.set_title('Training and Validation LogLoss')
        ax1.set_xlabel('Epochs (Boosting Rounds)')
        ax1.set_ylabel('LogLoss')
        ax1.legend()
        ax1.grid(True)
    else:
        ax1.set_title("LogLoss data not found")

    # Plot Error (Classification Error)
    if train_merror and val_merror:
        ax2.plot(epochs, train_merror, 'bo-', label='Training Classification Error', markersize=4)
        ax2.plot(epochs, val_merror, 'ro-', label='Validation Classification Error', markersize=4)
        ax2.set_title('Training and Validation Classification Error')
        ax2.set_xlabel('Epochs (Boosting Rounds)')
        ax2.set_ylabel('Classification Error (merror)')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title("Classification Error (merror) data not found")

    fig.suptitle('XGBoost Training History (No Graph Model)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_plot_file), exist_ok=True)
    plt.savefig(output_plot_file)
    logging.info(f"Training history plot saved to: {output_plot_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot XGBoost Training History")
    parser.add_argument('--history_in', type=str, required=True, help="Path to the training history JSON file.")
    parser.add_argument('--plot_out', type=str, required=True, help="Path to save the output plot PNG file.")
    args = parser.parse_args()

    plot_history(args.history_in, args.plot_out)