# src/layer_3_knowledge_and_ontology/generate_explanation_visuals.py
import matplotlib.pyplot as plt
import os

def create_visual_explanation(title, conditions, classification, output_filename):
    """
    Creates a simple flowchart-style image to explain an attack classification rule.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # --- Define styles for the boxes ---
    box_props = dict(boxstyle='round,pad=0.5', fc='lightblue', ec='b', lw=2)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', shrinkA=5, shrinkB=5,
                       patchA=None, patchB=None, color='black')

    # --- Create and place the boxes ---
    # 1. Input Traffic Box
    ax.text(5, 6, "Input Network Flow", ha='center', va='center', bbox=box_props, fontsize=12)

    # 2. Condition Boxes
    y_pos = 4.5
    for i, condition in enumerate(conditions):
        ax.text(5, y_pos, f"Condition: {condition}", ha='center', va='center', bbox=box_props, fontsize=11)
        # Arrow from previous box
        if i == 0:
            ax.annotate("", xy=(5, y_pos + 0.5), xytext=(5, 5.5), arrowprops=arrow_props)
        else:
            ax.annotate("", xy=(5, y_pos + 0.5), xytext=(5, y_pos + 1.5), arrowprops=arrow_props)
        y_pos -= 1.5

    # 3. Classification Box
    ax.text(5, 1, f"Classification:\n{classification}", ha='center', va='center', 
            bbox={**box_props, 'fc': 'lightgreen'}, fontsize=14, fontdict={'weight': 'bold'})
    ax.annotate("", xy=(5, 1.5), xytext=(5, y_pos + 1), arrowprops=arrow_props)


    plt.title(f"Identification Logic: {title}", fontsize=16)
    
    # --- Save the figure ---
    output_path = f'results/plots/{output_filename}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format="PNG", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"--- Explanation visual saved to: {output_path} ---")


if __name__ == "__main__":
    print("--- Generating Visual Explanations for Reasoning Engine Rules ---")

    # --- Rule 1: Unencrypted Login ---
    create_visual_explanation(
        title="Unencrypted Login Attempt",
        conditions=[
            "traffic_intent == 'Login Attempt'",
            "protocol_name == 'FTP'"
        ],
        classification="Unencrypted Login Attempt",
        output_filename="explanation_unencrypted_login.png"
    )

    # --- Rule 2: Potential Brute Force ---
    create_visual_explanation(
        title="Potential Brute Force",
        conditions=[
            "Label is 'FTP-Patator' OR 'SSH-Patator'"
        ],
        classification="Potential Brute Force",
        output_filename="explanation_brute_force.png"
    )
    
    # --- Rule 3: Anomalous Web Traffic ---
    create_visual_explanation(
        title="Anomalous Web Traffic",
        conditions=[
            "Label is 'Web Attack XSS' OR 'SQL Injection'..."
        ],
        classification="Anomalous Web Traffic",
        output_filename="explanation_web_attack.png"
    )

    print("\n--- All visual explanations have been generated. ---")
