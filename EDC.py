import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_path = 'Final_Result.csv'
data_df = pd.read_csv(file_path)

# Define function to calculate FNMR and plot EDC curve
def calculate_FNMR_EDC(df, quality_column, identity_column, predicted_identity_column, discard_step=0.01):
    discard_fractions = []
    fnmr_values = []
    total_count = len(df)
    if total_count == 0:
        raise ValueError("The input dataframe is empty.")
    # Sort by quality scores from low to high
    df = df.sort_values(by=quality_column, ascending=False).reset_index(drop=True)
    # Calculate initial FNMR
    total_attempts = total_count
    false_non_matches = sum(df[identity_column] != df[predicted_identity_column])
    initial_fnmr = false_non_matches / total_attempts
    discard_fractions.append(0)
    fnmr_values.append(initial_fnmr)
    for i in range(1, int(1 / discard_step) + 1):
        num_discard = int(discard_step * total_count)
        current_df = df.iloc[num_discard * i:]
        discard_fraction = i * discard_step
        # Calculate FNMR
        total_attempts = len(current_df)
        false_non_matches = sum(current_df[identity_column] != current_df[predicted_identity_column])
        if total_attempts > 0:
            fnmr = false_non_matches / total_attempts
        else:
            fnmr = 0  # If no sample pairs, set FNMR to 0
        discard_fractions.append(discard_fraction)
        fnmr_values.append(fnmr)
    return discard_fractions, fnmr_values, initial_fnmr
# Calculate and plot EDC curve
def plot_EDC_curve(data_df, quality_column, identity_column, predicted_identity_column, discard_step=0.025):
    plt.figure(figsize=(12, 6))
    discard_fractions, fnmr_values, initial_fnmr = calculate_FNMR_EDC(data_df, quality_column, identity_column,
                                                           predicted_identity_column, discard_step)
    plt.step(discard_fractions, fnmr_values, where='post', label='EDC Curve')
    # Add dashed line representing optimal result
    plt.plot([0, initial_fnmr], [initial_fnmr, 0], 'k--', linewidth=1, label='Optimal FNMR')
    plt.xlabel('Fraction of Discarded Comparisons')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title(f'EDC Curve ({quality_column} as Quality Function)')
    plt.legend()
    plt.ylim(0,0.03)
    plt.xlim(0,0.2)
    plt.grid()
    plt.show()
plot_EDC_curve(data_df, 'Quality Scores', 'TrueLabel', 'PredictedLabel', discard_step)
