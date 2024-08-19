import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
# Load the new CSV file
new_label_and_scores = pd.read_csv('comparison_scores.csv')
# Extract the necessary data
comparison_scores = new_label_and_scores['similarity']
true_labels = new_label_and_scores['score_type']
# Plot the distribution of comparison scores for mated and non-mated
plt.figure(figsize=(10, 6))
# Histogram for mated
mated_scores = comparison_scores[true_labels == 1]
plt.hist(mated_scores, bins=30, alpha=0.5, label='Mated', color='blue')
# Histogram for non-mated
non_mated_scores = comparison_scores[true_labels == 0]
plt.hist(non_mated_scores, bins=30, alpha=0.5, label='Non-Mated', color='red')
plt.title('Distribution of Comparison Scores')
plt.xlabel('Comparison Scores')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
# Compute DET curve
fpr, fnmr, thresholds = det_curve(true_labels, comparison_scores)
# Plot DET curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, fnmr, label='DET Curve', color='purple')
plt.title('DET Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('False Non-Match Rate (FNMR)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
