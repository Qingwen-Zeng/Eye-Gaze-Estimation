import pandas as pd

# Load the CSV files
comparison_scores_path = 'comparison_scores.csv'
all_predictions_path = 'all_predictions.csv'

comparison_scores_df = pd.read_csv(comparison_scores_path)
all_predictions_df = pd.read_csv(all_predictions_path)

# Merge the dataframes on Image1 and Image2
merged_df = pd.merge(comparison_scores_df, all_predictions_df, left_on=['image1', 'image2'], right_on=['Image1', 'Image2'])

# Drop the duplicate columns Image1 and Image2
merged_df = merged_df.drop(columns=['Image1', 'Image2'])

# Save the merged dataframe to a new CSV file
merged_df.to_csv('Face_Recognition_Result.csv', index=False)

print("Merged CSV file saved as 'Face_Recognition_Result.csv'.")
