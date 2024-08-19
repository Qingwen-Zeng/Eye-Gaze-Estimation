import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'Final_Result.csv'
data = pd.read_csv(file_path)

# Plot the distribution of 'Quality Scores' column
plt.figure(figsize=(10, 6))
plt.hist(data['Quality Scores'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Quality Scores')
plt.xlabel('Quality Scores')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
