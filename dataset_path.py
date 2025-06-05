import pandas as pd

# Load the dataset CSV file
dataset_path = 'train/_classes.csv'  # Update with the path to your dataset CSV file
df = pd.read_csv(dataset_path)

# Directory name to add in front of filenames
directory_name = 'train'  # Update with the directory name you want to add

# Add directory name in front of each filename
df['filename'] = directory_name + '\\' + df['filename']

# Save the modified dataset to a new CSV file
new_dataset_path = 'dataset.csv'  # Update with the path for the new modified dataset CSV file
df.to_csv(new_dataset_path, index=False)

print("Modified dataset saved to:", new_dataset_path)
