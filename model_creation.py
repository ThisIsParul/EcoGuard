import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import joblib

# Load the dataset
df = pd.read_csv('dataset.csv')

# Define a function to extract green density features from an image
def extract_green_density(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_channel = hsv[:, :, 0]
    green_channel = green_channel[(green_channel >= 40) & (green_channel <= 80)]
    green_density = np.mean(green_channel)
    return green_density

# Create a new column in the dataframe with the green density feature
df['green_density'] = df['filename'].apply(lambda x: extract_green_density(x.replace('train\\', 'train/')))

# Split the data into training and testing sets
X = df[['green_density']]  # feature matrix
y = df['deforestation']  # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.3f}')

# Save the trained model
joblib.dump(clf, 'deforest_model.joblib')
