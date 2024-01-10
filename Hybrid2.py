import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Function to extract simple statistics as features using Naive Bayes
def extract_features(image):
    # For demonstration purposes, extracting basic statistics (mean and standard deviation) from each channel
    features = []
    for channel in range(image.shape[2]):
        channel_mean = np.mean(image[:, :, channel])
        channel_std = np.std(image[:, :, channel])
        features.extend([channel_mean, channel_std])
    return features

# Paths to original and manipulated image folders
original_folder = 'path_to_original_images_directory'
manipulated_folder = 'path_to_manipulated_images_directory'

# Collecting original images and extracting features
original_images = []
original_labels = []
for filename in os.listdir(original_folder):
    img = cv2.imread(os.path.join(original_folder, filename))
    features = extract_features(img)
    original_images.append(features)
    original_labels.append(0)  # Label 0 for original images

# Collecting manipulated images and extracting features
manipulated_images = []
manipulated_labels = []
for filename in os.listdir(manipulated_folder):
    img = cv2.imread(os.path.join(manipulated_folder, filename))
    features = extract_features(img)
    manipulated_images.append(features)
    manipulated_labels.append(1)  # Label 1 for manipulated images

# Combine original and manipulated data
all_images = np.vstack((original_images, manipulated_images))
all_labels = np.hstack((original_labels, manipulated_labels))

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Naive Bayes for feature extraction
nb = GaussianNB()
nb.fit(X_train, y_train)
X_train_transformed = nb.predict_proba(X_train)  # Use probabilities as transformed features
X_test_transformed = nb.predict_proba(X_test)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_transformed, y_train)

# Predict labels for test set
predictions = rf_classifier.predict(X_test_transformed)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("Accuracy:", accuracy)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

conf_matrix = confusion_matrix(y_test, predictions)
print('Confusion Matrix:')
print(conf_matrix)
