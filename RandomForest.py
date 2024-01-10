import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Function to load and preprocess images
def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Replace 'directory' with the path to your real or fake image directory
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (224, 224))  # Resize to a consistent size
            images.append(img.flatten())  # Flatten the image to use it as a feature vector
            labels.append(label)
    return images, labels

# Load real and fake images
real_images, real_labels = load_images('path_to_real_images_directory', 'real')
fake_images, fake_labels = load_images('path_to_fake_images_directory', 'fake')

# Combine real and fake data
all_images = real_images + fake_images
all_labels = real_labels + fake_labels

# Convert labels to numerical format
le = LabelEncoder()
all_labels_encoded = le.fit_transform(all_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_encoded, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
