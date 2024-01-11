import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        # Replace 'folder' with the path to your original or tampered image directory
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # Read images in color
        img = cv2.resize(img, (256, 256))  # Resize images to a consistent size
        images.append(img)  # Don't flatten the image array
        labels.append(folder)  # Use folder name as label (original or tampered)
    return np.array(images), np.array(labels)

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features
    fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return fd

# Function to extract color histogram features from an image
def extract_color_histogram(image):
    hist_features = []
    for channel in cv2.split(image):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_features.extend(hist)
    return hist_features

# Load original and tampered images
original_images, original_labels = load_images('path_to_original_images_directory')
tampered_images, tampered_labels = load_images('path_to_tampered_images_directory')

# Combine data and labels
X = []
y = []

# Extract features for original images
for image in original_images:
    hog_features = extract_hog_features(image)
    color_hist_features = extract_color_histogram(image)
    combined_features = np.concatenate((hog_features, color_hist_features))
    X.append(combined_features)
    y.append("original")

# Extract features for tampered images
for image in tampered_images:
    hog_features = extract_hog_features(image)
    color_hist_features = extract_color_histogram(image)
    combined_features = np.concatenate((hog_features, color_hist_features))
    X.append(combined_features)
    y.append("tampered")

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Use StandardScaler to scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

pos_label = 'original'  # Change this to the positive class label
neg_label = 'tampered'  # Change this to the negative class label
precision = precision_score(y_test, y_pred, pos_label=pos_label)
recall = recall_score(y_test, y_pred, pos_label=pos_label)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
