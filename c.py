import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the directory where your image data is stored
data_dir = r'C:\Users\Manoj Abhiram\Desktop\Project IoT\New folder'

# Initialize lists to store flattened images and corresponding labels
flattened_images = []
labels = []

# Define the target image size
target_size = (256, 256)

# Iterate through each subdirectory (one per class) in the data directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)

    # Skip non-directory files in the data directory
    if not os.path.isdir(class_dir):
        continue

    # Iterate through image files in the current class directory
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        # Open and preprocess the image (resize to the target size and convert to grayscale)
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize(target_size)  # Resize to the target size
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

        # Flatten the image and append it to the list
        flattened_image = image.flatten()
        flattened_images.append(flattened_image)
        labels.append(class_name)

# Convert labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Convert the list of flattened images and labels to NumPy arrays
X = np.array(flattened_images)
y = np.array(numeric_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)
