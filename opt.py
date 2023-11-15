import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np

# Set the path to your data directory
data_dir = r'C:\Users\Manoj Abhiram\Desktop\Project IoT\New folder'

# Create data generators for training, validation, and testing with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% of the data will be used for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Load the dataset
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' class mode for one-hot encoding
    subset='training',
    shuffle=True,
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' class mode for one-hot encoding
    subset='validation',
    shuffle=False,
)

# Create the CNN model with hyperparameters from Trial 2
model = Sequential()
model.add(Conv2D(71, (5, 5), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(113, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(77, activation='relu'))
model.add(Dropout(0.04757701544620092))  # Dropout rate from Trial 2
model.add(Dense(4, activation='softmax'))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00013205851162020806)  # Learning rate from Trial 2
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=30,  # You can adjust the number of epochs
    validation_data=val_generator,
)

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' class mode for one-hot encoding
    shuffle=False,  # Don't shuffle test data
)

# Extract file names from the test generator
test_file_names = test_generator.filenames

# Extract labels from file names (assuming file names are labels)
test_labels = [os.path.dirname(file_name) for file_name in test_file_names]

# Predict and evaluate the model
y_true = np.array(test_generator.labels)
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes, target_names=train_generator.class_indices)

print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)
model.save("finalmodel3.h5")