import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

# Set your dataset paths
source_data_dir = 'path/to/your/dataset'
train_data_dir = 'path/to/train_data'
validation_data_dir = 'path/to/validation_data'

# Create train and validation directories if they don't exist
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(validation_data_dir, exist_ok=True)

# List subdirectories in the source directory (assuming each subdirectory represents a class)
class_names = os.listdir(source_data_dir)

# Split data into train and validation sets
for class_name in class_names:
    class_dir = os.path.join(source_data_dir, class_name)
    train_class_dir = os.path.join(train_data_dir, class_name)
    validation_class_dir = os.path.join(validation_data_dir, class_name)
    
    # Create class directories in train and validation directories
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(validation_class_dir, exist_ok=True)
    
    # List images in the class directory
    image_files = os.listdir(class_dir)
    
    # Split images into train and validation sets
    train_files, validation_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Move images to train and validation directories
    for file_name in train_files:
        src_path = os.path.join(class_dir, file_name)
        dst_path = os.path.join(train_class_dir, file_name)
        shutil.copy(src_path, dst_path)
    
    for file_name in validation_files:
        src_path = os.path.join(class_dir, file_name)
        dst_path = os.path.join(validation_class_dir, file_name)
        shutil.copy(src_path, dst_path)

print("Dataset split completed.")

# Define data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN architecture
def create_model(optimizer='adam', dropout_rate=0.2):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(len(class_names), activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create Keras classifier
keras_clf = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64],
    'epochs': [5, 10]
}

# Perform grid search
grid = GridSearchCV(estimator=keras_clf, param_grid=param_grid, cv=3)
grid_result = grid.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the best model on the validation set
y_pred = grid_result.best_estimator_.predict(validation_generator)
y_true = validation_generator.classes
validation_accuracy = accuracy_score(y_true, y_pred)
print(f"Validation accuracy: {validation_accuracy}")
