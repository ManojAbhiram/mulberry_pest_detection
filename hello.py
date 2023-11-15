import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

# Define the objective function to optimize
def objective(trial):
    # Define hyperparameter search spaces
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    num_filters_layers = []
    kernel_size_layers = []
    for i in range(num_conv_layers):
        num_filters_layers.append(trial.suggest_int(f'num_filters_layer{i}', 32, 128))  # Reduced max filter size
        kernel_size_layers.append(trial.suggest_int(f'kernel_size_layer{i}', 3, 5))
    num_units = trial.suggest_int('num_units', 32, 128)  # Reduced max units
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)  # Reduced max dropout
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)  # Adjusted learning rate range

    # Create the CNN model
    model = Sequential()
    model.add(Conv2D(num_filters_layers[0], (kernel_size_layers[0], kernel_size_layers[0]), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    for i in range(1, num_conv_layers):
        model.add(Conv2D(num_filters_layers[i], (kernel_size_layers[i], kernel_size_layers[i]), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(num_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, activation='softmax'))

    # Compile the model with hyperparameters
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Create data generators with a reduced batch size
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    data_dir = r'C:\Users\Manoj Abhiram\Desktop\Project IoT\New folder'

    batch_size = 16  # Reduced batch size
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,  # Reduced batch size
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,  # Reduced batch size
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,  # Reduced number of epochs
        verbose=0,
    )

    # Evaluate the model on validation data
    val_loss = history.history['val_loss'][-1]

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_jobs=1, n_trials=20)  # Reduced the number of trials and used 1 job

    # Print the best hyperparameters and results
    print("Best Hyperparameters:", study.best_params)
    print("Best Validation Loss:", study.best_value)