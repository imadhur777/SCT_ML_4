import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths to Train/Test (created by organize_leapgest.py)
train_dir = r"C:\Users\amrit\OneDrive\Desktop\SkillSoft4\Train"
test_dir  = r"C:\Users\amrit\OneDrive\Desktop\SkillSoft4\Test"

# Image size (LeapGestRecog images are 320x240, we resize to smaller size for CNN)
img_size = (64, 64)
batch_size = 32

# Data generators (normalize images between 0 and 1)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen  = ImageDataGenerator(rescale=1.0/255.0)

# Flow from directory (automatically detects gesture classes)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# Save model
model.save("hand_gesture_model.h5")
print("✅ Model saved as hand_gesture_model.h5")
