import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import cv2

# === Step 1: Dataset Path ===
DATA_DIR = 'dataset100'

# === Step 2: Calorie Dictionary (Edit as needed) ===
calorie_dict = {
    'dosa': 133,
    'idli': 39,
    'biryani': 290,
    'samosa': 308,
    'chapati': 104,
    'poha': 180,
    'puri': 150,
    'upma': 120,
    'vada': 160,
    'rice': 200
}

# === Step 3: Load Dataset Using ImageDataGenerator ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# === Step 4: Build the Model using EfficientNetB0 ===
if os.path.exists("food_model.h5"):
    print("✅ Loading existing model...")
    model = tf.keras.models.load_model("food_model.h5")
else:
    print("🔨 Training new model...")
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=1)  # Fast training
    model.save("food_model.h5")  # Save for future reuse

# === Step 5: Prediction Function ===
def predict_food(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image '{img_path}' not found or can't be opened.")
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)
    label = list(train_gen.class_indices.keys())[class_index]
    calories = calorie_dict.get(label.lower(), "Unknown")
    return label, calories

# === Step 6: Run Prediction on test.jpg ===
if __name__ == "__main__":
    test_image = 'test.jpg'  # Make sure this image is in the same folder

    if os.path.exists(test_image):
        food, cal = predict_food(test_image)
        print(f"\n✅ Predicted Food: {food}")
        print(f"🔥 Estimated Calories: {cal} kcal\n")
    else:
        print("❌ Error: 'test.jpg' not found in current folder.")

