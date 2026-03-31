import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import string
import random

DATA_PATH = "G:/allLabs/tensorflow_datasets"

dataset_name = "emnist/byclass"

dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True, data_dir=DATA_PATH)

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_data = dataset['train'].map(normalize_img).cache().shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
test_data = dataset['test'].map(normalize_img).batch(64).cache().prefetch(tf.data.AUTOTUNE)

num_classes = info.features['label'].num_classes
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Навчання моделі...")
model.fit(train_data, epochs=3)

def get_char(label):
    if label < 10: return str(label)
    elif label < 36: return string.ascii_uppercase[label - 10]
    else: return string.ascii_lowercase[label - 36]

def process_combined_word(model, test_dataset, word_length=6):
    samples = list(test_dataset.unbatch().shuffle(5000).take(word_length))
    
    images_list = [img.numpy() for img, lbl in samples]
    labels_list = [lbl.numpy() for img, lbl in samples]
    real_word = "".join([get_char(l) for l in labels_list])

    combined_image = np.hstack([img.squeeze() for img in images_list])

    print(f"\nСтворено єдине зображення слова. Довжина: {word_length} символів.")
    
    predicted_word = ""
    plt.figure(figsize=(12, 3))
    plt.imshow(combined_image, cmap='gray')
    plt.title("Єдине вхідне зображення (Слово)")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(15, 3))
    for i in range(word_length):
        start = i * 28
        end = start + 28
        char_crop = combined_image[:, start:end]
        
        char_input = char_crop.reshape(1, 28, 28, 1)
        pred = model.predict(char_input, verbose=0)
        char_p = get_char(np.argmax(pred))
        predicted_word += char_p

        plt.subplot(1, word_length, i + 1)
        plt.imshow(char_crop, cmap='gray')
        plt.title(f"Буква {i+1}: {char_p}")
        plt.axis('off')

    print("\n" + "="*40)
    print(f"РЕАЛЬНЕ СЛОВО:     {real_word}")
    print(f"РОЗПІЗНАНЕ СЛОВО:  {predicted_word}")
    print("="*40)
    plt.tight_layout()
    plt.show()

process_combined_word(model, test_data, word_length=8)