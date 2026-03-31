import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

_epochs = 4
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=_epochs, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nТочність на тестових даних: {test_acc:.2%}")

def plot_accuracy(history):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Точність на train')
    plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Точність на validation')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.xticks(range(1, len(history.history['accuracy']) + 1))
    plt.legend()
    plt.title('Графік точності')
    plt.show()

plot_accuracy(history)

def visualize_predictions(model, x_test, y_test, num_examples=40):
    plt.figure(figsize=(20,10))
    num_rows = num_examples / 5
    indices = np.random.choice(len(x_test), num_examples, replace=False)  
    predictions = model.predict(x_test[indices]) 

    for i, idx in enumerate(indices):
        plt.subplot(int(num_rows), 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.axis('off')
        pred_label = np.argmax(predictions[i])   
        true_label = y_test[idx]
        plt.title(f"Реальне : {true_label}\nПередбачено : {pred_label}",
                  color="green" if pred_label == true_label else "red")

    plt.tight_layout()
    plt.show()

visualize_predictions(model, x_test, y_test)
