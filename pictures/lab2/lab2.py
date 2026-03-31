import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import rotate
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


noise_factor_ = 0.3
weight_ = 0.1
num_samples = 3
epochs_= 4
angles = [-50, -30, -10, 0, 10, 30, 50]

def add_noise(images, noise_factor= noise_factor_):
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    return np.clip(noisy_images, 0., 1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)


def denoise_image(image):
    return denoise_tv_chambolle(image, weight=weight_)

x_train_denoised = np.array([denoise_image(img) for img in x_train_noisy])
x_test_denoised = np.array([denoise_image(img) for img in x_test_noisy])


x_train_denoised = x_train_denoised.reshape(-1, 28, 28, 1)
x_test_denoised = x_test_denoised.reshape(-1, 28, 28, 1)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_denoised, y_train, epochs=epochs_, validation_data=(x_test_denoised, y_test))


test_loss, test_acc = model.evaluate(x_test_denoised, y_test, verbose=2)
print(f'\nТочність на тестовому наборі: {test_acc:.4f}')


y_pred = model.predict(x_test_denoised)
y_pred_classes = np.argmax(y_pred, axis=1)


def rotate_and_predict(model, image):
    predictions = []

    for angle in angles:
        rotated = rotate(image, angle, reshape=False)
        rotated = np.expand_dims(rotated, axis=[0, -1])

        prediction = model.predict(rotated)
        predictions.append((np.argmax(prediction), np.max(prediction)))

    best_class = max(predictions, key=lambda x: x[1])[0]
    return best_class, predictions


indices = np.random.choice(len(x_test), num_samples, replace=False)

fig, axes = plt.subplots(num_samples, len(angles) + 3, figsize=(15, 10))

for i, idx in enumerate(indices):
    original_digit = y_test[idx] 
    denoised = x_test_denoised[idx].squeeze()

    axes[i, 0].imshow(x_test[idx], cmap="gray")
    axes[i, 0].set_title(f"Очікуємо: {original_digit}")
    axes[i, 1].imshow(x_test_noisy[idx], cmap="gray")
    axes[i, 2].imshow(denoised, cmap="gray")
    
    best_digit, predictions = rotate_and_predict(model, denoised)
    
    for j, (angle, (pred_digit, confidence)) in enumerate(zip(angles, predictions)):
        rotated = rotate(denoised, angle, reshape=False)
        ax = axes[i, j + 3]
        ax.imshow(rotated, cmap="gray")
        
        is_correct = (pred_digit == original_digit)
        color = "green" if is_correct else "red"
        
        ax.set_title(f"{angle}°: {pred_digit}\n({confidence:.2f})", color=color, fontsize=9)
        ax.axis("off")

    print(f"Приклад {i+1}: Очікувано {original_digit}, Найкращий вибір системи: {best_digit}")

plt.tight_layout()
plt.show()