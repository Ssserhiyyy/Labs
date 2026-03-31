import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS


avi_dict = {
    'alpha': 'A', 'bravo': 'B', 'charlie': 'C', 'delta': 'D', 'echo': 'E',
    'foxtrot': 'F', 'golf': 'G', 'hotel': 'H', 'india': 'I', 'juliett': 'J',
    'kilo': 'K', 'lima': 'L', 'mike': 'M', 'november': 'N', 'oscar': 'O',
    'papa': 'P', 'quebec': 'Q', 'romeo': 'R', 'sierra': 'S', 'tango': 'T',
    'uniform': 'U', 'victor': 'V', 'whiskey': 'W', 'xray': 'X', 'yankee': 'Y', 'zulu': 'Z',
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}

words = list(avi_dict.keys())
DATA_DIR = "C:/ling/dataset"


def generate_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Генерація аудіофайлів для навчання...")
        for word in words:
            for i in range(3): 
                tts = gTTS(text=word, lang='en', slow=(i%2==0))
                file_path = os.path.join(DATA_DIR, f"{word}_{i}.mp3")
                tts.save(file_path)
        print("Генерацію завершено!")


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Помилка обробки файлу {file_path}: {e}")
        return None
    return mfccs_processed


def prepare_data():
    X = []
    y = []
    print("Обробка аудіоданих (MFCC)...")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".mp3"):
            word = file.split("_")[0]
            file_path = os.path.join(DATA_DIR, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(word)
    return np.array(X), np.array(y)

generate_dataset()
X, y = prepare_data()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


print("Побудова та навчання нейромережі...")
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(40,)),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=150, batch_size=8, validation_data=(X_test, y_test), verbose=1)


print("\n--- СИМУЛЯЦІЯ РОЗПІЗНАВАННЯ ---")
test_sequence = ['hotel', 'alpha', 'victor', 'romeo', 'yankee', 'lima', 'kilo','oscar','two']
recognized_string = ""

print(f"Вхідна послідовність слів: {test_sequence}")

for word in test_sequence:
    test_file = os.path.join(DATA_DIR, f"{word}_0.mp3") 
    features = extract_features(test_file)
    features = features.reshape(1, -1) 
    
    prediction = model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_word = label_encoder.inverse_transform([predicted_class])[0]
    
    recognized_char = avi_dict[predicted_word]
    recognized_string += recognized_char

print(f"Розпізнаний бортовий номер: {recognized_string}")
