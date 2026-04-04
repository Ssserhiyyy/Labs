from gtts import gTTS
import pygame
import os
import time

def text_to_speech():
    print("=== Генератор мовлення запущено ===")
    text = input("Введіть текст для озвучення (або 'exit' для виходу): ")
    
    if text.lower() == 'exit':
        return False
        
    try:
        tts = gTTS(text=text, lang='uk', slow=False)
        filename = "output.mp3"
        tts.save(filename)
        print(f"Збережено у файл: {filename}")
        
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        print("Відтворюю...")
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
             time.sleep(0.1)
             
        pygame.mixer.quit()
        
    except Exception as e:
        print(f"Помилка: {e}")
        
    return True

while True:
    if not text_to_speech():
        break