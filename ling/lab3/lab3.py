from google import genai

client = genai.Client(api_key="AIzaSyCGmZF4DiaotjTLmGmcOTZgOAYq1_hmslA")

chat = client.chats.create(model="gemini-2.5-flash")

print("=== Чат-бот на базі оновленого Gemini SDK запущено! ===")
print("Напишіть 'вихід', 'exit' або 'quit' для завершення роботи.\n")

while True:
    user_input = input("Ви: ")
    
    if user_input.lower() in ['вихід', 'exit', 'quit']:
        print("Бот: Дякую за розмову. До побачення!")
        break
    
    if not user_input.strip():
        continue
        
    try:
        response = chat.send_message(user_input)
        print(f"Бот: {response.text}\n")
        
    except Exception as e:
        print(f"\n[Помилка]: Щось пішло не так. Деталі: {e}\n")