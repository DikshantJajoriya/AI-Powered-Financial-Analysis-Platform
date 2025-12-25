import ollama

print("⏳ Connecting to Ollama...")
try:
    # We use 'chat' because that is what your bot uses
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': 'Hi'}
    ])
    print("✅ SUCCESS! AI says:")
    print(response['message']['content'])
except Exception as e:
    print(f"❌ FAILED: {e}")