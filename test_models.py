import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY .env dosyasında bulunamadı!")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 Mevcut Gemini modelleri aranıyor...\n")

try:
    models = genai.list_models()
    
    # generateContent destekleyen modelleri bul
    available_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
    
    if available_models:
        print(f"✅ {len(available_models)} adet model bulundu:\n")
        for i, model_name in enumerate(available_models, 1):
            # Model adından gereksiz prefix'i temizle
            clean_name = model_name.replace('models/', '')
            print(f"{i}. {clean_name}")
        
        print("\n📝 Önerilen model adı (app.py'de kullanılacak):")
        # En yaygın modelleri önceliklendir
        if 'models/gemini-1.5-flash' in available_models:
            print("   → 'gemini-1.5-flash'")
        elif 'models/gemini-1.5-pro' in available_models:
            print("   → 'gemini-1.5-pro'")
        elif 'models/gemini-pro' in available_models:
            print("   → 'gemini-pro'")
        else:
            print(f"   → '{available_models[0].replace('models/', '')}'")
    else:
        print("❌ Hiçbir model bulunamadı!")
        
except Exception as e:
    print(f"❌ Hata: {e}")

