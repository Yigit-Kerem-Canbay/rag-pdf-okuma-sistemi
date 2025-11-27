# 📚 Mini RAG - Mevzuat Asistanı

PDF ve TXT belgelerini analiz eden, Türkçe sorulara cevap veren bir RAG (Retrieval-Augmented Generation) sistemi.

## ✨ Özellikler

- ✅ PDF ve TXT dosyası desteği
- ✅ Çoklu belge yükleme
- ✅ Türkçe soru-cevap
- ✅ Kaynak gösterimi (hangi belgeden, hangi sayfa)
- ✅ Ayarlanabilir chunk sayısı (Top K: 3-6)
- ✅ MMR (Maximal Marginal Relevance) desteği
- ✅ Gradio web arayüzü

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.8+
- Google Gemini API anahtarı

### 2. Projeyi İndir

```bash
git clone <repo-url>
cd "1- RAG ile pdf okuma sistemi"
```

### 3. Sanal Ortam Oluştur ve Etkinleştir

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# veya
source .venv/bin/activate  # Linux/Mac
```

### 4. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 5. API Anahtarını Ayarla

`.env.example` dosyasını `.env` olarak kopyala ve API anahtarını ekle:

```bash
cp env.example .env
```

`.env` dosyasını aç ve API anahtarını yaz:

```
GOOGLE_API_KEY=buraya_gemini_api_anahtarını_yaz
```

**ÖNEMLİ:** `.env` dosyası asla GitHub'a yüklenmemeli! API anahtarını paylaşma.

### 6. Uygulamayı Başlat

```bash
python app.py
```

Tarayıcıda `http://127.0.0.1:7860` adresini aç.

## 📖 Kullanım

1. **Belge Yükle:** PDF veya TXT dosyasını yükle (çoklu seçim desteklenir)
2. **Soru Sor:** Belgede geçen bir konu hakkında soru sor
3. **Kaynakları Kontrol Et:** Cevabın hangi belgeden, hangi sayfadan geldiğini gör

## ⚙️ Ayarlar

- **Top K:** Kaç parça (chunk) kullanılacağını belirler (3-6 arası)
- **MMR:** Çeşitlilik sağlamak için Maximal Marginal Relevance kullanır

## 🔒 Güvenlik

- API anahtarı `.env` dosyasında saklanır ve `.gitignore` ile korunur
- `.env` dosyasını asla paylaşma veya GitHub'a yükleme
- `env.example` dosyası örnek format gösterir

## 📝 Lisans

Bu proje eğitim amaçlıdır.

## 🤝 Katkıda Bulunma

1. Fork yap
2. Feature branch oluştur (`git checkout -b feature/amazing-feature`)
3. Commit yap (`git commit -m 'Add some amazing feature'`)
4. Push yap (`git push origin feature/amazing-feature`)
5. Pull Request aç

## 📧 İletişim

Sorular için issue açabilirsiniz.

