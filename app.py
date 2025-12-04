import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb import errors as chroma_errors
from chromadb.config import Settings
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Groq API için (opsiyonel)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
MMR_LAMBDA = 0.5
COLLECTION_NAME = "active_pdf"
DEFAULT_CITY = "Elazığ"

# Türkiye'nin 81 şehri (alfabetik sıralı)
TURKIYE_SEHIRLERI = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara", "Antalya",
    "Ardahan", "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik",
    "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum",
    "Denizli", "Diyarbakır", "Düzce", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir",
    "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul",
    "İzmir", "Kahramanmaraş", "Karabük", "Karaman", "Kars", "Kastamonu", "Kayseri", "Kırıkkale",
    "Kırklareli", "Kırşehir", "Kilis", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa",
    "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Osmaniye", "Rize",
    "Sakarya", "Samsun", "Şanlıurfa", "Siirt", "Sinop", "Şırnak", "Sivas", "Tekirdağ",
    "Tokat", "Trabzon", "Tunceli", "Uşak", "Van", "Yalova", "Yozgat", "Zonguldak"
]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

_gemini_model = None


def list_available_models():
    """Kullanılabilir Gemini modellerini listeler"""
    try:
        models = genai.list_models()
        available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        return available
    except Exception:
        return []


def get_gemini_model():
    global _gemini_model
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY değerini .env dosyasına ekleyip uygulamayı yeniden başlatın."
        )
    if _gemini_model is None:
        # Mevcut modeller: önce en güncel ve hızlı olanları dene
        model_candidates = [
            "gemini-2.5-flash",      # En güncel flash model (hızlı)
            "gemini-flash-latest",   # Her zaman en güncel flash
            "gemini-2.5-pro",        # Güçlü ama daha yavaş
            "gemini-pro-latest",     # Her zaman en güncel pro
            "gemini-1.5-flash",      # Eski ama stabil
            "gemini-1.5-pro",        # Eski ama stabil
        ]
        
        for model_name in model_candidates:
            try:
                _gemini_model = genai.GenerativeModel(model_name)
                # Model başarıyla oluşturuldu
                break
            except Exception:
                continue
        
        if _gemini_model is None:
            # Son çare: kullanılabilir modelleri listele
            available = list_available_models()
            if available:
                # İlk 5 model adını göster
                clean_names = [m.replace('models/', '') for m in available[:5]]
                error_msg = (
                    f"Hiçbir model yüklenemedi. "
                    f"Kullanılabilir modeller: {', '.join(clean_names)}"
                )
            else:
                error_msg = "Model bulunamadı. API anahtarınızı kontrol edin."
            raise RuntimeError(error_msg)
    return _gemini_model


_groq_client = None


def get_groq_client():
    """Groq API client'ını döndürür"""
    global _groq_client
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq paketi yüklü değil. 'pip install groq' komutu ile yükleyin.")
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY değerini .env dosyasına ekleyip uygulamayı yeniden başlatın. "
            "Ücretsiz API anahtarı için: https://console.groq.com/"
        )
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

active_collection = None
current_docs: Dict[str, Dict[str, str]] = {}

# Sohbet hafızası - önceki mesajları ve kullanıcı profilini tutar
conversation_history: List[Dict[str, str]] = []
user_profile: Dict[str, str] = {}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def sanitize_question(q: str) -> str:
    blockers = ["ceza", "suç", "yaptırım", "kovuşturma", "kimlik", "tc", "disiplin"]
    for b in blockers:
        q = q.replace(b, f"{b} (mevzuat kapsamında)")
    return q


def update_user_profile(message: str) -> None:
    """
    Kullanıcının verdiği basit kişisel bilgileri (ör. ad, üşürüm) profilde saklar.
    Bu bilgiler belgeye bağlı değildir ve belge içinde aranmaz.
    """
    global user_profile
    text = (message or "").strip()
    lower = text.lower()

    # Ad yakalama: "benim adım X", "adım X", "ismim X"
    name_patterns = [
        r"\bbenim ad[ıi]m\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)",
        r"\bad[ıi]m\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)",
        r"\bismim\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)",
    ]
    import re as _re

    for pat in name_patterns:
        m = _re.search(pat, text, flags=_re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # İlk harfi büyük yap, geri kalanı olduğu gibi bırak
            if name:
                user_profile["name"] = name[0].upper() + name[1:]
            break

    # Üşüme / sıcaklık hassasiyeti
    if "üşürüm" in lower or "çok üşürüm" in lower:
        user_profile["cold_sensitivity"] = "high"


def is_weather_related_question(question: str) -> bool:
    """Soru hava durumu, aktivite veya kıyafet önerisi ile ilgili mi kontrol eder"""
    question_lower = question.lower()
    weather_keywords = [
        "hava durumu", "hava", "sıcaklık", "yağmur", "kar", "rüzgar", "nem",
        "aktivite öner", "aktivite", "ne yapabilirim", "ne yapmalıyım",
        "kıyafet öner", "kıyafet", "giyin", "giyim", "nasıl giyinmeliyim",
        "bugünkü hava", "şu anki hava", "güncel hava", "hava durumuna göre",
        "havaya göre", "iklim", "mevsim"
    ]
    return any(keyword in question_lower for keyword in weather_keywords)


def read_pdf(file_path: str) -> List[Tuple[int, str]]:
    try:
        reader = PdfReader(file_path)
        pages: List[Tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""
            normalized = normalize_text(content)
            if normalized:
                pages.append((idx, normalized))
        return pages
    except Exception as e:
        raise RuntimeError(f"PDF okuma hatası: {str(e)}")


def read_txt(file_path: str) -> List[Tuple[int, str]]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        normalized = normalize_text(content)
        return [(1, normalized)] if normalized else []
    except Exception as e:
        raise RuntimeError(f"TXT okuma hatası: {str(e)}")


def chunk_pages(pages: List[Tuple[int, str]]) -> Tuple[List[str], List[Dict]]:
    chunks: List[str] = []
    metadatas: List[Dict] = []
    chunk_step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for page_num, text in pages:
        if not text:
            continue
        start = 0
        local_idx = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunk = text[start:end]
            chunks.append(chunk)
            metadatas.append(
                {
                    "page": page_num,
                    "chunk_id": f"P{page_num}-C{local_idx}",
                }
            )
            start += chunk_step
            local_idx += 1
    return chunks, metadatas


def reset_collection():
    global active_collection, current_docs
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Koleksiyon yoksa veya başka bir "bulunamadı" hatası gelirse göz ardı et
        pass
    active_collection = chroma_client.create_collection(name=COLLECTION_NAME)
    current_docs = {}


def ensure_collection():
    global active_collection
    if active_collection is None:
        try:
            active_collection = chroma_client.get_collection(COLLECTION_NAME)
        except (getattr(chroma_errors, "InvalidCollectionError", Exception),
                getattr(chroma_errors, "NotFoundError", Exception)):
            active_collection = chroma_client.create_collection(name=COLLECTION_NAME)


def ingest_file(file_obj):
    global current_docs
    if not file_obj:
        return "Lütfen PDF veya TXT dosyası yükleyin."

    try:
        # Gradio file nesnesini handle et
        # Gradio file nesnesi genellikle .name attribute'una sahiptir
        if hasattr(file_obj, 'name'):
            file_path_str = file_obj.name
        elif isinstance(file_obj, (str, Path)):
            file_path_str = str(file_obj)
        else:
            # Diğer durumlar için string'e çevir
            file_path_str = str(file_obj)
        
        file_path = Path(file_path_str)
        
        # Dosya yolunu kontrol et
        if not file_path.exists():
            return f"❌ Hata: Dosya bulunamadı: {file_path}"
        
        # Dosya boyutunu kontrol et
        file_size = file_path.stat().st_size
        if file_size == 0:
            return "❌ Hata: Dosya boş."
        
        # Dosya uzantısını kontrol et
        suffix = file_path.suffix.lower()
        if suffix not in [".pdf", ".txt"]:
            return "❌ Sadece PDF veya TXT destekleniyor."

        # Dosyayı oku
        if suffix == ".pdf":
            pages = read_pdf(str(file_path))
        else:  # .txt
            pages = read_txt(str(file_path))

        if not pages:
            return "❌ Belgeden metin çıkarılamadı."
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ Dosya okuma hatası: {str(e)}\n\nDetay: {error_detail[:200]}"

    ensure_collection()
    chunks, metadatas = chunk_pages(pages)
    # metadata'ya belge adını ekle
    for meta in metadatas:
        meta["doc"] = file_path.name

    embeddings = embedder.encode(chunks, batch_size=32, convert_to_numpy=True).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]
    active_collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    current_docs[file_path.name] = {
        "chunks": len(chunks),
        "path": str(file_path),
    }

    status = (
        f"✅ {file_path.name} yüklendi. "
        f"{len(chunks)} parça koleksiyona eklendi."
    )
    return status


def retrieve_chunks(question: str, top_k: int = DEFAULT_TOP_K, use_mmr: bool = False):
    if active_collection is None:
        raise RuntimeError("Önce bir belge yükleyin.")

    query_vec = embedder.encode([question], convert_to_numpy=True).tolist()
    query_params = {
        "query_embeddings": query_vec,
        "n_results": top_k,
    }
    if use_mmr:
        query_params.update({"mmr": True, "lambda_mult": MMR_LAMBDA})

    try:
        results = active_collection.query(**query_params)
    except TypeError as err:
        # MMR parametresi desteklenmiyorsa klasik sorguya düş
        if use_mmr and "unexpected keyword argument 'mmr'" in str(err):
            query_params.pop("mmr", None)
            query_params.pop("lambda_mult", None)
            results = active_collection.query(**query_params)
        else:
            raise
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return documents, metadatas


def format_context(docs: List[str], metas: List[Dict]) -> str:
    paired = []
    for doc, meta in zip(docs, metas):
        doc_label = meta.get("doc", "Belge")
        page_num = meta.get('page', '?')
        chunk_id = meta.get('chunk_id', '?')
        tag = f"📄 BELGE: {doc_label} | Sayfa: {page_num} | Parça: {chunk_id}"
        paired.append(f"[{tag}]\n{doc}")
    return "\n\n---\n\n".join(paired)


def build_sources(metas: List[Dict]) -> List[str]:
    unique = []
    seen = set()
    for meta in metas:
        doc_name = meta.get('doc', 'Belge')
        page_num = meta.get('page', '?')
        chunk_id = meta.get('chunk_id', '?')
        # Belge bazında grupla
        label = f"📄 {doc_name} | Sayfa {page_num} | Parça {chunk_id}"
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def get_weather_summary(city: str) -> Tuple[bool, str, str]:
    """
    OpenWeatherMap'ten hava durumu özetini döndürür.

    Returns:
        has_weather (bool): Gerçek API verisi başarıyla alındı mı?
        normalized_city (str): Kullanılan şehir adı
        summary (str): Kısa Türkçe özet veya hata bilgisi
    """
    city = (city or "").strip() or DEFAULT_CITY

    # API anahtarı yoksa akış bozulmasın, sadece belgeye dayanılacağını söyle
    if not WEATHER_API_KEY:
        return (
            False,
            city,
            "Hava durumu API anahtarı bulunamadı. Lütfen cevabını sadece belge bağlamına dayandır.",
        )

    try:
        params = {
            "q": city,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "lang": "tr",
        }
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=8,
        )
        if resp.status_code != 200:
            return (
                False,
                city,
                "Hava durumu şu anda alınamıyor. Lütfen cevabını sadece belge bağlamına dayandır.",
            )

        data = resp.json()
        weather_list = data.get("weather") or []
        main = data.get("main") or {}
        wind = data.get("wind") or {}

        description = (
            weather_list[0].get("description", "").capitalize()
            if weather_list
            else ""
        )
        temp = main.get("temp")
        feels = main.get("feels_like")
        humidity = main.get("humidity")
        wind_speed = wind.get("speed")

        parts = []
        if description:
            parts.append(description)
        if temp is not None:
            parts.append(f"Sıcaklık: {temp:.1f}°C")
        if feels is not None:
            parts.append(f"Hissedilen: {feels:.1f}°C")
        if humidity is not None:
            parts.append(f"Nem: %{humidity}")
        if wind_speed is not None:
            parts.append(f"Rüzgar: {wind_speed:.1f} m/sn")

        if not parts:
            return (
                False,
                city,
                "Hava durumu verisi alınamadı. Lütfen cevabını sadece belge bağlamına dayandır.",
            )

        summary = f"{city} için güncel hava durumu: " + ", ".join(parts) + "."
        return True, city, summary
    except Exception:
        # Her türlü hata durumunda sadece belgeye dayanılmasını sağla
        return (
            False,
            city,
            "Hava durumu servisine ulaşılamadı. Lütfen cevabını sadece belge bağlamına dayandır.",
        )


def format_conversation_history(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    """Sohbet geçmişini prompt formatına çevirir (son N mesaj)"""
    if not history:
        return ""
    
    # Son N mesajı al (çok uzun olmasın)
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    formatted = []
    for msg in recent_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            formatted.append(f"Kullanıcı: {content}")
        elif role == "assistant":
            formatted.append(f"Asistan: {content}")
    
    if formatted:
        return "\n".join(formatted)
    return ""


def format_user_profile(profile: Dict[str, str]) -> str:
    """Kullanıcı profilini (ad, üşüme vb.) prompt için okunur hale getirir."""
    if not profile:
        return ""
    parts = []
    name = profile.get("name")
    if name:
        parts.append(f"Kullanıcının adı: {name}")
    cold = profile.get("cold_sensitivity")
    if cold == "high":
        parts.append("Kullanıcı soğuğa karşı hassastır ve çabuk üşür.")
    if not parts:
        return ""
    return "\n".join(parts)


def call_gemini(
    context: str,
    question: str,
    sources: List[str],
    weather_summary: str,
    city: str,
    has_weather: bool,
    is_weather_question: bool = False,
    conversation_history: List[Dict[str, str]] = None,
    profile: Dict[str, str] | None = None,
) -> str:
    model = get_gemini_model()
    doc_list = ", ".join(sorted(current_docs.keys())) if current_docs else "Belge"

    weather_block = f"Hava durumu özeti (şehir: {city}):\n{weather_summary}\n"

    # Kullanıcı profilini formatla (ad, üşüme vb.)
    profile_text = ""
    if profile:
        profile_str = format_user_profile(profile)
        if profile_str:
            profile_text = (
                "\n\nKULLANICI PROFİLİ (belgeden bağımsız, sohbetten öğrenilen):\n"
                f"{profile_str}\n\n"
                "ÖNEMLİ:\n"
                "- Bu bilgiler belge içinde aranmaz; doğrudan doğru kabul edilir.\n"
                "- Kullanıcı kendi adını veya özelliklerini söylediyse, bunları belge yerine sohbet geçmişine göre kullan.\n"
            )

    # Önceki sohbet geçmişini formatla
    history_text = ""
    if conversation_history:
        history_text = format_conversation_history(conversation_history, max_turns=5)
        if history_text:
            history_text = f"\n\nÖNCEKİ SOHBET GEÇMİŞİ (bağlam için):\n{history_text}\n\nÖNEMLİ: Yukarıdaki önceki mesajları dikkate al ve kullanıcının önceki söylediklerini hatırla. Örneğin kullanıcı 'üşürüm' dediyse, kıyafet önerirken daha sıcak tutan kıyafetler öner."

    # Hava durumu soruları için özel kurallar
    if is_weather_question and has_weather:
        weather_rules = """
ÖNEMLİ - Hava Durumu Sorusu:
- Bu soru hava durumu, aktivite veya kıyafet önerisi ile ilgilidir.
- Belge bağlamında hava durumu bilgisi olmasa bile, hava durumu API'sinden gelen bilgiyi kullanabilirsin.
- Hava durumu API'sinden gelen bilgi (sıcaklık, yağış, rüzgar, nem vb.) bağımsız bir kaynaktır ve PDF'te olmasa bile kullanılabilir.
- Belge bağlamında hava durumu ile ilgili bilgi yoksa, sadece hava durumu API bilgisine dayanarak pratik tavsiyeler ver.
- Örnek: "Hava soğuk, kalın giyin" veya "Yağmur bekleniyor, yanınıza şemsiye alın" gibi direkt hava durumuna göre tavsiyeler verebilirsin.
"""
    else:
        weather_rules = """
- Belge bağlamında olmayan bir bilgiyi "Bu bilgi belgede yer almıyor." diyerek açıkça belirt.
- Hava durumu bilgisi yoksa veya alınamadıysa cevabını sadece belge bağlamına dayandır.
"""

    prompt = f"""
Sen bir akademik belge analiz ve tavsiye asistanısın.

KAYNAKLAR:
1) Belge bağlamı (PDF/TXT içeriği)
2) Hava durumu özeti (varsa)
3) Sohbet geçmişi ve kullanıcı profili (kullanıcının kendisiyle ilgili verdiği bilgiler: ad, üşüme vb.)

Kurallar:
{weather_rules}
- Kullanıcının kendisiyle ilgili verdiği kişisel bilgiler (ad, "üşürüm" gibi ifadeler) için bu bilgileri sohbet geçmişi / kullanıcı profilinden kullan; bu bilgiler için belgede geçme şartı YOKTUR.
- Belge içeriğiyle ilgili sorularda, sadece belge bağlamına dayan ve bağlamda yoksa "Bu bilgi belgede yer almıyor." de.
- Hava durumu bilgisi varsa ve soruda hava durumu ile ilgili bir istek varsa, cevabında hem belge kurallarına hem de hava durumu özetine dayanarak kısa, pratik bir tavsiye üret.
- Tahmin, uydurma veya belge/hava durumu/kullanıcı bilgisi dışında yorum yapma.

ÖNEMLİ: Her bilginin hangi kaynaktan geldiğini mutlaka belirt!
- Belge için: [Kaynak: belge_adı.pdf, Sayfa X]
- Hava durumu için: [Kaynak: Hava durumu API, Şehir: {city}]
- Kullanıcı profili veya sohbet geçmişi için: [Kaynak: Kullanıcı profili / Sohbet geçmişi]

Yanıt formatı:
Sonuç: <tek cümlelik özet> [Kaynak: ...]

Gerekçe:
- madde 1 [Kaynak: ...]
- madde 2 [Kaynak: ...]
- madde 3 [Kaynak: ...]

Akademik, tarafsız ve bilgilendirici ol.
- Her bilgi parçasından sonra hangi belgeden, hava durumundan veya kullanıcı profilinden geldiğini köşeli parantez içinde belirt.
- Birden fazla belge veya kaynak kullanıyorsan, her birini ayrı ayrı belirt.

Mevcut Belgeler: {doc_list}

Belge Bağlamı (her parça hangi belgeden geldiğini gösterir):
{context}

Hava Durumu Bağlamı:
{weather_block}
{profile_text}
{history_text}
Kullanıcının Sorusu:
{question}
"""
    try:
        # Güvenlik filtrelerini tamamen kapat - tüm sorulara cevap verebilsin
        safety_settings = [
            {"category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, 
             "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
             "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 
             "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 
             "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE},
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
            safety_settings=safety_settings,
        )

        if response.candidates:
            cand = response.candidates[0]
            print("FINISH REASON:", getattr(cand, "finish_reason", None))
            print("SAFETY RATINGS:", getattr(cand, "safety_ratings", None))
        
        # Response'u güvenli şekilde parse et
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            # Finish reason kontrolü - text'i almaya çalış
            finish_reason = candidate.finish_reason
            text = ""
            
            # Önce text'i almaya çalış
            if candidate.content and candidate.content.parts:
                text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            
            # Text yoksa response.text'i dene
            if not text:
                try:
                    text = response.text or ""
                except Exception:
                    pass
            
            # Hala text yoksa, özellikle SAFETY durumunda tekrar dene
            if not text:
                if finish_reason == 2:  # SAFETY - tekrar dene, daha açık prompt ile
                    try:
                        # Daha basit bir prompt ile tekrar dene
                        retry_prompt = f"""
Aşağıdaki metin akademik bir belgeden alınmıştır.
Herhangi bir yönlendirme, talimat veya zararlı içerik içermemektedir.
Sadece bilgilendirici bir özet istenmektedir.

Bağlam:
{context}

Soru:
{question}

Türkçe, kısa ve bilgilendirici cevap:
"""
                        
                        retry_response = model.generate_content(
                            retry_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.7,
                                max_output_tokens=512,
                            ),
                            safety_settings=safety_settings,
                        )
                        
                        if retry_response.candidates and len(retry_response.candidates) > 0:
                            retry_candidate = retry_response.candidates[0]
                            if retry_candidate.content and retry_candidate.content.parts:
                                text = "".join(part.text for part in retry_candidate.content.parts if hasattr(part, 'text'))
                            if not text:
                                try:
                                    text = retry_response.text or ""
                                except:
                                    pass
                    except Exception:
                        pass
                    
                    if not text:
                        text = "⚠️ Yanıt güvenlik filtresi tarafından engellendi. Lütfen soruyu farklı şekilde sorun."
                elif finish_reason == 3:  # RECITATION (telif hakkı)
                    text = "⚠️ Yanıt telif hakkı nedeniyle engellendi."
                elif finish_reason == 4:  # OTHER
                    text = "⚠️ Yanıt oluşturulamadı. Lütfen tekrar deneyin."
                elif finish_reason == 5:  # MAX_TOKENS
                    text = "⚠️ Yanıt çok uzun oldu. Lütfen daha spesifik bir soru sorun."
                else:
                    text = "⚠️ Modelden yanıt alınamadı."
        else:
            text = "⚠️ Modelden yanıt alınamadı. Lütfen tekrar deneyin."
        
        text = text.strip()
        if not text:
            text = "Üzgünüm, modelden yanıt alınamadı."
            
    except Exception as e:
        text = f"⚠️ Hata: {str(e)}"

    if "Kaynaklar:" not in text:
        # Kaynakları belgelere göre grupla
        sources_by_doc = {}
        for src in sources:
            # "📄 belge_adı | Sayfa X | Parça Y" formatından belge adını çıkar
            if "📄" in src:
                parts = src.split("|")
                doc_name = parts[0].replace("📄", "").strip()
                page_info = parts[1].strip() if len(parts) > 1 else ""
                chunk_info = parts[2].strip() if len(parts) > 2 else ""
            else:
                doc_name = "Bilinmeyen"
                page_info = src
            
            if doc_name not in sources_by_doc:
                sources_by_doc[doc_name] = []
            sources_by_doc[doc_name].append(f"{page_info} {chunk_info}".strip())
        
        formatted_sources = []
        for doc_name in sorted(sources_by_doc.keys()):
            formatted_sources.append(f"\n📄 {doc_name}:")
            for page_info in sources_by_doc[doc_name]:
                formatted_sources.append(f"  • {page_info}")
        
        text = f"{text.strip()}\n\n📚 Kaynaklar (Belge):{''.join(formatted_sources)}"

        # Hava durumu kaynağını da ekle
        if has_weather:
            text = f"{text}\n\n🌤 Hava Durumu Kaynağı:\n  • OpenWeatherMap API (Şehir: {city})"

    return text


def call_groq(
    context: str,
    question: str,
    sources: List[str],
    weather_summary: str,
    city: str,
    has_weather: bool,
    is_weather_question: bool = False,
    conversation_history: List[Dict[str, str]] = None,
    profile: Dict[str, str] | None = None,
) -> str:
    """Groq API kullanarak LLM çağrısı yapar (güvenlik filtreleri yok, çok hızlı)"""
    try:
        client = get_groq_client()
        doc_list = ", ".join(sorted(current_docs.keys())) if current_docs else "Belge"

        weather_block = f"Hava durumu özeti (şehir: {city}):\n{weather_summary}\n"

        # Kullanıcı profilini formatla (ad, üşüme vb.)
        profile_text = ""
        if profile:
            profile_str = format_user_profile(profile)
            if profile_str:
                profile_text = (
                    "\n\nKULLANICI PROFİLİ (belgeden bağımsız, sohbetten öğrenilen):\n"
                    f"{profile_str}\n\n"
                    "ÖNEMLİ:\n"
                    "- Bu bilgiler belge içinde aranmaz; doğrudan doğru kabul edilir.\n"
                    "- Kullanıcı kendi adını veya özelliklerini söylediyse, bunları belge yerine sohbet geçmişine göre kullan.\n"
                )

        # Önceki sohbet geçmişini formatla
        history_text = ""
        if conversation_history:
            history_text = format_conversation_history(conversation_history, max_turns=5)
            if history_text:
                history_text = f"\n\nÖNCEKİ SOHBET GEÇMİŞİ (bağlam için):\n{history_text}\n\nÖNEMLİ: Yukarıdaki önceki mesajları dikkate al ve kullanıcının önceki söylediklerini hatırla. Örneğin kullanıcı 'üşürüm' dediyse, kıyafet önerirken daha sıcak tutan kıyafetler öner."

        # Hava durumu soruları için özel kurallar
        if is_weather_question and has_weather:
            weather_rules = """
ÖNEMLİ - Hava Durumu Sorusu:
- Bu soru hava durumu, aktivite veya kıyafet önerisi ile ilgilidir.
- Belge bağlamında hava durumu bilgisi olmasa bile, hava durumu API'sinden gelen bilgiyi kullanabilirsin.
- Hava durumu API'sinden gelen bilgi (sıcaklık, yağış, rüzgar, nem vb.) bağımsız bir kaynaktır ve PDF'te olmasa bile kullanılabilir.
- Belge bağlamında hava durumu ile ilgili bilgi yoksa, sadece hava durumu API bilgisine dayanarak pratik tavsiyeler ver.
- Örnek: "Hava soğuk, kalın giyin" veya "Yağmur bekleniyor, yanınıza şemsiye alın" gibi direkt hava durumuna göre tavsiyeler verebilirsin.
"""
        else:
            weather_rules = """
- Belge bağlamında olmayan bir bilgiyi "Bu bilgi belgede yer almıyor." diyerek açıkça belirt.
- Hava durumu bilgisi yoksa veya alınamadıysa cevabını sadece belge bağlamına dayandır.
"""

        prompt = f"""Sen bir akademik belge analiz ve tavsiye asistanısın.

KAYNAKLAR:
1) Belge bağlamı (PDF/TXT içeriği)
2) Hava durumu özeti (varsa)
3) Sohbet geçmişi ve kullanıcı profili (kullanıcının kendisiyle ilgili verdiği bilgiler: ad, üşürüm vb.)

Kurallar:
{weather_rules}
- Kullanıcının kendisiyle ilgili verdiği kişisel bilgiler (ad, "üşürüm" gibi ifadeler) için bu bilgileri sohbet geçmişi / kullanıcı profilinden kullan; bu bilgiler için belgede geçme şartı YOKTUR.
- Belge içeriğiyle ilgili sorularda, sadece belge bağlamına dayan ve bağlamda yoksa "Bu bilgi belgede yer almıyor." de.
- Hava durumu bilgisi varsa ve soruda hava durumu ile ilgili bir istek varsa, cevabında hem belge kurallarına hem de hava durumu özetine dayanarak kısa, pratik bir tavsiye üret.
- Tahmin, uydurma veya belge/hava durumu/kullanıcı bilgisi dışında yorum yapma.

ÖNEMLİ: Her bilginin hangi kaynaktan geldiğini mutlaka belirt!
- Belge için: [Kaynak: belge_adı.pdf, Sayfa X]
- Hava durumu için: [Kaynak: Hava durumu API, Şehir: {city}]
- Kullanıcı profili veya sohbet geçmişi için: [Kaynak: Kullanıcı profili / Sohbet geçmişi]

Yanıt formatı:
Sonuç: <tek cümlelik özet> [Kaynak: ...]

Gerekçe:
- madde 1 [Kaynak: ...]
- madde 2 [Kaynak: ...]
- madde 3 [Kaynak: ...]

Akademik, tarafsız ve bilgilendirici ol.
- Her bilgi parçasından sonra hangi belgeden, hava durumundan veya kullanıcı profilinden geldiğini köşeli parantez içinde belirt.
- Birden fazla belge veya kaynak kullanıyorsan, her birini ayrı ayrı belirt.

Mevcut Belgeler: {doc_list}

Belge Bağlamı (her parça hangi belgeden geldiğini gösterir):
{context}

Hava Durumu Bağlamı:
{weather_block}
{profile_text}
{history_text}

Kullanıcının Sorusu:
{question}
"""

        # Groq API çağrısı - Güncel model listesi (eski model kullanımdan kaldırıldı)
        # Model adayları: önce en güçlü olanı dene
        model_candidates = [
            "llama-3.3-70b-versatile",      # Yeni versiyon (önerilen)
            "llama-3.1-70b-versatile",      # Eski (fallback)
            "llama-3.1-8b-instant",         # Daha hızlı ama küçük
            "mixtral-8x7b-32768",           # Uzun bağlam
            "gemma2-9b-it",                 # Google'ın modeli
        ]
        
        chat_completion = None
        last_error = None
        
        for model_name in model_candidates:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Sen Türkçe konuşan, akademik belge analiz ve tavsiye konusunda uzman bir asistansın. Her zaman kaynak belirtmeyi unutma."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model_name,
                    temperature=0.3,
                    max_tokens=1024,
                )
                # Başarılı oldu, döngüden çık
                break
            except Exception as e:
                last_error = e
                # Bu model çalışmadı, bir sonrakini dene
                continue
        
        if chat_completion is None:
            raise RuntimeError(f"Hiçbir Groq modeli çalışmadı. Son hata: {last_error}")

        text = chat_completion.choices[0].message.content.strip()

        if not text:
            text = "Üzgünüm, modelden yanıt alınamadı."

    except Exception as e:
        text = f"⚠️ Groq API Hatası: {str(e)}"

    # Kaynakları ekle (aynı format)
    if "Kaynaklar:" not in text and sources:
        sources_by_doc = {}
        for src in sources:
            if "📄" in src:
                parts = src.split("|")
                doc_name = parts[0].replace("📄", "").strip()
                page_info = parts[1].strip() if len(parts) > 1 else ""
                chunk_info = parts[2].strip() if len(parts) > 2 else ""
            else:
                doc_name = "Bilinmeyen"
                page_info = src
            
            if doc_name not in sources_by_doc:
                sources_by_doc[doc_name] = []
            sources_by_doc[doc_name].append(f"{page_info} {chunk_info}".strip())
        
        formatted_sources = []
        for doc_name in sorted(sources_by_doc.keys()):
            formatted_sources.append(f"\n📄 {doc_name}:")
            for page_info in sources_by_doc[doc_name]:
                formatted_sources.append(f"  • {page_info}")
        
        text = f"{text.strip()}\n\n📚 Kaynaklar (Belge):{''.join(formatted_sources)}"

    # Hava durumu kaynağını da ekle
    if has_weather:
        text = f"{text}\n\n🌤 Hava Durumu Kaynağı:\n  • OpenWeatherMap API (Şehir: {city})"

    return text


def answer_question(message, city, history, top_k, use_mmr, model_choice):
    global conversation_history, user_profile
    history = history or []
    if not message:
        return history, history, ""

    if active_collection is None:
        reply = "Önce bir PDF/TXT yükleyip indeks oluşturmalısın."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        # Hafızaya da ekle
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": reply})
        return history, history, ""

    try:
        # Önce kullanıcı profilini güncelle (ad, üşüme vb.)
        update_user_profile(message)

        message = sanitize_question(message)
        # Şehri hazırla (boşsa varsayılan Elazığ kullanılır)
        city = (city or "").strip() or DEFAULT_CITY

        # Hava durumunu al (gerçek API veya güvenli fallback)
        has_weather, normalized_city, weather_summary = get_weather_summary(city)

        # Sorunun hava durumu ile ilgili olup olmadığını kontrol et
        is_weather_question = is_weather_related_question(message)

        # Önce belge bağlamını getir
        documents, metadatas = retrieve_chunks(
            message, top_k=int(top_k), use_mmr=bool(use_mmr)
        )
        
        # Model seçimine göre çağrı fonksiyonunu belirle
        if model_choice == "Groq (Önerilen - Güvenlik Filtresi Yok)":
            call_llm = call_groq
        else:
            call_llm = call_gemini

        # Hava durumu sorularında belge bağlamı bulunamasa bile devam et
        if not documents:
            if is_weather_question and has_weather:
                # Hava durumu sorusu ve API'den veri var, sadece hava durumuna göre cevap ver
                context = "Belge bağlamı bulunamadı, ancak hava durumu bilgisi mevcut."
                sources = []
                reply = call_llm(
                    context=context,
                    question=message,
                    sources=sources,
                    weather_summary=weather_summary,
                    city=normalized_city,
                    has_weather=has_weather,
                    is_weather_question=True,
                    conversation_history=conversation_history,
                    profile=user_profile,
                )
            else:
                reply = "Bağlam bulunamadı. Daha farklı bir soru deneyebilirsin."
        else:
            raw_context = format_context(documents, metadatas)

            context = f"""
Bu içerik tamamen akademik ve bilgilendirme amaçlıdır.
Gerçek kişi, suç veya suistimal içermemektedir.

{raw_context}
"""

            sources = build_sources(metadatas)
            reply = call_llm(
                context=context,
                question=message,
                sources=sources,
                weather_summary=weather_summary,
                city=normalized_city,
                has_weather=has_weather,
                is_weather_question=is_weather_question,
                conversation_history=conversation_history,
                profile=user_profile,
            )
    except Exception as exc:
        reply = f"Hata: {exc}"

    # Hem Gradio history'ye hem de global conversation_history'ye ekle
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    conversation_history.append({"role": "user", "content": message})
    conversation_history.append({"role": "assistant", "content": reply})
    
    # Hafızayı çok uzamasın diye sınırla (son 20 mesaj)
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    
    return history, history, ""


def handle_upload(file_objs, history):
    history = history or []
    files = file_objs if isinstance(file_objs, list) else [file_objs]
    files = [f for f in files if f]
    if not files:
        return history, history, "Lütfen en az bir dosya seçin."

    statuses = []
    for file_obj in files:
        status = ingest_file(file_obj)
        statuses.append(status)

    doc_list = ", ".join(sorted(current_docs.keys())) or "-"
    summary = "\n".join(statuses)
    summary += f"\n📚 Toplam belge: {len(current_docs)} ({doc_list})"

    return history, history, summary


def clear_chat():
    """Sohbet geçmişini temizler (hem UI hem hafıza)"""
    global conversation_history
    conversation_history = []
    return [], [], ""


def clear_documents(history):
    reset_collection()
    history = history or []
    return history, history, "📁 Koleksiyon temizlendi. Yeni belgeler yükleyin."


with gr.Blocks(title="Mini RAG - Mevzuat") as demo:
    gr.Markdown(
        """
        # 📚 Mini RAG - Mevzuat Asistanı
        Tek bir PDF/TXT yükle, belgede geçen kural ve başlıkları anında sor.
        """
    )

    chatbot = gr.Chatbot(label="Sohbet", height=420)
    status_box = gr.Markdown("🔄 Önce belge yükleyin.")
    history_state = gr.State([])

    with gr.Row():
        file_input = gr.File(
            label="PDF veya TXT yükle",
            file_types=[".pdf", ".txt"],
            file_count="multiple",
        )
        clear_btn = gr.Button("Sohbeti temizle", variant="secondary")
        clear_docs_btn = gr.Button("Belgeleri temizle", variant="stop")

    # Model seçenekleri
    model_choices = ["Gemini (Google)"]
    if GROQ_AVAILABLE and GROQ_API_KEY:
        model_choices.insert(0, "Groq (Önerilen - Güvenlik Filtresi Yok)")
    
    with gr.Row():
        top_k_slider = gr.Slider(
            minimum=3,
            maximum=6,
            step=1,
            value=DEFAULT_TOP_K,
            label="Top K (kaç parça getirilsin?)",
        )
        mmr_checkbox = gr.Checkbox(
            label="MMR ile çeşitliliği artır",
            value=False,
            info="Benzerlik + çeşitlilik dengesi sağlar",
        )
        model_choice = gr.Dropdown(
            label="LLM Modeli",
            choices=model_choices,
            value=model_choices[0],
            info="Groq: Güvenlik filtresi yok, hızlı. Gemini: Google'ın modeli (bazen filtreler)",
        )

    with gr.Row():
        question_box = gr.Textbox(
            label="Sorunuzu yazın",
            placeholder="Örn. 'Belgede gizlilik kuralı ne?' veya 'Bugünkü hava durumuna göre ne önerirsin?'",
            scale=3,
        )
        city_box = gr.Dropdown(
            label="Şehir (hava durumu için)",
            choices=TURKIYE_SEHIRLERI,
            value=DEFAULT_CITY,
            allow_custom_value=True,
            scale=1,
            info="Türkiye'nin 81 şehrinden birini seçin veya yazın",
        )

    file_input.upload(
        fn=handle_upload,
        inputs=[file_input, history_state],
        outputs=[chatbot, history_state, status_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, city_box, history_state, top_k_slider, mmr_checkbox, model_choice],
        outputs=[chatbot, history_state, question_box],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, history_state, question_box],
    )

    clear_docs_btn.click(
        fn=clear_documents,
        inputs=[history_state],
        outputs=[chatbot, history_state, status_box],
    )

    gr.Markdown(
        "Cevaplar her zaman 'Bağlamda yoksa uydurma yok' ilkesine göre üretilir."
    )


if __name__ == "__main__":
    demo.launch()

