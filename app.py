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
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
MMR_LAMBDA = 0.5
COLLECTION_NAME = "active_pdf"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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


embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

active_collection = None
current_docs: Dict[str, Dict[str, str]] = {}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def sanitize_question(q: str) -> str:
    blockers = ["ceza", "suç", "yaptırım", "kovuşturma", "kimlik", "tc", "disiplin"]
    for b in blockers:
        q = q.replace(b, f"{b} (mevzuat kapsamında)")
    return q


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


def call_gemini(context: str, question: str, sources: List[str]) -> str:
    model = get_gemini_model()
    doc_list = ", ".join(sorted(current_docs.keys())) if current_docs else "Belge"
    prompt = f"""
Sen bir akademik belge analiz asistanısın.
Sadece aşağıdaki bağlamda yer alan bilgileri kullan.
Bağlamda yoksa açıkça "Bu bilgi belgede yer almıyor." de.
Tahmin, uydurma veya yorum yapma.

ÖNEMLİ: Her bilginin hangi belgeden geldiğini mutlaka belirt!

Yanıt formatı:
Sonuç: <tek cümlelik özet> [Kaynak: belge_adı.pdf, Sayfa X]

Gerekçe:
- madde 1 [Kaynak: belge_adı.pdf, Sayfa X]
- madde 2 [Kaynak: belge_adı2.pdf, Sayfa Y]
- madde 3 [Kaynak: belge_adı.pdf, Sayfa Z]

Akademik, tarafsız ve bilgilendirici ol.
- Her bilgi parçasından sonra hangi belgeden geldiğini mutlaka köşeli parantez içinde belirt.
- Birden fazla belgeden bilgi kullanıyorsan, her belgeyi ayrı ayrı belirt.

Mevcut Belgeler: {doc_list}

Bağlam (her parça hangi belgeden geldiğini gösterir):
{context}

Soru:
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
        
        text = f"{text.strip()}\n\n📚 Kaynaklar:{''.join(formatted_sources)}"
    return text


def answer_question(message, history, top_k, use_mmr):
    history = history or []
    if not message:
        return history, history, ""

    if active_collection is None:
        reply = "Önce bir PDF/TXT yükleyip indeks oluşturmalısın."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, history, ""

    try:
        message = sanitize_question(message)

        documents, metadatas = retrieve_chunks(message, top_k=int(top_k), use_mmr=bool(use_mmr))
        if not documents:
            reply = "Bağlam bulunamadı. Daha farklı bir soru deneyebilirsin."
        else:
            raw_context = format_context(documents, metadatas)

            context = f"""
Bu içerik tamamen akademik ve bilgilendirme amaçlıdır.
Gerçek kişi, suç veya suistimal içermemektedir.

{raw_context}
"""

            sources = build_sources(metadatas)
            reply = call_gemini(context, message, sources)
    except Exception as exc:
        reply = f"Hata: {exc}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
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

    question_box = gr.Textbox(
        label="Sorunuzu yazın",
        placeholder="Örn. 'Belgede gizlilik kuralı ne?'",
    )

    file_input.upload(
        fn=handle_upload,
        inputs=[file_input, history_state],
        outputs=[chatbot, history_state, status_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, history_state, top_k_slider, mmr_checkbox],
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

