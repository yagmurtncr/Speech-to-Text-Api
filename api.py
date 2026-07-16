# api.py — FastAPI HTTP katmanı
# ------------------------------------------------------------
# Amaç:
#   1) HTTP endpoint'lerini tanımlamak (upload, sonuç, konuşmacılar vb.)
#   2) Servis fonksiyonlarına delege etmek (iş mantığı servislerde)
#   3) Statik dosya ve template sunmak
# Not:
#   - Ağır işler (ASR, diarization, özet, duygu) background süreçlerde.
#   - Bu dosya yalın ve okunabilir olmalı; "thin controller" yaklaşımı.
# ------------------------------------------------------------

# 1) Ortam / log / uyarı ayarları (stabil ve sakin çalışsın)
import logging
import os
import warnings

_CPU_THREADS = str(max(2, (os.cpu_count() or 4)))
os.environ["OMP_NUM_THREADS"] = _CPU_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _CPU_THREADS
os.environ["MKL_NUM_THREADS"] = _CPU_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _CPU_THREADS

for name in ["uvicorn", "uvicorn.access", "transformers", "ctranslate2", "pyannote", "speechbrain", "elasticsearch"]:
    logging.getLogger(name).setLevel(logging.WARNING)  # Gürültülü logger’ları sustur

warnings.filterwarnings("ignore", module="pyannote")    # Belirli paket uyarılarını kapat
warnings.filterwarnings("ignore", module="speechbrain")
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# 2) FastAPI / Starlette / Pydantic importları
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# 3) Servis katmanı ve DB yardımcıları
from db import test_connection
from services.speaker_service import get_speakers, rename_speakers
from services.storage_service import get_job, get_result_blob, on_startup, results_cache
from services.transcription_service import upload_and_start

# 4) Uygulama ve dizinler
app = FastAPI(title="Speech to Text API", version="2.0.0")  # Temel FastAPI uygulaması

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Bu dosyanın bulunduğu klasör (mutlak yol)
STATIC_DIR = os.path.join(BASE_DIR, "static")               # Statik dosyalar için klasör
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")         # Jinja2 template klasörü
os.makedirs(STATIC_DIR, exist_ok=True)                      # Klasörler yoksa oluştur
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # /static altında dosya sun
templates = Jinja2Templates(directory=TEMPLATES_DIR)                    # Jinja2 ile HTML render

# 5) Lifecycle: Uygulama başlangıcında bir defa çalışır
@app.on_event("startup")
async def _startup():
    # - Mongo bağlantısını test eder
    # - Elasticsearch indexlerini hazırlar
    # - jobs koleksiyonunda TTL index (7 gün) oluşturur
    await on_startup()

# 6) UI Giriş: Basit HTML sayfası (templates/index.html)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Jinja2’de url_for vb. yardımcılar için request’i şablona geçir
    return templates.TemplateResponse("index.html", {"request": request})

# 7) Favicon (opsiyonel)
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")          # /static/favicon.ico konumu
    return FileResponse(path) if os.path.exists(path) else JSONResponse(status_code=404, content={"detail": "favicon not found"})

# 8) Healthcheck: Basit durum bilgisi
@app.get("/health")
async def health():
    # test_connection() True/False döndüğü için bool(...) ile güvenceye al
    return {"status": "healthy", "mongodb": bool(test_connection())}

# 9) Transcribe başlat: Dosyayı al, arka plan işini tetikle ve hemen media_id döndür
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), engine: str = Form("whisperx")):
    engine = (engine or "whisperx").lower()                 # Parametre normalization
    print(f"[TRANSCRIBE] engine={engine}")                  # Basit log
    # upload_and_start:
    #   - Dosyayı tmp’e yazar
    #   - RAM+Mongo jobs kaydı oluşturur (processing)
    #   - Ağır transcribe işini asyncio.create_task ile başlatır
    return await upload_and_start(file, options={"engine": engine})

# 10) Sonuç sorgu: UI polling ile iş durumu/çıktı
@app.get("/results/{media_id}")
async def results(media_id: str):
    # 10.1 RAM önbellek (en hızlı yol)
    info = results_cache().get(media_id)                    # RAM'de job var mı?
    if info and info.get("status") == "completed":          # Tamamlandıysa
        blob = get_result_blob(info.get("result_path"))     # Diskteki JSON'u oku
        if blob is not None:
            # "status": "completed" ile birlikte tam JSON dön
            return JSONResponse(status_code=200, content={"status": "completed", **blob})
        # JSON okunamadı -> 500
        return JSONResponse(status_code=500, content={"status": "error", "message": "Result JSON okunamadı (RAM)"})

    # 10.2 Mongo 'jobs' üzerinden sorgu
    job = get_job(media_id)                                 # Job dokümanı
    if not job:
        # 404 dön ki frontend doğru şekilde hatayı yakalasın
        raise HTTPException(status_code=404, detail="Media ID bulunamadı")

    if job.get("status") == "completed":                    # Mongo completed ise
        blob = get_result_blob(job.get("result_path"))      # JSON'u diskte aç
        if blob is not None:
            # RAM’i ısıt: Sonraki isteklerde disk okumayı azaltır
            results_cache()[media_id] = {
                "status": "completed",
                "result_path": job.get("result_path"),
                "mongo_id": job.get("mongo_id")
            }
            return JSONResponse(status_code=200, content={"status": "completed", **blob})
        return JSONResponse(status_code=500, content={"status": "error", "message": "Result JSON okunamadı (disk)"})

    # 10.3 Hâlâ tamamlanmadıysa geçerli durum/mesaj
    status_val = job.get("status") or "processing"
    if status_val == "error":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": job.get("error") or "İş hata ile sonuçlandı"}
        )
    # queued / processing vb. 200
    return JSONResponse(status_code=200, content={"status": status_val})

# 11) Konuşmacı dağılımı: Özet bilgi
@app.get("/speakers/{media_id}")
async def speakers(media_id: str):
    # get_speakers:
    #   - segments koleksiyonunda media_id ile filtreler
    #   - speaker bazında grup/istatistik çıkarır
    #   - {speaker, count, example_text, langs} listesi döndürür
    return await get_speakers(media_id)

# 12) Konuşmacı isimlerini topluca yeniden adlandır
class RenamePayload(BaseModel):
    mapping: dict[str, str] = Field(..., description="{'speaker01':'Ali','speaker02':'Ayşe'}")  # Gerekli: eski→yeni
    update_json: bool = True     # Sonuç JSON dosyasında da isimleri değiştir
    reindex_es: bool = True      # ES'ye yeniden yaz (opsiyonel, varsa)

@app.post("/speakers/{media_id}/rename")
async def speakers_rename(media_id: str, payload: RenamePayload):
    if not payload.mapping:                                     # Boş/geçersiz mapping kontrolü
        raise HTTPException(status_code=400, detail="Boş mapping")
    # rename_speakers:
    #   - Mongo segments'te updateMany pipeline ile speaker_raw korunarak speaker güncellenir
    #   - update_json=True ise result JSON içi de güncellenir
    #   - reindex_es=True ise ES’ye yeniden yazılır
    #   - Son durumda gruplama ile güncel dağılım döner
    return await rename_speakers(media_id, payload.mapping, payload.update_json, payload.reindex_es)
