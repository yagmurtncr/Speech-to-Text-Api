# ------------------------------------------------------------
# storage_service.py
# Amaç: Ortak saklama/erişim katmanı.
# Kullanım sırası:
# 1) on_startup()  -> jobs TTL + ES indexleri hazırlar.
# 2) set_processing()/set_completed()/set_error() -> RAM ve Mongo 'jobs' kaydı tutar.
# 3) get_result_blob() -> /results için JSON'u diskten okur.
# 4) refresh_result_json_speakers() -> rename sonrası JSON'u yerinde günceller.
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from db import media_col, segments_col, test_connection
from save_to_elastic import create_indices

# Yol / Önbellek Ayarları
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# os.path.dirname(__file__): Bu dosyanın bulunduğu klasörü verir (çalışma dizininden bağımsız).
# os.path.join(..., ".."): Bir üst klasöre çıkar (..).
# os.path.abspath(...): Ortaya çıkan yolu mutlak hâle getirir ve .. gibi göreli kısımları temizler.

TMP_DIR = os.path.join(BASE_DIR, "tmp_wavs")
# BASE_DIR altında tmp_wavs isimli bir klasör yolu üretir.
os.makedirs(TMP_DIR, exist_ok=True)

_results: Dict[str, Dict[str, Any]] = {}
# _results bir sözlüktür; dış anahtar str (genelde media_id), değer ise yine dict (özete/sonuca dair karma alanlar).

# MongoDB jobs koleksiyonu - güvenli başlatma
jobs = None
try:
    if media_col is not None:
        jobs = media_col.database["jobs"]
except Exception:
    jobs = None

def results_cache() -> Dict[str, Dict[str, Any]]:
    # Bu sözlüğün anahtarları str
    return _results

# 1) Startup: Uygulama başlarken çalışan fonksiyon
async def on_startup():
    """
    Uygulama başlarken çağrılır:
    - Mongo bağlantısı test edilir.
    - Elasticsearch indexleri oluşturulur.
    - Mongo 'jobs' koleksiyonu için 7 günlük TTL ayarlanır.
    """
    global jobs
    
    mongo_ok = test_connection()
    print("Startup: Mongo:", "OK" if mongo_ok else "FAILED")
    
    # Jobs koleksiyonu yeniden başlatma denemesi
    if mongo_ok and jobs is None:
        try:
            if media_col is not None:
                jobs = media_col.database["jobs"]
                print("+ Jobs koleksiyonu yeniden bağlandı")
        except Exception as e:
            print(f"- Jobs koleksiyonu başlatılamadı: {e}")
    
    try:
        if create_indices():
            print("+ ES indexleri hazır")
    except Exception as e:
        print(f"- ES index hatası: {e}")
    
    # Jobs TTL (7 gün)
    try:
        if jobs is not None:
            jobs.create_index("updated_at", expireAfterSeconds=60*60*24*7)
            print("+ Jobs TTL ayarlandı (7 gün)")
            # updated_at değeri, 7 gün'den eski olan belgeler TTL monitörü tarafından otomatik silinir.
    except Exception:
        pass

# 2) Job Durumunu Kaydetme ve Güncelleme (RAM + MongoDB)
def set_processing(media_id: str, tmp_path: str, filename: str):
    """
    Her medya işlenmeye başlandığında çağrılır.
    - RAM'de _results'a "processing" durumu yazılır.
    - Mongo jobs koleksiyonuna yeni kayıt eklenir.
    """
    _results[media_id] = {"status": "processing"}
    if jobs is not None:
        try:
            jobs.insert_one({
                "_id": media_id, "filename": filename, "tmp_path": tmp_path,
                "status": "processing", "error": None, "result_path": None, "mongo_id": None,
                "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()
            })
        except Exception as e:
            print(f"[WARN] Jobs koleksiyonuna yazılamadı: {e}")

def set_completed(media_id: str, json_path: str, mongo_id: Optional[str]):
    """
    Her medya başarıyla işlendiğinde çağrılır.
    - JSON dosyasının yolu ve Mongo medyanın ID'si RAM'e yazılır.
    - Mongo'daki job "completed" durumuna geçirilir.
    """
    _results[media_id] = {
        "status": "completed",
        "result_path": json_path,
        "mongo_id": mongo_id
    }
    if jobs is not None:
        try:
            jobs.update_one({"_id": media_id}, {
                "$set": {
                    "status": "completed",
                    "result_path": json_path,
                    "mongo_id": mongo_id,
                    "updated_at": datetime.utcnow()
                }
            })
        except Exception as e:
            print(f"[WARN] Jobs durumu güncellenemedi: {e}")

def set_error(media_id: str, message: str):
    """
    Herhangi bir aşamada hata oluştuğunda çağrılır.
    - Hata mesajı RAM'e ve Mongo'ya yazılır.
    - Eğer job hiç yoksa Mongo'da matched_count = 0 olur.
    """
    _results[media_id] = {
        "status": "error",
        "message": message
    }
    if jobs is not None:
        try:
            jobs.update_one({"_id": media_id}, {
                "$set": {
                    "status": "error",
                    "error": message,
                    "updated_at": datetime.utcnow()
                }
            })
        except Exception as e:
            print(f"[WARN] Jobs hata durumu güncellenemedi: {e}")

def get_job(media_id: str) -> Optional[dict]:
    """
    Bir job kaydını MongoDB üzerinden döner.
    - refresh_result_json_speakers gibi işlemler için kullanılır.
    """
    if jobs is not None:
        try:
            return jobs.find_one({"_id": media_id})
        except Exception as e:
            print(f"[WARN] Job kaydı okunamadı: {e}")
    return None

# 3) Sonuç JSON’unu Diskten Okuma
def get_result_blob(result_path: Optional[str]) -> Optional[dict]:
    """
    Verilen dosya yolu üzerinden JSON sonucu okur.
    - /results endpoint’inde veya testlerde kullanılır.
    """
    if not result_path or not os.path.exists(result_path):
        return None
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 4) Konuşmacı Etiketlerini JSON Üzerinde Güncelleme
def refresh_result_json_speakers(media_id: str, mapping: dict[str, str]) -> bool:
    """
    Konuşmacı adlarını yeniden adlandırır (örneğin: speaker01 → Ahmet).
    - Transkripsiyon sonrası yeniden isimlendirme yapılmak istenirse çağrılır.
    - Güncelleme sonucu dosyaya tekrar yazılır.
    """
    path = None

    # Önce RAM’deki _results içinden dosya yolu alınır.
    info = _results.get(media_id)
    if info and info.get("result_path") and os.path.exists(info["result_path"]):
        path = info["result_path"]
    else:
        # RAM’de bulunamazsa Mongo jobs kaydına bakılır.
        job = get_job(media_id)
        if job and job.get("result_path") and os.path.exists(job["result_path"]):
            path = job["result_path"]

    # Yol bulunamazsa işlem iptal edilir.
    if not path:
        return False

    # JSON dosyasını aç ve segmentler üzerinde konuşmacı adlarını değiştir.
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = 0
        for s in data.get("segments", []) or []:
            old = s.get("speaker")
            if old in mapping:
                s["speaker"] = mapping[old]
                changed += 1

        # Güncellenmişse tekrar dosyaya yaz.
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                # ensure_ascii=False: Türkçe karakterler kaçışsız yazılsın (okunabilir)
        return True
    except Exception:
        return False
