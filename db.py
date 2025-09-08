# ===============================================
# db.py - MongoDB bağlantı ve koleksiyon yönetimi
# - Tek bir MongoClient örneğini (singleton benzeri) yönetir
# - Varsayılan DB ve koleksiyon referanslarını hazırlar
# - Temel indeksleri oluşturur
# - Bağlantı sağlık kontrolü sağlar
# ===============================================

# Standart kütüphaneler
import os                  # Ortam değişkenleri ve yol işlemleri için
import logging             # Loglama için
from typing import Optional  # İsteğe bağlı (nullable) tip açıklamaları için

# Üçüncü parti kütüphaneler
from dotenv import load_dotenv        # .env dosyasından ortam değişkenlerini yüklemek için
from pymongo import MongoClient       # MongoDB istemcisi
# Not: Gerekirse spesifik hata tipleri için from pymongo.errors import ... ekleyebilirsin

# -------------------------------
# LOG YAPILANDIRMASI
# -------------------------------
log = logging.getLogger("db")         # Bu modül için isimlendirilmiş bir logger
logging.basicConfig(level=logging.INFO)  # Basit bir log konfigürasyonu: INFO ve üstünü yaz

# -------------------------------
# ORTAM DEĞİŞKENLERİNİ YÜKLE
# -------------------------------
load_dotenv()  # .env dosyasını okuyup os.environ'a yükler

# -------------------------------
# KONFİG: URI ve DB adı
# -------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")   # .env'de yoksa localhost
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "speech_to_text")      # Varsayılan DB adı

# -------------------------------
# TEKİL (SINGLETON BENZERİ) REFERANSLAR
# -------------------------------
_client: Optional[MongoClient] = None  # Tek bir MongoClient örneği tutulacak
_db = None                             # Varsayılan Database referansı (lazy init)
media_col = None                       # "media" koleksiyonu referansı (lazy init)
segments_col = None                    # "segments" koleksiyonu referansı (lazy init)

def get_client() -> MongoClient:
    """
    Tek MongoClient örneğini döner.
    Yoksa oluşturur, varsa aynı örneği geri verir.
    """
    global _client
    # Eğer henüz bir client oluşturulmamışsa:
    if _client is None:
        # MongoClient örneğini parametrelerle oluştur
        _client = MongoClient(
            MONGO_URI,                 # Bağlanılacak MongoDB adresi
            serverSelectionTimeoutMS=5000,  # Sunucu seçimi için zaman aşımı (ms)
            connectTimeoutMS=10000,         # TCP bağlantı kurulumu için zaman aşımı (ms)
            socketTimeoutMS=20000,          # Soket üzerinden okuma/yazma zaman aşımı (ms)
            maxPoolSize=20,                 # Bağlantı havuzu maksimum bağlantı sayısı
            retryWrites=True,               # Bazı yazma işlemlerini tekrar dene (idempotent senaryolarda faydalı)
        )
        # Bağlantının gerçekten kurulabildiğini doğrulamak için bir ping at
        _client.admin.command("ping")  # Hata atarsa üstteki try/except'te yakalanacak (çağıran yerlerde)
        log.info("✅ MongoDB client oluşturuldu")  # Başarılıysa bilgi logu
    # Mevcutta oluşturulmuş client'ı döndür
    return _client

def get_db():
    """
    Varsayılan veritabanı (Database) referansını döner.
    İlk çağrıda client üzerinden alınır ve cache'lenir.
    """
    global _db
    # Eğer henüz DB referansı alınmadıysa:
    if _db is None:
        # Önce client'ı al (yoksa oluşturur)
        _db = get_client()[MONGO_DB_NAME]  # İlgili isimdeki veritabanına bağlan
        log.info(f"+ DB seçildi: {MONGO_DB_NAME}")  # Bilgi logu
    # DB referansını döndür
    return _db

def init_collections():
    """
    Koleksiyon referanslarını hazırlar ve gerekli temel indexleri oluşturur.
    Uygulama import edildiğinde bir defa çalışması amaçlanır.
    """
    global media_col, segments_col
    db = get_db()              # Varsayılan DB referansını al

    # Koleksiyon referanslarını hazırla
    media_col = db["media"]        # Medya meta verileri (dosya adı, süre, dil, özet vb.)
    segments_col = db["segments"]  # Parça (segment) detayları (speaker, start, end, text vb.)

    # Temel index oluşturma (create_index idempotent: varsa tekrar yaratmaz)
    media_col.create_index("created_at")                 # Tarihe göre sorguları hızlandırır
    segments_col.create_index([("media_id", 1), ("start", 1)])  # media_id + start sıralı sorguları hızlandırır

def test_connection() -> bool:
    """
    MongoDB'ye bağlanabildiğimizi hızlıca kontrol eder.
    Bağlantı başarılıysa True, hata varsa False döner.
    """
    try:
        # Client'ı al ve admin veritabanına ping at
        get_client().admin.command("ping")
        return True  # Başarılı bağlantı
    except Exception as e:
        # Herhangi bir hata durumunda logla ve False dön
        log.error(f"MongoDB test hatası: {e}")
        return False

# --------------------------------------------------------
# MODÜL İMPORT EDİLDİĞİNDE KOLEKSİYONLARI HAZIRLA
# --------------------------------------------------------
try:
    init_collections()  # Koleksiyon referanslarını ve indexleri hazırla
except Exception as e:
    # MongoDB ayakta değilse burada hata loglanır ama uygulama tamamen çökmez
    log.error(f"Init hatası: {e}")
