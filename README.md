# 🎤 Speech-to-Text API

**Çoklu ASR Motor Desteği ile Gelişmiş Ses Analiz Platformu**

Bu proje, 3 farklı ASR motoru (WhisperX, WhisperX+NeMo, Nvidia Parakeet) ile ses dosyalarını metne dönüştüren, konuşmacı ayırma (diarization), duygu analizi ve Kafka, MongoDB, Elasticsearch entegrasyonu ile çalışan enterprise-grade bir API sistemidir.

## ✨ Temel Özellikler
- 🎯 **3 ASR Motoru**: WhisperX, WhisperX+NeMo, Nvidia Parakeet
- 🎤 **Konuşmacı Ayrımı**: Pyannote.audio ile hassas segmentasyon
- 🧠 **Duygu Analizi**: 6 kategorili HuggingFace tabanlı analiz
- 💾 **Çoklu Depolama**: MongoDB + Elasticsearch entegrasyonu
- 🚀 **Event-Driven**: Kafka ile asenkron işleme
- 🐳 **Container Ready**: Docker Compose ile kolay deployment

## 🚀 Özellikler

### 🎯 ASR Motorları
- **WhisperX Engine**: OpenAI Whisper tabanlı, kelime hizalamalı (alignment) yüksek kaliteli transkripsiyon
- **WhisperX + NeMo**: WhisperX ile NVIDIA NeMo entegrasyonu, gelişmiş çok dilli destek
- **Nvidia Parakeet**: NVIDIA'nın özel NeMo tabanlı ASR motoru, enterprise-grade performans

### 🎤 Konuşmacı Analizi
- **Gelişmiş Konuşmacı Ayırma**: Pyannote.audio ile hassas konuşmacı segmentasyonu ve etiketleme
- **Konuşmacı Yönetimi**: Konuşmacı isimlerini değiştirme, gruplama ve istatistik çıkarma
- **Kronolojik Sıralama**: Konuşmacıları ilk görülme zamanına göre otomatik sıralama

### 🧠 Duygu Analizi
- **6 Kategorili Analiz**: anger, fear, joy, sadness, surprise, neutral duygu kategorileri
- **Segment Bazlı**: Her konuşma segmenti için ayrı duygu tahmini
- **HuggingFace Entegrasyonu**: j-hartmann/emotion-english-distilroberta-base modeli

### 💾 Veri Yönetimi
- **Çoklu Depolama**: 
  - **MongoDB**: Medya metadata'sı ve segment verilerinin kalıcı saklanması
  - **Elasticsearch**: Gelişmiş metin arama, filtreleme ve indexleme
- **Event-Driven Architecture**: Kafka ile asenkron event processing ve sistemler arası bildirim

### 🌐 Modern API
- **FastAPI Tabanlı**: RESTful API, otomatik OpenAPI dokumentasyonu
- **Web UI**: Jinja2 template ile kullanıcı dostu arayüz
- **Asenkron İşleme**: Background task processing ile non-blocking operasyonlar

### 🐳 DevOps & Performans
- **Container Orkestrasyon**: Docker Compose ile tüm servislerin yönetimi
- **Paralel İşleme**: ProcessPoolExecutor ve ThreadPoolExecutor ile performans optimizasyonu
- **Çok Formatı Desteği**: MP3, MP4, WAV, WebM, M4A formatlarında ses dosyası işleme

## 🏗️ Sistem Mimarisi

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│   Client    │───▶│   FastAPI   │───▶│   ASR Engine       │
└─────────────┘    └─────────────┘    │   Selection         │
                           │          └─────────────────────┘
                           │                    │
                    ┌──────┴──────┐    ┌───────▼───────┐
                    │             │    │               │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │  MongoDB  │ │Elasticsearch│ │WhisperX │ │WhisperX+ │ │Nvidia   │
              └───────────┘ └────────────┘ │ Engine  │ │NeMo     │ │Parakeet │
                           │               └─────────┘ └─────────┘ └─────────┘
                    ┌──────▼──────┐              │           │           │
                    │    Kafka    │              └───────────┼───────────┘
                    └─────────────┘                        │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   Pyannote      │
                                                  │   Diarization   │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   Emotion       │
                                                  │   Analysis      │
                                                  └─────────────────┘
```

### 🔧 ASR Motor Seçimi

Sistem 3 farklı ASR motorunu destekler:

1. **WhisperX Engine** (`engine: "whisperx"`)
   - OpenAI Whisper large-v3 modeli
   - Kelime hizalamalı (word alignment) çıktı
   - Pyannote ile konuşmacı ayrımı

2. **WhisperX + NeMo** (`engine: "whisperx_nemo"`)
   - WhisperX tabanlı transkripsiyon
   - NVIDIA NeMo entegrasyonu
   - Gelişmiş çok dilli destek

3. **Nvidia Parakeet** (`engine: "parakeet"`)
   - NVIDIA'nın özel NeMo tabanlı ASR
   - Enterprise-grade performans
   - Yüksek doğruluk oranı

## 📋 Gereksinimler

- Python 3.10+
- Docker & Docker Compose
- FFmpeg (ses dönüştürme için)
- CUDA (opsiyonel, GPU hızlandırma için)

## 🛠️ Kurulum

### 1. Docker Servislerini Başlat

```bash
docker-compose up -d
docker-compose ps
```

### 2. Python Bağımlılıklarını Yükle

```bash
# Sanal ortam oluştur
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements-312-app.txt

# PyTorch CPU versiyonu (özel index gerekli)
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" "torchaudio==2.6.0+cpu"
```

**Önemli Bağımlılıklar:**
- `whisperx==3.4.2` - WhisperX ASR engine
- `pyannote.audio==3.3.2` - Konuşmacı ayrımı
- `fastapi==0.115.5` - Web API framework
- `pymongo==4.10.1` - MongoDB driver
- `elasticsearch==9.1.0` - Elasticsearch client

### 3. Environment Değişkenlerini Ayarla

`.env` dosyası oluşturun:

```env
# MongoDB Ayarları
MONGO_URI=mongodb://mongoadmin:secret123@localhost:27017
MONGO_DB=speech_to_text

# Elasticsearch Ayarları
ELASTIC_URL=http://localhost:9200
INDEX_TO_ES=true

# Kafka Ayarları
KAFKA_BOOTSTRAP=localhost:9093
KAFKA_TOPIC=media_processed

# İşleme Ayarları
HEAVY_WORKERS=2
HEAVY_TIMEOUT=1800

# HuggingFace Token (duygu analizi için)
HUGGINGFACE_TOKEN=hf_your_token_here

# GPU Kullanımı (opsiyonel)
CUDA_VISIBLE_DEVICES=0
```

## 🔄 İş Akışı

1. **Ses Yükleme** → API veya batch script ile ses dosyası yüklenir.
2. **Format Dönüşümü** → `convert_to_wav` ile WAV’a dönüştürülür.
3. **ASR + Diarization** → WhisperX + Pyannote ile metin çıkarılır ve konuşmacılar etiketlenir.
4. **Duygu Analizi** → `EmotionJSONAnalyzer` ile segmentlere duygu etiketi eklenir.
5. **Veri Saklama** → MongoDB’ye kayıt, Elasticsearch’e index.
6. **Event Gönderimi** → Kafka’ya “processed” eventi.
7. **Sonuç Döndürme** → API üzerinden JSON yanıt.

## 🌐 API Endpoints

### Ana Endpoints
- `GET /` → Ana sayfa (HTML UI)
- `GET /favicon.ico` → Site ikonu
- `GET /health` → Servis sağlık durumu kontrolü

### Transkripsiyon İşlemleri
- `POST /transcribe` → Ses dosyası yükle ve işleme başlat
  - **Parameters**: `file` (multipart/form-data), `engine` (optional: "whisperx", "parakeet", "faster-whisper")
  - **Response**: `{"media_id": "uuid"}`
- `GET /results/{media_id}` → İşlenmiş sonuçları getir
  - **Response**: İş durumu (processing/completed/error) ve tam transkripsiyon verileri

### Konuşmacı Yönetimi
- `GET /speakers/{media_id}` → Konuşmacı dağılımı ve istatistikleri
  - **Response**: Konuşmacı listesi, segment sayıları ve örnek metinler
- `POST /speakers/{media_id}/rename` → Konuşmacı isimlerini toplu değiştir
  - **Body**: `{"mapping": {"speaker01": "Ali", "speaker02": "Ayşe"}, "update_json": true, "reindex_es": true}`

### Statik Dosyalar
- `GET /static/*` → CSS, JavaScript ve diğer statik dosyalar

## 📊 Docker Servisleri

| Servis         | Port  | Açıklama            |
|----------------|-------|--------------------|
| MongoDB        | 27017 | Veritabanı         |
| Elasticsearch  | 9200  | Arama motoru       |
| Kafka          | 9093  | Mesaj kuyruğu      |
| Kafka UI       | 8080  | Kafka yönetim UI   |

## 🚀 Kullanım ve Başlatma

### 1. Servisleri Başlat

```bash
# Docker servislerini başlat
docker-compose up -d

# API sunucusunu başlat
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Web arayüzüne erişim: http://localhost:8000
```

### 2. API Kullanım Örnekleri

#### Tek Dosya İşleme

```python
import requests
import time

# Dosya yükleme ve işleme başlatma
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"engine": "whisperx"}  # veya "parakeet", "faster-whisper"
    )

media_id = response.json()["media_id"]
print(f"İş başlatıldı: {media_id}")

# İşlem durumunu kontrol et
while True:
    result = requests.get(f"http://localhost:8000/results/{media_id}")
    status = result.json()["status"]
    
    if status == "completed":
        # Tam sonuçları al
        data = result.json()
        print(f"Transkripsiyon tamamlandı: {len(data['segments'])} segment")
        break
    elif status == "error":
        print("Hata oluştu:", result.json().get("message"))
        break
    else:
        print(f"Durum: {status}")
        time.sleep(5)
```

#### Konuşmacı Yönetimi

```python
# Konuşmacı listesi
speakers = requests.get(f"http://localhost:8000/speakers/{media_id}")
print("Konuşmacılar:", speakers.json())

# Konuşmacı isimlerini değiştir
rename_payload = {
    "mapping": {
        "speaker01": "Ali Yılmaz", 
        "speaker02": "Ayşe Demir"
    },
    "update_json": True,
    "reindex_es": True
}
response = requests.post(
    f"http://localhost:8000/speakers/{media_id}/rename",
    json=rename_payload
)
```

### 3. Toplu İşleme (Batch Processing)

```bash
# main.py ile toplu dosya işleme
python main.py

# Özelleştirme:
# 1. main.py içinde input_dir değişkenini düzenleyin
# 2. Maksimum dosya sayısını ayarlayın (varsayılan: 100)
# 3. Thread havuzu boyutunu değiştirin (varsayılan: 4)
```

### 4. CLI Araçları

```bash
# Veritabanı bağlantısını test et
python -c "from db import test_connection; print('MongoDB:', test_connection())"

# Elasticsearch durumunu kontrol et
curl http://localhost:9200/_cluster/health

# Kafka topic'lerini listele
docker exec speech-kafka kafka-topics --bootstrap-server localhost:9092 --list
```

## 🔧 Sorun Giderme

### Docker Servisleri
```bash
# Tüm servislerin durumunu kontrol et
docker-compose ps

# Log'ları görüntüle
docker-compose logs mongo
docker-compose logs elasticsearch
docker-compose logs kafka
docker-compose logs kafka-ui

# Servisleri yeniden başlat
docker-compose restart mongo
docker-compose down && docker-compose up -d
```

### Veritabanı Bağlantıları
```bash
# MongoDB bağlantısı test
python -c "from db import test_connection; print('MongoDB OK' if test_connection() else 'MongoDB FAILED')"

# Elasticsearch sağlık durumu
curl -X GET "http://localhost:9200/_cluster/health?pretty"
curl -X GET "http://localhost:9200/_cat/indices"

# Kafka topic durumu
docker exec speech-kafka kafka-topics --bootstrap-server localhost:9092 --describe --topic media_processed
```

### Yaygın Sorunlar
1. **Port Çakışması**: 27017, 9200, 9093, 8080 portlarının boş olduğundan emin olun
2. **Bellek Yetersizliği**: Elasticsearch için en az 4GB RAM gerekli
3. **HuggingFace Token**: Duygu analizi için geçerli token gerekli
4. **FFmpeg Eksik**: `sudo apt install ffmpeg` veya `choco install ffmpeg`
5. **Pyannote Auth**: Pyannote modellerine erişim için HF token gerekli

### Debug Modları
```bash
# FastAPI debug modu
python -m uvicorn api:app --reload --log-level debug

# Tek dosya test etme
python -c "
from engines.transcribe_large_multil import transcribe_large
result = transcribe_large('test.wav')
print(result)
"

# MongoDB koleksiyonlarını kontrol et
python -c "
from db import media_col, segments_col
print('Media count:', media_col.count_documents({}))
print('Segments count:', segments_col.count_documents({}))
"
```

## 📈 Performans Optimizasyonu

### Donanım Önerileri
- **CPU**: En az 8 çekirdek (çoklu dosya işleme için)
- **RAM**: En az 16GB (Elasticsearch + ASR modelleri için)
- **GPU**: NVIDIA RTX serisi (CUDA desteği ile 5-10x hızlanma)
- **SSD**: Hızlı disk I/O için NVMe SSD önerili

### Yazılım Ayarları
```env
# .env dosyasında performans ayarları
HEAVY_WORKERS=4                    # CPU çekirdek sayısına göre ayarlayın
HEAVY_TIMEOUT=3600                 # Uzun dosyalar için süreyi artırın
CUDA_VISIBLE_DEVICES=0             # GPU kullanımı için
OMP_NUM_THREADS=4                  # CPU thread sayısı
```

### Elasticsearch Optimizasyonu
```bash
# Index ayarları (development)
curl -X PUT "localhost:9200/segments" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "30s"
  }
}
'
```

### MongoDB Performansı
```bash
# Index oluşturma (production için)
python -c "
from db import media_col, segments_col
media_col.create_index('filename')
segments_col.create_index([('media_id', 1), ('start', 1)])
segments_col.create_index('speaker')
"
```

## 📁 Proje Yapısı

```
speech_to_text/
├── 📁 engines/                     # ASR Engine implementasyonları
│   ├── whisperx_whit_nemo.py      # WhisperX + NeMo entegrasyonu
│   ├── nvidia_parakeet.py         # Nvidia Parakeet ASR
│   ├── transcribe_large_multil.py # Ana transkripsiyon logic
│   └── transcription_worker.py    # Alt süreç işleyicisi
├── 📁 services/                    # İş mantığı servisleri
│   ├── transcription_service.py   # Transkripsiyon orkestrasyon
│   ├── storage_service.py         # Veri saklama yönetimi
│   └── speaker_service.py         # Konuşmacı yönetimi
├── 📁 templates/                   # HTML şablonları
│   └── index.html                 # Web UI ana sayfa
├── 📁 static/                      # CSS/JS/resim dosyaları
├── 📁 tmp_wavs/                    # Geçici ses dosyaları
├── 📁 logs/                        # Sistem log dosyaları
├── 📁 tests/                       # Unit test'ler
├── api.py                          # FastAPI ana uygulama
├── main.py                         # Batch işleme scripti
├── convert_audio.py               # Ses format dönüştürücü
├── emotion_detection.py           # Duygu analizi modülü
├── db.py                          # MongoDB connection helper
├── save_to_mongo.py               # MongoDB yazma işlemleri
├── save_to_elastic.py             # Elasticsearch indexleme
├── kafka_producer.py              # Kafka event publisher
├── kafka_consumer.py              # Kafka event consumer
├── docker-compose.yml             # Container orkestrasyon
├── requirements-312-app.txt       # Python bağımlılıkları
└── .env                           # Environment değişkenleri
```

### Ana Modüller

**`api.py`**: FastAPI web sunucusu, HTTP endpoints
**`engines/`**: Farklı ASR motorlarının implementasyonları
**`services/`**: İş mantığı ve orkestrasyon katmanı
**`main.py`**: Toplu dosya işleme ve batch processing
**`emotion_detection.py`**: HuggingFace tabanlı duygu analizi
**`db.py`**: MongoDB bağlantı yönetimi
**`convert_audio.py`**: FFmpeg ile ses dönüştürme

## 🧪 Test Etme

```bash
# Unit test'leri çalıştır
python -m pytest tests/ -v

# Belirli test dosyası
python -m pytest tests/test_transcription.py -v

# API endpoint test
curl -X GET http://localhost:8000/health
curl -X POST -F "file=@test.wav" http://localhost:8000/transcribe

# Performans testi
python tests/performance_test.py
```

## 🚀 Production Deployment

### Docker ile Production
```bash
# Production build
docker-compose -f docker-compose.prod.yml up -d

# Load balancer ile
docker-compose -f docker-compose.prod.yml --scale api=3 up -d
```

### Environment Production Ayarları
```env
# Production .env
DEBUG=false
HEAVY_WORKERS=8
HEAVY_TIMEOUT=7200
INDEX_TO_ES=true

# Güvenlik
JWT_SECRET=your-secret-key
ALLOWED_ORIGINS=yourdomain.com

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
LOG_LEVEL=INFO
```

### Commit Kuralları
```
feat: yeni özellik ekleme
fix: hata düzeltme
docs: dokümantasyon güncelleme
style: kod formatlama
refactor: kod yeniden düzenleme
test: test ekleme/güncelleme
perf: performans iyileştirme
```


