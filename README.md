# 🎤 Speech-to-Text API

<p>
  <img src="https://github.com/yagmurtncr/Speech-to-Text-Api/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/WhisperX%20%7C%20NeMo%20%7C%20Parakeet-ASR-EE4C2C" />
  <img src="https://img.shields.io/badge/Kafka-231F20?logo=apachekafka&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-47A248?logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/Elasticsearch-005571?logo=elasticsearch&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

> **Enterprise-grade speech-analysis API.** Transcribes audio with three interchangeable ASR
> engines (WhisperX, WhisperX+NeMo, NVIDIA Parakeet), plus speaker diarization, emotion analysis
> and summarization — event-driven via Kafka with MongoDB + Elasticsearch storage.

An advanced speech-analysis platform that converts audio to text with three different ASR
engines, performs speaker diarization and emotion analysis, and persists results through a
Kafka → MongoDB + Elasticsearch pipeline.

## 🏗️ Architecture

```mermaid
flowchart TD
    Client["Client / Web UI"] -->|"upload audio"| API["FastAPI"]
    API --> Prod["Kafka Producer"]
    Prod --> Topic(["Kafka Topic"])
    Topic --> Cons["Kafka Consumer (worker)"]

    Cons --> Conv["Audio conversion"]
    Conv --> ASR{"ASR Engine"}
    ASR -->|option 1| WX["WhisperX"]
    ASR -->|option 2| WXN["WhisperX + NeMo"]
    ASR -->|option 3| PK["NVIDIA Parakeet"]

    WX & WXN & PK --> Diar["Speaker diarization (Pyannote)"]
    Diar --> Post["Post-processing / hypothesis fix"]
    Post --> Emo["Emotion analysis (6 classes)"]
    Emo --> Sum["Summarization"]

    Sum --> Mongo[("MongoDB")]
    Sum --> ES[("Elasticsearch")]
    Mongo & ES --> API
    API -->|"structured JSON"| Client
```

## ✨ Features

**ASR engines** — choose one per request:
- **WhisperX** (`whisperx`) — OpenAI Whisper large-v3 with word-level alignment
- **WhisperX + NeMo** (`whisperx_nemo`) — WhisperX transcription with NVIDIA NeMo, richer multilingual support
- **NVIDIA Parakeet** (`parakeet`) — NVIDIA's NeMo-based engine, enterprise-grade performance

**Analysis**
- 🎤 **Speaker diarization** — precise segmentation & labelling with Pyannote.audio, speaker renaming, grouping and chronological ordering
- 🧠 **Emotion analysis** — 6 classes (anger, fear, joy, sadness, surprise, neutral) per segment, HuggingFace-based
- 📝 **Post-processing & summarization** — hypothesis fixing and summary generation

**Platform**
- 🚀 **Event-driven** — asynchronous processing with Kafka
- 💾 **Dual storage** — MongoDB (metadata & segments) + Elasticsearch (search & indexing)
- 🌐 **FastAPI** — REST API with auto OpenAPI docs and a Jinja2 web UI
- ⚡ **Parallelism** — `ProcessPoolExecutor` / `ThreadPoolExecutor` for throughput
- 🎬 **Formats** — MP3, MP4, WAV, WebM, M4A
- 🐳 **Docker Compose** — one command brings up all services

## 🛠️ Setup

**1. Start the infrastructure**
```bash
docker-compose up -d
docker-compose ps
```

**2. Install Python dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements-312-app.txt
# CPU-only PyTorch (needs a dedicated index):
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" "torchaudio==2.6.0+cpu"
```

**3. Configure environment** — create a `.env` file:
```env
# MongoDB
MONGO_URI=mongodb://mongoadmin:secret123@localhost:27017
MONGO_DB=speech_to_text
# Elasticsearch
ELASTIC_URL=http://localhost:9200
# Kafka
KAFKA_BOOTSTRAP=localhost:9093
KAFKA_TOPIC=media_processed
# HuggingFace token (for emotion analysis)
HF_TOKEN=your_hf_token
```

## 🚀 Run

```bash
docker-compose up -d
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Web UI: http://localhost:8000
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Home page (HTML UI) |
| `GET`  | `/health` | Service health check |
| `POST` | `/transcribe` | Upload an audio file and start processing |
| `GET`  | `/results/{media_id}` | Fetch processed results |
| `GET`  | `/speakers/{media_id}` | Speaker distribution and statistics |
| `POST` | `/speakers/{media_id}/rename` | Bulk-rename speakers |

**Example — upload & poll:**
```python
import requests

# start processing
with open("audio.mp3", "rb") as f:
    media_id = requests.post("http://localhost:8000/transcribe", files={"file": f}).json()["media_id"]

# fetch results
result = requests.get(f"http://localhost:8000/results/{media_id}").json()
```

## 🧪 Tests & CI

```bash
ruff check .     # lint
pytest -q        # dependency-free unit tests (segment merging, hypothesis extraction)
```

GitHub Actions runs `ruff` + `pytest` on every push/PR. The scripts under `tests/*.py` are live
integration checks (Kafka / Mongo / models) and are kept separate from the CI unit tests.

## 📁 Project Structure

```text
api.py                 FastAPI app, routes, background tasks
engines/               WhisperX / WhisperX+NeMo / Parakeet ASR engines
services/              transcription, speaker and storage services
kafka_producer.py      publishes media events
kafka_consumer.py      worker that runs the pipeline
post_processor.py      segment merging & cleanup
hypothesis_fixer.py    transcript hypothesis extraction
emotion_detection.py   6-class emotion analysis
text_summarizer.py     summarization
save_to_mongo.py       MongoDB persistence
save_to_elastic.py     Elasticsearch indexing
tests/unit/            fast, dependency-free unit tests (run in CI)
docker-compose.yml     Kafka, MongoDB, Elasticsearch
```

## 🚀 Production Notes

- GPU is recommended for the heavy ASR/diarization models.
- Tune worker/thread pool sizes and Elasticsearch/MongoDB indexes for your load.
- Manage secrets via environment variables; never commit `.env`.

## 📄 License

Released under the [MIT License](LICENSE).
