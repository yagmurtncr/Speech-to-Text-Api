import os

from dotenv import load_dotenv
from transcribe_whisperx import transcribe_whisperx

# .env dosyasını yükle
load_dotenv()

# Test edilecek dosya
file_path = r"C:\Users\ytunc\Downloads\audio_1.wav"

# Çalıştır
result = transcribe_whisperx(file_path)

# Çıktı
print("\n=== TEST SONUCU ===")
print("Dil:", result.get("language", "unknown"))
print("Özet:", result.get("summary", ""))
print("Toplam Segment:", len(result.get("segments", [])))

for seg in result.get("segments", []):
    print(f"\n[Speaker {seg['speaker']}] {seg['start']:.2f}s - {seg['end']:.2f}s")
    print("Metin:", seg['text'])
