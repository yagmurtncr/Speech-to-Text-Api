# ============================================================
# convert_audio.py
# FFmpeg kullanarak ses dosyalarını WAV formatına çevirir.
# - 16kHz örnekleme hızı
# - Mono kanal
# - PCM 16-bit LE formatı
# ============================================================

import os
import subprocess   # Dış komut çalıştırmak (FFmpeg) için
import uuid         # Benzersiz çıktı dosya adları üretmek için
from dotenv import load_dotenv  # Ortam değişkenlerini .env dosyasından yüklemek için

class AudioConverter:
    """
    FFmpeg ile ses dosyalarını WAV formatına dönüştüren yardımcı sınıf.
    """
    def __init__(self, ffmpeg_path=None):
        """
        AudioConverter başlatılırken:
        - .env dosyasını yükler
        - FFmpeg binary yolunu belirler
        """
        # ENV_PATH ortam değişkeni ile özel bir .env yolu belirtilebilir.
        env_path = os.getenv("ENV_PATH", ".env")
        print(f"[INFO] .env dosyası yüklendi: {env_path}")
        load_dotenv(env_path)  # .env dosyasını yükle

        # FFmpeg binary yolu:
        # 1. Parametre ile verilmişse onu kullan
        # 2. Ortam değişkenlerinden (FFMPEG_BIN) oku
        # 3. Yoksa default Windows FFmpeg yolunu kullan
        self.FFMPEG_BIN = ffmpeg_path or os.getenv(
            "FFMPEG_BIN",
            r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
        )
        print(f"[INFO] FFmpeg ikili yolu: {self.FFMPEG_BIN}")

    def convert_to_wav(self, input_path: str, output_path: str = None) -> str:
        """
        Verilen ses dosyasını WAV formatına dönüştürür.
        - input_path: Dönüştürülecek dosyanın tam yolu
        - output_path: Çıktı dosyasının yolu (None ise otomatik üretilir)
        """
        # Eğer output_path belirtilmemişse, input dosya adına benzersiz ek eklenir
        if output_path is None:
            base, _ = os.path.splitext(input_path)  # Dosya adı ve uzantıyı ayır
            output_path = f"{base}_{uuid.uuid4().hex}.wav"  # Benzersiz isim ekle

        # FFmpeg komutunu oluştur
        cmd = [
            self.FFMPEG_BIN, "-y", "-i", input_path,  # -y: Mevcut dosya varsa üzerine yaz
            "-ar", "16000",  # 16 kHz örnekleme hızı
            "-ac", "1",      # Tek kanal (mono)
            "-c:a", "pcm_s16le",  # PCM signed 16-bit little-endian formatı
            output_path
        ]

        # İşlem başlıyor
        print(f"[INFO] WAV dönüşümü başlıyor: {input_path} -> {output_path}")

        # FFmpeg komutunu çalıştır (hata olursa exception fırlatır)
        subprocess.run(cmd, check=True)

        # Başarılı dönüşüm
        print(f"[SUCCESS] WAV dönüşümü tamamlandı: {output_path}")
        return output_path  # Çıktı dosyasının yolunu döndür


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Geriye dönük uyumluluk için (Backward compatibility wrapper).
    """
    return AudioConverter().convert_to_wav(input_path, output_path)
