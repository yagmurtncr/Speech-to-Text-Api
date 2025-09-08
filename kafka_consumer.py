# -----------------------------------------------------------
# Kafka Consumer
# - Kafka'dan 'media_processed' gibi event'leri dinler
# - Mesajları JSON olarak çözümler ve statüye göre loglar/işler
# - Bağlantı testi ve topic bilgisi alma yardımcıları içerir
# -----------------------------------------------------------

from kafka import KafkaConsumer          # Kafka tüketici (consumer) istemcisi
import os                                # Ortam değişkenleri için
import json                              # Mesaj gövdelerini JSON'a çevirmek için
from dotenv import load_dotenv           # .env içindeki değişkenleri yüklemek için
import logging                           # Loglama altyapısı
from datetime import datetime            # (Şu an kullanılmıyor ama zaman damgası için faydalı)

# .env dosyasını belleğe yükle (KAFKA_BOOTSTRAP, KAFKA_TOPIC vb. için)
load_dotenv()

# ---------------------------
# LOG YAPILANDIRMASI
# ---------------------------
logging.basicConfig(level=logging.INFO)        # Basit konfigurasyon: INFO ve üzerini yaz
logger = logging.getLogger(__name__)           # Modül özelinde logger

# ---------------------------
# KONFİG / ORTAM DEĞİŞKENLERİ
# ---------------------------
# Bootstrap server: docker-compose'da PLAINTEXT_HOST -> "localhost:9093"
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP", "localhost:9093")
# Dinlenecek topic adı
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "media_processed")

def get_kafka_consumer():
    """
    Kafka consumer oluşturur ve temel bağlantı testini yapar.
    Başarılıysa consumer nesnesini döndürür; hata varsa None döner.
    """
    try:
        # KafkaConsumer: Topic'e abone olarak başlatıyoruz
        consumer = KafkaConsumer(
            TOPIC_NAME,                           # Dinlenecek topic
            bootstrap_servers=KAFKA_BROKER,       # Kafka broker adresi
            auto_offset_reset='earliest',         # Topic'te offset yoksa en baştan başla
            enable_auto_commit=True,              # Offsetleri otomatik commit et (manuel commit de yapıyoruz)
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),  # Byte -> str -> JSON
            consumer_timeout_ms=10000,            # 10 sn mesaj gelmezse for döngüsü biter (StopIteration)
            group_id='speech_to_text_consumer'    # Consumer group adı (aynı grup offset paylaşır)
        )
        logger.info("+ Kafka consumer başarıyla oluşturuldu!")
        return consumer
    except Exception as e:
        # Örn. broker erişilemiyorsa burada hata alırsın
        logger.error(f"- Kafka consumer oluşturma hatası: {e}")
        return None

def process_message(message):
    """
    Kafka'dan gelen tek bir mesajı işler.
    Beklenen message.value: dict (ör. {"status": "processed", "filename": "...", "media_id": "..."})
    """
    try:
        data = message.value                          # Deserializer sayesinde direkt dict
        logger.info(f" Mesaj alındı: {data}")

        # Mesaj türüne / statüsüne göre yönlendir
        if data.get("status") == "processed":
            # Başarılı işlenmiş medya olayı
            logger.info(f"+ İşlenen medya: {data.get('filename')} - {data.get('media_id')}")

            # Buraya iş kurallarını ekleyebilirsin:
            # - Elasticsearch'e yaz
            # - Webhook gönder
            # - E-posta bildirimi
            # - Analytics / metrik güncelleme

        elif data.get("status") == "error":
            # Üretim hattında bir hata oluşmuş
            logger.error(f"- Hata durumu: {data.get('filename')} - {data.get('message', 'Bilinmeyen hata')}")

        else:
            # Beklenen alanlar yoksa veya farklı bir türde mesaj geldiyse
            logger.info(f" Bilinmeyen durum: {data}")

    except Exception as e:
        # Mesaj formatı beklenmedikse, deserialization veya işleme hataları vb.
        logger.error(f"- Mesaj işleme hatası: {e}")

def start_consumer():
    """
    Consumer'ı başlatır, topic'ten gelen mesajları dinler ve process_message ile işler.
    Ctrl+C ile durdurulabilir.
    """
    consumer = get_kafka_consumer()
    if not consumer:
        logger.error("Consumer oluşturulamadı, çıkılıyor...")
        return

    try:
        logger.info(f" Kafka consumer dinleniyor... Topic: {TOPIC_NAME}")
        logger.info(f" Broker: {KAFKA_BROKER}")
        logger.info(" Durdurmak için Ctrl+C kullanın")

        # KafkaConsumer iterable'dır; her mesaj geldiğinde döngüye düşer
        for message in consumer:
            try:
                # Tek mesajı işle
                process_message(message)

                # Başarıyla işlendiyse offset'i commit et (enable_auto_commit True olsa da manuel kontrol iyidir)
                consumer.commit()

            except Exception as e:
                # process_message içinde yakalanmayan hatalar için son savunma
                logger.error(f"- Mesaj işleme hatası: {e}")
                # Not: Hata halinde commit ETMEMEK mesajın tekrar işlenmesini sağlar

    except KeyboardInterrupt:
        # Kullanıcı manuel durdurdu
        logger.info("⏹ Consumer durduruluyor...")
    except Exception as e:
        # Döngü dışı beklenmedik hata
        logger.error(f"- Consumer hatası: {e}")
    finally:
        # Kaynakları bırak
        consumer.close()
        logger.info("+ Consumer kapatıldı")

def test_kafka_connection():
    """
    Basit bir bağlantı testi yapar:
    - Consumer oluşturmayı dener
    - Broker'dan topic listesini çeker
    Başarılıysa True döner.
    """
    try:
        consumer = get_kafka_consumer()
        if consumer:
            topics = consumer.topics()  # Broker'daki topic setini getir
            logger.info(f"+ Kafka bağlantısı başarılı! Mevcut topic'ler: {list(topics)}")
            consumer.close()
            return True
        return False
    except Exception as e:
        logger.error(f"- Kafka bağlantı testi başarısız: {e}")
        return False

def get_topic_info():
    """
    İlgili topic hakkında temel bilgiler döker:
    - Partition sayısı / listesi
    - Her partition için end offset ve current offset
    """
    try:
        consumer = get_kafka_consumer()
        if consumer:
            # Topic partititon bilgilerini al
            partitions = consumer.partitions_for_topic(TOPIC_NAME)
            if partitions:
                logger.info(f" Topic: {TOPIC_NAME}")
                logger.info(f" Partition sayısı: {len(partitions)}")
                logger.info(f" Partition'lar: {list(partitions)}")

                # Her partition için uç (end) ve mevcut (current) offsetleri yaz
                for partition in partitions:
                    # end_offsets: { (topic, partition): end_offset_int }
                    end_offset = consumer.end_offsets([(TOPIC_NAME, partition)])
                    # position: Bu consumer'ın mevcut pozisyonu (commit + fetch sonrası)
                    current_offset = consumer.position([(TOPIC_NAME, partition)])

                    logger.info(
                        f"   Partition {partition}: "
                        f"End={end_offset[(TOPIC_NAME, partition)]}, "
                        f"Current={current_offset[(TOPIC_NAME, partition)]}"
                    )

            consumer.close()
            return True
        return False
    except Exception as e:
        logger.error(f"- Topic bilgi alma hatası: {e}")
        return False

# ---------------------------
# Komut satırından çalıştırma
# ---------------------------
if __name__ == "__main__":
    print(" Kafka bağlantısı test ediliyor...")

    # 1) Bağlantı testi
    if test_kafka_connection():
        print(" Topic bilgileri alınıyor...")
        # 2) Topic özet bilgilerini al
        get_topic_info()

        print("\n Consumer başlatılıyor...")
        # 3) Mesaj dinlemeye başla
        start_consumer()
    else:
        print("- Kafka bağlantısı kurulamadı, consumer başlatılamıyor")
