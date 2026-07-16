import json
import time

from kafka_producer import send_bulk_events, send_media_event, test_kafka_connection


def test_single_events():
    """Tek tek event gönderimi test eder"""
    print("🧪 Tek tek event gönderimi test ediliyor...")
    
    for i in range(5):
        event_data = {
            "media_id": f"test_{i + 1}",
            "filename": f"test_video_{i+1}.mp4",
            "status": "processed",
            "duration": round(10 + i * 2.5, 2),
            "language": "tr",
            "summary": f"Test video {i+1} özeti",
            "segments_count": 5 + i
        }
        
        success = send_media_event(**event_data)
        if success:
            print(f"✅ Event {i+1} başarıyla gönderildi")
        else:
            print(f"❌ Event {i+1} gönderilemedi")
        
        time.sleep(0.5)  # Daha iyi gözlemlemek için bekleme

def test_bulk_events():
    """Toplu event gönderimi test eder"""
    print("\n🧪 Toplu event gönderimi test ediliyor...")
    
    events_list = []
    for i in range(10):
        event = {
            "media_id": f"bulk_test_{i + 1}",
            "filename": f"bulk_video_{i+1}.mp4",
            "status": "processed",
            "duration": round(15 + i * 1.5, 2),
            "language": "en",
            "summary": f"Bulk test video {i+1} summary",
            "segments_count": 8 + i
        }
        events_list.append(event)
    
    success = send_bulk_events(events_list)
    if success:
        print("✅ Tüm bulk eventler başarıyla gönderildi")
    else:
        print("❌ Bazı bulk eventler gönderilemedi")

def test_error_scenarios():
    """Hata senaryolarını test eder"""
    print("\n🧪 Hata senaryoları test ediliyor...")
    
    # Geçersiz veri ile test
    try:
        success = send_media_event(
            media_id=None,  # Geçersiz media_id
            filename="",     # Boş filename
            status="",       # Boş status
            duration=-1      # Geçersiz duration
        )
        print(f"⚠️ Geçersiz veri testi sonucu: {success}")
    except Exception as e:
        print(f"✅ Geçersiz veri hatası yakalandı: {e}")

def main():
    """Ana test fonksiyonu"""
    print("🚀 Kafka Producer Test Başlatılıyor...")
    print("=" * 50)
    
    # Önce bağlantıyı test et
    if not test_kafka_connection():
        print("❌ Kafka bağlantısı kurulamadı, testler yapılamıyor")
        return
    
    print("✅ Kafka bağlantısı başarılı, testler başlatılıyor...")
    
    # Testleri çalıştır
    test_single_events()
    test_bulk_events()
    test_error_scenarios()
    
    print("\n" + "=" * 50)
    print("🎉 Tüm testler tamamlandı!")
    print("📊 Sonuçları Kafka UI'dan kontrol edebilirsiniz: http://localhost:8080")

if __name__ == "__main__":
    main()
