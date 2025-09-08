#!/usr/bin/env python3
"""
Tüm servis bağlantılarını test eden script
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_mongodb():
    """MongoDB bağlantısını test eder"""
    print("🔍 MongoDB bağlantısı test ediliyor...")
    try:
        from db import test_connection
        if test_connection():
            print("✅ MongoDB bağlantısı başarılı!")
            return True
        else:
            print("❌ MongoDB bağlantısı başarısız!")
            return False
    except Exception as e:
        print(f"❌ MongoDB test hatası: {e}")
        return False

def test_elasticsearch():
    """Elasticsearch bağlantısını test eder"""
    print("🔍 Elasticsearch bağlantısı test ediliyor...")
    try:
        from save_to_elastic import get_elasticsearch_client
        es_client = get_elasticsearch_client()
        if es_client:
            print("✅ Elasticsearch bağlantısı başarılı!")
            return True
        else:
            print("❌ Elasticsearch bağlantısı başarısız!")
            return False
    except Exception as e:
        print(f"❌ Elasticsearch test hatası: {e}")
        return False

def test_kafka():
    """Kafka bağlantısını test eder"""
    print("🔍 Kafka bağlantısı test ediliyor...")
    try:
        from kafka_producer import test_kafka_connection
        if test_kafka_connection():
            print("✅ Kafka bağlantısı başarılı!")
            return True
        else:
            print("❌ Kafka bağlantısı başarısız!")
            return False
    except Exception as e:
        print(f"❌ Kafka test hatası: {e}")
        return False

def test_elasticsearch_indices():
    """Elasticsearch indexlerini test eder"""
    print("🔍 Elasticsearch indexleri test ediliyor...")
    try:
        from save_to_elastic import create_indices
        if create_indices():
            print("✅ Elasticsearch indexleri başarıyla oluşturuldu!")
            return True
        else:
            print("❌ Elasticsearch indexleri oluşturulamadı!")
            return False
    except Exception as e:
        print(f"❌ Elasticsearch index test hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 Tüm Servis Bağlantıları Test Ediliyor...")
    print("=" * 60)
    
    results = {}
    
    # MongoDB testi
    results['mongodb'] = test_mongodb()
    print()
    
    # Elasticsearch testi
    results['elasticsearch'] = test_elasticsearch()
    print()
    
    # Kafka testi
    results['kafka'] = test_kafka()
    print()
    
    # Elasticsearch index testi
    if results['elasticsearch']:
        results['elasticsearch_indices'] = test_elasticsearch_indices()
        print()
    
    # Sonuçları özetle
    print("=" * 60)
    print("📊 TEST SONUÇLARI:")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for service, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{service:20} : {status}")
    
    print("=" * 60)
    print(f"Toplam Test: {total_tests}")
    print(f"Başarılı: {passed_tests}")
    print(f"Başarısız: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 Tüm testler başarılı! Sistem hazır.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test başarısız. Lütfen kontrol edin.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
