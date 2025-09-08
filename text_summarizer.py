from transformers import pipeline

# HuggingFace 'transformers' kütüphanesinden BART-large-cnn modelini kullanarak özetleme pipeline'ı oluşturuyoruz.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_summary(text, max_length=120, min_length=30):
    """
    Verilen metnin özetini döner. 
    Tekrar eden cümleleri filtreler, daha kısa ve anlamlı bir özet üretir.
    
    Parametreler:
        text (str): Özetlenecek metin
        max_length (int): Özetin maksimum token uzunluğu
        min_length (int): Özetin minimum token uzunluğu

    Dönüş:
        str: Özet metin
    """
    
    # Eğer metin yoksa veya çok kısa ise (50 karakterden az),
    # özetleme yapmadan orijinal metni döner.
    if not text or len(text.strip()) < 50:
        return text
    
    # Metni cümlelere ayırıyoruz. Ayırıcı olarak '. ' kullanıyoruz.
    # dict.fromkeys ile sırayı koruyarak tekrar eden cümleleri kaldırıyoruz.
    sentences = list(dict.fromkeys(text.split('. ')))  
    
    # Tekrarları temizledikten sonra cümleleri tekrar birleştiriyoruz.
    unique_text = '. '.join(sentences)
    
    # Özetleme işlemi.
    # do_sample=False -> deterministic (rastgelelik yok, tekrarlanabilir sonuç)
    summary = summarizer(
        unique_text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False
    )
    
    # HuggingFace pipeline çıktısı bir liste/dict formatında gelir, 
    # ilk elemanın 'summary_text' alanını döndürüyoruz.
    return summary[0]["summary_text"]
