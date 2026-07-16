# ---------------------------------------------------------
# hypotheses.json içindeki "Hypothesis(...)" türü stringlerden
# gerçek metni (text='...') ayıklayıp düz metne çevirir.
# - hypotheses.json'u okur
# - Her kaydın "text" alanını regex ile temizler
# - Sonucu hypotheses_fixed.json olarak yazar
# - Ardından orijinal hypotheses.json dosyasını da günceller
# ---------------------------------------------------------

import json
import re


def extract_text_from_hypothesis(hypothesis_str: str) -> str:
    """
    Verilen Hypothesis nesnesini temsil eden stringten gerçek metni çıkarır.
    Beklenen desen: ... text='...metin...' ...
    Eşleşme yoksa, gelen stringi aynen geri döndürür.

    Ör: "Hypothesis(text='Merhaba', score=0.98)" -> "Merhaba"
    """
    # 'text=' ve tek tırnaklar arasını eşle (temel durum)
    match = re.search(r"text='([^']*)'", hypothesis_str)
    if match:
        # 1. yakalama grubu -> tırnaklar içindeki metin
        return match.group(1)
    else:
        # Eşleşme bulunamazsa orijinali koru (bozmayalım)
        return hypothesis_str

def fix_hypotheses_file() -> None:
    """
    Çalışma klasöründeki 'hypotheses.json' dosyasını okur,
    her öğenin 'text' alanını extract_text_from_hypothesis ile temizler,
    çıktıyı hem 'hypotheses_fixed.json' olarak yazar,
    hem de orijinal 'hypotheses.json' dosyasını günceller.
    """
    try:
        # 1) hypotheses.json'u oku
        with open("hypotheses.json", "r", encoding="utf-8") as f:
            data = json.load(f)  # Genelde liste beklenir: [{"text": "...", ...}, ...]

        # 2) Her kayıt için 'text' alanını temizle
        for item in data:
            if "text" in item:
                # item["text"] bir Hypothesis(...) stringi olabilir => gerçek metni çek
                item["text"] = extract_text_from_hypothesis(item["text"])

        # 3) Dönüştürülmüş çıktıyı yeni bir dosyaya yaz (yedek/karşılaştırma için)
        with open("hypotheses_fixed.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Fixed hypotheses saved to hypotheses_fixed.json")

        # 4) Orijinal dosyayı da güncelle (isteniyorsa tek dosyayla devam edilsin)
        with open("hypotheses.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Original hypotheses.json has been updated")

    except Exception as e:
        # Genel hata yakalama (dosya yok, JSON hatalı, izin yok vb.)
        print(f"Error: {e}")

# Komut satırından çalıştırıldığında düzeltme fonksiyonunu tetikle
if __name__ == "__main__":
    fix_hypotheses_file()
