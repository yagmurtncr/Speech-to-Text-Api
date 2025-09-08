# postprocess_merge.py
# ------------------------------------------------------------
# Bu script, diarization (konuşmacı ayrımı) sonrası üretilen segment listesini
# düzenleyip birleştirir:
#   - Tek konuşmacı varsa tüm segmentleri tek metinde birleştirir.
#   - Birden fazla konuşmacı varsa her (konuşmacı, dil) için metinleri birleştirir.
# ------------------------------------------------------------

import json
from collections import defaultdict
from pathlib import Path

# Girdi ve çıktı JSON dosya yolları
IN_PATH  = "hypotheses_multil_diarized_raw.json"
OUT_PATH = "hypotheses_multil_diarized_merged.json"

def merge_segments(segs):
    """
    Segmentleri birleştirir:
    1) Zaman sırasına göre sıralar.
    2) Tek konuşmacı varsa tek metin haline getirir.
    3) Birden fazla konuşmacı varsa her (speaker, lang) kombinasyonu için ayrı metin oluşturur.
    """
    # 1) Zaman sırasına göre sırala (start zamanı küçükten büyüğe)
    segs = sorted(segs, key=lambda x: x["start"])

    # Tüm konuşmacıları listele (varsayılan speaker01)
    speakers = [s.get("speaker", "speaker01") for s in segs]
    uniq_speakers = sorted(set(speakers))  # Benzersiz konuşmacılar

    # --- TEK SPEAKER DURUMU ---
    if len(uniq_speakers) == 1:
        # Metinleri boş olmayanlar ile birleştir (aralarda boşluk)
        text = " ".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())
        if not segs:
            return []  # Hiç segment yoksa boş dön
        return [{
            "speaker": uniq_speakers[0],           # Tek konuşmacı ID'si
            "start": segs[0]["start"],             # İlk segment başlangıcı
            "end":   segs[-1]["end"],               # Son segment bitişi
            "text":  text,                         # Birleştirilmiş metin
            "lang":  segs[0].get("lang", "unknown") # İlk segment dil bilgisi
        }]

    # --- BİRDEN FAZLA SPEAKER DURUMU ---
    # Her konuşmacı + dil kombinasyonu ayrı bir "bölme"de tutulur
    buckets = defaultdict(list)  # (speaker, lang) → [(start, end, text), ...]
    starts, ends = {}, {}        # Her (speaker, lang) için başlangıç/bitiş zamanları

    for s in segs:
        spk  = s.get("speaker", "speaker01")   # Konuşmacı ID'si
        lang = s.get("lang", "unknown")        # Dil bilgisi
        key = (spk, lang)                      # Grup anahtarı
        t = (s["start"], s["end"], (s.get("text") or "").strip())

        # Metin boş değilse bucket'a ekle
        if t[2]:
            buckets[key].append(t)

        # Başlangıç zamanını kaydet (ilk kez görüyorsak)
        if key not in starts:
            starts[key] = s["start"]

        # Her zaman son end değerini güncelle (en büyük end alınır)
        ends[key] = s["end"]

    # Gruplanmış veriyi birleştir
    merged = []
    for (spk, lang), items in buckets.items():
        # Segmentleri kendi başlangıç zamanına göre sırala
        items.sort(key=lambda t: t[0])
        # Metinleri boşlukla birleştir
        text = " ".join(t[2] for t in items)
        merged.append({
            "speaker": spk,
            "start": starts[(spk, lang)],
            "end":   ends[(spk, lang)],
            "text":  text,
            "lang":  lang,
        })

    # Sonuçları başlangıç zamanına göre sırala
    merged.sort(key=lambda x: x["start"])
    return merged

def main():
    # Girdi dosyasını oku
    in_path = Path(IN_PATH)
    with in_path.open("r", encoding="utf-8") as f:
        segs = json.load(f)

    # Segmentleri birleştir
    merged = merge_segments(segs)

    # Çıktı dosyasına yaz
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[OK] Birleştirildi → {OUT_PATH} | satır={len(merged)}")

if __name__ == "__main__":
    main()
