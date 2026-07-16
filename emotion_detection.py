# ---------------------------------------------------------------
# emotion_detection.py
# - HF (Transformers) pipeline ile 6 etiketli duygu tahmini
# - Segment bazında zenginleştirme (emotion_pred, emotion_dist)
# - Konuşmacı bazında özetleme
# ---------------------------------------------------------------

from __future__ import annotations  # İleriye referans verilen type hint'ler için (py3.7+)

import json  # JSON okuma/yazma yardımcıları
from dataclasses import dataclass  # Kolay veri sınıfı tanımı için
from pathlib import Path  # (Şu an kullanılmıyor ama dosya yolları için faydalı olabilir)
from typing import Any, Dict, List, Optional, Tuple  # Tip ipuçları

# Transformers ekosistemi: tokenizer, model ve pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 6 etiketli nihai duygu kümesi
# Not: 'disgust' bu sınıfta yok; orijinal model 7 etiketli.
TARGET6 = ["anger", "fear", "joy", "sadness", "surprise", "neutral"]  # 'disgust' hariç

def _to_six(all_scores: List[Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """
    HF pipeline'dan gelen *tüm* skorları 6 etikete indirger.
    Girdi: [{'label': 'joy', 'score': 0.8}, ...] (modelin tüm etiketleri)
    Çıktı:
      - pred: en yüksek skorlu etiket (str)
      - norm: {label: score} şeklinde 6 etikete indirgenmiş dağılım
    """
    # Sadece TARGET6 içinde olan etiketleri tut
    keep = [s for s in all_scores if s["label"].lower() in TARGET6]

    # Etiket->skor sözlüğü (float'a döndürüp normalize değil, direkt model skoru)
    norm: Dict[str, float] = {}
    for s in keep:
        lbl = s["label"].lower()           # Etiketi küçük harfe indir
        norm[lbl] = float(s["score"])      # Skoru float'a çevirip koy

    # Eksik etiket olmasın: olmayanları 0.0 ile tamamla
    for k in TARGET6:
        norm.setdefault(k, 0.0)

    # En yüksek skor hangi etiketteyse onu tahmin olarak seç
    pred = max(norm.items(), key=lambda kv: kv[1])[0]
    return pred, norm

@dataclass
class EmotionJSONAnalyzer:
    """
    j-hartmann/emotion-english-distilroberta-base modeli ile
    JSON segmentlerinden (text alanı) 6 etiketli duygu analizi yapar.

    Notlar:
    - model_id: Hugging Face model kimliği (sequence classification)
    - device: None/-1 -> CPU, 0 -> CUDA:0, 1 -> CUDA:1 ...
    - batch_size: pipeline içinde toplu işleme için ipucu (pipeline kendi yönetebilir)
    - max_text_len: çok uzun metinler için güvenli kırpma limiti (token değil, karakter)
    """
    model_id: str = "j-hartmann/emotion-english-distilroberta-base"
    device: Optional[int] = None     # CPU: None/-1, GPU: 0 vb.
    batch_size: int = 16
    max_text_len: int = 4000         # Güvenli kırpma (karakter bazlı)

    def __post_init__(self):
        """
        dataclass init'inden sonra çağrılır.
        - Tokenizer ve modeli yükler
        - text-classification pipeline'ını (return_all_scores=True) hazırlar
        """
        # Tokenizer ve model yükle
        tok = AutoTokenizer.from_pretrained(self.model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(self.model_id)

        # Pipeline: tüm etiket skorlarını döndür (return_all_scores=True, top_k=None)
        # device: None ise -1 (CPU) kullan
        self.clf = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            return_all_scores=True,                # Tüm etiketler ve skorları gelsin
            top_k=None,                            # Sıralama/filtreleme yapma
            device=self.device if self.device is not None else -1,
        )

    def _predict_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Metin listesini (batch) modele gönderir ve 6 etiketli çıktıya çevirir.
        Dönüş: [(pred_label, {label:score}), ...]
        """
        # Çok uzun metinleri kes (karakter bazlı)
        texts = [(t or "")[: self.max_text_len] for t in texts]

        # Pipeline çağrısı -> List[List[{'label','score'}]]
        # Her metin için bir liste, içinde o metne dair tüm etiketler ve skorları
        raw = self.clf(texts)

        out: List[Tuple[str, Dict[str, float]]] = []
        for scores in raw:
            # Her öğe: [{'label': 'joy', 'score': 0.87}, ...]
            # Label'ları küçük harfe indirerek 6 etikete map et
            pred, dist = _to_six(
                [{"label": s["label"].lower(), "score": float(s["score"])} for s in scores]
            )
            out.append((pred, dist))
        return out

    # ---------- Genel kullanım API'leri ----------
    def analyze_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Segment listesi alır, her segmentin 'text' alanına duygu tahmini yapar.
        Çıkışta her segmente şu alanları ekler:
          - emotion_pred: str (en olası duygu)
          - emotion_dist: Dict[str, float] (6 etiket dağılım)
        Diğer segment alanları korunur.
        """
        # Segmentlerden metinleri çek
        texts = [seg.get("text", "") for seg in segments]

        # Toplu tahmin
        preds = self._predict_batch(texts)

        # Orijinal segment + yeni alanlar
        enriched: List[Dict[str, Any]] = []
        for seg, (pred, dist) in zip(segments, preds):
            s = dict(seg)                 # Segmenti kopyala (yan etkiden kaçın)
            s["emotion_pred"] = pred      # En yüksek skorlu etiket
            s["emotion_dist"] = dist      # 6 etiket dağılımı
            enriched.append(s)
        return enriched

    def analyze_file(self, input_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Diskten JSON (list of segments) okur, segmentleri zenginleştirir ve
        istenirse sonucu diske yazar.
        """
        # JSON oku
        with open(input_path, "r", encoding="utf-8") as f:
            segs = json.load(f)

        # Kök mutlaka liste olmalı (segment listesi)
        assert isinstance(segs, list), "JSON kökü liste olmalı."

        # Segment analizi
        enriched = self.analyze_segments(segs)

        # Çıktı dosyası istenmişse yaz
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)

        return enriched

    def summarize_by_speaker(self, segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Konuşmacı (speaker) bazında metinleri zaman sırasına göre birleştirir
        ve her konuşmacı için tek bir duygu tahmini üretir.

        Dönüş:
        {
          speakerX: {
            "text": "<birleştirilmiş metin>",
            "emotion_pred": "<etiket>",
            "emotion_dist": {label: score, ...}
          },
          ...
        }
        """
        # Önce konuşmacıya göre segmentleri topla (start zamanı ile birlikte)
        by_spk: Dict[str, List[Tuple[float, str]]] = {}
        for s in segments:
            spk = s.get("speaker", "speaker01")  # Varsayılan speaker adı
            by_spk.setdefault(spk, []).append(
                (float(s.get("start", 0.0)), s.get("text", ""))  # (zaman, metin)
            )

        # Zaman sırası ile birleştir
        spk_texts: Dict[str, str] = {}
        for spk, items in by_spk.items():
            items.sort(key=lambda x: x[0])                # start'a göre sırala
            spk_texts[spk] = " ".join(t for _, t in items if t)  # metinleri birleştir

        # Her konuşmacının birleştirilmiş metnini toplu halde tahmin et
        preds = self._predict_batch([spk_texts[k] for k in spk_texts])

        # Sonucu konuşmacı->özet sözlüğüne dök
        summary: Dict[str, Dict[str, Any]] = {}
        for (spk, txt), (pred, dist) in zip(spk_texts.items(), preds):
            summary[spk] = {"text": txt, "emotion_pred": pred, "emotion_dist": dist}
        return summary

def _group_by_emotion(enriched_segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Zenginleştirilmiş segmentleri (emotion_pred alanı olanlar)
    tahmin etiketine göre gruplar.

    Dönüş:
      {
        "joy": [
           {"speaker": "...", "start": ..., "end": ..., "text": "...", "lang": "..."},
           ...
        ],
        ...
      }

    Not:
    - Boş gruplar sonuçtan filtrelenir (yani hiç eleman yoksa anahtar da yoktur).
    """
    # Başlangıçta 6 etiket için boş liste hazırla
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TARGET6}

    # Her segmenti kendi emotion_pred anahtarına it
    for s in enriched_segments:
        pred = s.get("emotion_pred", "neutral")
        if pred not in grouped:
            grouped[pred] = []  # (Güvenlik için) beklenmeyen etiket varsa yine de ekle
        grouped[pred].append({
            "speaker": s.get("speaker", "speaker01"),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": s.get("text", ""),
            "lang": s.get("lang", "unknown"),
        })

    # Boş grupları kaldır (yalnızca dolu olanlar kalsın)
    grouped = {k: v for k, v in grouped.items() if v}
    return grouped

# -----------------------
# ÖRNEK KULLANIM (Script)
# -----------------------
if __name__ == "__main__":
    """
    1) Segment bazında zenginleştirilmiş JSON üretir:
       input : hypotheses_multil_diarized_raw.json
       output: hypotheses_multil_diarized_emotions.json

    2) Aynı veriyi duygu başlıklarına göre gruplar:
       output: hypotheses_multil_diarized_emotions_grouped.json

    3) (Opsiyonel) Konuşmacı bazında özet duygu (stdout'a yazar).
    """

    # Girdi/çıktı dosya adları (aynı klasörde varsayılır)
    INPUT = "hypotheses_multil_diarized_raw.json"
    OUT_ENRICHED = "hypotheses_multil_diarized_emotions.json"
    OUT_GROUPED = "hypotheses_multil_diarized_emotions_grouped.json"

    # Analyzer oluştur (CPU için device=None/-1; GPU için 0 yaz)
    analyzer = EmotionJSONAnalyzer(device=None)

    # 1) JSON dosyasını oku, segmentleri zenginleştir ve diske yaz
    enriched = analyzer.analyze_file(INPUT, OUT_ENRICHED)

    # 2) Zenginleştirilmiş segmentleri etiketlere göre grupla ve diske yaz
    grouped = _group_by_emotion(enriched)
    with open(OUT_GROUPED, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    # 3) Konuşmacı bazında özetle ve ekrana yaz
    summary = analyzer.summarize_by_speaker(enriched)
    print("=== Speaker Summary ===")
    for spk, info in summary.items():
        print(spk, "->", info["emotion_pred"])
