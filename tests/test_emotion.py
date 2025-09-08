from speechbrain.inference.interfaces import foreign_class

classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    run_opts={"device": "cpu"}
)

out_prob, score, index, text_lab = classifier.classify_file("03-01-04-02-02-01-05.wav")
print("Emotion:", text_lab)
