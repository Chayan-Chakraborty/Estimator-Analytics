from transformers import pipeline

def detect_and_translate(text: str) -> str:
    """
    Detects the language of the input text (including Hinglish/code-mixed)
    and translates it into natural English.
    """

    # 1️⃣ Use a multilingual model trained to handle many languages.
    # NLLB-200 is good for Hindi, Hinglish, and many others.
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="auto",    # let model auto-detect the source language
        tgt_lang="eng_Latn" # translate to English
    )

    # Translate text
    result = translator(text, max_length=512)
    return result[0]['translation_text']

if __name__ == "__main__":
    # Example inputs
    samples = [
        "mujhe thoda rest chahiye yaar, kal bohot kaam tha",
        "आज मौसम बहुत अच्छा है",
        "Kal meeting hai at 5pm, don't forget!",
        "Quiero un café por favor"
    ]

    for s in samples:
        print(f"Input: {s}")
        print(f"English: {detect_and_translate(s)}\n")
