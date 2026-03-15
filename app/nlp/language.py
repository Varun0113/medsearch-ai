from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati"
}

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except:
        return "en"
