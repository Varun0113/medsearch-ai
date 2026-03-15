from transformers import pipeline


class Translator:
    def __init__(self):
        # Keep startup and English-only queries fast.
        self.translator = None

    def _get_translator(self):
        if self.translator is None:
            self.translator = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M"
            )
        return self.translator

    def to_english(self, text: str, source_lang: str) -> str:
        if source_lang in {"en", "eng_Latn"}:
            return text

        translator = self._get_translator()
        return translator(
            text,
            src_lang=source_lang,
            tgt_lang="eng_Latn",
            max_length=256,
            truncation=True
        )[0]["translation_text"]

    def from_english(self, text: str, target_lang: str) -> str:
        if target_lang in {"en", "eng_Latn"}:
            return text

        translator = self._get_translator()
        return translator(
            text,
            src_lang="eng_Latn",
            tgt_lang=target_lang,
            max_length=512,
            truncation=True
        )[0]["translation_text"]
