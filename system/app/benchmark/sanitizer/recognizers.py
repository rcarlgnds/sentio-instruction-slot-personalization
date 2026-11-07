import re

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

PHONE_CONTEXT = ["telepon", "telp", "phone", "hp", "wa", "whatsapp", "kontak", "contact", "hubungi", "nomor"]

BPS_PROVINCE_CODES = {
    11, 12, 13, 14, 15, 16, 17, 18, 19, 21,
    31, 32, 33, 34, 35, 36,
    51, 52, 53,
    61, 62, 63, 64, 65,
    71, 72, 73, 74, 75, 76,
    81, 82,
    91, 92, 93, 94, 95, 96,
}


class NimRecognizer(EntityRecognizer):
    DIGIT_RUN = re.compile(r"(?<!\d)\d{10}(?!\d)")
    CONTEXT = re.compile(r"\b(nim|npm|nomor\s+induk|mahasiswa|student)\b", re.IGNORECASE)
    WINDOW = 60

    def __init__(self):
        super().__init__(supported_entities=["ID_NIM"], name="id_nim_recognizer")

    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        if "ID_NIM" not in entities:
            return []
        results = []
        for match in self.DIGIT_RUN.finditer(text):
            lo = max(0, match.start() - self.WINDOW)
            hi = min(len(text), match.end() + self.WINDOW)
            if self.CONTEXT.search(text[lo:hi]):
                results.append(RecognizerResult("ID_NIM", match.start(), match.end(), 0.85))
        return results


class NikRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="ID_NIK",
            name="id_nik_recognizer",
            patterns=[Pattern(name="nik_16_digit", regex=r"(?<!\d)\d{16}(?!\d)", score=0.6)],
        )

    def validate_result(self, pattern_text: str):
        province = int(pattern_text[0:2])
        day = int(pattern_text[6:8])
        month = int(pattern_text[8:10])
        day_ok = 1 <= day <= 31 or 41 <= day <= 71
        month_ok = 1 <= month <= 12
        if province in BPS_PROVINCE_CODES and day_ok and month_ok:
            return True
        return False


class PhoneIdRecognizer(PatternRecognizer):
    STRICT = r"(?<!\d)(?<!\d[.\-])(?:\+62|62|0)8[1-9](?:[.\-]?\d){7,11}(?!\d)(?!\.\d)(?!-\d)"
    SPACED = r"(?<!\d)(?<!\d[.\-])(?:\+62|62|0)[\s\-]?8[1-9](?:[\s.\-]?\d){6,10}(?!\d)(?!\.\d)(?!-\d)"

    def __init__(self):
        super().__init__(
            supported_entity="ID_PHONE_NUMBER",
            name="id_phone_recognizer",
            patterns=[
                Pattern(name="phone_id_contiguous", regex=self.STRICT, score=0.6),
                Pattern(name="phone_id_spaced", regex=self.SPACED, score=0.35),
            ],
            context=PHONE_CONTEXT,
        )

    def validate_result(self, pattern_text: str):
        digits = re.sub(r"\D", "", pattern_text)
        if digits.startswith("62"):
            digits = "0" + digits[2:]
        if not 10 <= len(digits) <= 13:
            return False
        if pattern_text.lstrip().startswith("+62"):
            return True
        if any(ch.isspace() for ch in pattern_text):
            return None
        return True


def custom_recognizers():
    return [NimRecognizer(), NikRecognizer(), PhoneIdRecognizer()]
