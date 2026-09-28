"""Checksum-based and context-bound clinical identifiers."""

import re

from presidio_analyzer import EntityRecognizer, RecognizerResult

NPI_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d{10}(?![A-Za-z0-9])")
NPI_CONTEXT = re.compile(r"\b(?:npi|provider|prescriber)\b", re.IGNORECASE)
MRN_PATTERN = re.compile(
    r"(?:\bMRN\b|\bmedical\s+record\b|\brecord\s+no\b|\bchart\s*#)"
    r"\s*(?:number\b|no\b)?\s*[:#=-]?\s*([A-Za-z0-9]{6,12})\b",
    re.IGNORECASE,
)


def valid_npi(value: str) -> bool:
    if not re.fullmatch(r"\d{10}", value):
        return False
    digits = "80840" + value
    total = 0
    for index, character in enumerate(reversed(digits)):
        digit = int(character)
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class NpiRecognizer(EntityRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entities=["US_NPI"],
            name="US NPI recognizer",
            supported_language="en",
        )

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list:
        if "US_NPI" not in entities:
            return []
        results = []
        for match in NPI_PATTERN.finditer(text):
            if not valid_npi(match.group()):
                continue
            window = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)]
            score = 0.95 if NPI_CONTEXT.search(window) else 0.7
            results.append(
                RecognizerResult("US_NPI", match.start(), match.end(), score)
            )
        return results


class MedicalRecordRecognizer(EntityRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entities=["MEDICAL_RECORD_NUMBER"],
            name="Medical record recognizer",
            supported_language="en",
        )

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list:
        if "MEDICAL_RECORD_NUMBER" not in entities:
            return []
        return [
            RecognizerResult(
                "MEDICAL_RECORD_NUMBER", match.start(1), match.end(1), 0.95
            )
            for match in MRN_PATTERN.finditer(text)
        ]
