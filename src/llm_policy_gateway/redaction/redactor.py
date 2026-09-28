"""Presidio analysis and request-scoped reversible placeholders."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import spacy
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from llm_policy_gateway.redaction.recognizers import (
    NPI_CONTEXT,
    NPI_PATTERN,
    MedicalRecordRecognizer,
    NpiRecognizer,
    valid_npi,
)
from llm_policy_gateway.schemas import ChatRequest, EmbedRequest

STANDARD_ENTITIES = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_SSN",
        "IBAN_CODE",
        "IP_ADDRESS",
        "US_BANK_NUMBER",
    }
)
PHI_ENTITIES = STANDARD_ENTITIES | {
    "PERSON",
    "DATE_TIME",
    "LOCATION",
    "US_DRIVER_LICENSE",
    "MEDICAL_RECORD_NUMBER",
    "US_NPI",
}
PLACEHOLDER = re.compile(r"<([A-Z][A-Z0-9_]*)_(\d+)>")


@dataclass(frozen=True)
class RedactionReport:
    counts: dict[str, int]


class RedactionUnavailable(Exception):
    pass


class Redactor:
    def __init__(
        self, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, threshold: float
    ):
        self.analyzer = analyzer
        self.anonymizer = anonymizer
        self.threshold = threshold
        self.language = "en"

    @classmethod
    def create(cls, model_name: str, language: str = "en", threshold: float = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("REDACTION_SCORE_THRESHOLD must be between 0 and 1")
        if not spacy.util.is_package(model_name):
            raise RuntimeError(f"spaCy model {model_name} is missing; run make models")
        analyzer, anonymizer = _load_engines(model_name, language)
        redactor = cls(analyzer, anonymizer, threshold)
        redactor.language = language
        return redactor

    def session(
        self,
        profile: Literal["standard", "phi"],
        allow_terms: list[str] | None = None,
    ) -> "RedactionSession":
        return RedactionSession(self, profile, allow_terms or [])


@lru_cache(maxsize=4)
def _load_engines(model_name: str, language: str):
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": model_name}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    nlp_engine.load()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
    analyzer.registry.add_recognizer(NpiRecognizer())
    analyzer.registry.add_recognizer(MedicalRecordRecognizer())
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            name="Dashed SSN recognizer",
            patterns=[
                Pattern(
                    name="Dashed nine-digit identifier",
                    regex=r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
                    score=0.7,
                )
            ],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="North American phone recognizer",
            patterns=[
                Pattern(
                    name="Three-three-four phone number",
                    regex=(
                        r"(?<!\d)(?:\+1[ -]?)?\(?[2-9]\d{2}\)?"
                        r"[ .-]?[2-9]\d{2}[ .-]?\d{4}(?!\d)"
                    ),
                    score=0.7,
                )
            ],
        )
    )
    return analyzer, AnonymizerEngine()


class RedactionSession:
    def __init__(self, redactor: Redactor, profile: str, allow_terms: list[str]):
        self.redactor = redactor
        self.entities = STANDARD_ENTITIES if profile == "standard" else PHI_ENTITIES
        self.allow_terms = allow_terms
        self.reserved: dict[str, set[int]] = defaultdict(set)
        self.counters: dict[str, int] = defaultdict(int)
        self.to_placeholder: dict[tuple[str, str], str] = {}
        self.issued: dict[str, str] = {}
        self.counts: Counter[str] = Counter()

    def _reserve(self, texts: list[str]) -> None:
        for text in texts:
            for match in PLACEHOLDER.finditer(text):
                self.reserved[match.group(1)].add(int(match.group(2)))

    def _allow_spans(self, text: str) -> list[tuple[int, int]]:
        spans = []
        for term in self.allow_terms:
            if not term:
                continue
            start = 0
            while (start := text.find(term, start)) != -1:
                spans.append((start, start + len(term)))
                start += len(term)
        return spans

    def _select(
        self, text: str, results: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        allow_spans = self._allow_spans(text)
        invalid_npi_spans = []
        for match in NPI_PATTERN.finditer(text):
            window = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)]
            if NPI_CONTEXT.search(window) and not valid_npi(match.group()):
                invalid_npi_spans.append((match.start(), match.end()))
        candidates = [
            result
            for result in results
            if not any(
                start <= result.start and result.end <= end
                for start, end in allow_spans
            )
            and not any(
                result.start < end and result.end > start
                for start, end in invalid_npi_spans
            )
        ]
        candidates.sort(
            key=lambda result: (
                -result.score,
                -(result.end - result.start),
                result.start,
            )
        )
        selected = []
        for candidate in candidates:
            if all(
                candidate.end <= existing.start or candidate.start >= existing.end
                for existing in selected
            ):
                selected.append(candidate)
        return sorted(selected, key=lambda result: result.start)

    def _placeholder_for(self, entity: str, value: str) -> str:
        key = (entity, value)
        if key not in self.to_placeholder:
            index = self.counters[entity] + 1
            while index in self.reserved[entity]:
                index += 1
            self.counters[entity] = index
            placeholder = f"<{entity}_{index}>"
            self.to_placeholder[key] = placeholder
            self.issued[placeholder] = value
        return self.to_placeholder[key]

    def _redact_text(self, text: str) -> str:
        findings = self.redactor.analyzer.analyze(
            text=text,
            language=self.redactor.language,
            entities=sorted(self.entities),
            score_threshold=self.redactor.threshold,
        )
        selected = self._select(text, findings)
        if not selected:
            return text
        for result in selected:
            value = text[result.start : result.end]
            self._placeholder_for(result.entity_type, value)
        operators = {
            entity: OperatorConfig(
                "custom",
                {
                    "lambda": lambda value, entity=entity: self.to_placeholder[
                        (entity, value)
                    ]
                },
            )
            for entity in {result.entity_type for result in selected}
        }
        anonymized = self.redactor.anonymizer.anonymize(
            text=text,
            analyzer_results=selected,
            operators=operators,
            merge_entities_with_spaces=False,
        )
        self.counts.update(result.entity_type for result in selected)
        return anonymized.text

    def redact_chat(self, request: ChatRequest) -> tuple[ChatRequest, RedactionReport]:
        copy = request.model_copy(deep=True)
        texts = [
            message.content
            if isinstance(message.content, str)
            else " ".join(part.text for part in message.content)
            for message in copy.messages
        ]
        self._reserve(texts)
        for message in copy.messages:
            if isinstance(message.content, str):
                message.content = self._redact_text(message.content)
            else:
                for part in message.content:
                    part.text = self._redact_text(part.text)
        return copy, RedactionReport(dict(self.counts))

    def redact_embed(
        self, request: EmbedRequest
    ) -> tuple[EmbedRequest, RedactionReport]:
        copy = request.model_copy(deep=True)
        texts = [copy.input] if isinstance(copy.input, str) else copy.input
        self._reserve(texts)
        copy.input = (
            self._redact_text(copy.input)
            if isinstance(copy.input, str)
            else [self._redact_text(value) for value in copy.input]
        )
        return copy, RedactionReport(dict(self.counts))

    def reidentify(self, response_text: str) -> str:
        return PLACEHOLDER.sub(
            lambda match: self.issued.get(match.group(), match.group()), response_text
        )
