"""Offline redaction behavior with synthetic values."""

import pytest
from fastapi.testclient import TestClient

from llm_policy_gateway.app import create_app
from llm_policy_gateway.redaction.recognizers import (
    MedicalRecordRecognizer,
    NpiRecognizer,
    valid_npi,
)
from llm_policy_gateway.redaction.redactor import Redactor
from llm_policy_gateway.schemas import ChatRequest, EmbedRequest


@pytest.fixture(scope="module")
def redactor() -> Redactor:
    return Redactor.create("en_core_web_sm")


def chat(text: str) -> ChatRequest:
    return ChatRequest(
        model="clinical-local", messages=[{"role": "user", "content": text}]
    )


def test_custom_recognizers_use_checksum_and_context() -> None:
    assert valid_npi("0000000006")
    assert not valid_npi("0000000007")
    assert not valid_npi("not-a-npi")
    npi = NpiRecognizer()
    assert npi.analyze("NPI 0000000006", ["US_NPI"])[0].score == 0.95
    assert npi.analyze("0000000006", ["US_NPI"])[0].score == 0.7
    assert npi.analyze("NPI 0000000007", ["US_NPI"]) == []
    assert npi.analyze("NPI 0000000006", ["PERSON"]) == []
    mrn = MedicalRecordRecognizer()
    assert len(mrn.analyze("MRN: ZZZ12345", ["MEDICAL_RECORD_NUMBER"])) == 1
    assert mrn.analyze("ZZZ12345", ["MEDICAL_RECORD_NUMBER"]) == []
    assert mrn.analyze("MRN: ZZZ12345", ["PERSON"]) == []


def test_phi_synthetic_values_and_wrong_identifiers(redactor: Redactor) -> None:
    raw = (
        "Jane Roe, DOB 01/02/1900, SSN 123-45-6789, "
        "jane.roe@example.com, 202-555-0143, "
        "NPI 0000000006, MRN: ZZZ12345"
    )
    result, report = redactor.session("phi").redact_chat(chat(raw))
    text = result.messages[0].content
    for value in (
        "Jane Roe",
        "01/02/1900",
        "123-45-6789",
        "jane.roe@example.com",
        "202-555-0143",
        "0000000006",
        "ZZZ12345",
    ):
        assert value not in text
    for entity in (
        "PERSON",
        "DATE_TIME",
        "US_SSN",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_NPI",
        "MEDICAL_RECORD_NUMBER",
    ):
        assert f"<{entity}_1>" in text
        assert report.counts[entity] == 1
    wrong, _ = redactor.session("phi").redact_chat(chat("NPI 0000000007"))
    assert "<US_NPI_" not in wrong.messages[0].content
    assert "<MEDICAL_RECORD_NUMBER_" not in wrong.messages[0].content


def test_repeated_values_collision_and_reidentification(redactor: Redactor) -> None:
    request = ChatRequest(
        model="clinical-local",
        messages=[
            {"role": "system", "content": "<PERSON_1>"},
            {"role": "user", "content": "Jane Roe met Alex Smith"},
            {"role": "assistant", "content": [{"type": "text", "text": "Alex Smith"}]},
            {"role": "tool", "content": "Jane Roe"},
        ],
    )
    session = redactor.session("phi")
    result, report = session.redact_chat(request)
    contents = [
        m.content if isinstance(m.content, str) else m.content[0].text
        for m in result.messages
    ]
    assert contents[0] == "<PERSON_1>"
    assert "<PERSON_2>" in contents[1] and "<PERSON_2>" in contents[3]
    assert "<PERSON_3>" in contents[1] and "<PERSON_3>" in contents[2]
    assert report.counts["PERSON"] == 4
    assert (
        session.reidentify("<PERSON_1> <PERSON_2> <PERSON_3> <PERSON_999>")
        == "<PERSON_1> Jane Roe Alex Smith <PERSON_999>"
    )


def test_allow_terms_standard_and_embeddings(redactor: Redactor) -> None:
    allowed, report = redactor.session("phi", ["Jane Roe"]).redact_chat(
        chat("Jane Roe and jane.roe@example.com")
    )
    assert "Jane Roe" in allowed.messages[0].content
    assert "jane.roe@example.com" not in allowed.messages[0].content
    assert report.counts == {"EMAIL_ADDRESS": 1}
    request = EmbedRequest(
        model="clinical-local", input=["jane.roe@example.com", "jane.roe@example.com"]
    )
    result, report = redactor.session("standard").redact_embed(request)
    assert result.input == ["<EMAIL_ADDRESS_1>", "<EMAIL_ADDRESS_1>"]
    assert report.counts == {"EMAIL_ADDRESS": 2}
    single, _ = redactor.session("standard").redact_embed(
        EmbedRequest(model="clinical-local", input="jane.roe@example.com")
    )
    assert single.input == "<EMAIL_ADDRESS_1>"


def test_missing_model_stops_startup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PRESIDIO_NLP_MODEL", "missing_english_model")
    with pytest.raises(RuntimeError, match="run make models"):
        with TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'unused.db'}")):
            pass


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="REDACTION_SCORE_THRESHOLD"):
        Redactor.create("en_core_web_sm", threshold=1.1)
