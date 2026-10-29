import pytest

from sensai.core.guardrails.detectors import DETECTORS


def found(name: str, text: str) -> list[str]:
    return [text[span.start : span.end] for span in DETECTORS[name].find(text)]


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("email", "Écrire à marc.vidal@alvor-consulting.com demain", ["marc.vidal@alvor-consulting.com"]),
        ("phone", "Appeler le 06 12 34 56 78 ou le +33 6 12 34 56 78", ["06 12 34 56 78", "+33 6 12 34 56 78"]),
        ("phone", "Bureau : 01.23.45.67.89", ["01.23.45.67.89"]),
        ("iban", "IBAN FR76 3000 6000 0112 3456 7890 189 du compte", ["FR76 3000 6000 0112 3456 7890 189"]),
        ("iban", "compte FR7630006000011234567890189", ["FR7630006000011234567890189"]),
        ("card", "carte 4111 1111 1111 1111 expirée", ["4111 1111 1111 1111"]),
        ("nir", "numéro 2 69 05 49 588 157 80 de la salariée", ["2 69 05 49 588 157 80"]),
        ("ip", "connexion depuis 192.168.1.20 la nuit", ["192.168.1.20"]),
    ],
)
def test_personal_data_is_found(name, text, expected):
    assert found(name, text) == expected


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("iban", "FR76 3000 6000 0112 3456 7890 180"),
        ("card", "4111 1111 1111 1112"),
        ("nir", "1 85 05 78 006 084 37"),
        ("ip", "version 1.2.3.4.5"),
    ],
)
def test_invalid_checksums_are_not_personal_data(name, text):
    assert found(name, text) == []


def test_amounts_dates_and_references_are_not_personal_data():
    text = "Virement de 480 000 EUR le 03/04/2023, facture 2023-0412, ligne 12, total 1 234 567,89 EUR"
    assert {name: found(name, text) for name in DETECTORS} == {name: [] for name in DETECTORS}
