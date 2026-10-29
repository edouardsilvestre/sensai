import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn(text: str) -> bool:
    digits = [int(d) for d in _digits(text)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def _iban(text: str) -> bool:
    iban = re.sub(r"\s", "", text).upper()
    number = "".join(str(int(c, 36)) for c in iban[4:] + iban[:4])
    return int(number) % 97 == 1


def _nir(text: str) -> bool:
    # French social security number: 13 characters then a key, 97 - (number mod 97). Corsica uses 2A and 2B.
    raw = re.sub(r"\s", "", text).upper()
    body, key = raw[:13].replace("2A", "19").replace("2B", "18"), int(raw[13:])
    return 97 - int(body) % 97 == key


@dataclass(frozen=True)
class Detector:
    label: str
    pattern: re.Pattern
    # Checks a candidate (checksum...), so that ordinary numbers are not taken for personal data.
    valid: Callable[[str], bool] = lambda text: True

    def find(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), self.label) for m in self.pattern.finditer(text) if self.valid(m.group())]


DETECTORS = {
    "email": Detector("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", re.I)),
    "phone": Detector(
        "PHONE",
        re.compile(r"(?<![\w+])(?:(?:\+|00)33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)|(?<![\w+])\+\d{1,3}(?:[\s.-]?\d{2,4}){3,5}(?!\d)"),
    ),
    "iban": Detector("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b"), _iban),
    "card": Detector("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), _luhn),
    "nir": Detector(
        "NIR", re.compile(r"(?<!\d)[12]\s?\d{2}\s?(?:0[1-9]|1[0-2])\s?(?:\d{2}|2A|2B)\s?\d{3}\s?\d{3}\s?\d{2}(?!\d)", re.I), _nir
    ),
    "ip": Detector("IP", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")),
}
