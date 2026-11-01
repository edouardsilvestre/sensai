import re
import unicodedata
from dataclasses import dataclass

MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
COMMON_WORDS = {
    "a", "ainsi", "alors", "apres", "au", "aucun", "aucune", "aussi", "aux", "avant", "car", "cependant", "ce",
    "ces", "cette", "comme", "dans", "de", "des", "donc", "du", "elle", "elles", "en", "enfin", "ensuite", "et",
    "il", "ils", "je", "la", "le", "les", "leur", "lorsque", "mais", "nous", "non", "on", "oui", "ou", "par", "pas",
    "pour", "puis", "quand", "selon", "si", "son", "sa", "ses", "toutefois", "un", "une", "vous",
    "according", "also", "an", "and", "however", "he", "in", "it", "no", "she", "the", "there", "these", "they",
    "this", "yes",
}
IGNORED_NAMES = {"eur", "euro", "euros", "usd", "gbp", "chf", "sa", "sas", "sarl", "m", "mme", "mr", "mrs", "ms"}

_CITATION = re.compile(r"\[\d+(?:\s*[,;]\s*\d+)*\]")
_MONTH = "|".join(MONTHS)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WRITTEN_DATE = re.compile(rf"\b(?:(\d{{1,2}})(?:er)?\s+)?({_MONTH})\s+(\d{{4}})\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_AMOUNT = re.compile(
    r"\b\d{1,3}(?:[ .]\d{3})+(?:,\d+)?\b|\b\d+,\d+\b|\b\d{4,}\b|\b\d+(?=\s?(?:%|€|eur\b|euros?\b))"
)
_CAPITAL = "A-ZÀ-ÖØ-Þ"
_NAME = re.compile(rf"(?<![\w'’])[{_CAPITAL}][\w'’-]*(?:\s+[{_CAPITAL}][\w'’-]*)*")
_ELISION = re.compile(r"^[LlDd]['’]")


@dataclass(frozen=True)
class Fact:
    kind: str
    text: str
    value: str


def plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.replace(" ", " ").replace(" ", " "))
    return "".join(char for char in text if not unicodedata.combining(char)).lower()


def _amount(text: str) -> str:
    whole, _, cents = re.sub(r"[ .]", "", text).partition(",")
    cents = cents.rstrip("0")
    return f"{int(whole)}.{cents}" if cents else str(int(whole))


def _dates(text: str) -> tuple[list[Fact], str]:
    found = []

    def keep(match: re.Match, value: str) -> str:
        found.append(Fact("date", match.group(0), value))
        return " "

    text = _ISO_DATE.sub(lambda m: keep(m, f"{m[1]}-{m[2]}-{m[3]}"), text)
    text = _NUMERIC_DATE.sub(lambda m: keep(m, f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}"), text)
    text = _WRITTEN_DATE.sub(
        lambda m: keep(m, f"{m[3]}-{MONTHS[m[2]]:02d}" + (f"-{int(m[1]):02d}" if m[1] else "")), text
    )
    text = _YEAR.sub(lambda m: keep(m, m[0]), text)
    return found, text


def _names(text: str) -> list[Fact]:
    names = []
    for match in _NAME.finditer(text):
        words = [_ELISION.sub("", word) for word in match.group(0).split()]
        start = match.start() == 0 or re.search(r"[.!?:]\s*$", text[:match.start()])
        if words and start and plain(words[0]) in COMMON_WORDS:
            words = words[1:]
        words = [
            word for word in words
            if word[:1].isupper() and plain(word) not in IGNORED_NAMES and plain(word) not in MONTHS
        ]
        if words:
            name = " ".join(words)
            names.append(Fact("name", name, plain(name)))
    return names


def facts(text: str) -> list[Fact]:
    text = _CITATION.sub(" ", text.replace(" ", " ").replace(" ", " "))
    dates, rest = _dates(plain(text))
    names = _names(_CITATION.sub(" ", text))
    amounts = [Fact("amount", match.group(0), _amount(match.group(0))) for match in _AMOUNT.finditer(rest)]
    return amounts + dates + names


def known(evidence: str) -> tuple[set[str], str]:
    values = set()
    for fact in facts(evidence):
        values.add(fact.value)
        if fact.kind == "date":
            values.update(fact.value[:length] for length in (4, 7))
    return values, plain(evidence)


def missing(sentence: str, evidence: str) -> list[Fact]:
    values, text = known(evidence)
    absent = []
    for fact in facts(sentence):
        if fact.kind == "name":
            found = re.search(rf"\b{re.escape(fact.value)}\b", text) is not None
        else:
            found = fact.value in values
        if not found and fact not in absent:
            absent.append(fact)
    return absent
