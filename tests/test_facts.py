import pytest

from sensai.core.evaluation.facts import Fact, facts, missing

EVIDENCE = """[1] statuts_orion.txt:
Gérance : la société Orion Delta Consulting est gérée par M. Marc Vidal, nommé gérant pour une durée illimitée.

[2] releve_2023-04.csv, row 2:
05/04/2023;VIR NORDBANK SA -> ORION DELTA CONSULTING conseil stratégique;;480 000,00"""


def values(text: str, kind: str) -> list[str]:
    return [fact.value for fact in facts(text) if fact.kind == kind]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("480 000,00", "480000"),
        ("480 000 €", "480000"),
        ("480.000 EUR", "480000"),
        ("1 500,50 euros", "1500.5"),
        ("12500", "12500"),
        ("35 €", "35"),
        ("12 %", "12"),
    ],
)
def test_amounts_are_compared_whatever_their_format(text, expected):
    assert values(f"Il a reçu {text}.", "amount") == [expected]


def test_small_numbers_and_citations_are_not_amounts():
    assert values("Les 2 virements cités en [1][2] et [3, 4].", "amount") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("le 05/04/2023", "2023-04-05"),
        ("le 2023-04-05", "2023-04-05"),
        ("le 5 avril 2023", "2023-04-05"),
        ("le 1er août 2023", "2023-08-01"),
        ("en avril 2023", "2023-04"),
        ("en 2022", "2022"),
        ("on April 2023", "2023-04"),
    ],
)
def test_dates_are_compared_whatever_their_format(text, expected):
    assert values(f"Le virement a eu lieu {text}.", "date") == [expected]


def test_a_date_is_not_read_as_an_amount():
    assert values("Le 05/04/2023, en 2023.", "amount") == []


def test_names_are_the_words_in_capitals():
    assert values("Selon le relevé, Marc Vidal et Mme Sophie Vidal ont écrit à Nordbank SA.", "name") == [
        "marc vidal", "sophie vidal", "nordbank"
    ]


@pytest.mark.parametrize(
    "sentence",
    [
        "Les documents ne le disent pas.",
        "Cependant, il est gérant.",
        "L'article dit qu'il est gérant.",
        "However, he runs the company.",
        "En avril 2023, il a été payé.",
    ],
)
def test_a_common_word_opening_a_sentence_is_not_a_name(sentence):
    assert values(sentence, "name") == []


def test_a_name_opening_a_sentence_is_kept():
    assert values("Nordbank a payé Orion.", "name") == ["nordbank", "orion"]


def test_nothing_is_missing_from_a_right_answer():
    for sentence in [
        "Orion Delta Consulting est gérée par Marc Vidal [1].",
        "Elle a reçu un virement de 480 000 EUR de Nordbank le 5 avril 2023 [2].",
        "Le virement date d'avril 2023.",
        "Les documents ne mentionnent pas le salaire de Marc Vidal.",
    ]:
        assert missing(sentence, EVIDENCE) == []


def test_what_the_evidence_does_not_hold_is_missing():
    absent = missing("Paul Girard a touché 35 000 EUR de Nordbank en 2022 [2].", EVIDENCE)
    assert absent == [
        Fact("amount", "35 000", "35000"), Fact("date", "2022", "2022"), Fact("name", "Paul Girard", "paul girard")
    ]


def test_a_name_must_be_a_whole_word_of_the_evidence():
    assert [fact.text for fact in missing("Orio a reçu l'argent.", EVIDENCE)] == ["Orio"]


def test_an_invented_claim_without_new_facts_is_left_to_the_judge():
    assert missing("Marc Vidal est le seul associé d'Orion [1].", EVIDENCE) == []
