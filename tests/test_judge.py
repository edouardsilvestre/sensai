import json

import pytest

from sensai.core.evaluation.judge import (
    CLAIM_SCHEMA,
    NO_EVIDENCE,
    SCORE_SCHEMA,
    Claim,
    Judgement,
    JudgeError,
    evidence,
    judge,
    parse_claim,
    parse_scores,
    sentences,
)
from sensai.core.rag.models import SearchResult

STATUTES = SearchResult(1, "Gérance : M. Marc Vidal, nommé gérant.", "p. 2", "statuts_orion.pdf", "", 0.9)
STATEMENT = SearchResult(2, "05/04/2023 VIR NORDBANK SA -> ORION 480 000,00", "", "releve.csv", "", 0.8)
ANSWER = "Marc Vidal dirige Orion [1]. Il est le seul associé [1]."


def verdict(value: str, reason: str = "the evidence says so") -> str:
    return json.dumps({"reason": reason, "verdict": value})


def scores(relevance: int = 5, coherence: int = 4) -> str:
    return json.dumps({"relevance": relevance, "coherence": coherence})


class Recorder:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, model, messages, options=None, think=None, usage=None, format=None) -> str:
        self.calls.append({"model": model, "prompt": messages[0]["content"], "options": options, "think": think,
                           "format": format})
        return self.replies.pop(0)


def test_an_answer_is_cut_into_sentences():
    answer = "Marc Vidal dirige Orion [1]. Est-ce légal ? Oui !\n- Le siège est à Lyon.\n\n* Fin\n[1]"
    assert sentences(answer) == [
        "Marc Vidal dirige Orion [1].", "Est-ce légal ?", "Oui !", "Le siège est à Lyon.", "Fin"
    ]


def test_an_amount_or_a_date_does_not_cut_a_sentence():
    assert sentences("Il a reçu 480 000,00 EUR le 05.04.2023 de Nordbank.") == [
        "Il a reçu 480 000,00 EUR le 05.04.2023 de Nordbank."
    ]


def test_an_initial_does_not_cut_a_sentence():
    assert sentences("M. Marc Vidal dirige Orion. J. Martin a signé.") == [
        "M. Marc Vidal dirige Orion.", "J. Martin a signé."
    ]


def test_each_sentence_is_checked_on_its_own_then_the_answer_is_scored():
    client = Recorder(verdict("supported"), verdict("contradicted", "he is the manager"), scores())
    judgement = judge(client, "qwen3:1.7b", "Qui dirige Orion ?", ANSWER, evidence([STATUTES, STATEMENT]),
                      {"num_ctx": 8192}, think=True)

    assert judgement == Judgement(
        [
            Claim("Marc Vidal dirige Orion [1].", "supported", "the evidence says so"),
            Claim("Il est le seul associé [1].", "contradicted", "he is the manager"),
        ],
        5, 4,
    )
    first, second, last = client.calls
    assert [call["format"] for call in client.calls] == [CLAIM_SCHEMA, CLAIM_SCHEMA, SCORE_SCHEMA]
    assert {(call["model"], call["think"], call["options"]["num_ctx"]) for call in client.calls} == {
        ("qwen3:1.7b", True, 8192)
    }
    assert "Sentence: Marc Vidal dirige Orion [1]." in first["prompt"] and "seul associé" not in first["prompt"]
    assert "Evidence:\n[1] statuts_orion.pdf, p. 2:" in second["prompt"]
    assert "Question: Qui dirige Orion ?" in last["prompt"] and ANSWER in last["prompt"]


def test_a_sentence_with_an_amount_a_date_or_a_name_not_in_the_evidence_is_not_sent_to_the_model():
    client = Recorder(verdict("supported"), scores())
    answer = "Marc Vidal dirige Orion [1]. Paul Girard a reçu 35 000 EUR de Nordbank en 2022 [2]."
    judgement = judge(client, "qwen3:1.7b", "Qui a reçu de l'argent ?", answer, evidence([STATUTES, STATEMENT]))

    assert judgement.claims[1] == Claim(
        "Paul Girard a reçu 35 000 EUR de Nordbank en 2022 [2].", "unsupported",
        "Not in the evidence: 35 000, 2022, Paul Girard.",
    )
    assert [call["format"] for call in client.calls] == [CLAIM_SCHEMA, SCORE_SCHEMA]
    assert "Paul Girard" not in client.calls[0]["prompt"]


def test_faithfulness_is_the_share_of_supported_sentences():
    judgement = Judgement(
        [Claim("a", "supported"), Claim("b", "supported"), Claim("c", "contradicted"), Claim("d", "unsupported")], 4, 4
    )
    assert judgement.faithfulness == 0.5
    assert [claim.text for claim in judgement.flagged] == ["c", "d"]


def test_an_answer_without_sentences_has_no_faithfulness():
    client = Recorder(scores(1, 1))
    judgement = judge(client, "qwen3:1.7b", "Qui dirige Orion ?", "[1]", NO_EVIDENCE)
    assert (judgement.claims, judgement.faithfulness, len(client.calls)) == ([], None, 1)


@pytest.mark.parametrize(
    "reply, message",
    [
        ("not json", "did not reply in JSON"),
        ("[]", "did not reply with a JSON object"),
        ('{"reason": "x", "verdict": "maybe"}', "invalid verdict: 'maybe'"),
        ('{"reason": "x"}', "invalid verdict: None"),
    ],
)
def test_an_invalid_verdict_is_an_error(reply, message):
    with pytest.raises(JudgeError, match=message):
        parse_claim("Marc Vidal dirige Orion.", reply)


@pytest.mark.parametrize(
    "reply, message",
    [
        ('{"relevance": 0, "coherence": 5}', "invalid relevance score: 0"),
        ('{"relevance": 5, "coherence": "high"}', "invalid coherence score: 'high'"),
        ('{"relevance": true, "coherence": 5}', "invalid relevance score: True"),
        ('{"coherence": 5}', "invalid relevance score: None"),
    ],
)
def test_invalid_scores_are_an_error(reply, message):
    with pytest.raises(JudgeError, match=message):
        parse_scores(reply)


def test_evidence_numbers_the_excerpts_like_the_answer_cites_them():
    assert evidence([STATUTES, STATEMENT], ["Orion Delta Consulting: SARL, Lyon"]) == (
        "[1] statuts_orion.pdf, p. 2:\nGérance : M. Marc Vidal, nommé gérant.\n\n"
        "[2] releve.csv:\n05/04/2023 VIR NORDBANK SA -> ORION 480 000,00\n\n"
        "Tool result:\nOrion Delta Consulting: SARL, Lyon"
    )


def test_no_evidence_is_said_plainly():
    assert evidence([]) == NO_EVIDENCE
