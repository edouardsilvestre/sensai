import json
import re
from dataclasses import dataclass

from sensai.core.evaluation.facts import missing
from sensai.core.llm.ollama import OllamaClient
from sensai.core.rag.models import SearchResult

VERDICTS = ("supported", "contradicted", "unsupported")
FLAGGED = ("contradicted", "unsupported")
SCORES = ("relevance", "coherence")

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["reason", "verdict"],
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": list(SCORES),
}

CLAIM_PROMPT = """Evidence:
{evidence}

Sentence: {sentence}

Does the evidence state what the sentence says? Compare each detail of the sentence (who, what, how much, when)
with the evidence.
supported: the evidence states every detail of the sentence. contradicted: the evidence states something else
for a detail. unsupported: the evidence says nothing about a detail.
A sentence saying that the documents do not contain some information is supported if the evidence indeed does
not contain it.
First give the reason, then the verdict. Reply with JSON only: {{"reason": REASON, "verdict": VERDICT}}"""

SCORE_PROMPT = """Score the answer below, given to the question of a journalist, from 1 to 5:
- relevance: 5 when it answers the question asked, 1 when it answers another one;
- coherence: 5 when it is clear and does not contradict itself, 1 when it cannot be followed.
Reply with JSON only: {{"relevance": N, "coherence": N}}

Question: {question}

Answer:
{answer}"""

NO_EVIDENCE = "None: the answer was given without any document excerpt nor tool result."

_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)\s+|\n+")


class JudgeError(Exception):
    pass


@dataclass
class Claim:
    text: str
    verdict: str
    reason: str = ""


@dataclass
class Judgement:
    claims: list[Claim]
    relevance: int
    coherence: int

    @property
    def flagged(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.verdict in FLAGGED]

    @property
    def faithfulness(self) -> float | None:
        if not self.claims:
            return None
        return sum(claim.verdict == "supported" for claim in self.claims) / len(self.claims)


def sentences(answer: str) -> list[str]:
    parts = (part.strip().lstrip("-*• ").strip() for part in _SENTENCE_END.split(answer))
    return [part for part in parts if re.search(r"\w{2}", part)]


def evidence(sources: list[SearchResult], results: list[str] | None = None) -> str:
    parts = []
    for number, source in enumerate(sources, start=1):
        place = f", {source.location}" if source.location else ""
        parts.append(f"[{number}] {source.source}{place}:\n{source.text}")
    parts += [f"Tool result:\n{result}" for result in results or []]
    return "\n\n".join(parts) or NO_EVIDENCE


def _json(reply: str) -> dict:
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        raise JudgeError("The judge did not reply in JSON") from None
    if not isinstance(data, dict):
        raise JudgeError("The judge did not reply with a JSON object")
    return data


def parse_claim(sentence: str, reply: str) -> Claim:
    data = _json(reply)
    if data.get("verdict") not in VERDICTS:
        raise JudgeError(f"The judge gave an invalid verdict: {data.get('verdict')!r}")
    return Claim(sentence, data["verdict"], str(data.get("reason", "")).strip())


def parse_scores(reply: str) -> tuple[int, int]:
    data = _json(reply)
    scores = [data.get(name) for name in SCORES]
    for name, score in zip(SCORES, scores):
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise JudgeError(f"The judge gave an invalid {name} score: {score!r}")
    return scores[0], scores[1]


def judge(
    client: OllamaClient, model: str, question: str, answer: str, given: str,
    options: dict | None = None, think: bool | None = None,
) -> Judgement:
    claims = []
    for sentence in sentences(answer):
        absent = missing(sentence, given)
        if absent:
            reason = f"Not in the evidence: {', '.join(fact.text for fact in absent)}."
            claims.append(Claim(sentence, "unsupported", reason))
            continue
        prompt = CLAIM_PROMPT.format(evidence=given, sentence=sentence)
        reply = client.chat(model, [{"role": "user", "content": prompt}], options, think=think, format=CLAIM_SCHEMA)
        claims.append(parse_claim(sentence, reply))
    prompt = SCORE_PROMPT.format(question=question, answer=answer)
    reply = client.chat(model, [{"role": "user", "content": prompt}], options, think=think, format=SCORE_SCHEMA)
    return Judgement(claims, *parse_scores(reply))
