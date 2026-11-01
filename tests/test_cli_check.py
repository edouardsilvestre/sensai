from sensai.core.agent import Agent
from sensai.core.evaluation.judge import UNCHECKED, Claim, Judgement
from sensai.core.events import CheckEvent, CheckFailedEvent, CheckStartedEvent
from sensai.interfaces.cli.answer import chat_turn, show_events
from sensai.interfaces.cli.session import Session

from test_agent_check import ANSWER, JudgeClient, Retriever

JUDGEMENT = Judgement(
    [
        Claim("Orion Delta Consulting est gérée par Marc Vidal [1].", "supported", "the statutes say so"),
        Claim("Il a touché 480 000 EUR en 2022 [2].", "unsupported", "2022", by="code"),
        Claim("Marc Vidal est le seul associé d'Orion [1].", "contradicted", "he is the manager, not an associate"),
        Claim("Il dirige aussi Nordbank.", "unsupported", "the evidence says nothing about it"),
        Claim("Il est gérant depuis 2019.", UNCHECKED, "The judge did not reply in JSON"),
    ],
    5, 4,
)


def show(session, *events) -> None:
    show_events(session, iter(events))


def test_the_check_says_what_is_certain_and_what_is_only_a_doubt(config, client, capsys):
    session = Session(config, Agent(config, client))
    show(session, CheckEvent(JUDGEMENT))
    assert capsys.readouterr().out.splitlines() == [
        "[check] 5 sentences: 1 supported, faithfulness 25%, relevance 5/5, coherence 4/5",
        '  not in the excerpts: "Il a touché 480 000 EUR en 2022 [2]." (2022)',
        '  contradicted: "Marc Vidal est le seul associé d\'Orion [1]." (he is the manager, not an associate)',
        '  to verify: "Il dirige aussi Nordbank." (the evidence says nothing about it)',
        '  unchecked: "Il est gérant depuis 2019." (The judge did not reply in JSON)',
    ]


def test_a_judgement_without_checked_sentences_nor_scores(config, client, capsys):
    session = Session(config, Agent(config, client))
    show(session, CheckEvent(Judgement([Claim("Il est gérant.", UNCHECKED)], None, None)))
    assert capsys.readouterr().out.splitlines() == [
        "[check] 1 sentence: 0 supported, faithfulness -, relevance -/5, coherence -/5",
        '  unchecked: "Il est gérant."',
    ]


def test_the_check_announces_that_it_takes_time(config, client, capsys):
    session = Session(config, Agent(config, client))
    show(session, CheckStartedEvent(3, "qwen3:1.7b"))
    assert capsys.readouterr().out == (
        "[check] Checking 3 sentences with qwen3:1.7b, this can take a few minutes (Ctrl+C to skip)...\n"
    )


def test_a_failed_check_is_a_warning(config, client, capsys):
    session = Session(config, Agent(config, client))
    show(session, CheckFailedEvent("Cannot reach Ollama"))
    assert "[warning] The answer could not be checked: Cannot reach Ollama" in capsys.readouterr().err


def test_a_chat_turn_shows_the_answer_then_its_check(config, capsys):
    config.features.update(document=False, fact_check=True)
    session = Session(config, Agent(config, JudgeClient("supported"), retriever=Retriever()))
    chat_turn(session, "Qui dirige Orion ?")
    out = capsys.readouterr().out
    assert out.index(ANSWER) < out.index("[check] Checking 1 sentence") < out.index("[check] 1 sentence: 1 supported")
