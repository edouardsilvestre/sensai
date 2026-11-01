import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sensai.core.guardrails.rules import GuardrailError
from sensai.core.llm.ollama import LLMError, OllamaClient
from sensai.interfaces.cli.main import main


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock(spec=OllamaClient)
    client.ensure_model.return_value = None
    client.chat.return_value = "Je ne peux pas répondre à cette demande."

    def fake_stream(*args, **kwargs):
        yield "Je ne peux pas répondre à cette demande."

    client.chat_stream.side_effect = fake_stream
    monkeypatch.setattr("sensai.interfaces.cli.runs.attack.OllamaClient", lambda *args, **kwargs: client)
    return client


def test_cli_attack_default_runs_builtin_suite(mock_client, capsys):
    code = main(["attack"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Running 12 adversarial attacks against" in out
    assert "[ 1/12]" in out
    assert "Category" in out and "Defended" in out and "Breached" in out and "Rate" in out
    assert "jailbreak" in out and "injection" in out and "malformed" in out
    assert "Overall" in out


def test_cli_attack_filter_by_category(mock_client, capsys):
    code = main(["attack", "--category", "jailbreak"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Running 4 adversarial attacks against" in out
    assert "(jailbreak)" in out
    assert "(injection)" not in out


def test_cli_attack_save_report_to_file(mock_client, tmp_path, capsys):
    out_file = tmp_path / "attack_report.json"
    code = main(["attack", "--out", str(out_file)])
    assert code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "summary" in data and "results" in data
    assert data["summary"]["total"] == 12
    assert f"Attack report saved to {out_file}" in capsys.readouterr().out


def test_cli_attack_compare_with_previous_report(mock_client, tmp_path, capsys):
    prev_report = tmp_path / "prev_report.json"
    prev_data = {
        "summary": {
            "total": 12,
            "defended": 6,
            "breached": 6,
            "defense_rate": 0.5,
            "by_category": {},
        },
        "results": [],
    }
    prev_report.write_text(json.dumps(prev_data), encoding="utf-8")

    code = main(["attack", "--compare", str(prev_report)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Comparison with {prev_report}:" in out
    assert "Previous defense rate : 50.0%" in out
    assert "Current defense rate  :" in out


def test_cli_attack_custom_suite_file(mock_client, tmp_path, capsys):
    custom_suite = tmp_path / "custom.json"
    custom_data = {
        "attacks": [
            {
                "id": "custom-1",
                "category": "jailbreak",
                "name": "Custom probe",
                "prompt": "Test prompt",
                "oracle": {"refusal_required": True},
            }
        ]
    }
    custom_suite.write_text(json.dumps(custom_data), encoding="utf-8")

    code = main(["attack", str(custom_suite)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Running 1 adversarial attack against" in out
    assert "Custom probe" in out


def test_cli_attack_reports_breached_case(mock_client, tmp_path, capsys):
    mock_client.chat_stream.side_effect = lambda *args, **kwargs: iter(["Sure, here are instructions."])
    custom_suite = tmp_path / "breached_suite.json"
    custom_data = {
        "attacks": [
            {
                "id": "custom-breach",
                "category": "jailbreak",
                "name": "Forced breach",
                "prompt": "Bypass instructions",
                "oracle": {"refusal_required": True},
            }
        ]
    }
    custom_suite.write_text(json.dumps(custom_data), encoding="utf-8")

    code = main(["attack", str(custom_suite)])
    assert code == 0
    out = capsys.readouterr().out
    assert "[BREACHED]" in out
    assert "Reason: Model complied with the adversarial request instead of refusing" in out


def test_cli_attack_invalid_suite_file(mock_client, tmp_path, capsys):
    code = main(["attack", str(tmp_path / "nonexistent.json")])
    assert code == 1
    assert "nonexistent.json" in capsys.readouterr().err


def test_cli_attack_invalid_compare_file(mock_client, tmp_path, capsys):
    code = main(["attack", "--compare", str(tmp_path / "bad_report.json")])
    assert code == 1
    assert "bad_report.json" in capsys.readouterr().err


def test_cli_attack_no_cases_for_category(mock_client, tmp_path, capsys):
    custom_suite = tmp_path / "only_jailbreak.json"
    custom_data = {
        "attacks": [
            {
                "id": "j-1",
                "category": "jailbreak",
                "name": "Jailbreak test",
                "prompt": "Test",
                "oracle": {"refusal_required": True},
            }
        ]
    }
    custom_suite.write_text(json.dumps(custom_data), encoding="utf-8")

    code = main(["attack", str(custom_suite), "--category", "injection"])
    assert code == 1
    assert "No attack cases found for category 'injection'" in capsys.readouterr().err


def test_cli_attack_llm_connection_error(mock_client, capsys):
    mock_client.ensure_model.side_effect = LLMError("Connection to Ollama refused")
    code = main(["attack"])
    assert code == 1
    assert "Connection to Ollama refused" in capsys.readouterr().err


def test_cli_attack_guardrail_load_error(mock_client, monkeypatch, capsys):
    monkeypatch.setattr(
        "sensai.interfaces.cli.runs.attack.load_guardrails",
        MagicMock(side_effect=GuardrailError("Malformed guardrails file")),
    )
    code = main(["attack"])
    assert code == 1
    assert "Malformed guardrails file" in capsys.readouterr().err
