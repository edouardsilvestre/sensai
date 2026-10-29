from sensai.core.approval import Approval, refusal


def test_refusal_gives_the_reason_to_the_model():
    result = refusal(Approval(False, "keep the source secret"))
    assert result.content.startswith("The user refused this tool call: keep the source secret.")
    assert result.summary == "refused" and not result.error


def test_refusal_without_reason():
    assert refusal(Approval(False)).content.startswith("The user refused this tool call. Do not call it again")


def test_refusal_tells_the_model_not_to_guess():
    content = refusal(Approval(False)).content
    assert "do not make up what it would have found" in content
    assert "only state what you already know for sure" in content
