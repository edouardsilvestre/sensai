import os

import pytest

from sensai.core.tools.permissions import PermissionDenied, Permissions


@pytest.fixture
def case(tmp_path):
    for name in ["leak/mail.eml", "leak/sources/contact.txt", "leak/keys/id.key", "drafts/article.md", "secret.txt",
                 ".env", ".sensai/guardrails.json", "leak/.git/config"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return tmp_path


def permissions(case, **settings) -> Permissions:
    defaults = {"read": ["leak"], "write": ["drafts"], "deny": []}
    return Permissions(base=case, **(defaults | settings))


def reason(perms: Permissions, path: str, write: bool = False) -> str:
    with pytest.raises(PermissionDenied) as error:
        perms.check(path, write=write)
    return str(error.value)


def test_files_in_a_read_folder_can_be_read(case):
    assert permissions(case).check("leak/mail.eml") == (case / "leak" / "mail.eml").resolve()


def test_a_write_folder_can_also_be_read(case):
    assert permissions(case).check("drafts/article.md")


def test_a_read_folder_cannot_be_written(case):
    assert reason(permissions(case), "leak/mail.eml", write=True) == (
        "leak/mail.eml is outside the folders you may write (SENSAI_FILES_WRITE: drafts)"
    )


def test_a_file_outside_the_folders_is_denied(case):
    assert reason(permissions(case), "secret.txt") == (
        "secret.txt is outside the folders you may read (SENSAI_FILES_READ: drafts, leak)"
    )


@pytest.mark.parametrize("path", ["leak/../secret.txt", "leak/sources/../../secret.txt", "../secret.txt"])
def test_dot_dot_cannot_leave_a_folder(case, path):
    assert "outside the folders" in reason(permissions(case), path)


def test_an_absolute_path_is_checked_like_the_others(case):
    assert permissions(case).check(str(case / "leak" / "mail.eml"))
    assert "outside the folders" in reason(permissions(case), str(case / "secret.txt"))


def test_a_link_cannot_lead_out_of_a_folder(case):
    link = case / "leak" / "shortcut.txt"
    try:
        os.symlink(case / "secret.txt", link)
    except OSError:
        pytest.skip("creating links needs more rights on this system")
    assert "outside the folders" in reason(permissions(case), "leak/shortcut.txt")


@pytest.mark.parametrize(
    ("path", "pattern"),
    [("leak/sources/contact.txt", "sources"), ("leak/sources", "sources"), ("leak/keys/id.key", "*.key"),
     ("leak/SOURCES/contact.txt", "sources")],
)
def test_denied_patterns_apply_inside_the_allowed_folders(case, path, pattern):
    perms = permissions(case, deny=["sources/", "*.key"])
    assert reason(perms, path) == f"{path} is denied by the pattern '{pattern}'"


@pytest.mark.parametrize(("path", "pattern"), [(".env", ".env"), (".sensai/guardrails.json", ".sensai"),
                                               ("leak/.git/config", ".git")])
def test_settings_and_sensai_data_are_always_denied(case, path, pattern):
    perms = permissions(case, read=["."])
    assert reason(perms, path) == f"{path} is denied by the pattern '{pattern}'"


def test_sensai_data_stays_denied_when_allowed_explicitly(case):
    perms = permissions(case, read=[".sensai"])
    assert "denied by the pattern '.sensai'" in reason(perms, ".sensai/guardrails.json")


def test_user_patterns_do_not_apply_above_the_folder(case):
    perms = Permissions(read=[case / "leak" / "sources"], deny=["sources"], base=case)
    assert perms.check("leak/sources/contact.txt")


def test_no_folder_allows_nothing(case):
    assert reason(Permissions(base=case), "leak/mail.eml") == (
        "leak/mail.eml is outside the folders you may read (SENSAI_FILES_READ: none)"
    )


def test_an_empty_path_is_refused(case):
    assert reason(permissions(case), "  ") == "no path given"
