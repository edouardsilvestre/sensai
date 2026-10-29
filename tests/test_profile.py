import pytest

from sensai.core.memory.profile import Profile, ProfileError


def test_missing_file_is_an_empty_profile(tmp_path):
    profile = Profile(tmp_path / "profile.md")
    assert profile.entries == [] and profile.text() == ""


def test_entries_are_read_one_per_line(tmp_path):
    path = tmp_path / "profile.md"
    path.write_text("- Answer in French.\n\n* I work for Le Monde.\nQuote amounts in EUR.\n", encoding="utf-8")
    assert Profile(path).entries == ["Answer in French.", "I work for Le Monde.", "Quote amounts in EUR."]


def test_added_entries_are_saved(tmp_path):
    path = tmp_path / "sub" / "profile.md"
    profile = Profile(path)
    profile.add("  Answer   in French. ")
    profile.add("Never name the source.")
    assert Profile(path).entries == ["Answer in French.", "Never name the source."]
    assert profile.text() == "- Answer in French.\n- Never name the source."


def test_empty_entry_is_rejected(tmp_path):
    with pytest.raises(ProfileError, match="empty"):
        Profile(tmp_path / "profile.md").add("   ")


def test_entries_are_removed_by_number(tmp_path):
    path = tmp_path / "profile.md"
    profile = Profile(path)
    profile.add("one")
    profile.add("two")
    assert profile.remove(1) == "one"
    assert Profile(path).entries == ["two"]


@pytest.mark.parametrize("number", [0, 2])
def test_unknown_number_is_rejected(tmp_path, number):
    profile = Profile(tmp_path / "profile.md")
    profile.add("one")
    with pytest.raises(ProfileError, match=f"No instruction {number}"):
        profile.remove(number)


def test_unreadable_profile_is_reported(tmp_path):
    path = tmp_path / "profile.md"
    path.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(ProfileError, match="Cannot read the profile"):
        Profile(path)
