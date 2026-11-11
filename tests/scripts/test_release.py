"""scripts/release.py: bump, check and notes, run against copies of the
repository's real version files."""

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "release", ROOT / "scripts" / "release.py"
)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

FILES = [*release.VERSION_SPOTS, release.CHANGELOG]


@pytest.fixture
def repo(tmp_path):
    """A copy of the version files, with an [Unreleased] entry to release."""
    for name in FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    changelog = tmp_path / release.CHANGELOG
    changelog.write_text(re.sub(
        r"^## \[Unreleased\]\n.*?(?=^## \[)",
        "## [Unreleased]\n\n### Added\n\n- A test entry.\n\n",
        changelog.read_text(),
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    ))
    return tmp_path


def test_real_files_have_every_version_line():
    versions = release.read_versions(ROOT)
    assert None not in versions.values(), versions


def test_bump_updates_every_file(repo):
    previous = release.read_versions(repo)["pyproject.toml"]
    release.bump("9.0.0", "2030-01-02", root=repo)

    assert set(release.read_versions(repo).values()) == {"9.0.0"}
    assert 'date-released: "2030-01-02"' in (repo / "CITATION.cff").read_text()
    changelog = (repo / release.CHANGELOG).read_text()
    assert "## [Unreleased]\n\n## [9.0.0] - 2030-01-02\n" in changelog
    assert "[Unreleased]: https://github.com/terrahydro-esdt/crest/" \
        "compare/v9.0.0...HEAD" in changelog
    assert "[9.0.0]: https://github.com/terrahydro-esdt/crest/" \
        f"compare/v{previous}...v9.0.0" in changelog
    assert release.check(repo, tag="v9.0.0") == []


def test_bump_changes_only_the_version_lines(repo):
    current = release.read_versions(repo)["pyproject.toml"]
    before = (repo / "pyproject.toml").read_text().splitlines()
    release.bump("9.0.0", "2030-01-02", root=repo)
    after = (repo / "pyproject.toml").read_text().splitlines()

    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after)
    assert changed == [(f'version = "{current}"', 'version = "9.0.0"')]


def test_notes_are_the_section_body(repo):
    release.bump("9.0.0", "2030-01-02", root=repo)

    notes = release.release_notes("9.0.0", repo)
    assert notes == "### Added\n\n- A test entry."
    notes = release.release_notes("0.1.0", repo)
    assert notes.startswith("### Summary") and "]: https://" not in notes


@pytest.mark.parametrize("version, message", [
    ("9.0", "not X.Y.Z"),
    ("0.0.1", "not newer"),
])
def test_bump_refuses_bad_versions(repo, version, message):
    with pytest.raises(release.ReleaseError, match=message):
        release.bump(version, "2030-01-02", root=repo)


def test_bump_refuses_empty_unreleased(repo):
    release.bump("9.0.0", "2030-01-02", root=repo)
    with pytest.raises(release.ReleaseError, match="nothing under"):
        release.bump("9.0.1", "2030-01-03", root=repo)


def test_failed_bump_writes_nothing(repo):
    about = repo / "crest" / "__about__.py"
    about.write_text("# version line removed\n")
    before = {name: (repo / name).read_text() for name in FILES}

    with pytest.raises(release.ReleaseError, match="crest/__about__.py"):
        release.bump("9.0.0", "2030-01-02", root=repo)
    assert {name: (repo / name).read_text() for name in FILES} == before


def test_check_reports_problems(repo):
    release.bump("9.0.0", "2030-01-02", root=repo)
    about = repo / "crest" / "__about__.py"
    about.write_text('__version__ = "9.0"\n')

    problems = release.check(repo, tag="v9.0.0")
    assert len(problems) == 1 and "versions disagree" in problems[0]

    about.write_text('__version__ = "9.0.0"\n')
    assert any("does not match" in p for p in release.check(repo, tag="v9.1.0"))


def test_check_requires_matching_citation_date(repo):
    release.bump("9.0.0", "2030-01-02", root=repo)
    citation = repo / "CITATION.cff"
    text = citation.read_text()
    citation.write_text(text.replace("2030-01-02", "2030-01-03"))

    assert any("date-released" in p for p in release.check(repo))


def test_main_exit_codes(repo, monkeypatch, capsys):
    monkeypatch.setattr(release, "ROOT", repo)
    monkeypatch.setattr(release.bump, "__defaults__", (repo,))
    monkeypatch.setattr(release.check, "__defaults__", (repo, None))
    monkeypatch.setattr(release.read_versions, "__defaults__", (repo,))
    monkeypatch.setattr(release.release_notes, "__defaults__", (repo,))

    assert release.main(["bump", "9.0.0", "--date", "2030-01-02"]) == 0
    assert release.main(["check", "--tag", "v9.0.0"]) == 0
    assert release.main(["check", "--tag", "v1.0.0"]) == 1
    capsys.readouterr()
    assert release.main(["notes"]) == 0
    assert capsys.readouterr().out == "### Added\n\n- A test entry.\n"
