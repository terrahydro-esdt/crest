#!/usr/bin/env python
"""
Bump, check and print the notes of a CREST release.

The version is written in several files (pyproject.toml, crest/__about__.py,
CITATION.cff, the BibTeX block in README.rst) and each release has a section in
CHANGELOG.md. This script keeps them in step:

    python scripts/release.py bump 0.2.0 [--date 2026-11-12]
        Sets every version to 0.2.0, sets CITATION.cff's date-released, turns
        the [Unreleased] section of CHANGELOG.md into "## [0.2.0] - <date>"
        (leaving an empty [Unreleased] above it) and updates the compare
        links at the bottom.

    python scripts/release.py check [--tag v0.2.0]
        Fails unless every version agrees, CHANGELOG.md has a dated, non-empty
        section for it, and (with --tag) the tag is "v" + that version.

    python scripts/release.py notes [0.2.0]
        Prints that version's CHANGELOG.md section (default: the current
        version), for use as the GitHub release notes.

Standard library only, so CI can run it without installing CREST.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# file -> pattern of the line holding the version (group "v")
VERSION_SPOTS = {
    "pyproject.toml": r'^version = "(?P<v>[^"]+)"',
    "crest/__about__.py": r'^__version__ = "(?P<v>[^"]+)"',
    "CITATION.cff": r"^version: (?P<v>\S+)",
    "README.rst": r"^\s+version = \{(?P<v>[^}]+)\}",
}
CITATION_DATE = r'^date-released: "(?P<v>[^"]+)"'
CHANGELOG = "CHANGELOG.md"

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
HEADING = re.compile(
    r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\S+))?$", re.MULTILINE
)
LINK_REFS = re.compile(r"^\[[^\]]+\]: \S+$", re.MULTILINE)
UNRELEASED_LINK = re.compile(
    r"^\[Unreleased\]: (?P<base>\S+)/compare/(?P<prev>\S+)\.\.\.HEAD$",
    re.MULTILINE,
)


class ReleaseError(Exception):
    """A problem that stops a bump, check or notes command."""


def _find(text: str, pattern: str):
    return re.search(pattern, text, re.MULTILINE)


def _replace(root: Path, name: str, pattern: str, value: str) -> None:
    path = root / name
    text = path.read_text()
    match = _find(text, pattern)
    if match is None:
        raise ReleaseError(f"{name}: no line matches {pattern!r}")
    start, end = match.span("v")
    path.write_text(text[:start] + value + text[end:])


def read_versions(root: Path = ROOT) -> dict:
    """Return {file: version found in it, or None}."""
    versions = {}
    for name, pattern in VERSION_SPOTS.items():
        match = _find((root / name).read_text(), pattern)
        versions[name] = match["v"] if match else None
    return versions


def _sections(text: str) -> dict:
    """Return {heading name: (date, body)} for CHANGELOG.md's ## sections."""
    headings = list(HEADING.finditer(text))
    sections = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.end():end]
        refs = LINK_REFS.search(body)
        if refs:
            body = body[:refs.start()]
        sections[heading["name"]] = (heading["date"], body.strip())
    return sections


def release_notes(version: str, root: Path = ROOT) -> str:
    """Return the body of ``version``'s CHANGELOG.md section."""
    sections = _sections((root / CHANGELOG).read_text())
    if version not in sections:
        raise ReleaseError(f"{CHANGELOG}: no section for {version}")
    return sections[version][1]


def check(root: Path = ROOT, tag: str | None = None) -> list:
    """Return the problems that should stop a release (empty when ready)."""
    problems = []
    versions = read_versions(root)
    for name, found in versions.items():
        if found is None:
            problems.append(f"{name}: version line not found")
    found = {v for v in versions.values() if v is not None}
    if len(found) > 1:
        listing = ", ".join(f"{n}={v}" for n, v in versions.items())
        problems.append(f"versions disagree: {listing}")
    if len(found) != 1:
        return problems

    version = found.pop()
    if not SEMVER.match(version):
        problems.append(f"version {version!r} is not X.Y.Z")
    if tag is not None and tag != f"v{version}":
        problems.append(f"tag {tag!r} does not match version v{version}")

    changelog = (root / CHANGELOG).read_text()
    date, body = _sections(changelog).get(version, (None, ""))
    if date is None:
        problems.append(f"{CHANGELOG}: no dated section for {version}")
    elif not body:
        problems.append(f"{CHANGELOG}: section for {version} is empty")
    if not re.search(rf"^\[{re.escape(version)}\]: ", changelog, re.MULTILINE):
        problems.append(f"{CHANGELOG}: no link for [{version}]")

    cited = _find((root / "CITATION.cff").read_text(), CITATION_DATE)
    if date and (cited is None or cited["v"] != date):
        problems.append(
            f"CITATION.cff: date-released is "
            f"{cited['v'] if cited else 'missing'}, {CHANGELOG} says {date}"
        )
    return problems


def bump(version: str, date: str, root: Path = ROOT) -> None:
    """Set every version to ``version`` and roll [Unreleased] into it."""
    if not SEMVER.match(version):
        raise ReleaseError(f"version {version!r} is not X.Y.Z")
    datetime.date.fromisoformat(date)

    # check everything before writing anything, so a failed bump changes nothing
    versions = read_versions(root)
    missing = [name for name, found in versions.items() if found is None]
    if _find((root / "CITATION.cff").read_text(), CITATION_DATE) is None:
        missing.append("CITATION.cff (date-released)")
    if missing:
        raise ReleaseError(f"version line not found in {', '.join(missing)}")
    current = versions["pyproject.toml"]
    if SEMVER.match(current) and tuple(map(int, version.split("."))) <= tuple(
        map(int, current.split("."))
    ):
        raise ReleaseError(f"{version} is not newer than {current}")

    path = root / CHANGELOG
    text = path.read_text()
    unreleased = re.search(r"^## \[Unreleased\]\n", text, re.MULTILINE)
    if unreleased is None:
        raise ReleaseError(f"{CHANGELOG}: no ## [Unreleased] heading")
    if not _sections(text).get("Unreleased", (None, ""))[1]:
        raise ReleaseError(f"{CHANGELOG}: nothing under [Unreleased]")
    link = UNRELEASED_LINK.search(text)
    if link is None:
        raise ReleaseError(f"{CHANGELOG}: no [Unreleased] compare link")

    # links first: they come after the heading, so its offsets stay valid
    base, prev = link["base"], link["prev"]
    text = (
        text[:link.start()]
        + f"[Unreleased]: {base}/compare/v{version}...HEAD\n"
        + f"[{version}]: {base}/compare/{prev}...v{version}"
        + text[link.end():]
    )
    text = (
        text[:unreleased.end()]
        + f"\n## [{version}] - {date}\n"
        + text[unreleased.end():]
    )
    path.write_text(text)

    for name, pattern in VERSION_SPOTS.items():
        _replace(root, name, pattern, version)
    _replace(root, "CITATION.cff", CITATION_DATE, date)


def main(argv=None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p_bump = commands.add_parser("bump", help="start a release")
    p_bump.add_argument("version", help="new version, X.Y.Z")
    p_bump.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),  # noqa: DTZ011 (local date)
        help="release date, YYYY-MM-DD (default: today)",
    )
    p_check = commands.add_parser("check", help="verify a release")
    p_check.add_argument("--tag", help="tag being released, e.g. v0.2.0")
    p_notes = commands.add_parser("notes", help="print release notes")
    p_notes.add_argument("version", nargs="?", help="default: current")
    args = parser.parse_args(argv)

    try:
        if args.command == "bump":
            bump(args.version, args.date)
            print(f"Bumped to {args.version}; review with: git diff")
        elif args.command == "check":
            problems = check(tag=args.tag)
            for problem in problems:
                print(f"release check: {problem}", file=sys.stderr)
            if problems:
                return 1
            print("release check: OK")
        else:
            version = args.version or read_versions()["pyproject.toml"]
            print(release_notes(version))
    except (ReleaseError, ValueError) as error:
        print(f"release.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
