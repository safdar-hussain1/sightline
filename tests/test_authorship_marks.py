"""The page says who built it, in its source and in what is published.

Four marks, each written into the page template or its entry so that every
build carries them, and each asserted again in ``docs/``. A mark present in
``web/`` but lost by the build would pass a source-only check while the
published page went unsigned, so both halves are checked:

- ``<meta name="author" content="Safdar Hussain">`` in the head;
- an HTML comment naming the author and his GitHub profile, first in the head;
- one ``console.info`` line from the app entry, which is the script the
  published ``index.html`` actually loads, and the entry's first statement;
- a visible footer credit, "Built by Safdar Hussain", linking to the profile.

The footer also carries the two notices the dataset's terms call for: the page
is for research and education only, never for controlling a vehicle, and the
footage is credited to its author with a link.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "index.html"
ENTRY = ROOT / "web" / "src" / "main.ts"
PUBLISHED = ROOT / "docs" / "index.html"

PAGES = [TEMPLATE, PUBLISHED]
PAGE_IDS = ["template", "published"]

PROFILE = "https://github.com/safdar-hussain1"
DATASET = "https://github.com/SullyChen/driving-datasets"
AUTHOR_META = re.compile(r'<meta\s+name="author"\s+content="Safdar Hussain"\s*/?>')
COMMENT = f"<!-- Sightline · built by Safdar Hussain · {PROFILE} -->"
CREDIT = f'<a href="{PROFILE}">Built by Safdar Hussain</a>'
CONSOLE_TEXT = f"Sightline — built by Safdar Hussain · {PROFILE}/sightline"
#: The bundler is free to re-quote a string literal (it emits backticks today),
#: so the quote character is matched, not assumed. The text inside is exact.
CONSOLE = re.compile(r"console\.info\(([\"'`])" + re.escape(CONSOLE_TEXT) + r"\1\)")


def _head(html: str) -> str:
    start, end = html.find("<head>"), html.find("</head>")
    assert 0 <= start < end, "no <head> element"
    return html[start:end]


def _footer(html: str) -> str:
    start, end = html.find("<footer"), html.find("</footer>")
    assert 0 <= start < end, "no <footer> element"
    return html[start:end]


def _published_entry() -> Path:
    """The module script the published page loads -- its built entry."""
    html = PUBLISHED.read_text(encoding="utf-8")
    scripts = re.findall(r'<script type="module"[^>]*\ssrc="\./(assets/[^"]+\.js)"', html)
    assert len(scripts) == 1, f"expected one entry script in index.html, found {scripts}"
    path = ROOT / "docs" / scripts[0]
    assert path.is_file(), f"index.html loads {scripts[0]}, which is not in docs/"
    return path


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_the_page_names_its_author_in_the_head(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    assert AUTHOR_META.search(_head(html)), f"{page.relative_to(ROOT)} has no author meta"


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_the_head_opens_with_the_authorship_comment(page: Path) -> None:
    head = _head(page.read_text(encoding="utf-8"))
    assert head[len("<head>"):].lstrip().startswith(COMMENT)


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_the_footer_credits_the_author_visibly(page: Path) -> None:
    footer = _footer(page.read_text(encoding="utf-8"))
    assert CREDIT in footer, f"{page.relative_to(ROOT)}: no credit in the footer"
    opening_tag = footer[: footer.index(">") + 1]
    assert "hidden" not in opening_tag, "the footer carrying the credit is hidden"


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_the_footer_carries_the_research_notice_and_the_dataset_credit(page: Path) -> None:
    footer = _footer(page.read_text(encoding="utf-8"))
    text = " ".join(re.sub(r"<[^>]+>", " ", footer).split())
    assert "research and education only" in text
    assert "vehicle" in text
    assert re.search(rf'<a href="{re.escape(DATASET)}">[^<]*Sully Chen', footer)
    assert "MIT licence" in text


def test_the_app_entry_signs_the_console_first_and_once() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    assert len(CONSOLE.findall(source)) == 1
    # The first statement after the imports: strip comments and imports, and
    # what is left must start with the signature.
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*import\b[^;]*;", "", code, flags=re.MULTILINE)
    assert CONSOLE.match(code.strip()), "console.info is not the entry's first statement"


def test_the_published_entry_bundle_signs_the_console_once() -> None:
    bundle = _published_entry().read_text(encoding="utf-8")
    assert len(CONSOLE.findall(bundle)) == 1, (
        "the script index.html loads does not carry the console line; "
        "run `npm run build` in web/"
    )
