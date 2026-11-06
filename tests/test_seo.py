"""The page's search and sharing metadata, in the template and in what is published.

The head block is authored in ``web/index.html`` and its files in ``web/public/``;
``npm run build`` carries both into ``docs/``, which is what GitHub Pages serves.
Every assertion runs on both sides, because a tag present in the template but
dropped or rewritten by the build would pass a source-only check while the live
page went without it.

What is asserted is the portfolio's standard for every project page: a title of
at most 60 characters, a description of 120 to 160, the author, the Search
Console token, a canonical URL, a favicon that resolves, Open Graph and Twitter
tags that agree with the title and description, a 1200x630 ``og-image.png`` under
500 KB (and the README's picture of the demo beside it), JSON-LD describing the
page as a ``WebApplication`` and the repository as ``SoftwareSourceCode`` with
the author written out on both, exactly one ``<h1>``
in the HTML the server sends, a sitemap whose ``<lastmod>`` is a literal date
rather than a build-time one, and no ``robots.txt``. Scripts may not rewrite the
title a crawler reads, except the in-page self-test, which only runs on request.
"""

import datetime as dt
import json
import math
import re
import struct
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://safdar-hussain1.github.io/sightline/"
REPO = "https://github.com/safdar-hussain1/sightline"
PROFILE = "https://github.com/safdar-hussain1"
LINKEDIN = "https://www.linkedin.com/in/safdar-hussain-a8a61b248"
SEARCH_CONSOLE_TOKEN = "0SIEfExLTSQj1qvnHWF5A5fY58KVl2lpIEnePP9CtI0"
TITLE = "Sightline — a steering model that shows where it looks"

#: (page, the directory its site-relative files resolve against)
SIDES = {
    "template": (ROOT / "web" / "index.html", ROOT / "web" / "public"),
    "published": (ROOT / "docs" / "index.html", ROOT / "docs"),
}


class _Page(HTMLParser):
    """Head metadata, JSON-LD blocks, inline scripts and the ``<h1>`` elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.title = None
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict] = []
        self.ld: list[str] = []
        self.inline_scripts: list[str] = []
        self.h1: list[dict] = []
        self._in_head = self._in_title = self._in_ld = self._in_script = False
        self._in_h1 = False
        self._buffer = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.lang = attributes.get("lang")
        elif tag == "head":
            self._in_head = True
        elif tag == "title" and self._in_head and self.title is None:
            self._in_title, self._buffer = True, ""
        elif tag == "meta" and self._in_head:
            key = attributes.get("name") or attributes.get("property")
            if key:
                self.meta.setdefault(key.lower(), []).append(
                    (attributes.get("content") or "").strip()
                )
        elif tag == "link" and self._in_head:
            self.links.append(attributes)
        elif tag == "script" and attributes.get("type") == "application/ld+json":
            self._in_ld, self._buffer = True, ""
        elif tag == "script" and "src" not in attributes:
            self._in_script, self._buffer = True, ""
        elif tag == "h1":
            self._in_h1, self._buffer = True, ""
            self.h1.append({"attributes": attributes, "text": ""})

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        elif tag == "title" and self._in_title:
            self._in_title, self.title = False, " ".join(self._buffer.split())
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.ld.append(self._buffer)
        elif tag == "script" and self._in_script:
            self._in_script = False
            self.inline_scripts.append(self._buffer)
        elif tag == "h1" and self._in_h1:
            self._in_h1 = False
            self.h1[-1]["text"] = " ".join(self._buffer.split())

    def handle_data(self, data):
        if self._in_title or self._in_ld or self._in_script or self._in_h1:
            self._buffer += data

    def one(self, key: str) -> str:
        values = self.meta.get(key, [])
        assert len(values) == 1, f"expected exactly one {key!r} tag, found {values}"
        return values[0]


def _parse(side: str) -> _Page:
    page, _ = SIDES[side]
    parser = _Page()
    parser.feed(page.read_text(encoding="utf-8"))
    return parser


def _resolve(side: str, href: str) -> Path:
    """A site-relative or absolute on-site URL, as a file on that side."""
    _, base = SIDES[side]
    relative = href.replace(SITE, "")
    relative = relative[2:] if relative.startswith("./") else relative.lstrip("/")
    return base / relative


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    return struct.unpack(">II", data[16:24])


sides = pytest.mark.parametrize("side", sorted(SIDES))


@sides
def test_title_and_description_fit_what_a_search_result_shows(side):
    page = _parse(side)
    assert page.lang == "en"
    assert page.title == TITLE
    assert len(page.title) <= 60, (len(page.title), page.title)
    description = page.one("description")
    assert 120 <= len(description) <= 160, (len(description), description)
    assert "width=device-width" in page.one("viewport")


@sides
def test_author_search_console_and_canonical(side):
    page = _parse(side)
    assert page.one("author") == "Safdar Hussain"
    assert page.one("google-site-verification") == SEARCH_CONSOLE_TOKEN
    canonical = [link.get("href") for link in page.links if link.get("rel") == "canonical"]
    assert canonical == [SITE]


@sides
def test_the_favicon_is_a_file_that_ships(side):
    page = _parse(side)
    icons = [link for link in page.links if "icon" in (link.get("rel") or "").split()]
    assert len(icons) == 1, icons
    href = icons[0]["href"]
    assert not href.startswith("data:"), "the favicon must be a file a crawler can fetch"
    target = _resolve(side, href)
    assert target.is_file(), f"{href} does not resolve to a file on the {side} side"
    assert target.read_text(encoding="utf-8").lstrip().startswith("<svg")


@sides
def test_open_graph_and_twitter_agree_with_the_page(side):
    page = _parse(side)
    title, description = page.title, page.one("description")
    assert page.one("og:type") == "website"
    assert page.one("og:site_name") == "Safdar Hussain"
    assert page.one("og:url") == SITE
    for key in ("og:title", "twitter:title"):
        assert page.one(key) == title, key
    for key in ("og:description", "twitter:description"):
        assert page.one(key) == description, key
    assert page.one("og:image") == SITE + "og-image.png"
    assert page.one("twitter:image") == page.one("og:image")
    assert (page.one("og:image:width"), page.one("og:image:height")) == ("1200", "630")
    assert len(page.one("og:image:alt")) > 40
    assert page.one("twitter:image:alt") == page.one("og:image:alt")
    assert page.one("twitter:card") == "summary_large_image"


@sides
def test_the_share_image_is_1200_by_630_and_under_500_kb(side):
    image = _resolve(side, _parse(side).one("og:image"))
    data = image.read_bytes()
    assert _png_size(data) == (1200, 630)
    assert len(data) < 500 * 1024, f"{len(data)} bytes"


@pytest.mark.parametrize("name", ["og-image.png", "readme-hero.png"])
def test_the_published_pictures_are_the_authored_ones(name):
    """docs/ must serve the files web/public/ holds, not stale copies of them."""
    assert (ROOT / "docs" / name).read_bytes() == (ROOT / "web" / "public" / name).read_bytes()


def test_the_readme_picture_is_a_png_of_at_most_300_kb():
    data = (ROOT / "docs" / "readme-hero.png").read_bytes()
    width, height = _png_size(data)
    assert width >= 900 and 1.6 <= width / height <= 2.0, (width, height)
    assert len(data) <= 300 * 1024, f"{len(data)} bytes"


#: What the share image shows, in words: the demo's first screen, which
#: scripts/capture.mjs pauses on a frame of the 2018 clip where the model's
#: heat is brightest on the centre line and the guardrail. A new picture
#: means new words here and in web/index.html.
SHARE_ALT = (
    "The Sightline demo paused on a bend to the left: a heat map over the road shows where the "
    "model looks, brightest on the centre line and the guardrail, and the model’s violet steering "
    "wheel with the driver’s pale one drawn over it."
)

#: The page's words for a ratio, as web/src/sections/words.ts writes them:
#: rounded to the nearest half, in words up to twelve.
_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve"]


def _times_in_words(ratio: float) -> str:
    half = math.floor(ratio * 2 + 0.5) / 2
    whole = math.floor(half)
    if whole > 12:
        return f"{math.floor(ratio + 0.5):,} times"
    return f"{_NUMBER_WORDS[whole]} times" if half == whole else f"{_NUMBER_WORDS[whole]} and a half times"


@sides
def test_the_description_says_what_the_paired_measurement_found(side):
    """The description is static text, so its "about three times" is held to
    results.json here: the paired measurement's ratio for SightlineNet, the
    network the demo runs, in the words the page's own sentence uses."""
    ratio = json.loads((ROOT / "web" / "public" / "data" / "results.json").read_text(encoding="utf-8"))[
        "paired_leakage"]["sightlinenet"]["ratio"]["mean"]
    assert ratio >= 1.25, "below 1.25 the page says 'a little', not 'about'"
    description = _parse(side).one("description")
    assert description == (
        "Watch a steering model read road footage in your browser and see where it looks. "
        f"Tested on shuffled frames, it looks about {_times_in_words(ratio)} better than it is."
    )


@sides
def test_the_share_image_alt_says_what_the_image_shows(side):
    page = _parse(side)
    assert page.one("og:image:alt") == SHARE_ALT
    assert page.one("twitter:image:alt") == SHARE_ALT


@sides
def test_the_head_text_carries_no_typed_figure(side):
    """Every figure on the page is read from the data; the head's words carry
    none in digits, and the one count they make in words is pinned above."""
    page = _parse(side)
    for key in ("description", "og:description", "twitter:description", "og:image:alt", "twitter:image:alt"):
        assert not re.search(r"\d", page.one(key)), key
    assert not re.search(r"\d", page.title)


@sides
def test_structured_data_names_the_application_the_source_and_the_author(side):
    page = _parse(side)
    assert len(page.ld) == 1, "expected one JSON-LD block"
    graph = json.loads(page.ld[0])["@graph"]
    nodes = {node["@type"]: node for node in graph}
    assert set(nodes) == {"WebApplication", "SoftwareSourceCode"}

    app = nodes["WebApplication"]
    assert app["name"] == "Sightline"
    assert app["url"] == SITE
    assert app["description"] == page.one("description")
    assert app["image"] == SITE + "og-image.png"
    assert app["applicationCategory"] == "ReferenceApplication"
    assert app["operatingSystem"] == "Any (runs in a web browser)"
    assert app["browserRequirements"] == "Requires JavaScript and WebAssembly"
    assert app["isAccessibleForFree"] is True
    assert app["offers"] == {"@type": "Offer", "price": "0", "priceCurrency": "USD"}

    source = nodes["SoftwareSourceCode"]
    assert source["name"] == "Sightline"
    assert source["codeRepository"] == REPO
    assert source["license"] == "https://opensource.org/licenses/MIT"
    assert source["programmingLanguage"] == ["Python", "TypeScript"]

    for node in graph:
        author = node["author"]
        assert author["@type"] == "Person"
        assert author["name"] == "Safdar Hussain"
        assert author["url"] == PROFILE
        assert set(author["sameAs"]) == {PROFILE, LINKEDIN}


def test_the_served_page_has_exactly_one_h1_naming_the_product():
    """A single-page app's heading must be in the HTML the server sends, not
    injected by a script: a crawler that does not run scripts reads only that."""
    headings = _parse("published").h1
    assert [h["text"] for h in headings] == ["Sightline"]
    attributes = headings[0]["attributes"]
    assert "hidden" not in attributes and "aria-hidden" not in attributes
    assert "visually-hidden" not in (attributes.get("class") or "")


def test_the_template_leaves_the_heading_to_the_build():
    """The template holds an empty masthead the build writes the heading into;
    a second, hand-written <h1> there would make two in the served page."""
    assert _parse("template").h1 == []
    template = SIDES["template"][0].read_text(encoding="utf-8")
    assert template.count('<header class="masthead" id="masthead"></header>') == 1


@pytest.mark.parametrize("base", [ROOT / "web" / "public", ROOT / "docs"], ids=["template", "published"])
def test_the_sitemap_lists_the_page_with_a_literal_lastmod(base):
    text = (base / "sitemap.xml").read_text(encoding="utf-8")
    entries = re.findall(r"<url>(.*?)</url>", text, re.DOTALL)
    locations = [re.search(r"<loc>([^<]*)</loc>", entry).group(1) for entry in entries]
    assert locations == [SITE]
    dates = re.findall(r"<lastmod>([^<]*)</lastmod>", entries[0])
    assert len(dates) == 1, dates
    # A real calendar date written into the file, so the build stays
    # byte-reproducible; the parse fails on anything else. A day of slack for
    # a machine whose clock is a time zone behind the one that wrote it.
    assert dt.date.fromisoformat(dates[0]) <= dt.date.today() + dt.timedelta(days=1)


@pytest.mark.parametrize("base", [ROOT / "web" / "public", ROOT / "docs"], ids=["template", "published"])
def test_there_is_no_per_project_robots_txt(base):
    """Crawlers read robots.txt only at the host root, never under /sightline/,
    so one here would do nothing while looking as if it did something."""
    assert not (base / "robots.txt").exists()


#: Reading or writing the title through the document, in any spelling a
#: script might use.
_TITLE_ACCESS = re.compile(r"document\s*(\.\s*title\b|\[\s*[\"'`]title[\"'`]\s*\])")


def test_no_script_rewrites_the_title_except_the_self_test():
    """Crawlers that run scripts index the title a script sets, so only the
    self-test -- which runs on ``?selftest=1`` alone -- may touch it. Test files
    never ship, so they are not counted."""
    sources = [
        path
        for path in (ROOT / "web" / "src").rglob("*.ts")
        if not path.name.endswith(".test.ts") and path.name != "selftest.ts"
    ]
    assert sources, "no sources found under web/src"
    offenders = [
        str(path.relative_to(ROOT))
        for path in sources
        if _TITLE_ACCESS.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@sides
def test_no_inline_script_touches_the_title(side):
    page = _parse(side)
    assert page.inline_scripts, "expected the inline theme script"
    for script in page.inline_scripts:
        assert not _TITLE_ACCESS.search(script)
