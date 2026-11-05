"""The documents the published page links to are published beside it.

The demo view and the Results view link the model card (``model-card.md``,
served as it is beside the page: the site has a ``.nojekyll``), and the Results
view links the data file every one of its numbers is read from, and each report
those numbers come from where it lives in the repository. The links are
written into ``docs/index.html`` by the build, from the same markup the page
uses; the model card itself is a file in ``web/public/`` that the build copies.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "docs" / "index.html"
REPOSITORY = "https://github.com/safdar-hussain1/sightline"


def test_the_page_links_the_model_card_and_the_data_file():
    html = PUBLISHED.read_text(encoding="utf-8")
    assert html.count('href="model-card.md"') >= 2, "the demo view and the Results view each link the model card"
    assert 'href="data/results.json"' in html


def test_every_link_into_the_repository_names_a_file_it_tracks():
    """A report renamed or moved would leave the page linking to nothing: each
    file link names a tracked file, each folder link a folder holding some."""
    html = PUBLISHED.read_text(encoding="utf-8")
    links = re.findall(r'href="' + re.escape(REPOSITORY) + r'/(blob|tree)/main/([^"#?]+)"', html)
    assert len(links) >= 6, links
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    tracked = set(listed.decode("utf-8").split("\0")) - {""}
    for kind, path in links:
        if kind == "blob":
            assert path in tracked, f"the page links {path}, which the repository does not track"
        else:
            assert any(name.startswith(path.rstrip("/") + "/") for name in tracked), f"the page links the folder {path}, which holds no tracked file"


def test_the_page_asks_for_the_demo_code_as_early_as_its_own():
    """The demo's code is a chunk of its own, imported once the page's first
    chunk runs; a modulepreload lets the browser fetch the two side by side."""
    html = PUBLISHED.read_text(encoding="utf-8")
    preloads = re.findall(r'<link rel="modulepreload"[^>]*href="\./(assets/cockpit-[\w-]+\.js)"', html)
    assert len(preloads) == 1, preloads
    assert (ROOT / "docs" / preloads[0]).is_file()
    assert html.index(preloads[0]) < html.index("</head>")


@pytest.mark.parametrize("base", [ROOT / "web" / "public", ROOT / "docs"], ids=["template", "published"])
def test_the_model_card_is_published_beside_the_page(base):
    text = (base / "model-card.md").read_text(encoding="utf-8")
    assert text.lstrip().startswith("# "), "the model card opens with its title"
