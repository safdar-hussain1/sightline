"""The documents the published page links to are published beside it.

The demo view and the Results view link the model card (``model-card.md``,
served as it is beside the page: the site has a ``.nojekyll``), and the Results
view links the data file every one of its numbers is read from. The links are
written into ``docs/index.html`` by the build, from the same markup the page
uses; the model card itself is a file in ``web/public/`` that the build copies.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "docs" / "index.html"


def test_the_page_links_the_model_card_and_the_data_file():
    html = PUBLISHED.read_text(encoding="utf-8")
    assert html.count('href="model-card.md"') >= 2, "the demo view and the Results view each link the model card"
    assert 'href="data/results.json"' in html


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason="the model card is not written yet",
)
@pytest.mark.parametrize("base", [ROOT / "web" / "public", ROOT / "docs"], ids=["template", "published"])
def test_the_model_card_is_published_beside_the_page(base):
    text = (base / "model-card.md").read_text(encoding="utf-8")
    assert text.lstrip().startswith("# "), "the model card opens with its title"
