"""``docs/`` must be the build output of the sources committed beside it.

``docs/`` is the published GitHub Pages site, and it is a build output that lives in the
repository: the suite tests ``web/src``, and visitors run ``docs/``. ``npm run build`` writes
``docs/build-manifest.json``, a digest of every file the build reads and every file it writes, and
this module recomputes both sides from the working tree. Change a source and forget to build again,
edit a file in ``docs/`` by hand, or delete a file the site needs, and one of these fails.

What it does not do is run ``npm run build`` itself and compare the result. That would need Node in
the Python environment, and it would be a slower way of asking the same question, since a build from
unchanged inputs writes unchanged outputs. CI asks that one directly: its web job builds and fails
if ``docs/`` changed.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = DOCS / "build-manifest.json"
WEB = ROOT / "web"

#: floors, so no assertion here can pass by finding nothing: an empty manifest would otherwise
#: satisfy every comparison below
MINIMUM_INPUTS = 60
MINIMUM_OUTPUTS = 50

#: the files outside web/src and web/public that decide what the build writes: the page, the Vite
#: config (which also writes the page's markup into it), the TypeScript config, the package file and
#: the lockfile that pins every tool's version
CONFIG_INPUTS = ("web/index.html", "web/vite.config.ts", "web/tsconfig.json", "web/package.json",
                 "web/package-lock.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), (
        "docs/build-manifest.json is missing: run `npm run build` in web/, which writes it as part "
        "of the build"
    )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_manifest_records_a_real_build(manifest: dict) -> None:
    assert set(manifest) == {"note", "inputs", "outputs"}
    assert len(manifest["inputs"]) >= MINIMUM_INPUTS
    assert len(manifest["outputs"]) >= MINIMUM_OUTPUTS
    assert "docs/index.html" in manifest["outputs"]


def test_the_manifest_is_a_pure_function_of_the_tree(manifest: dict) -> None:
    """Sorted paths, digests only, and nothing that changes from one build to the next (a time, a
    host, a tool version), so that the same sources give the same bytes and CI can require that the
    build leaves docs/ as committed. The file is exactly the canonical form of what it holds."""
    for side in ("inputs", "outputs"):
        paths = list(manifest[side])
        assert paths == sorted(paths), f"the {side} are not in sorted order"
        for path, sha in manifest[side].items():
            assert "\\" not in path and not path.startswith(("/", "./")), path
            assert len(sha) == 64 and set(sha) <= set("0123456789abcdef"), f"{path}: {sha!r}"
    canonical = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    assert MANIFEST.read_text(encoding="utf-8") == canonical


def test_every_recorded_output_is_present_and_unchanged(manifest: dict) -> None:
    """Nothing in docs/ was edited by hand or lost after the build wrote it."""
    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.is_file(), f"{relative} is in the manifest but missing from the tree"
        assert _digest(path) == expected, f"{relative} does not match the build that wrote it; run `npm run build`"


def test_docs_holds_nothing_the_build_did_not_write(manifest: dict) -> None:
    """A stray file in docs/ is published to visitors, so it is a defect even when everything the
    manifest lists is intact."""
    on_disk = {path.relative_to(ROOT).as_posix() for path in DOCS.rglob("*") if path.is_file() and path != MANIFEST}
    assert on_disk == set(manifest["outputs"])


def test_every_build_input_still_hashes_to_what_was_built(manifest: dict) -> None:
    """The half that catches a stale site: a source edited after the last build. This is the
    assertion that ties docs/ to ``npm run build`` rather than to itself."""
    stale = []
    for relative, expected in manifest["inputs"].items():
        path = ROOT / relative
        if not path.is_file():
            stale.append(f"{relative} (deleted)")
        elif _digest(path) != expected:
            stale.append(relative)
    assert not stale, (
        "docs/ was built from other sources than the ones in the tree; run `npm run build` in web/. "
        f"Changed since the build: {sorted(stale)}"
    )


def test_no_file_the_build_reads_is_missing_from_the_inputs(manifest: dict) -> None:
    """A file added to web/src or web/public without a fresh build would otherwise be invisible to the
    check above: the manifest would not mention it, and every input it records would still match."""
    recorded = set(manifest["inputs"])
    expected = set(CONFIG_INPUTS)
    for base in (WEB / "src", WEB / "public"):
        expected |= {path.relative_to(ROOT).as_posix() for path in base.rglob("*")
                     if path.is_file() and not path.name.endswith(".test.ts")}
    assert not expected - recorded, (
        f"the last build never saw these files; build again: {sorted(expected - recorded)}"
    )


def test_the_inputs_leave_out_the_test_files_and_nothing_else(manifest: dict) -> None:
    """Vitest specs are outside the bundler's module graph, so they are not inputs; every other
    input is a file of web/src or web/public, or one of the config files."""
    recorded = set(manifest["inputs"])
    assert not [path for path in recorded if path.endswith(".test.ts")]
    assert any(path.startswith("web/src/") for path in recorded)
    assert any(path.startswith("web/public/") for path in recorded)
    outside = {path for path in recorded if not path.startswith(("web/src/", "web/public/"))}
    assert outside == set(CONFIG_INPUTS)


def test_every_file_of_web_public_ships_unchanged(manifest: dict) -> None:
    """Vite copies web/public into docs/ as it is, so each of its files is an output with the same
    digest at the same place under docs/."""
    for relative, sha in manifest["inputs"].items():
        if relative.startswith("web/public/"):
            shipped = "docs/" + relative.removeprefix("web/public/")
            assert manifest["outputs"].get(shipped) == sha, f"{relative} did not ship as {shipped}"


def test_the_published_index_references_only_files_that_shipped(manifest: dict) -> None:
    """The site's entry point, checked against what is in docs/. A link to a part of the page
    (``./#results``) or to the page with a query names the page itself, whose index.html must have
    shipped too."""
    index = (DOCS / "index.html").read_text(encoding="utf-8")
    outputs = set(manifest["outputs"])
    referenced = [fragment.split('"')[0] for marker in ('src="./', 'href="./')
                  for fragment in index.split(marker)[1:]]
    assert len(referenced) >= 3, "docs/index.html references almost no files"
    for target in referenced:
        path = target.split("#")[0].split("?")[0] or "index.html"
        assert f"docs/{path}" in outputs, f"index.html points at {target}, which is not in docs/"
