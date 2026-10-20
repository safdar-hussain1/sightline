"""Structural guards over the repository itself.

These tests police the repo's public surface: what is tracked, what is ignored, and what words may
never appear in tracked text, in tracked paths or in the commits of its history. Those words are held
here as sha256 digests of their normal form and never spelled, so this file names nothing it keeps
out of the repository and is scanned like every other file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[1]

#: formats that are never text, fonts among them
BINARY_SUFFIXES = {
    ".bin", ".gif", ".ico", ".jpeg", ".jpg", ".mp4", ".npy", ".npz", ".onnx", ".otf", ".pdf", ".png",
    ".ttf", ".wasm", ".webm", ".webp", ".woff", ".woff2",
}
#: tracked files without a binary suffix that may hold a NUL byte, each with the reason it may; a
#: NUL byte in any other of them fails the guards, as the text scans could not read the file
NUL_ALLOWED: dict[str, str] = {}
#: an inline base64 payload (``data:...;base64,...``), up to the first character base64 never uses
BASE64_PAYLOAD = re.compile(r";base64,[A-Za-z0-9+/]+=*")
#: the keys of a JSON tree that hold base64 payloads: a name that ends in ``_b64``, and a notebook
#: output's binary image types
PAYLOAD_KEY = re.compile(r".*_b64|image/(?:png|jpeg|gif|webp)")
#: a string in a JSON file's text as it stands, from its opening quote to its closing one
JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def _tracked_paths() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    return [path for path in out.stdout.split("\0") if path]


def _tracked_files() -> list[Path]:
    return [REPO / path for path in _tracked_paths()]


def not_authored(rel: str) -> bool:
    """Whether the tracked file at ``rel`` (from the repository root) is one nobody writes by hand,
    matched exactly: an npm lock file, a bundle the web build emits into ``docs/assets/``, or a file
    of an installed node module."""
    path = PurePosixPath(rel)
    return path.name == "package-lock.json" or rel.startswith("docs/assets/") or "node_modules" in path.parts


def _scanned_files() -> list[Path]:
    """Every tracked file the text scans read: all but the ones nobody writes by hand and the
    binary formats."""
    return [path for path in _tracked_files()
            if path.is_file() and not not_authored(path.relative_to(REPO).as_posix())
            and path.suffix.lower() not in BINARY_SUFFIXES]


def is_base64(text: str) -> bool:
    """Whether ``text`` is base64 and nothing else (line breaks aside): its letters, then at most two
    ``=``, in a whole number of four-character groups."""
    packed = text.replace("\n", "")
    return len(packed) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", packed) is not None


def _strings(tree, where: str, payload: bool = False, payloads: set | None = None):
    """Every key and every string of a parsed JSON tree, with where it sits, and every list of two strings
    or more joined into one as well: a notebook keeps its text one line to a string, so a phrase wrapped
    onto the next line is whole only there. Under a key that names a base64 payload (``PAYLOAD_KEY``) a
    string, or a list's joined text, that is base64 and nothing else is left out, a string so left out
    being added to ``payloads``; any other string there is read, and the walk goes into every mapping
    and list whatever its key."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield f"{where}.{key}", key
            yield from _strings(value, f"{where}.{key}", payload or PAYLOAD_KEY.fullmatch(key) is not None,
                                payloads)
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from _strings(value, f"{where}[{i}]", payload, payloads)
        if len(tree) > 1 and all(isinstance(value, str) for value in tree):
            joined = "".join(tree)
            if not (payload and is_base64(joined)):
                yield where, joined
    elif isinstance(tree, str):
        if payload and is_base64(tree):
            if payloads is not None:
                payloads.add(tree)
        else:
            yield where, tree


class DuplicateKey(ValueError):
    """A JSON object that holds one key twice: a parsed tree keeps only the last of its values."""


def _refuse_duplicates(pairs: list) -> dict:
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise DuplicateKey(f"the key {key!r} appears twice in one object")
        seen.add(key)
    return dict(pairs)


def read_json(text: str):
    """``text`` parsed as JSON; an object that holds a key twice is refused (``DuplicateKey``)."""
    return json.loads(text, object_pairs_hook=_refuse_duplicates)


def readable_text(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """What a reader of ``path`` can read, as ``(where, text)`` pieces, and why any of it cannot be
    read. The file's text as it stands comes first, its inline base64 payloads blanked. A notebook or a
    JSON file adds every key and string as it decodes, its escapes read, its metadata, every output
    and every traceback included, and each list of strings joined (a cell's lines, an output's), but
    no base64 payload; in its text as it stands, each string that holds one of those payloads is
    blanked too. A JSON file that holds a key twice in one object is
    read as text alone and named, as its parsed tree would keep one value only; a file with the JSON
    suffix that is not JSON (JSON with comments, say) is read as text alone."""
    rel = path.relative_to(REPO).as_posix()
    text = path.read_text(errors="replace")
    raw, pieces, problems = BASE64_PAYLOAD.sub(";base64,", text), [], []
    if path.suffix.lower() in (".ipynb", ".json"):
        try:
            tree = read_json(text)
        except DuplicateKey as exc:
            problems.append(f"{rel}: {exc}")
        except ValueError:  # not JSON as it stands: read as text alone
            pass
        else:
            payloads: set = set()
            pieces = list(_strings(tree, rel, payloads=payloads))
            if payloads:
                raw = JSON_STRING.sub(lambda m: '""' if json.loads(m.group()) in payloads else m.group(), raw)
    return [(rel, raw), *pieces], problems


def _line(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


#: files every scan must read, so that a scan of nothing cannot pass
SCANNED_FOR_SURE = {"LICENSE", ".gitignore", "pyproject.toml", "src/sightline/cli.py", "tests/test_guards.py",
                    "notebooks/02-results.ipynb", "reports/summary.json", "web/public/data/results.json"}


def test_the_scans_read_every_file_but_the_generated_ones_and_the_binary_formats():
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in _scanned_files()}


@pytest.mark.parametrize("rel, skipped", [
    ("package-lock.json", True), ("web/package-lock.json", True),
    ("notes/package-lock.json.md", False), ("web/package-lock.json.orig", False),
    ("docs/assets/index-3f2a.js", True), ("web/docs/assets/index.js", False), ("docs/assets-notes.md", False),
    ("web/node_modules/vite/index.js", True), ("web/my_node_modules.md", False), ("notes/node_modules.md", False),
])
def test_only_the_files_nobody_writes_are_left_out_and_exactly(rel, skipped):
    assert not_authored(rel) is skipped


def test_no_tracked_file_holds_a_nul_byte_unless_its_format_is_binary():
    allowed = {rel for rel, reason in NUL_ALLOWED.items() if reason.strip()}
    holding = [path.relative_to(REPO).as_posix() for path in _scanned_files() if b"\0" in path.read_bytes()]
    assert [rel for rel in holding if rel not in allowed] == [], "files the text scans cannot read"


# --- words the public text never holds ------------------------------------------------------------
#
# Every word and phrase refused here is held as the sha256 of its normal form, never spelled, so that
# this file does not name what it keeps out of the repository. A text's normal form is its tokens:
# each a run of ASCII letters and digits in lower case, or a section sign on its own, with a token of
# digits, alone or followed by one or two letters (an item number such as ``11a``), written ``#``; a
# phrase is one to three consecutive tokens joined by single spaces. The tokens of a phrase are parted
# in the text by whitespace alone (spaces and line breaks, and after a line break the mark that carries
# a comment or a quote on to the next line: ``#``, ``//``, ``*``, ``>``), so that a phrase wrapped onto
# the next line is still one, or by hyphens, underscores and slashes alone, which stand for spaces in a
# name or a path; any other mark parts them, so code such as ``self.word`` or ``count[word] += 1`` is
# never a phrase. An EXACT digest names a phrase that is refused whole. A STEMS digest names the
# beginning, four characters or more, of a token or of a two- or three-token phrase, reaching into its
# last token, so that one stem refuses every inflection of a word; a word whose stem would also begin
# innocent words is held exact instead. Two tokens joined by hyphens or underscores alone are also
# read as one word, for the stems. A word whose digest is in FLAGS may stand as a command-line flag,
# ``--word`` in lower case and a token of its own; anywhere else it is refused like the rest. Each
# digest carries the kind of rule it keeps, which a finding names. The scans read every tracked text
# file as it stands and as it decodes (``readable_text``), every tracked path, and the names and
# message of every commit.
#
# The probes further down plant made-up words whose digests they write out: they show that the
# machinery finds each form a rule covers and leaves the rest alone, not that the lists below name the
# right words, which no test can read here. The lists are pinned instead (``PINNED``), so that a digest
# dropped, added or changed fails; the readable list they are made from is kept privately, outside the
# repository, where a script checks that these digests are the digests of its entries.

EXACT = {
    "history": {
        "3bfc269594ef649228e9a74bab00f042efc91d5acc6fbee31a382e80d42388fe",
        "43f8412030b2f413a34a85a43ddcb3d57be95e801aba1340130ccb9045aed2b3",
        "fb04dcb6970e4c3d1873de51fd5a50d7bb46b3383113602665c350ec40b5f990",
    },
    "attribution": {
        "32e83e92d45d71f69dcf9d214688f0375542108631b45d344e5df2eb91c11566",
    },
    "process": {
        "05bbda5599ad82ce328ad50ec7364811ff9e41387e9e8988bed64952a5600a96",
        "1331d54359ce62ad154cd83fcdf682916ecfea3852cc4352e167c6a0cfd4174d",
        "2cc80c031e19249bd930b952b27967f49329d8ca25c12bf31474c4eb6c232e21",
        "32503795554682dd08640f18b380b9e9476a96efd5cd743f8bcf728fac232d1d",
        "5249804f2a9a1499425c32a6ffe94c43984cdcf7cd2a7165344e8bb765aa4fba",
        "53cfadbb46a7703e2d3606085b994cdd6a8a754a39112da968143687ad57bd00",
        "9d6e63ad6b8e50d3b09932c2f2605cb45b125968f317361d3ecbb34dd5ba69ef",
        "a6a226508be3fddffce96a315b70b6b91e46aae8aae7d1b273167ba61c29cfdc",
        "d30275523e5516ce8bc110627e40622229593f53e3c2096c89fb47d71e9cd36e",
        "e3238faccb7585ba292c430afadbc021ee7b1828551378ee39d3be416655f1fd",
        "ec3f4a4cd18a80ef0278399034b7262590a4c80ed67cadddb91c6e8769e387b9",
    },
}
STEMS = {
    "history": {
        "5cdb778cb86d9a0aef849d3f9fbe121982c486e69b29abac9d10a4811da8770b",
        "76cfec1c272a6aa7f2e559a0d72bf106db3e74ab77f8b70c32955769422edf0b",
        "776af9401679287d8d332e26e99062919d58a135f4ef5ba85d4ec981b50f58ae",
        "86dc882c2e0d0e59a954e96a6464186fcc6f64c69406a52787b6fb19abd5d4f7",
    },
    "attribution": {
        "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c",
        "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef",
        "8aa1e92a0f8c8e6100a68def958736e6dda170ce2092fa2252a93a4d860ca2b3",
        "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
        "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
    },
    "process": {
        "132a0519893f35d86aafbf5cd9863a34867ef0c6595f32fb2e1d8c1e1951fdc5",
        "530338f42218dbd3a9cc12ed2d6a91673cf7cab5b0568325956737b2c576789c",
        "94a168d2da574b00ec6c787377278465271cf2b97e68af076fc3eee87eba8664",
        "a85785ab0c6b89ed491e8108e32545e37111a4a4c77c93a7bf9910361856b79b",
        "b8fa9efbc8c8f043afca1b60f8f4c5245c54b5ff5bfb0603a71071f66c1ef295",
        "c8358edd8e151c797bc6ff328aea24d86ff1379d171abd778e47ee77b76cd6bf",
        "eef0f7354969b40c58ba6f3c27314e007446b46b8c1328b8d0df9baa32b443ca",
        "f7024509f545785ce604950d9073a158db75237837b279cdc8a0e306b84d3353",
    },
}
FLAGS = {
    "7fef9479c8bc5a3c86891cb82969cbf2ffc73c7350366c82286345d8292eebf5",
}
#: each list of digests pinned, kind by kind: how many it holds, and the sha256 of its digests sorted
#: and joined by line breaks (``pins``), which names no word
PINNED = {
    "exact history": (3, "63f248469bd91edbd0a804f952a0fdffa9e935198d53aa1d1b01046f3c54bc35"),
    "exact attribution": (1, "b742a79fb229d19f5c296cf13106eb936edcbe3edcbe1b3f1debb98ab15902f6"),
    "exact process": (11, "8d3b46a26d1836622f841bf01d2f993ccc51d206515549e80fff85af8b2784f0"),
    "stems history": (4, "116db9156e2f5e1ca7cb7ebc7002f5d29741de6d185bb61c14db8802e2b1249b"),
    "stems attribution": (5, "3fd7a305f084ce166ada1e7feaea2a2cebe3cf4c84bf6ce1a4097a98f4e3ae43"),
    "stems process": (8, "0a0efc2c0da768b9ff216e43c1123c786bdd342f1bb6a75e1c1cbf763071e846"),
    "flags": (1, "cd61ad43b90e0e7d414497241533e4e45d901b323e88d402bbe565b8583e8c3d"),
}

#: a token of the normal form: a run of ASCII letters and digits, or a section sign
TOKEN = re.compile(r"[A-Za-z0-9]+|§")
#: a token written ``#``: digits, alone or followed by one or two letters (``12``, ``11a``, ``3rd``)
NUMBER = re.compile(r"[0-9]+[A-Za-z]{0,2}")
#: what may part two tokens of a phrase: whitespace, and after each line break the mark that carries a
#: comment or a quote on to that line; nothing at all where a section sign touches a token. Each run of
#: whitespace has one way to match, so a long gap that fails does so at once.
SPACE = re.compile(r"[^\S\n]*(?:\n[^\S\n]*(?:(?:#+|/{2,}|\*+|>+)[^\S\n]*)?)*")
#: what else may part them: hyphens, underscores and slashes alone, which stand for spaces in a name
#: or a path
LINK = re.compile(r"[-_/]+")
#: what may stand between two tokens that are read as one word for the stems
JOINER = re.compile(r"[-_]+")
#: a command-line flag: two hyphens, then a lower-case token of its own
FLAG = re.compile(r"(?<![\w-])--([a-z0-9]+)(?![\w-])")
#: the length of the shortest beginning a stem digest is held against
SHORTEST_STEM = 4


def pins(exact: dict, stems: dict, flags) -> dict:
    """Each list's pin (``PINNED``), kind by kind: how many digests it holds, and the sha256 of its
    digests sorted and joined by line breaks."""
    lists = {**{f"exact {kind}": digests for kind, digests in exact.items()},
             **{f"stems {kind}": digests for kind, digests in stems.items()}, "flags": flags}
    return {name: (len(digests), hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest())
            for name, digests in lists.items()}


def _normal(token: str) -> str:
    return "#" if NUMBER.fullmatch(token) else token.lower()


def normal_form(text: str) -> str:
    """The tokens of ``text`` in normal form, joined by single spaces."""
    return " ".join(_normal(token) for token in TOKEN.findall(text))


def digest(phrase: str) -> str:
    """sha256 of a phrase in normal form, in hex."""
    return hashlib.sha256(phrase.encode()).hexdigest()


class Banned:
    """Digests of refused phrases (``exact``) and beginnings (``stems``), each by kind, and of the words
    allowed as a command-line flag (``flags``). Each answer is kept, as a text repeats its phrases."""

    def __init__(self, exact: dict, stems: dict, flags=()):
        self.exact = {sha: kind for kind, digests in exact.items() for sha in digests}
        self.stems = {sha: kind for kind, digests in stems.items() for sha in digests}
        self.flags = frozenset(flags)
        self._answers: dict = {}

    def _stem(self, phrase: str, shortest: int) -> str | None:
        """The kind of a stem that begins ``phrase`` and is ``shortest`` characters long at least."""
        for k in range(max(SHORTEST_STEM, shortest), len(phrase) + 1):
            kind = self.stems.get(digest(phrase[:k]))
            if kind is not None:
                return kind
        return None

    def _kept(self, key: tuple, answer):
        if key not in self._answers:
            self._answers[key] = answer()
        return self._answers[key]

    def token(self, a: str) -> str | None:
        """The kind of the refused word ``a`` is, or of a stem that begins it."""
        return self._kept(("token", a), lambda: self.exact.get(digest(a)) or self._stem(a, 0))

    def pair(self, a: str, b: str) -> str | None:
        """The kind of the refused phrase ``a b`` is, or of a stem that begins it and reaches into ``b``."""
        phrase = f"{a} {b}"
        return self._kept(("pair", a, b), lambda: self.exact.get(digest(phrase)) or self._stem(phrase, len(a) + 2))

    def triple(self, a: str, b: str, c: str) -> str | None:
        """The kind of the refused phrase ``a b c`` is, or of a stem that begins it and reaches into ``c``."""
        phrase = f"{a} {b} {c}"
        return self._kept(("triple", a, b, c),
                          lambda: self.exact.get(digest(phrase)) or self._stem(phrase, len(a) + len(b) + 3))

    def joined(self, a: str, b: str) -> str | None:
        """The kind of a stem that begins the word ``a`` and ``b`` make together and reaches into ``b``."""
        return self._kept(("joined", a, b), lambda: self._stem(a + b, len(a) + 1))


BANNED = Banned(EXACT, STEMS, FLAGS)


def in_one_phrase(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]``, all that stands between two tokens, lets them be read as words of
    one phrase: whitespace (``SPACE``), or hyphens, underscores and slashes alone (``LINK``)."""
    return SPACE.fullmatch(text, start, end) is not None or LINK.fullmatch(text, start, end) is not None


def banned_words(text: str, banned: Banned = BANNED) -> list[tuple[int, str, str]]:
    """Each refused word or phrase in ``text``: its offset, the text it spans and its kind. A word
    ``banned`` allows as a flag is left out where it stands as one."""
    tokens = [(m.start(), m.end(), _normal(m.group())) for m in TOKEN.finditer(text)]
    flags = {m.span(1) for m in FLAG.finditer(text) if digest(_normal(m.group(1))) in banned.flags}
    found = []
    for i, (start, end, a) in enumerate(tokens):
        kind = banned.token(a)
        if kind and (start, end) not in flags:
            found.append((start, end, kind))
        if i + 1 < len(tokens):
            b_start, b_end, b = tokens[i + 1]
            phrase = in_one_phrase(text, end, b_start)
            kind = (banned.pair(a, b) if phrase else None) \
                or (banned.joined(a, b) if JOINER.fullmatch(text, end, b_start) else None)
            if kind:
                found.append((start, b_end, kind))
            if phrase and i + 2 < len(tokens):
                c_start, c_end, c = tokens[i + 2]
                kind = banned.triple(a, b, c) if in_one_phrase(text, b_end, c_start) else None
                if kind:
                    found.append((start, c_end, kind))
    return [(start, text[start:end], kind) for start, end, kind in found]


def scan(paths: list[Path], banned: Banned = BANNED) -> list[str]:
    """Every refused word in what a reader of each of ``paths`` can read, as ``<where>:<line>: <text>
    (<kind>)``, and every file whose parsed text could hide one."""
    offenders = []
    for path in paths:
        pieces, problems = readable_text(path)
        offenders += problems
        for where, text in pieces:
            offenders += [f"{where}:{_line(text, start)}: {word!r} ({kind})"
                          for start, word, kind in banned_words(text, banned)]
    return offenders


def scan_paths(paths: list[str], banned: Banned = BANNED) -> list[str]:
    """Every refused word in each of ``paths``, read as text in any case."""
    return [f"{path} (path): {word!r} ({kind})" for path in paths for _, word, kind in banned_words(path, banned)]


def test_no_banned_words_in_tracked_files():
    files = _scanned_files()
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in files}
    offenders = scan(files)
    assert offenders == [], f"banned words in tracked files: {offenders}"


def test_no_tracked_path_holds_a_banned_word():
    """Every tracked path, binary files and the ones nobody writes by hand included: the scan that keeps
    the private notes folder and working-notes file out of the repository."""
    paths = _tracked_paths()
    assert SCANNED_FOR_SURE <= set(paths)
    offenders = scan_paths(paths)
    assert offenders == [], f"banned words in tracked paths: {offenders}"


def test_the_lists_of_digests_are_the_pinned_ones():
    """A digest dropped, added or changed fails here, and so does a stem made to need more characters;
    that the lists name the right words is checked outside the repository, against the readable list."""
    every = [*FLAGS, *(sha for table in (EXACT, STEMS) for digests in table.values() for sha in digests)]
    assert [sha for sha in every if not re.fullmatch(r"[0-9a-f]{64}", sha)] == [], "not a sha256 in hex"
    assert pins(EXACT, STEMS, FLAGS) == PINNED
    assert SHORTEST_STEM == 4


# --- the probes: made-up words, with the digests written out ------------------------------------------

PROBE_EXACT = {
    "zebrafish": "d075da24a84b9f608363799992193ca14c055b24f2e41b7bbc18d8c18d1568ff",
    "purple heron": "7491777e1f876fa84212fdf39e8c5e0ede079b67b21a695d8c3ccbf56b8d83c1",
    "slow amber tide": "6ff0aad45023d3076e4e34ca71e862a62cb269883eda412b6cceb5ee33476afb",
    "lantern #": "3c13cec09bdbd44f7eb77192d0e2ef44c0f94a447c29f393712a69aca05cab52",
    "lantern §": "7007ef96f2a4e1761eb71f24216764a9dee6199a14f6a3a2e6f64c878e554392",
}
PROBE_STEMS = {
    "marigold": "a5055bbfef782d417823b27963363a96ccd48a9df697398805c6eaa8d30eddd7",
    "velvet moth": "51ec23d7f3e54a326de9ebb83479e9f0c592cfb903e6511a49920af2ce7036ff",
    "quiet # brook": "276651eef78d5837af9413eb68883c1bba20d896580777c1dca3284a58886d22",
    "fern": "f0a5da17e0ffa3a21c2b07c964d465799eb0deb62905ccb1b271ce7716614746",  # as short as a stem may be
    "basalt": "8369762ae90ec0b2252931df46d51aa11a4c0727669c41c7157e6da8a88430f3",  # six letters
    "owl": "10f7127b980b0b06ef157749f8a02cca1c606d22642729dcb2d24dbbf2bf30df",  # shorter than any stem held
}
PROBE = Banned({"probe": set(PROBE_EXACT.values())}, {"probe": set(PROBE_STEMS.values())},
               flags={PROBE_EXACT["zebrafish"]})
#: an image's bytes in base64 that hold a probe word by chance, between a plus and a slash
PAYLOAD = "AAAA+zebrafish/BBBBB"


def test_a_probe_word_is_held_as_the_sha256_of_its_normal_form():
    probes = {**PROBE_EXACT, **PROBE_STEMS}
    assert {word: digest(normal_form(word.replace("#", "7"))) for word in probes} == probes
    assert normal_form("Lantern 2026!") == "lantern #"
    assert normal_form("PURPLE\n  heron") == normal_form("purple_heron") == normal_form("purple-heron") == "purple heron"
    assert normal_form("§12 and §§") == "§ # and § §"
    assert normal_form("quill_x9.1, drive2018") == "quill x9 # drive2018"
    # digits followed by one or two letters are an item number, written as digits are
    assert normal_form("lantern 11a, 3RD, 64px, 12abc and x9") == "lantern # # # 12abc and x9"


@pytest.mark.parametrize("text, found", [
    # one word, exact: in any case, and only whole
    ("a zebrafish swims", ["zebrafish"]),
    ("ZebraFish, ZEBRAFISH.", ["ZebraFish", "ZEBRAFISH"]),
    ("zebrafishes, xzebrafish, zebrafish2 and zebra fish", []),
    # a stem: every word it begins, and a word split by hyphens or underscores
    ("marigold, Marigolds and marigolded", ["marigold", "Marigolds", "marigolded"]),
    ("mari-golds and mari__gold", ["mari-golds", "mari__gold"]),
    # a whole stem joined to the next word is found in its own word, and once
    ("marigold-like and marigold_ish", ["marigold", "marigold"]),
    # a stem only begins a token, and a space or a stop does not join two
    ("marigol, xmarigold, mari golds and mari.gold", []),
    # a stem as short as a stem may be, and one of six letters: each finds itself and the words it begins
    ("fern, Ferns, basalt and BASALTIC", ["fern", "Ferns", "basalt", "BASALTIC"]),
    ("fer, xfern, basal and bas alt", []),
    # a stem shorter than four characters finds nothing
    ("an owl and owls", []),
    # a one-word stem ends a phrase at its own word
    ("a marigold grows", ["marigold"]),
    # two words, only whole, parted by whitespace (line breaks included, and after a line break the mark
    # that carries a comment or a quote on) or by hyphens, underscores and slashes, as in a name or a path
    ("a purple heron, a PURPLE\n  HERON, purple-heron, purple_heron and purple/heron",
     ["purple heron", "PURPLE\n  HERON", "purple-heron", "purple_heron", "purple/heron"]),
    ("purple\n    # heron, purple\r\n// heron, purple\n * heron and purple\n> heron",
     ["purple\n    # heron", "purple\r\n// heron", "purple\n * heron", "purple\n> heron"]),
    ("purple herons and purpleheron", []),
    # any other mark parts them, so code such as ``this.word`` or ``count[word] += 1`` is left alone
    ("purple.heron, purple(heron), purple[heron], purple=heron, purple+heron, purple, heron, purple # heron "
     "and purple - heron", []),
    ("if (this.purple) heron(); count[lantern] += 1; lantern(7)", []),
    # a section sign may touch the word before it
    ("lantern §4, Lantern§, lantern\n§ and lantern-§", ["lantern §", "Lantern§", "lantern\n§", "lantern-§"]),
    ("lantern.§ and lantern(§)", []),
    # a two-word stem: its last word inflected
    ("velvet moths, a Velvet Moth flies and velvet_moths", ["velvet moths", "Velvet Moth", "velvet_moths"]),
    ("velvet and moth, velvet.moths", []),
    # three words
    ("slow amber tide; slow\namber\n  tide; slow_amber_tide and slow-amber/tide",
     ["slow amber tide", "slow\namber\n  tide", "slow_amber_tide", "slow-amber/tide"]),
    ("Slow, amber; tide, slow, amber tide, slow amber.tide and slow amber", []),
    # a three-word stem: its last word inflected, once the stem reaches into it
    ("quiet 4 brooks, Quiet 11b Brook, quiet 2\nbrooklets and quiet-4-brooks",
     ["quiet 4 brooks", "Quiet 11b Brook", "quiet 2\nbrooklets", "quiet-4-brooks"]),
    ("quiet 4, quiet 4 broo and quiet.4 brooks", []),
    # digits, alone or followed by one or two letters (an item number), are one sign
    ("lantern 7, Lantern 2026, lantern 11a:, LANTERN 3RD and lantern-12.5",
     ["lantern 7", "Lantern 2026", "lantern 11a", "LANTERN 3RD", "lantern-12"]),
    ("lantern7, lantern 7abc, lantern seven and lantern(12)", []),
    # a word allowed as a flag: as a lower-case token of its own after two hyphens, and nowhere else
    ("pip install --zebrafish pip", []),
    ("`--zebrafish`", []),
    ("--zebrafish-strategy eager", ["zebrafish"]),
    ("x--zebrafish", ["zebrafish"]),
    ("--Zebrafish", ["Zebrafish"]),
])
def test_the_rule_for_banned_words(text, found):
    assert [word for _, word, _ in banned_words(text, PROBE)] == found


def _planted(tmp_path, monkeypatch, files: dict[str, str]) -> list[str]:
    """What the scan finds, with the probe digests, in ``files`` (name to text) planted in a stand-in
    repository."""
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return scan([tmp_path / name for name in files], PROBE)


def test_the_scan_reads_a_json_files_text_and_its_decoded_strings(tmp_path, monkeypatch):
    found = _planted(tmp_path, monkeypatch, {
        "escaped.json": '{"note": "a \\u007aebrafish"}',  # an escape hides it in the text as it stands
        "plain.json": '{"note": "a zebrafish"}',
        "key.json": '{"purple heron": 1}',
        "comments.json": '// a zebrafish\n{"a": 1}',  # not JSON: read as text alone
        "notes.md": "the\nslow amber tide",
    })
    assert found == [
        "escaped.json.note:1: 'zebrafish' (probe)",
        "plain.json:1: 'zebrafish' (probe)", "plain.json.note:1: 'zebrafish' (probe)",
        "key.json:1: 'purple heron' (probe)", "key.json.purple heron:1: 'purple heron' (probe)",
        "comments.json:1: 'zebrafish' (probe)",
        "notes.md:2: 'slow amber tide' (probe)",
    ]


def test_a_json_file_that_holds_a_key_twice_is_refused(tmp_path, monkeypatch):
    text = '{"note": "a \\u007aebrafish", "note": "a calm pond"}'
    # parsed the usual way, the first value is gone; in the text as it stands, an escape hides it
    assert json.loads(text) == {"note": "a calm pond"} and banned_words(text, PROBE) == []
    assert _planted(tmp_path, monkeypatch, {"twice.json": text}) == ["twice.json: the key 'note' appears twice in one object"]
    with pytest.raises(DuplicateKey):
        read_json('{"a": {"b": 1, "b": 2}}')


def test_the_scan_reads_everything_a_notebook_holds_but_its_images(tmp_path, monkeypatch):
    # a notebook keeps its text one line to a string: a phrase wrapped onto the next line, in a cell or in
    # an output, is whole only where the lines are joined
    notebook = {"metadata": {"title": "the purple heron notes"}, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["the slow amber\n", "tide, wrapped"]},
        {"cell_type": "code", "metadata": {"tags": ["marigolds"]},
         "source": ["print(1)\n", "# lantern 7, a purple\n", "# heron\n"],
         "outputs": [
             {"output_type": "stream", "name": "stdout", "text": ["a velvet\n", "moth\n"]},
             {"output_type": "display_data", "data": {"image/png": PAYLOAD, "text/plain": ["the velvet moths"],
                                                      "application/json": {"note": "slow amber tide"}}},
             {"output_type": "error", "ename": "ValueError", "evalue": "ZEBRAFISH",
              "traceback": ["line one", "a zebrafish"]},
         ]}]}
    # the error's value is written with an escape, as a notebook's text may hold it
    text = json.dumps(notebook).replace("ZEBRAFISH", "\\u005aEBRAFISH")
    found = _planted(tmp_path, monkeypatch, {"n.ipynb": text})
    code, outputs = "n.ipynb.cells[1]", "n.ipynb.cells[1].outputs"
    # decoded: every string, and every list of strings joined
    assert sorted(entry for entry in found if not entry.startswith("n.ipynb:")) == sorted([
        "n.ipynb.metadata.title:1: 'purple heron' (probe)",
        "n.ipynb.cells[0].source:1: 'slow amber\\ntide' (probe)",
        f"{code}.metadata.tags[0]:1: 'marigolds' (probe)",
        f"{code}.source[1]:1: 'lantern 7' (probe)",
        f"{code}.source:2: 'lantern 7' (probe)", f"{code}.source:2: 'purple\\n# heron' (probe)",
        f"{outputs}[0].text:1: 'velvet\\nmoth' (probe)",
        f"{outputs}[1].data.text/plain[0]:1: 'velvet moths' (probe)",
        f"{outputs}[1].data.application/json.note:1: 'slow amber tide' (probe)",
        f"{outputs}[2].evalue:1: 'ZEBRAFISH' (probe)",
        f"{outputs}[2].traceback[1]:1: 'zebrafish' (probe)", f"{outputs}[2].traceback:1: 'zebrafish' (probe)",
    ])
    # the text as it stands has neither the wrapped phrases nor the escaped word, and neither has the
    # image's bytes
    assert sorted(entry for entry in found if entry.startswith("n.ipynb:")) == sorted(
        f"n.ipynb:1: {word!r} (probe)"
        for word in ["purple heron", "marigolds", "lantern 7", "velvet moths", "slow amber tide", "zebrafish"])


def test_the_scan_leaves_out_only_strings_that_are_base64_under_a_payload_key(tmp_path, monkeypatch):
    cases = {
        "skipped.json": {"saliency_f32_b64": {"m": PAYLOAD}, "frames_b64": PAYLOAD},
        "not-base64.json": {"saliency_f32_b64": "a zebrafish, not a payload"},
        "nested.json": {"saliency_f32_b64": {"note": ["a zebrafish"]}},
        "plain-key.json": {"title": PAYLOAD},
        # a list of strings, joined, under the same rule: the payload's bytes cut in two
        "cut-payload.json": {"frames_b64": ["AAAA+zebra", "fish/BBBBB"]},
        "cut-plain-key.json": {"title": ["AAAA+zebra", "fish/BBBBB"]},
    }
    found = _planted(tmp_path, monkeypatch, {name: json.dumps(tree) for name, tree in cases.items()})
    assert {name: [entry.split(":", 1)[0] for entry in found if entry.startswith(name)] for name in cases} == {
        "skipped.json": [],
        "not-base64.json": ["not-base64.json", "not-base64.json.saliency_f32_b64"],
        "nested.json": ["nested.json", "nested.json.saliency_f32_b64.note[0]"],
        "plain-key.json": ["plain-key.json", "plain-key.json.title"],
        "cut-payload.json": [],
        "cut-plain-key.json": ["cut-plain-key.json.title"],
    }
    assert is_base64(PAYLOAD) and is_base64("QUJD\nREVG") and not is_base64("zebrafish") \
        and not is_base64("QUJDRA=") and not is_base64("QU JD")


def test_the_scan_blanks_an_inline_payload_and_reads_what_follows_it(tmp_path, monkeypatch):
    found = _planted(tmp_path, monkeypatch, {
        "p.html": f'<img src="data:image/png;base64,{PAYLOAD}">a zebrafish',
        "q.md": f"data:text/plain;base64,{PAYLOAD}\nthe purple heron page",
        "r.css": f"a {{ background: url(data:image/png;base64,{PAYLOAD}) }} /* the zebra */",
    })
    assert found == ["p.html:1: 'zebrafish' (probe)", "q.md:2: 'purple heron' (probe)"]


@pytest.mark.parametrize("path, found", [
    ("notes/Zebrafish-Notes.md", ["Zebrafish"]),
    ("web/MARIGOLDS/index.js", ["MARIGOLDS"]),
    ("web/mari-golds_11a.css", ["mari-golds"]),
    ("data/lantern-12.txt", ["lantern-12"]),
    ("data/lantern/12b.txt", ["lantern/12b"]),
    ("docs/purple_heron.png", ["purple_heron"]),
    ("docs/purple/heron.png", ["purple/heron"]),
    ("docs/purple.heron.png", []),
    ("src/sightline/cli.py", []),
])
def test_a_tracked_path_is_scanned_like_text_in_any_case(path, found):
    assert scan_paths([path], PROBE) == [f"{path} (path): {word!r} (probe)" for word in found]


def test_no_absolute_user_paths_in_tracked_files():
    pattern = re.compile(r"/Use" + r"rs/[A-Za-z]")
    files = _scanned_files()
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in files}
    offenders = [where for path in files for where, text in readable_text(path)[0] if pattern.search(text)]
    assert offenders == [], f"absolute user paths leaked into: {offenders}"


def test_no_process_documents_tracked():
    banned_names = re.compile(
        r"(spec|plan|handbook|session|ledger|numbers|todo|prompt)[-_a-z0-9]*\.(md|txt|pdf)$",
        re.IGNORECASE,
    )
    allowed = {"README.md"}
    offenders = [
        str(p.relative_to(REPO))
        for p in _tracked_files()
        if p.name not in allowed and banned_names.search(p.name)
    ]
    assert offenders == [], f"process documents tracked: {offenders}"


@pytest.mark.parametrize(
    "path",
    [
        "data/drive2018/data.txt",
        "data/drive2017/0.jpg",
        "runs/cache/drive2018/meta.json",
        "runs/anything/best.pt",
        "weights.npz",
        "model.ckpt.index",
        "web/.public-k2x9q1/data/results.json",
        "reports/.export.json.4242-1a2b3c4d.new",
        "reports/.export.json.4242-1a2b3c4d.previous",
        "demo.mp4",  # what `sightline demo` writes by default
        "demo-c2018.mp4",
        "notebooks/.ipynb_checkpoints/02-results-checkpoint.ipynb",
    ],
)
def test_private_paths_are_ignored(path: str):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", path], cwd=REPO
    )
    assert result.returncode == 0, f"{path} is NOT git-ignored"


def _matched_by_an_ignore_rule(path: str) -> bool:
    """Whether the ignore rules alone match ``path``: ``--no-index`` asks the rules even about
    a tracked file, which ``git check-ignore`` otherwise leaves out."""
    return subprocess.run(["git", "check-ignore", "--no-index", "--quiet", path], cwd=REPO).returncode == 0


def test_the_demo_clip_videos_are_tracked_and_no_ignore_rule_matches_them():
    tracked = {p.relative_to(REPO).as_posix() for p in _tracked_files()}
    for clip in ("c2017", "c2018"):
        path = f"web/public/data/clips/{clip}/clip.mp4"
        assert path in tracked, f"{path} is not tracked"
        assert not _matched_by_an_ignore_rule(path), f"an ignore rule matches {path}"
    # the rule for the demo command's output is anchored to the repository root
    assert _matched_by_an_ignore_rule("demo.mp4")
    assert not _matched_by_an_ignore_rule("web/public/demo.mp4"), "the demo.mp4 rule is not root-only"
    assert not _matched_by_an_ignore_rule("notebooks/demo.mp4"), "the demo.mp4 rule is not root-only"


def test_the_export_record_is_tracked_and_no_ignore_rule_matches_it():
    assert "reports/export.json" in {p.relative_to(REPO).as_posix() for p in _tracked_files()}
    assert not _matched_by_an_ignore_rule("reports/export.json")


def test_no_dataset_or_weight_files_tracked():
    heavy = [
        str(p.relative_to(REPO))
        for p in _tracked_files()
        if p.suffix in {".pt", ".pth", ".npz", ".ckpt"} or ".ckpt." in p.name
    ]
    assert heavy == [], f"weight/dataset files tracked: {heavy}"


# --- the history ----------------------------------------------------------------------------------

#: who authors and commits every commit
AUTHOR = ("Safdar Hussain", "safdar.eryx@gmail.com")
#: who commits a merge GitHub makes, of a pull request among them; GitHub sets its author
GITHUB = ("GitHub", "noreply@github.com")


@dataclass(frozen=True)
class Commit:
    sha: str
    parents: tuple[str, ...]
    author: tuple[str, str]
    committer: tuple[str, str]
    message: str


def history(repo: Path) -> list[Commit]:
    """Every commit of every branch of ``repo``, as ``git log --all`` gives it."""
    out = subprocess.run(["git", "log", "--all", "-z", "--format=%H%x1f%P%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    commits = []
    for record in out.split("\0"):
        if record.strip():
            sha, parents, author, author_email, committer, committer_email, message = record.split("\x1f", 6)
            commits.append(Commit(sha.strip(), tuple(parents.split()), (author, author_email),
                                  (committer, committer_email), message))
    return commits


def is_shallow(repo: Path) -> bool:
    """Whether ``repo`` is a shallow clone: it holds part of the history, and its oldest commits show
    no parents."""
    out = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo, capture_output=True,
                         text=True, check=True).stdout
    return out.strip() == "true"


def authorship_problems(commits: list[Commit], banned: Banned = BANNED) -> list[str]:
    """What is wrong with ``commits``: a refused word in a commit's names or message, and an author or
    committer other than ``AUTHOR``. A merge GitHub committed, with two parents and ``GITHUB`` as its
    committer, is excused its authorship, which GitHub sets, when both its parents are among
    ``commits`` and checked like the rest; its message is checked all the same."""
    present = {commit.sha for commit in commits}
    problems = []
    for commit in commits:
        (author, author_email), (committer, committer_email) = commit.author, commit.committer
        names = f"{author} <{author_email}> {committer} <{committer_email}>"
        problems += [f"{commit.sha[:7]}: {word!r} ({kind})"
                     for _, word, kind in banned_words(f"{names}\n{commit.message}", banned)]
        if commit.committer == GITHUB and len(commit.parents) == 2:
            missing = [parent[:7] for parent in commit.parents if parent not in present]
            if missing:
                problems.append(f"{commit.sha[:7]}: a merge GitHub committed, whose parent {', '.join(missing)} "
                                "is not in the history checked")
            continue
        problems += [f"{commit.sha[:7]}: {role} {name} <{email}>"
                     for role, (name, email) in (("author", commit.author), ("committer", commit.committer))
                     if (name, email) != AUTHOR]
    return problems


def test_commit_authorship_is_clean():
    commits = history(REPO)
    if not commits:
        pytest.skip("no commits yet")
    assert not is_shallow(REPO), "a shallow clone holds part of the history only; fetch all of it to check it"
    assert authorship_problems(commits) == []


def _commit(name: str, parents: str = "", committer=AUTHOR, author=AUTHOR, message="Add the page\n") -> Commit:
    """A commit of a made-up history: ``name`` and each letter of ``parents`` stand for whole hashes."""
    return Commit(name * 40, tuple(parent * 40 for parent in parents), author, committer, message)


#: the author GitHub may record on a merge it makes
SOMEONE = ("someone", "someone@users.noreply.github.com")


def test_a_merge_github_committed_is_excused_its_authorship_when_both_parents_are_checked():
    base, side = _commit("a"), _commit("b", "a")
    merge = _commit("c", "ab", committer=GITHUB, author=SOMEONE, message="Merge b into a\n")
    assert authorship_problems([merge, side, base]) == []
    # its parents are checked like every other commit
    stranger = _commit("b", "a", author=("Someone Else", "else@example.com"))
    assert authorship_problems([merge, stranger, base]) == ["bbbbbbb: author Someone Else <else@example.com>"]
    # and its message too
    said = _commit("c", "ab", committer=GITHUB, author=SOMEONE, message="Merge the zebrafish branch\n")
    assert authorship_problems([said, side, base], PROBE) == ["ccccccc: 'zebrafish' (probe)"]


@pytest.mark.parametrize("merge, problems", [
    # a commit with one parent: not a merge
    (_commit("c", "a", committer=GITHUB, author=SOMEONE),
     [f"ccccccc: author {SOMEONE[0]} <{SOMEONE[1]}>", "ccccccc: committer GitHub <noreply@github.com>"]),
    # two parents, but someone else committed it
    (_commit("c", "ab", committer=SOMEONE),
     [f"ccccccc: committer {SOMEONE[0]} <{SOMEONE[1]}>"]),
    # GitHub's name with another address
    (_commit("c", "ab", committer=("GitHub", "someone@github.com")),
     ["ccccccc: committer GitHub <someone@github.com>"]),
    # a parent that is not in the history, so not checked
    (_commit("c", "ad", committer=GITHUB, author=SOMEONE),
     ["ccccccc: a merge GitHub committed, whose parent ddddddd is not in the history checked"]),
])
def test_only_a_merge_github_committed_with_both_parents_checked_is_excused(merge, problems):
    assert authorship_problems([merge, _commit("b", "a"), _commit("a")]) == problems


def test_the_history_of_a_real_repository_with_a_merge_by_github(tmp_path):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": AUTHOR[0], "GIT_AUTHOR_EMAIL": AUTHOR[1],
           "GIT_COMMITTER_NAME": AUTHOR[0], "GIT_COMMITTER_EMAIL": AUTHOR[1]}
    repo = tmp_path / "repo"

    def git(*args, **extra):
        subprocess.run(["git", *args], cwd=repo, env={**env, **extra}, check=True, capture_output=True)

    repo.mkdir()
    git("init", "-q", "-b", "main")
    for name, branch in (("a", None), ("b", "side"), ("c", "main")):
        if branch:
            git("checkout", "-q", *(["-b"] if branch == "side" else []), branch)
        (repo / name).write_text(name)
        git("add", name)
        git("commit", "-q", "-m", f"Add {name}")
    git("merge", "-q", "--no-ff", "side", "-m", "Merge side into main",
        GIT_COMMITTER_NAME=GITHUB[0], GIT_COMMITTER_EMAIL=GITHUB[1])
    commits = history(repo)
    (merge,) = [commit for commit in commits if len(commit.parents) == 2]
    assert len(commits) == 4 and merge.committer == GITHUB and not is_shallow(repo)
    assert authorship_problems(commits) == []
    # a shallow clone holds the merge alone, and shows it with no parents: never excused
    clone = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(clone)], env=env, check=True,
                   capture_output=True)
    assert is_shallow(clone) and [commit.parents for commit in history(clone)] == [()]
    assert authorship_problems(history(clone)) == [f"{merge.sha[:7]}: committer GitHub <noreply@github.com>"]
