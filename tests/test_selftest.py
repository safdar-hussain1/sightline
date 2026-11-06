"""The page's self-test as the build publishes it, and the script that reads its verdict.

``?selftest=1`` runs the in-page self-test (``web/src/selftest.ts``), which writes its verdict into
the tab title: ``PASS k/k threads=1``, or ``FAIL k/m <check>: <why>``. The published build in
``docs/`` must keep that code out of the page's normal load: it is a chunk of its own, the served
HTML never loads or preloads it, the first chunk reaches it only through an import guarded by the
address, and it is the only published script that writes the title.

``scripts/verify_page.sh`` opens a URL in headless Chrome and exits on the verdict. Its exit status
is what CI and a release check read, so its rule is tested here against a stand-in for Chrome that
serves the DevTools target list with a chosen title: 0 only for a PASS in its exact shape, 1 for a
FAIL or any other title that starts with PASS, 4 when no verdict comes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
VERIFY = ROOT / "scripts" / "verify_page.sh"

#: a string delimiter as a minifier may write it
Q = r"[`\"']"
TITLE_WRITE = re.compile(r"document\s*\.\s*title\s*=(?!=)")
GUARDED_IMPORT = re.compile(
    rf"\.get\(\s*{Q}selftest{Q}\s*\)\s*===\s*{Q}1{Q}\s*&&[^;]{{0,80}}?import\(\s*{Q}\./(selftest-[\w-]+\.js){Q}\s*\)"
)


def _selftest_chunks() -> list[Path]:
    return sorted(ASSETS.glob("selftest-*.js"))


def test_the_self_test_is_one_chunk_of_its_own():
    assert [path.name for path in _selftest_chunks()] != []
    assert len(_selftest_chunks()) == 1


def test_the_served_page_never_loads_or_preloads_the_self_test():
    assert "selftest" not in (DOCS / "index.html").read_text(encoding="utf-8")


def test_the_first_chunk_imports_the_self_test_only_under_the_flag():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    entries = re.findall(r'<script type="module"[^>]*\bsrc="\./(assets/index-[\w-]+\.js)"', html)
    assert len(entries) == 1, entries
    entry = (DOCS / entries[0]).read_text(encoding="utf-8")
    [chunk] = _selftest_chunks()
    assert GUARDED_IMPORT.findall(entry) == [chunk.name]
    # The only other mention is the preload list of that same guarded import.
    assert entry.count(chunk.name) == 2


def test_no_published_script_imports_the_self_test_statically():
    for script in ASSETS.glob("*.js"):
        text = script.read_text(encoding="utf-8")
        assert not re.search(rf"from\s*{Q}\./selftest-", text), script.name
        assert not re.search(rf"\bimport\s*{Q}\./selftest-", text), script.name


def test_only_the_self_test_writes_the_title():
    writers = sorted(path.name for path in ASSETS.glob("*.js") if TITLE_WRITE.search(path.read_text(encoding="utf-8")))
    assert writers == [chunk.name for chunk in _selftest_chunks()]


def test_the_verifier_is_tracked_as_an_executable():
    listed = subprocess.run(["git", "ls-files", "-s", "scripts/verify_page.sh"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.split()
    assert listed[:1] == ["100755"]
    assert VERIFY.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")


# -- the verifier's exit rule, against a stand-in for Chrome ------------------------------------

#: A stand-in for Chrome: it serves FAKE_TARGETS as the DevTools target list (the page URL it was
#: started on replaces "{url}"), writes the port into DevToolsActivePort in its profile as Chrome
#: does, and runs until it is killed. FAKE_EXIT makes it exit at once instead, and FAKE_IGNORE_TERM makes it ignore SIGTERM.
FAKE_CHROME = textwrap.dedent("""
    import http.server, json, os, signal, sys
    if os.environ.get("FAKE_EXIT"):
        sys.exit(1)
    if os.environ.get("FAKE_IGNORE_TERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    profile = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--user-data-dir="))
    url = sys.argv[-1]
    body = os.environ["FAKE_TARGETS"].replace("{url}", url).encode()

    class Targets(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/json/list" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Targets)
    with open(os.path.join(profile, "DevToolsActivePort"), "w") as port:
        port.write(f"{server.server_address[1]}\\n/devtools/browser/stand-in\\n")
    server.serve_forever()
""")

URL = "http://127.0.0.1:8765/?selftest=1#results"

needs_tools = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("curl") is None,
                                 reason="the verifier needs bash and curl")


def _page(title: str, url: str = "{url}", kind: str = "page") -> dict:
    return {"type": kind, "url": url, "title": title}


def _verify(tmp_path: Path, targets: list[dict], *args: str, exit_now: bool = False,
            **extra_env: str) -> subprocess.CompletedProcess:
    script = tmp_path / "fake_chrome.py"
    script.write_text(FAKE_CHROME, encoding="utf-8")
    chrome = tmp_path / "chrome"
    chrome.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    chrome.chmod(0o755)
    env = {**os.environ, "CHROME": str(chrome), "PYTHON": sys.executable, "TMPDIR": str(tmp_path),
           "FAKE_TARGETS": json.dumps(targets)}
    if exit_now:
        env["FAKE_EXIT"] = "1"
    env.update(extra_env)
    return subprocess.run([shutil.which("bash") or "bash", str(VERIFY), *args], env=env, capture_output=True, text=True, timeout=60)


@needs_tools
@pytest.mark.parametrize("title", ["PASS 6/6 threads=1", "PASS 12/12 threads=1"])
def test_the_verifier_passes_a_pass_in_its_exact_shape(tmp_path, title):
    done = _verify(tmp_path, [_page(title)], URL, "PASS ", "20")
    assert (done.returncode, done.stdout.strip()) == (0, title)
    assert list(tmp_path.glob("verify-page.*")) == [], "the throwaway profile must be removed"


@needs_tools
@pytest.mark.parametrize("title", [
    "FAIL 5/6 decode-hash: c2018: frame 7 of 422 decodes to other bytes than Python's",
    "FAIL 6/6 threads=1",
    "PASS 5/6 selftest-inputs: shuffled angles off by up to 0.1",
    "PASS 6/6 threads=2",
    "PASS 0/0 threads=1",
    "PASS 6/6 threads=1 and more",
    "PASSED",
])
def test_the_verifier_fails_every_other_verdict(tmp_path, title):
    done = _verify(tmp_path, [_page(title)], URL, "PASS ", "20")
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert done.stdout.strip() == title


@needs_tools
def test_the_verifier_reads_only_the_page_it_opened(tmp_path):
    targets = [
        _page("PASS 6/6 threads=1", kind="other"),
        _page("PASS 6/6 threads=1", url="http://127.0.0.1:8765/elsewhere"),
        _page("FAIL 1/6 metadata: shuffled is run &quot;x&quot;", url="http://127.0.0.1:8765/?selftest=1"),
    ]
    done = _verify(tmp_path, targets, URL, "PASS ", "20")
    # The fragment is not part of the comparison; the title is unescaped as the page wrote it.
    assert (done.returncode, done.stdout.strip()) == (1, 'FAIL 1/6 metadata: shuffled is run "x"')


@needs_tools
def test_the_verifier_gives_up_without_a_verdict(tmp_path):
    done = _verify(tmp_path, [_page("RUNNING 3/6 selftest-inputs")], URL, "PASS ", "2")
    assert done.returncode == 4
    assert "RUNNING 3/6 selftest-inputs" in done.stderr


@needs_tools
def test_the_verifier_reports_a_chrome_that_exits(tmp_path):
    assert _verify(tmp_path, [], URL, "PASS ", "10", exit_now=True).returncode == 3


@needs_tools
@pytest.mark.parametrize("args", [[], [URL, "FAIL"], [URL, "PASS ", "soon"]])
def test_the_verifier_refuses_bad_usage(tmp_path, args):
    assert _verify(tmp_path, [_page("PASS 6/6 threads=1")], *args).returncode == 2


@needs_tools
def test_the_verifier_names_a_missing_curl_before_it_starts_chrome(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    done = _verify(tmp_path, [_page("PASS 6/6 threads=1")], URL, "PASS ", "10", PATH=str(bare))
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert "curl" in done.stderr
    assert list(tmp_path.glob("verify-page.*")) == [], "Chrome must not have been started"


@needs_tools
def test_the_verifier_does_not_wait_forever_for_a_chrome_that_ignores_sigterm(tmp_path):
    started = time.monotonic()
    done = _verify(tmp_path, [_page("PASS 6/6 threads=1")], URL, "PASS ", "20", FAKE_IGNORE_TERM="1")
    assert (done.returncode, done.stdout.strip()) == (0, "PASS 6/6 threads=1")
    assert time.monotonic() - started < 25
    assert list(tmp_path.glob("verify-page.*")) == []


@needs_tools
def test_the_verifier_says_when_no_page_target_matches_the_url(tmp_path):
    targets = [_page("PASS 6/6 threads=1", url="http://127.0.0.1:8765/?selftest=1&x=a%20b"),
               _page("", url="about:blank", kind="other")]
    done = _verify(tmp_path, targets, URL, "PASS ", "2")
    assert done.returncode == 4
    assert "no page target matched" in done.stderr
    assert URL in done.stderr
    assert "http://127.0.0.1:8765/?selftest=1&x=a%20b" in done.stderr
