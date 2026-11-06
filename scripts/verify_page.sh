#!/usr/bin/env bash
# Open a page in headless Chrome and read its verdict from the tab title.
#
# Usage:
#   scripts/verify_page.sh <url> [expected-prefix] [timeout-seconds]
#
#   scripts/verify_page.sh "http://127.0.0.1:8765/?selftest=1"
#   scripts/verify_page.sh "http://127.0.0.1:8765/?selftest=1#results"
#
# The page's self-test (?selftest=1) writes its verdict into document.title:
# "PASS k/k threads=1" when every check passed, "FAIL k/m <check>: <why>"
# otherwise. This script waits for a verdict, prints it, and exits on it:
#
#   0  a PASS in its exact shape (every check passed, on one thread) that
#      starts with the expected prefix ("PASS " unless given);
#   1  a FAIL, a title that starts with PASS in any other shape, or a PASS
#      without the expected prefix;
#   2  bad usage, or no Chrome or Python to run;
#   3  Chrome exited early, or never opened its DevTools port;
#   4  no verdict before the timeout (120 s unless given).
#
# How it reads the title, and why:
#
#   * The DevTools HTTP endpoint (/json/list) lists the open targets with
#     their titles, so nothing is injected into the page and no DevTools
#     session is held open. Chrome picks a free port (--remote-debugging-port=0)
#     and writes it into DevToolsActivePort in its own profile, so a Chrome
#     already listening somewhere can never answer for this one.
#   * The endpoint lists more than the tab: a fresh headless Chrome also
#     reports targets such as an omnibox popup, whose title is no verdict. Only
#     targets of type "page" on the URL that was asked for are read.
#   * A failure is matched before the expected prefix, and a title starting
#     with PASS counts only in its exact shape, so no prefix can let a failing
#     page through.
#   * --virtual-time-budget is not used: it stalls the media stack, and the
#     page waits on a <video>.
#   * Each run has a throwaway --user-data-dir, removed on exit, and Chrome is
#     killed however the script ends. macOS has no `timeout`; the polling loop
#     keeps its own deadline.

set -euo pipefail

URL="${1:-}"
EXPECT="${2:-PASS }"
TIMEOUT="${3:-120}"

if [[ -z "$URL" ]]; then
  echo "usage: $0 <url> [expected-prefix] [timeout-seconds]" >&2
  exit 2
fi
if [[ "$EXPECT" != PASS* ]]; then
  echo "verify_page: the expected prefix must start with PASS, not '$EXPECT'" >&2
  exit 2
fi
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "verify_page: the timeout must be a whole number of seconds, not '$TIMEOUT'" >&2
  exit 2
fi

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [[ ! -x "$CHROME" ]]; then
  CHROME="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)"
fi
if [[ -z "$CHROME" || ! -x "$CHROME" ]]; then
  echo "verify_page: no Chrome executable found; set CHROME=<path>" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" > /dev/null; then
  echo "verify_page: $PYTHON not found; set PYTHON=<path>" >&2
  exit 2
fi

# Extra Chrome flags, space separated: for example CHROME_FLAGS="--no-sandbox"
# where Chrome runs as root in a container.
read -r -a EXTRA_FLAGS <<< "${CHROME_FLAGS:-}"

PROFILE="$(mktemp -d "${TMPDIR:-/tmp}/verify-page.XXXXXX")"
LOG="$PROFILE/chrome.log"
CHROME_PID=""

cleanup() {
  if [[ -n "$CHROME_PID" ]]; then
    kill "$CHROME_PID" 2>/dev/null || true
    wait "$CHROME_PID" 2>/dev/null || true
  fi
  # Helper processes that outlived the browser still name the profile.
  pkill -f -- "--user-data-dir=$PROFILE" 2>/dev/null || true
  rm -rf "$PROFILE"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"$CHROME" \
  --headless=new \
  --remote-debugging-port=0 \
  --user-data-dir="$PROFILE" \
  --no-first-run \
  --no-default-browser-check \
  --disable-features=Translate,MediaRouter \
  --autoplay-policy=no-user-gesture-required \
  --mute-audio \
  --window-size=1512,841 \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  "$URL" > "$LOG" 2>&1 &
CHROME_PID=$!

# The title of the "page" target on $URL, compared without the fragment and
# with an empty path read as "/"; nothing while there is none. The endpoint
# HTML-escapes titles, so the title is unescaped back to what the page wrote.
read_title() {
  curl -s --max-time 5 "http://127.0.0.1:$1/json/list" | "$PYTHON" -c '
import html, json, sys
from urllib.parse import urlsplit

def key(url):
    parts = urlsplit(url)
    return (parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query)

try:
    targets = json.load(sys.stdin)
except Exception:
    sys.exit(0)
wanted = key(sys.argv[1])
for target in targets:
    if target.get("type") == "page" and key(target.get("url", "")) == wanted:
        print(html.unescape(target.get("title", "")))
        break
' "$URL" || true
}

PASS_SHAPE='^PASS ([0-9]+)/([0-9]+) threads=1$'
deadline=$(( $(date +%s) + TIMEOUT ))
port=""
title=""
while (( $(date +%s) < deadline )); do
  if ! kill -0 "$CHROME_PID" 2>/dev/null; then
    echo "verify_page: Chrome exited early" >&2
    tail -5 "$LOG" >&2 || true
    exit 3
  fi
  if [[ -z "$port" && -s "$PROFILE/DevToolsActivePort" ]]; then
    port="$(head -n 1 "$PROFILE/DevToolsActivePort")"
  fi
  if [[ -n "$port" ]]; then
    title="$(read_title "$port")"
    case "$title" in
      FAIL*)
        echo "$title"
        exit 1
        ;;
      PASS*)
        echo "$title"
        if ! [[ "$title" =~ $PASS_SHAPE ]] || [[ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" || "${BASH_REMATCH[1]}" == 0 ]]; then
          echo "verify_page: a title that starts with PASS but is not 'PASS k/k threads=1'" >&2
          exit 1
        fi
        if [[ "$title" != "$EXPECT"* ]]; then
          echo "verify_page: the verdict does not start with '$EXPECT'" >&2
          exit 1
        fi
        exit 0
        ;;
    esac
  fi
  sleep 0.5
done

if [[ -z "$port" ]]; then
  echo "verify_page: Chrome never opened its DevTools port" >&2
  exit 3
fi
echo "verify_page: no verdict after ${TIMEOUT}s; last title: ${title:-<none>}" >&2
exit 4
