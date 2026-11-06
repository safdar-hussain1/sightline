#!/usr/bin/env bash
# Take the page's screenshots in headless Chrome: the share image, the
# README's picture of the demo, and both views whole in the dark and the light
# theme and on a phone (scripts/capture.mjs says which, and how).
#
# Usage:
#   scripts/capture.sh <url> [out-dir]
#
#   scripts/capture.sh "http://127.0.0.1:8765/" /tmp/sightline-shots
#
# Serve docs/ first (npx vite preview from web/, or any static server): the
# demo's model and clips do not load from file://. The pictures land in
# out-dir (a new temporary folder unless given), never in the repository;
# og-image.png and readme-hero.png are published by copying them into
# web/public/ and building. Look at every one before it is published.
#
# Exit codes: 0 every screenshot taken; 1 a shot failed; 2 bad usage or no
# Chrome or Node; 3 Chrome exited early or never opened its DevTools port.

set -euo pipefail

URL="${1:-}"
OUT="${2:-}"
if [[ -z "$URL" ]]; then
  echo "usage: $0 <url> [out-dir]" >&2
  exit 2
fi
if [[ -z "$OUT" ]]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/sightline-shots.XXXXXX")"
fi

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [[ ! -x "$CHROME" ]]; then
  CHROME="$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)"
fi
if [[ -z "$CHROME" || ! -x "$CHROME" ]]; then
  echo "capture: no Chrome executable found; set CHROME=<path>" >&2
  exit 2
fi
if ! command -v node > /dev/null; then
  echo "capture: node not found (22 or later)" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Extra Chrome flags, space separated: for example CHROME_FLAGS="--no-sandbox"
# where Chrome runs as root in a container.
read -r -a EXTRA_FLAGS <<< "${CHROME_FLAGS:-}"

PROFILE="$(mktemp -d "${TMPDIR:-/tmp}/capture.XXXXXX")"
CHROME_PID=""

cleanup() {
  if [[ -n "$CHROME_PID" ]]; then
    kill "$CHROME_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$CHROME_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$CHROME_PID" 2>/dev/null; then
      kill -9 "$CHROME_PID" 2>/dev/null || true
    fi
    wait "$CHROME_PID" 2>/dev/null || true
  fi
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
  --hide-scrollbars \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  about:blank > "$PROFILE/chrome.log" 2>&1 &
CHROME_PID=$!

port=""
for _ in $(seq 1 100); do
  if ! kill -0 "$CHROME_PID" 2>/dev/null; then
    echo "capture: Chrome exited early" >&2
    tail -5 "$PROFILE/chrome.log" >&2 || true
    exit 3
  fi
  if [[ -s "$PROFILE/DevToolsActivePort" ]]; then
    port="$(head -n 1 "$PROFILE/DevToolsActivePort")"
    break
  fi
  sleep 0.1
done
if [[ -z "$port" ]]; then
  echo "capture: Chrome never opened its DevTools port" >&2
  exit 3
fi

if ! node "$HERE/capture.mjs" "$port" "$URL" "$OUT"; then
  echo "capture: a screenshot failed" >&2
  exit 1
fi
echo "capture: the screenshots are in $OUT"
