#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${ROOT}/datasets"
LIMIT=0
EXTRACT=0
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --list-only)
      LIST_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash tests/data/download_data.sh [--output DIR] [--limit N] [--extract] [--list-only]" >&2
      exit 2
      ;;
  esac
done

URLS=()
while IFS= read -r url; do
  [[ -n "${url}" ]] && URLS+=("${url}")
done < "${ROOT}/tests/data/suitesparse_urls.txt"
if [[ ${#URLS[@]} -ne 820 ]]; then
  echo "ERROR: expected 820 dataset URLs, found ${#URLS[@]}." >&2
  exit 3
fi
if [[ "${LIMIT}" -gt 0 && "${LIMIT}" -lt "${#URLS[@]}" ]]; then
  URLS=("${URLS[@]:0:${LIMIT}}")
fi

if [[ "${LIST_ONLY}" -eq 1 ]]; then
  printf '%s\n' "${URLS[@]}"
  exit 0
fi

mkdir -p "${OUTPUT}"
if command -v wget >/dev/null 2>&1; then
  DOWNLOADER=wget
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER=curl
else
  echo "ERROR: wget or curl is required." >&2
  exit 2
fi

for url in "${URLS[@]}"; do
  archive="${OUTPUT}/$(basename "${url}")"
  if [[ ! -s "${archive}" ]]; then
    echo "Downloading ${url}"
    if [[ "${DOWNLOADER}" == wget ]]; then
      wget -c -O "${archive}" "${url}"
    else
      curl -fL --retry 3 -C - -o "${archive}" "${url}"
    fi
  else
    echo "Keeping existing ${archive}"
  fi
  if [[ "${EXTRACT}" -eq 1 ]]; then
    marker="${archive}.extracted"
    if [[ ! -f "${marker}" ]]; then
      tar -xzf "${archive}" -C "${OUTPUT}"
      touch "${marker}"
    fi
  fi
done

echo "Dataset archives: ${OUTPUT}"
