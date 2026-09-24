#!/usr/bin/env bash
# Snapshot the integrator's Claude Code session transcript into evidence/<which>/.
#
#   tools/evidence/snapshot_session.sh before|after [--from DIR | --file FILE] [--force]
#
# Copies the newest *.jsonl directly in the Claude Code project directory of the main
# checkout (~/.claude/projects/<main path with every non-alphanumeric character
# replaced by '-'>, e.g. -home-vedant-TabPack; override with --from DIR or
# $TABPACK_SESSION_DIR, or name the transcript with --file) to
# evidence/<which>/session.jsonl of the current checkout, redacts personal email
# addresses with tools/evidence/redact_emails.py and prints the sha256 row for the
# table in evidence/README.md. An existing evidence/before/session.jsonl is only
# replaced with --force (the "before" snapshot is taken once).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

which="" from="${TABPACK_SESSION_DIR:-}" file="" force=0
while (($#)); do
  case "$1" in
    --from | --file)
      [[ $# -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 2; }
      if [[ $1 == --from ]]; then from="$2"; else file="$2"; fi
      shift 2 ;;
    --force) force=1; shift ;;
    -h | --help) usage; exit 0 ;;
    before | after)
      [[ -z $which ]] || { echo "error: give before or after once" >&2; exit 2; }
      which="$1"; shift ;;
    *) echo "error: unexpected argument $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [[ -z $which ]]; then
  echo "usage: snapshot_session.sh before|after [--from DIR | --file FILE] [--force]" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
MAIN="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"

if [[ -z $file ]]; then
  from="${from:-$HOME/.claude/projects/${MAIN//[^A-Za-z0-9]/-}}"
  if [[ ! -d $from ]]; then
    echo "error: no Claude Code project directory at $from (use --from DIR)" >&2
    exit 1
  fi
  file="$(find "$from" -maxdepth 1 -type f -name '*.jsonl' -printf '%T@ %p\n' |
    sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -z $file ]]; then
    echo "error: no *.jsonl transcript in $from" >&2
    exit 1
  fi
fi
if [[ ! -f $file ]]; then
  echo "error: no such transcript: $file" >&2
  exit 1
fi

dest_dir="$ROOT/evidence/$which"
dest="$dest_dir/session.jsonl"
if [[ $which == before && -e $dest && $force -eq 0 ]]; then
  echo "error: $dest exists; the before snapshot is taken once (use --force)" >&2
  exit 1
fi
mkdir -p "$dest_dir"

# Copy first (the live transcript keeps growing), then hash and redact the copy.
tmp="$dest.tmp.$$"
trap 'rm -f "$tmp"' EXIT
cp "$file" "$tmp"
raw_sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
python3 "$SCRIPT_DIR/redact_emails.py" "$tmp" >/dev/null
red_sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
n_red="$({ grep -o '\[redacted-email\]' "$tmp" || true; } | wc -l)"
mv "$tmp" "$dest"

echo "snapshot_session: $file"
echo "  -> $dest ($(wc -l <"$dest") lines, $n_red '[redacted-email]' in total)"
echo "Row for the redaction table in evidence/README.md:"
echo "| \`$which/session.jsonl\` | \`$raw_sha\` | \`$red_sha\` |"
