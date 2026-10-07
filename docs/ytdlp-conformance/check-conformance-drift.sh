#!/usr/bin/env bash
# Drift control for the copied yt-dlp conformance test: the canonical
# region still hashes to the value recorded in
# [tool.ytdlp_conformance].canonical_region_sha256.
#
# The region is hashed rather than the whole file because additions
# below the end sentinel are legitimate, and a check that cries wolf
# teaches people to ignore it. Both sentinels are inside the hashed
# span, so moving one is itself drift.
#
# It also checks the xfail budget, by loading
# conformance_xfail_plugin.py into a collect-only pytest run and
# reading the count it writes. Neither the file's text nor pytest's
# output is parsed, and both of those were tried:
#
# A text scan cannot see what pytest sees, and the gap is exploitable
# with ordinary pytest: two lines outside the hashed region --
#
#     from pytest import mark
#     pytestmark = mark.xfail(strict=False, reason="...")
#
# -- turn every test in the module into an xfail. Measured against an
# earlier version that grepped for pytest.mark.xfail: it reported
# "0/0 xfail" and exited 0 while pytest reported "3 skipped, 4 xfailed,
# 21 xpassed". The whole suite was neutralised and the check called it
# clean.
#
# The count is taken here rather than in a test inside the conformance
# module, because the same two lines mark that test as well: its
# failure is then reported as an xfail and the run stays green. A check
# cannot police the module it lives in.
#
# And it is taken from a plugin rather than from pytest's listing,
# because parsing the listing made the answer depend on how the path
# was spelled and how verbose pytest was. Measured on a module with 35
# xfail items: a plain relative path reported all 35, while
# ./tests/..., an absolute path, and the same path under
# PYTEST_ADDOPTS=-q each reported 0 and exited 0.
#
# Copy this and conformance_xfail_plugin.py next to the test, and run
# it from the repository root in CI with the project's test environment
# active. Spec: docs/ytdlp-invariants.md in
# fdff87554/youtube-downloader.
#
# Usage: check-conformance-drift.sh [test-path] [manifest-path]

set -euo pipefail

TEST_PATH="${1:-tests/test_ytdlp_conformance.py}"
MANIFEST="${2:-pyproject.toml}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

BEGIN_SENTINEL='# --- ytdlp-conformance:canonical-begin ---'
END_SENTINEL='# --- ytdlp-conformance:canonical-end ---'

die() {
	printf 'conformance drift: %s\n' "$1" >&2
	exit 1
}

[ -f "$TEST_PATH" ] || die "no conformance test at $TEST_PATH"
[ -f "$MANIFEST" ] || die "no manifest at $MANIFEST"

# tomllib rather than grep: the profile is TOML, and a key matched by
# regex would also match one inside a comment or a different table.
read_profile_key() {
	python3 - "$MANIFEST" "$1" <<'PY'
import sys
import tomllib

manifest, key = sys.argv[1], sys.argv[2]
with open(manifest, "rb") as handle:
    profile = tomllib.load(handle).get("tool", {}).get("ytdlp_conformance")
if profile is None:
    sys.exit(f"{manifest} declares no [tool.ytdlp_conformance]")
if key not in profile:
    sys.exit(f"{manifest} declares no {key}; every profile key is mandatory")

value = profile[key]
if key == "xfail_budget":
    # Checked before anything compares against it. An earlier version
    # read it untyped, and `[ 0 -gt "oops" ]` printed an error to
    # stderr and left the script exiting 0, so an unusable budget read
    # as a passing one.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        sys.exit(f"{manifest}: xfail_budget is {value!r}, not a non-negative integer")
print(value)
PY
}

expected_hash="$(read_profile_key canonical_region_sha256)"
xfail_budget="$(read_profile_key xfail_budget)"

# Inclusive of both sentinels, so moving a sentinel is itself drift.
region="$(
	awk -v b="$BEGIN_SENTINEL" -v e="$END_SENTINEL" '
		$0 == b { inside = 1 }
		inside  { print }
		$0 == e { inside = 0; found = 1 }
		END     { if (!found) exit 3 }
	' "$TEST_PATH"
)" || die "no canonical region in $TEST_PATH; both sentinels must be present"

actual_hash="$(printf '%s\n' "$region" | sha256sum | cut -d' ' -f1)"

if [ "$actual_hash" != "$expected_hash" ]; then
	die "$(
		cat <<MSG
the canonical region of $TEST_PATH has changed.
  recorded: $expected_hash
  actual:   $actual_hash
Repository-specific tests belong BELOW the end sentinel. If the region
itself needs to change, change it upstream in
docs/ytdlp-conformance/ first, then re-copy here and update
canonical_region_sha256 in $MANIFEST -- that line moving is the visible
part of this mechanism, not a formality to clear.
MSG
	)"
fi

# pytest's own collection and marker resolution, so an aliased import,
# a module-level pytestmark and a marker added from a conftest.py all
# count. The plugin lives beside this script; PYTHONPATH is extended
# rather than replaced so the project's own imports still resolve.
plugin_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
count_file="$workdir/xfail-count"

collect_status=0
collect_output="$(
	PYTHONPATH="$plugin_dir${PYTHONPATH:+:$PYTHONPATH}" \
		CONFORMANCE_XFAIL_COUNT_FILE="$count_file" \
		python3 -m pytest "$TEST_PATH" --collect-only -p conformance_xfail_plugin \
		-p no:cacheprovider 2>&1
)" || collect_status=$?

# 0 means items were collected; 5 means none were, which for a
# conformance module is itself wrong. Anything else is a real failure.
if [ "$collect_status" -ne 0 ]; then
	printf '%s\n' "$collect_output" >&2
	die "could not collect $TEST_PATH to count xfail markers (pytest exit $collect_status)"
fi

# The plugin writes the count on the first line and the marked node ids
# after it. A missing file means the plugin never ran, which must not
# read as a count of zero.
if [ ! -f "$count_file" ]; then
	printf '%s\n' "$collect_output" >&2
	die "conformance_xfail_plugin did not run; it must be importable beside $0"
fi

xfail_count="$(head -n 1 "$count_file")"
case "$xfail_count" in
'' | *[!0-9]*)
	die "conformance_xfail_plugin wrote '$xfail_count', which is not a count"
	;;
esac

if [ "$xfail_count" -gt "$xfail_budget" ]; then
	die "$(
		cat <<MSG
$TEST_PATH holds $xfail_count xfailed item(s), over a budget of $xfail_budget:
$(tail -n +2 "$count_file" | sed 's/^/  /')
The list is the backlog and may only shrink. Fix the aspect, or -- if
the gap is genuinely accepted -- waive it in $MANIFEST, which has to
cite an issue, rather than raising the budget.
MSG
	)"
fi

printf 'conformance: region %s matches %s, %s/%s xfailed\n' \
	"${actual_hash:0:12}" "$TEST_PATH" "$xfail_count" "$xfail_budget"
