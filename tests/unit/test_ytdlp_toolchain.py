"""The yt-dlp toolchain has to be complete, because incomplete fails quietly.

Both call sites pass ``quiet`` and ``no_warnings``, so a missing or
mismatched challenge-solver package does not surface as an error: extraction
reports "Signature solving failed", the formats that need signature
deciphering disappear, and the download either degrades or fails with an
unrelated-looking message. Nothing else in the suite can catch that, because
the download tests replace yt-dlp with a mock.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re

import pytest

EJS_PACKAGE = "yt-dlp-ejs"

# yt-dlp names the version it expects in its own extras. We depend on the
# package directly rather than through them -- they pull in eight packages
# this project does not use -- so the coupling is checked here instead of by
# the resolver. Both the "default" and "pin" extras carry the same line; match
# either so a reshuffle upstream does not silently disable this check.
_PINNED_EJS = re.compile(r"^yt-dlp-ejs\s*==\s*(?P<version>[\w.]+)\s*;.*extra\s*==\s*['\"](?:default|pin)['\"]")


def _expected_ejs_version() -> str:
    requirements = metadata.metadata("yt-dlp").get_all("Requires-Dist") or []
    for requirement in requirements:
        match = _PINNED_EJS.match(requirement)
        if match:
            return match.group("version")
    pytest.fail(
        "the installed yt-dlp declares no pinned yt-dlp-ejs version; its extras "
        "may have been restructured, so this check needs revisiting"
    )


class TestChallengeSolversAreInstalled:
    def test_ejs_package_is_installed(self) -> None:
        # metadata() raises PackageNotFoundError when absent, which is the
        # failure worth reporting -- not an ImportError somewhere later.
        assert metadata.version(EJS_PACKAGE)

    def test_ejs_matches_the_installed_yt_dlp(self) -> None:
        # Relocking after a yt-dlp bump can leave the solvers behind, and a
        # mismatched pair only fails at runtime unless something checks here.
        assert metadata.version(EJS_PACKAGE) == _expected_ejs_version()
