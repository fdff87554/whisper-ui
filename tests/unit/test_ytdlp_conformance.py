"""Conformance test for the shared yt-dlp invariants.

Copy this file into the repository's test path, drop the ``.example``
suffix, and supply the ``ytdlp_subject`` fixture from
``conftest-fixture.py.example``. The spec is
``docs/ytdlp-invariants.md`` in fdff87554/youtube-downloader.

Everything between the two canonical sentinels below is shared across
all three repositories and must stay byte-identical: a drift check
hashes exactly that span. Repository-specific tests belong *below* the
end sentinel, where they are expected and do not move the hash.

The expected values are data, held in ``[tool.ytdlp_conformance]`` in
the repository's ``pyproject.toml``. The logic here is a constant, and
this file imports nothing from the repository it is testing -- the
adapter fixture is the only seam.

This test never touches the network. Where it reaches into yt-dlp's
private attributes it guards with ``hasattr`` and **fails** rather than
skipping, because a silent skip is the failure mode the whole mechanism
exists to prevent.
"""

# --- ytdlp-conformance:canonical-begin ---
from __future__ import annotations

import importlib.metadata as metadata
import itertools
import os
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yt_dlp

if TYPE_CHECKING:
    from collections.abc import Iterable

SPEC_VERSION = "1.0.0"

REQUIRED_PROFILE_KEYS = frozenset(
    {
        "spec_version",
        "canonical_region_sha256",
        "extractor_spec",
        "accepts_urls",
        "rejects_urls",
        "cachedir",
        "codec_policy",
        "socket_timeout_seconds",
        "max_bytes",
        "js_runtime",
        "js_runtime_provided_by",
        "ejs_coupling",
        "markers_verified_against",
        "po_token_hook",
        "player_clients",
        "needs_ffmpeg",
        "needs_ffprobe",
        "xfail_budget",
    }
)
OPTIONAL_PROFILE_KEYS = frozenset({"waivers"})

# Every aspect the spec defines. C7 and C8 are not single checks -- a
# ceiling that holds on one downloader path is not a ceiling -- so they
# are accounted for in parts, and the rest are single aspects under
# their bare ID.
SPEC_ASPECTS = frozenset(
    {
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7.selectable",
        "C7.metered",
        "C7.external",
        "C7.landed",
        "C8.socket",
        "C8.retries",
        "C8.deadline",
        "C9",
        "C10",
    }
)

# What this file checks mechanically. Anything in SPEC_ASPECTS and not
# here has to be waived explicitly, which is what stops "no test" from
# reading the same as "passing". Three are absent on purpose and are
# expected to be waived by every repository:
#
#   C5           error-message content is not visible through the adapter
#   C7.landed    a streaming repository lands no file, and one that does
#                exposes no hook for a shared test to inspect
#   C8.deadline  none of the three has a wall-clock bound, and a shared
#                test cannot invent the mechanism it would check
#
# C7.external is listed as covered, but what covers it is a test that
# demands a waiver: a subprocess reports nothing back, so the only
# honest outcome for a repository that downloads that way is a named
# waiver with an issue behind it.
COVERED_ASPECTS = frozenset(
    {
        "C1",
        "C2",
        "C3",
        "C4",
        "C6",
        "C7.selectable",
        "C7.metered",
        "C7.external",
        "C8.socket",
        "C8.retries",
        "C9",
        "C10",
    }
)

# yt-dlp declares the yt-dlp-ejs version it expects in an extra. Which
# extra has changed between releases, so both spellings are accepted;
# what matters is the pinned version, not where it is parked.
PINNED_EJS_REQUIREMENT = re.compile(
    r"^yt-dlp-ejs\s*==\s*(?P<version>[\w.]+)\s*;.*extra\s*==\s*"
    r"['\"](?:pin|default)['\"]"
)

# C4's vectors. The last two are U+2028 and U+2029, written with chr()
# so that neither this file nor a reviewer's diff carries a character
# that renders as a line break. They are here deliberately: they are
# the two that ``str.splitlines()`` breaks on while Unicode's ``C``
# category does not cover them, so a neutraliser written against
# ``unicodedata.category(ch).startswith("C")`` lets them through and
# still produces a forged line.
CONTROL_CHARACTER_VECTORS = (
    "\r",
    "\n",
    "\x1b",
    "\x00",
    chr(0x2028),
    chr(0x2029),
)

# C7.selectable: a downloader can only be metered if the bytes pass
# through Python. HttpFD and DashSegmentsFD do; anything yt-dlp hands
# to ffmpeg does not, and m3u8_native hands the whole transfer over
# whenever HlsFD cannot handle the manifest itself.
_METERED_PROTOCOL_PREFIXES = ("http", "ftp")

# The offline pool each download selector is evaluated against. The
# unmeterable renditions carry the *higher* bitrate on purpose: proto
# sits after quality, tbr, filesize, vbr, height and width in yt-dlp's
# default sort order, so it only breaks a tie. A pool of equal-quality
# renditions shows https winning and invites the false conclusion that
# an unconstrained selector is safe.
# The unmeterable renditions carry the HIGHER bitrate on purpose: proto
# sits after quality, tbr, filesize, vbr, height and width in yt-dlp's
# default sort order, so it only breaks a tie. A pool of equal-quality
# renditions shows https winning and invites the false conclusion that
# an unconstrained selector is safe.
_HIGH_TBR = 6000
_LOW_TBR = 3000

# Every height a repository here selects on, so that a tier-limited
# selector finds something at its own tier instead of failing for want
# of a format. 1080 doubles as the unbounded tier.
_PROBE_HEIGHTS = (1080, 720, 480)
_FORMAT_KINDS = ("video", "audio", "progressive")

# http and ftp pass through HttpFD or DashSegmentsFD, which count bytes
# in Python. Anything yt-dlp hands to ffmpeg does not, and m3u8_native
# hands the whole transfer over whenever HlsFD cannot handle the
# manifest itself.
_UNMETERED_PROBE_PROTOCOLS = ("m3u8_native", "m3u8")


def _probe_format(
    format_id: str, protocol: str, kind: str, tbr: int, height: int
) -> dict[str, Any]:
    common = {
        "format_id": format_id,
        "protocol": protocol,
        "url": f"https://conformance.invalid/{format_id}",
        "tbr": tbr,
        "filesize": tbr * 1000,
    }
    if kind == "audio":
        return common | {
            "ext": "m4a",
            "vcodec": "none",
            "acodec": "mp4a.40.2",
            "abr": tbr,
        }
    video = {
        "ext": "mp4",
        "vcodec": "avc1.4d401f",
        "height": height,
        "width": height * 16 // 9,
        "fps": 30,
    }
    if kind == "video":
        return common | video | {"acodec": "none"}
    return common | video | {"acodec": "mp4a.40.2"}


def _pool(include_metered: bool) -> list[dict[str, Any]]:
    """Renditions of every shape, at every height a tier selects on.

    With ``include_metered`` false nothing meterable is on offer, which
    is the only way to exercise a selector's fallback branches: a
    selector whose first branch is constrained and whose fallback is
    not matches on the first branch against a mixed pool, so the
    fallback is never reached there.
    """
    pool = []
    for kind in _FORMAT_KINDS:
        heights = _PROBE_HEIGHTS if kind != "audio" else (0,)
        for height in heights:
            suffix = f"{kind}-{height}" if height else kind
            if include_metered:
                pool.append(
                    _probe_format(f"http-{suffix}", "https", kind, _LOW_TBR, height)
                )
            for protocol in _UNMETERED_PROBE_PROTOCOLS:
                pool.append(
                    _probe_format(
                        f"{protocol}-{suffix}", protocol, kind, _HIGH_TBR, height
                    )
                )
    return pool


def _mixed_pool() -> list[dict[str, Any]]:
    """Meterable and unmeterable renditions, the unmeterable ones better."""
    return _pool(include_metered=True)


def _unmeterable_only_pool() -> list[dict[str, Any]]:
    """Nothing meterable on offer, which is what exercises a fallback."""
    return _pool(include_metered=False)


def _selected_formats(
    selector: str, sort: list[str] | None, pool: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """What yt-dlp's own selector picks out of ``pool``.

    Evaluating the selector is the only way to check C7.selectable.
    Searching the string for a protocol filter passes
    ``bestaudio[protocol^=http]/best``, whose second branch is
    unconstrained, and ``best[protocol=m3u8]``, which contains a
    protocol filter that selects exactly the wrong thing.

    Raises whatever yt-dlp raises when nothing matches; callers decide
    whether that is the answer they wanted.
    """
    ydl = yt_dlp.YoutubeDL(
        {"quiet": True, "no_warnings": True, **({"format_sort": sort} if sort else {})}
    )
    info = {
        "formats": [dict(entry) for entry in pool],
        "id": "conformance",
        "title": "conformance",
        "extractor": "youtube",
        "extractor_key": "Youtube",
        "webpage_url": "https://conformance.invalid/conformance",
        "incomplete_formats": False,
    }
    if not hasattr(ydl, "build_format_selector"):
        pytest.fail(
            "yt_dlp.YoutubeDL no longer exposes build_format_selector, so "
            "selectors cannot be evaluated; this test needs rewriting "
            "against whatever replaced it"
        )
    ydl.sort_formats(info)
    return list(ydl.build_format_selector(selector)(info))


def _transfer_components(formats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each separately downloaded stream behind a selection.

    A ``video+audio`` selection comes back as one merged entry whose
    ``requested_formats`` holds the parts that are actually fetched, and
    whose own ``protocol`` is their names joined: ``https+m3u8_native``.
    Reading only that top-level field is how an HLS audio track passed
    for metered -- it begins with "http", so the string test said yes.
    """
    components = []
    for entry in formats:
        parts = entry.get("requested_formats")
        components.extend(parts if parts else [entry])
    return components


def _oversized_events(ceiling: int) -> list[tuple[str, dict[str, Any]]]:
    """Transfers past ``ceiling``, in the shapes yt-dlp really reports.

    The first shape alone is not enough. ``total_bytes`` is
    Content-Length, and a hook that reads only that one passes while
    being unable to stop a transfer of unknown length -- which is the
    case the ceiling exists for, since ``max_filesize`` already covers
    the declared-length one. The last shape is the external-downloader
    report: a single ``finished`` event once the child process exits.
    """
    over = ceiling + 1
    return [
        (
            "a transfer reporting its length",
            {
                "status": "downloading",
                "filename": "conformance.part",
                "downloaded_bytes": over,
                "total_bytes": over,
            },
        ),
        (
            "a transfer whose length is None",
            {
                "status": "downloading",
                "filename": "conformance.part",
                "downloaded_bytes": over,
                "total_bytes": None,
            },
        ),
        (
            "a transfer that never reports a length",
            {
                "status": "downloading",
                "filename": "conformance.part",
                "downloaded_bytes": over,
            },
        ),
        (
            "a finished transfer",
            {
                "status": "finished",
                "filename": "conformance.part",
                "downloaded_bytes": over,
                "total_bytes": over,
            },
        ),
    ]


def _drive_hooks(hooks: Iterable[Any], status: dict[str, Any]) -> BaseException | None:
    """Run the hooks the way yt-dlp does, and report what stopped them.

    ``downloader/common.py:488-495`` iterates every hook in order with
    **one** status object, so the first one to raise ends the transfer
    and the rest never run. The question this answers is whether the
    *set* stops it, not whether any particular member does.

    The copy is taken once, before the loop, and never inside it.
    Sharing the object is deliberate upstream -- the source says so in
    as many words, at ``:489-493``: "Ideally we want to make a copy of
    the dict, but that is too slow", then "youtube-dl passes the same
    status object to all the hooks ... So keep this behavior if
    possible". A hook that rewrites the event therefore changes what
    every later hook sees, and copying per hook hid exactly that: a
    pair whose first member zeroes the counters was reported as
    stopping a transfer it cannot stop. Copying once still keeps
    ``status`` itself untouched, so the caller can reuse one event
    across several hook sets.
    """
    event = dict(status)
    for hook in hooks:
        try:
            hook(event)
        except BaseException as error:  # noqa: BLE001
            return error
    return None


def _unmetered(formats: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The downloaded components no Python-side downloader can meter."""
    return [
        (str(part.get("format_id")), str(part.get("protocol") or ""))
        for part in _transfer_components(formats)
        if not str(part.get("protocol") or "").startswith(_METERED_PROTOCOL_PREFIXES)
    ]


EXTRACT = "extract"
DOWNLOAD = "download"


def _flag_value(arguments: list[str], flag: str) -> str | None:
    for current, value in itertools.pairwise(arguments):
        if current == flag:
            return value
    return None


@dataclass(frozen=True)
class CallSite:
    """One place where the repository hands work to yt-dlp.

    A repository usually has several, and they do not all have the same
    obligations: a site that only extracts metadata cannot be required
    to meter a transfer, and a site that downloads cannot be excused
    from it. Declaring them separately is what keeps the checks honest
    rather than approximately right.

    ``name`` appears in the assertion message, so make it the name of
    the function it comes from. Exactly one of ``options`` and
    ``arguments`` is set: ``options`` for yt-dlp driven in-process,
    ``arguments`` for the full argv of a subprocess.
    """

    name: str
    purpose: str
    options: dict[str, Any] | None = None
    arguments: list[str] | None = None

    def __post_init__(self) -> None:
        if self.purpose not in {EXTRACT, DOWNLOAD}:
            raise ValueError(
                f"{self.name}: purpose must be {EXTRACT!r} or {DOWNLOAD!r}"
            )
        if (self.options is None) == (self.arguments is None):
            raise ValueError(f"{self.name}: set exactly one of options and arguments")

    @property
    def drives_in_process(self) -> bool:
        return self.options is not None

    @property
    def format_sort(self) -> list[str] | None:
        """The ``format_sort`` this call site applies, if any.

        Derived rather than declared, so the adapter cannot hand the
        selector check a different sort from the one the code uses. It
        matters because sort decides which of several matching formats
        wins, and protocol is only a tiebreak within it.
        """
        if self.options is not None:
            sort = self.options.get("format_sort")
            return list(sort) if sort else None
        value = _flag_value(self.arguments or [], "-S") or _flag_value(
            self.arguments or [], "--format-sort"
        )
        return value.split(",") if value else None


def _repository_root() -> Path:
    """The nearest ancestor directory holding a ``pyproject.toml``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    pytest.fail(
        "no pyproject.toml above this test, so the conformance profile "
        "cannot be located; the test was copied to the wrong place"
    )


def _load_profile() -> dict[str, Any]:
    manifest = _repository_root() / "pyproject.toml"
    with manifest.open("rb") as handle:
        data = tomllib.load(handle)
    profile = data.get("tool", {}).get("ytdlp_conformance")
    if profile is None:
        pytest.fail(
            f"{manifest} declares no [tool.ytdlp_conformance]; every key "
            "is mandatory and none has a default, by design"
        )
    return profile


@pytest.fixture(scope="session")
def profile() -> dict[str, Any]:
    return _load_profile()


@pytest.fixture(scope="session")
def waivers(profile: dict[str, Any]) -> dict[str, str]:
    return profile.get("waivers", {})


# Which aspect each test stands for. Kept in one place rather than
# spread over decorators so that the mapping is reviewable as a whole,
# and so that an aspect listed as covered with no test behind it fails
# a test of its own below. Tests not listed here are about the profile
# or the mechanism rather than an invariant, and always run.
ASPECT_OF_TEST = {
    "test_the_pin_is_present_and_not_empty": "C1",
    "test_the_pin_does_not_enable_the_generic_extractor": "C1",
    "test_the_rejection_matrix_can_actually_fail": "C1",
    "test_every_call_site_passes_the_declared_pin": "C1",
    "test_every_declared_acceptance_is_real": "C1",
    "test_every_declared_rejection_is_real": "C1",
    "test_the_coupling_is_declared_one_way_or_the_other": "C2",
    "test_the_solvers_match_the_installed_yt_dlp": "C2",
    "test_absent_is_asserted_in_reverse": "C2",
    "test_the_declaration_is_internally_consistent": "C3",
    "test_a_host_provided_runtime_is_actually_present": "C3",
    "test_no_vector_survives": "C4",
    "test_neutralising_keeps_the_legible_text": "C4",
    "test_an_accepted_url_survives_the_rebuild": "C6",
    "test_a_rejected_url_reaches_no_extractor": "C6",
    "test_the_ceiling_is_a_positive_number": "C7.metered",
    "test_every_downloading_site_rejects_unmeterable_formats": "C7.selectable",
    "test_no_downloading_selector_falls_back_to_an_unmeterable_format": "C7.selectable",
    "test_a_mixed_selection_is_judged_by_its_components": "C7.selectable",
    "test_the_pool_can_exercise_a_height_limited_selector": "C7.selectable",
    "test_every_in_process_downloading_site_meters_mid_transfer": "C7.metered",
    "test_a_metering_hook_lets_a_transfer_within_the_ceiling_through": "C7.metered",
    "test_a_hook_that_rewrites_the_event_defeats_the_one_after_it": "C7.metered",
    "test_a_subprocess_download_declares_its_unverifiable_bound": "C7.external",
    "test_the_declared_timeout_is_positive": "C8.socket",
    "test_every_call_site_sets_the_declared_timeout": "C8.socket",
    "test_every_in_process_downloading_site_sets_retries_explicitly": "C8.retries",
    "test_no_call_site_disables_extraction_retry": "C8.retries",
    "test_every_in_process_site_supplies_a_logger": "C9",
    "test_no_call_site_silences_warnings": "C9",
    "test_any_marker_coupling_names_the_version_it_was_checked_against": "C10",
}


@pytest.fixture(autouse=True)
def _honour_waivers(request: pytest.FixtureRequest, waivers: dict[str, str]) -> None:
    """Skip the tests of a waived aspect, by name and with the reason."""
    aspect = ASPECT_OF_TEST.get(request.function.__name__)
    if aspect is not None:
        _require_not_waived(waivers, aspect)


def _require_not_waived(waivers: dict[str, str], aspect: str) -> None:
    """Skip when ``aspect`` is waived, loudly and by name.

    A waiver is the visible, countable alternative to deleting a test:
    it still has to cite an issue, it still shows up in the profile,
    and the reason travels into the test report rather than leaving a
    silent gap.

    Called explicitly at the top of each test rather than applied by an
    autouse fixture off a class attribute, because the gate is
    per-aspect while C7 and C8 each span several in one class.
    """
    reason = waivers.get(aspect)
    if reason is not None:
        pytest.skip(f"{aspect} waived: {reason}")


@pytest.fixture(scope="session")
def call_sites(ytdlp_subject: Any) -> list[CallSite]:
    sites = list(ytdlp_subject.call_sites())
    if not sites:
        pytest.fail(
            "the adapter declared no call sites, so nothing about how "
            "this repository drives yt-dlp can be checked"
        )
    return sites


def _sites_for(sites: list[CallSite], purpose: str) -> list[CallSite]:
    return [site for site in sites if site.purpose == purpose]


def _in_process(sites: list[CallSite]) -> list[CallSite]:
    return [site for site in sites if site.drives_in_process]


def _as_subprocess(sites: list[CallSite]) -> list[CallSite]:
    return [site for site in sites if not site.drives_in_process]


def _enabled_extractors(spec: list[str]) -> list[Any]:
    """The extractor instances ``spec`` leaves enabled, in order."""
    ydl = yt_dlp.YoutubeDL(
        {"allowed_extractors": spec, "quiet": True, "no_warnings": True}
    )
    if not hasattr(ydl, "_ies"):
        pytest.fail(
            "yt_dlp.YoutubeDL no longer exposes _ies, so the extractor "
            "matrix cannot be checked; this test needs rewriting against "
            "whatever replaced it"
        )
    return list(ydl._ies.values())


def _extractors_matching(spec: list[str], url: str) -> list[str]:
    """Every extractor name that would accept ``url`` under ``spec``.

    Built from the real registry. ``extract_info`` is not usable here:
    once an extractor matches it issues actual HTTP requests, and this
    test has to stay offline.

    ``generic`` is **not** filtered out. An earlier version did, on the
    grounds that it matches almost any URL, and that quietly defeated
    the rejection matrix: with a pin of ``["youtube.*", "generic"]``,
    the malicious-authority URL and the link-local metadata address
    both match ``generic`` and the filtered helper returned ``[]``, so
    every rejection passed. The pin not enabling ``generic`` in the
    first place is checked separately, as its own assertion rather than
    as a silent assumption inside a helper.
    """
    return [ie.IE_NAME for ie in _enabled_extractors(spec) if ie.suitable(url)]


def _expected_ejs_version() -> str:
    requirements = metadata.metadata("yt-dlp").get_all("Requires-Dist") or []
    for requirement in requirements:
        match = PINNED_EJS_REQUIREMENT.match(requirement)
        if match:
            return match.group("version")
    pytest.fail(
        "the installed yt-dlp pins no yt-dlp-ejs version in a 'pin' or "
        "'default' extra; its extras may have been restructured, so this "
        "check needs revisiting rather than relaxing"
    )


def _format_selectors(site: CallSite) -> list[str]:
    """Every format selector string this call site hands to yt-dlp."""
    if site.options is not None:
        selector = site.options.get("format")
        return [] if selector is None else [str(selector)]
    arguments = site.arguments or []
    return [
        value
        for flag, value in itertools.pairwise(arguments)
        if flag in {"-f", "--format"}
    ]


class TestTheProfileIsComplete:
    def test_every_mandatory_key_is_declared(self, profile: dict[str, Any]) -> None:
        # No key has a default. Adding one upstream therefore turns all
        # three repositories red at once instead of passing silently in
        # the two that did not notice.
        missing = sorted(REQUIRED_PROFILE_KEYS - profile.keys())
        assert not missing, f"profile is missing {missing}"

    def test_no_unrecognised_key_is_declared(self, profile: dict[str, Any]) -> None:
        # Catches a typo, which would otherwise read as a key left at
        # no value at all.
        allowed = REQUIRED_PROFILE_KEYS | OPTIONAL_PROFILE_KEYS
        unknown = sorted(profile.keys() - allowed)
        assert not unknown, f"profile declares unknown keys {unknown}"

    def test_each_waiver_names_an_aspect_and_an_issue(
        self, waivers: dict[str, str]
    ) -> None:
        # A waiver is the visible, countable alternative to omitting a
        # test. One that names neither the invariant nor a ticket is an
        # omission with extra steps.
        for aspect, reason in waivers.items():
            assert aspect in SPEC_ASPECTS, (
                f"waiver key {aspect!r} is not an aspect the spec defines; "
                f"it has to be one of {sorted(SPEC_ASPECTS)}"
            )
            assert "#" in reason, f"waiver for {aspect} cites no issue: {reason!r}"


class TestEveryAspectIsAccountedFor:
    def test_nothing_is_silently_uncovered(self, waivers: dict[str, str]) -> None:
        # The spec's rule, made mechanical: an aspect with no test and
        # no waiver is a failure, not a gap. Accounting at aspect
        # granularity is what stops a partial check from clearing a
        # whole invariant -- C7 used to count as covered by a check
        # that only asserted a positive ceiling and a present hook.
        unaccounted = sorted(SPEC_ASPECTS - COVERED_ASPECTS - waivers.keys())
        assert not unaccounted, (
            f"{unaccounted} have neither a test in this file nor a waiver "
            "in the profile"
        )

    def test_every_covered_aspect_has_a_test_behind_it(self) -> None:
        # Guards the claim COVERED_ASPECTS makes. Without this, adding
        # an aspect to that set is enough to stop anyone having to
        # waive it, which is the same overstatement in a new place.
        claimed = COVERED_ASPECTS - set(ASPECT_OF_TEST.values())
        assert not claimed, (
            f"{sorted(claimed)} are listed as covered but no test in "
            "ASPECT_OF_TEST stands for them"
        )

    def test_no_test_claims_an_unknown_aspect(self) -> None:
        unknown = sorted(set(ASPECT_OF_TEST.values()) - SPEC_ASPECTS)
        assert not unknown, f"ASPECT_OF_TEST names aspects the spec does not: {unknown}"


class TestTheSpecVersionIsCoupled:
    def test_the_profile_tracks_this_copy_of_the_spec(
        self, profile: dict[str, Any]
    ) -> None:
        # The red is the migration notice: a spec bump breaks every
        # repository until someone reads what changed.
        assert profile["spec_version"] == SPEC_VERSION


class TestTheExtractorPinHolds:
    def test_the_pin_is_present_and_not_empty(self, profile: dict[str, Any]) -> None:
        assert profile["extractor_spec"], (
            "an empty extractor spec falls back to GenericIE, which "
            "follows off-site redirects"
        )

    def test_the_pin_does_not_enable_the_generic_extractor(
        self, profile: dict[str, Any]
    ) -> None:
        # C1 exists to keep extraction off GenericIE, so a pin that
        # enables it defeats the invariant no matter how precise the
        # rest of the pin is. Checked by name against the real
        # registry, because a spec entry can enable it without the
        # string "generic" appearing in the profile.
        enabled = [ie.IE_NAME for ie in _enabled_extractors(profile["extractor_spec"])]
        assert "generic" not in enabled, (
            f"{profile['extractor_spec']} leaves the generic extractor "
            "enabled, so a URL no YouTube extractor claims is still "
            "fetched, and off-site redirects are followed"
        )

    def test_the_rejection_matrix_can_actually_fail(
        self, profile: dict[str, Any]
    ) -> None:
        # A guard on the check above rather than on the repository: it
        # proves the matrix is sensitive to generic being enabled, so
        # that a helper filtering generic back out cannot pass
        # unnoticed. If this ever stops holding, the rejections below
        # are no longer evidence of anything.
        rejections = profile["rejects_urls"]
        if not rejections:
            pytest.skip("profile declares no rejections")
        with_generic = [*profile["extractor_spec"], "generic"]
        caught = [url for url in rejections if _extractors_matching(with_generic, url)]
        assert caught, (
            "no declared rejection matches anything even with generic "
            "enabled, so the rejection matrix cannot fail and is not "
            "testing the pin"
        )

    def test_every_call_site_passes_the_declared_pin(
        self, profile: dict[str, Any], call_sites: list[CallSite]
    ) -> None:
        # The profile is a declaration; this is the part that checks
        # the code actually carries it, at every site rather than at
        # the one that was easiest to remember.
        expected = list(profile["extractor_spec"])
        for site in call_sites:
            if site.options is not None:
                assert list(site.options.get("allowed_extractors") or []) == expected, (
                    f"{site.name} does not pass the declared extractor pin"
                )
            else:
                value = _flag_value(site.arguments or [], "--use-extractors")
                assert value is not None, (
                    f"{site.name} passes no --use-extractors, so yt-dlp's "
                    "whole extractor registry is reachable"
                )
                assert value.split(",") == expected, (
                    f"{site.name} passes --use-extractors {value!r}, which "
                    f"is not the declared {expected}"
                )

    def test_every_declared_acceptance_is_real(self, profile: dict[str, Any]) -> None:
        spec = profile["extractor_spec"]
        for url in profile["accepts_urls"]:
            assert _extractors_matching(spec, url), (
                f"{url} is declared accepted but no pinned extractor "
                f"matches it under {spec}"
            )

    def test_every_declared_rejection_is_real(self, profile: dict[str, Any]) -> None:
        spec = profile["extractor_spec"]
        for url in profile["rejects_urls"]:
            matched = _extractors_matching(spec, url)
            assert not matched, (
                f"{url} is declared rejected but {matched} accepts it under {spec}"
            )


class TestTheChallengeSolversMatch:
    def test_the_coupling_is_declared_one_way_or_the_other(
        self, profile: dict[str, Any]
    ) -> None:
        assert profile["ejs_coupling"] in {"pinned", "absent"}

    def test_the_solvers_match_the_installed_yt_dlp(
        self, profile: dict[str, Any]
    ) -> None:
        if profile["ejs_coupling"] != "pinned":
            pytest.skip("profile declares ejs_coupling = absent")
        # Regenerating a lock after a yt-dlp bump can leave the scripts
        # behind, and a mismatched pair fails at runtime, not here,
        # unless something checks it.
        assert metadata.version("yt-dlp-ejs") == _expected_ejs_version()

    def test_absent_is_asserted_in_reverse(self, profile: dict[str, Any]) -> None:
        if profile["ejs_coupling"] != "absent":
            pytest.skip("profile declares ejs_coupling = pinned")
        # "Not needed" has to be checked too, or the package arrives as
        # a transitive dependency at a mismatched version and the
        # declaration quietly becomes false.
        with pytest.raises(metadata.PackageNotFoundError):
            metadata.version("yt-dlp-ejs")


class TestTheJsRuntimeIsDeclared:
    def test_the_declaration_is_internally_consistent(
        self, profile: dict[str, Any]
    ) -> None:
        runtime = profile["js_runtime"]
        provider = profile["js_runtime_provided_by"]
        assert provider in {"image", "host", "none"}
        if runtime == "absent":
            assert provider == "none", "no runtime cannot have a provider"
        else:
            assert provider != "none", (
                f"{runtime} is declared but nothing is said to provide it"
            )

    def test_a_host_provided_runtime_is_actually_present(
        self, profile: dict[str, Any]
    ) -> None:
        if profile["js_runtime_provided_by"] != "host":
            pytest.skip(
                "runtime is not host-provided; provider is "
                f"{profile['js_runtime_provided_by']}"
            )
        runtime = profile["js_runtime"]
        assert shutil.which(runtime), (
            f"{runtime} is declared host-provided but is not on PATH; "
            "availability has to be declared, not discovered"
        )


class TestControlCharactersAreNeutralised:
    def test_no_vector_survives(self, ytdlp_subject: Any) -> None:
        for vector in CONTROL_CHARACTER_VECTORS:
            cleaned = ytdlp_subject.neutralize(f"title{vector}INJECTED")
            assert vector not in cleaned, (
                f"{vector!r} survived neutralisation: {cleaned!r}"
            )

    def test_neutralising_keeps_the_legible_text(self, ytdlp_subject: Any) -> None:
        # A neutraliser that returns "" passes the test above and is
        # useless, so pin the other side of it too.
        cleaned = ytdlp_subject.neutralize(f"a{chr(0x2028)}b")
        assert "a" in cleaned
        assert "b" in cleaned


class TestUrlsAreCanonicalisedBeforeYtDlp:
    def test_an_accepted_url_survives_the_rebuild(
        self, profile: dict[str, Any], ytdlp_subject: Any
    ) -> None:
        spec = profile["extractor_spec"]
        for url in profile["accepts_urls"]:
            rebuilt = ytdlp_subject.canonicalize(url)
            assert rebuilt is not None, f"{url} was declared accepted"
            # The cross-check with C1: a rebuild producing something the
            # pin rejects is worse than no rebuild, since it turns an
            # accepted URL into a GenericIE candidate.
            assert _extractors_matching(spec, rebuilt), (
                f"{url} rebuilt to {rebuilt}, which the pin rejects"
            )

    def test_a_rejected_url_reaches_no_extractor(
        self, profile: dict[str, Any], ytdlp_subject: Any
    ) -> None:
        # The rebuild and the pin are two layers, and the invariant is
        # about the pair: a rejected URL must be stopped by one of
        # them. Requiring the rebuild alone to stop everything would be
        # wrong as well as unmet -- measured in Youtube-Downloader,
        # whose URL check passes "youtube.com/about/" through on
        # purpose, because it accepts the URL as typed, while the pin
        # matches no extractor for it.
        spec = profile["extractor_spec"]
        for url in profile["rejects_urls"]:
            rebuilt = ytdlp_subject.canonicalize(url)
            if rebuilt is None:
                continue
            matched = _extractors_matching(spec, rebuilt)
            assert not matched, (
                f"{url} is declared rejected but rebuilt to {rebuilt}, "
                f"which {matched} accepts"
            )


class TestTheByteCeilingIsDeclaredAndReachable:
    def test_the_ceiling_is_a_positive_number(self, profile: dict[str, Any]) -> None:
        assert profile["max_bytes"] > 0

    def test_every_downloading_site_rejects_unmeterable_formats(
        self, call_sites: list[CallSite]
    ) -> None:
        # C7.selectable, checked by evaluating each selector against an
        # offline pool in which the unmeterable renditions are the
        # better ones. Every branch of the selector is exercised,
        # because the pool offers something for each.
        for site in _sites_for(call_sites, DOWNLOAD):
            selectors = _format_selectors(site)
            assert selectors, f"{site.name} downloads but exposes no selector"
            for selector in selectors:
                try:
                    chosen = _selected_formats(
                        selector, site.format_sort, _mixed_pool()
                    )
                except Exception as error:  # noqa: BLE001
                    pytest.fail(
                        f"{site.name} selector {selector!r} matched nothing "
                        "in a pool offering meterable and unmeterable "
                        "renditions of every shape, so it could not be "
                        f"exercised: {type(error).__name__}"
                    )
                assert chosen, (
                    f"{site.name} selector {selector!r} matched nothing in "
                    "the conformance pool, so it could not be exercised"
                )
                unmetered = _unmetered(chosen)
                assert not unmetered, (
                    f"{site.name} selector {selector!r} preferred "
                    f"{unmetered} over the meterable renditions on offer; "
                    "no Python-side downloader meters those protocols, so "
                    "the byte ceiling cannot hold on them"
                )

    def test_the_pool_can_exercise_a_height_limited_selector(self) -> None:
        # A guard on the pool. It once held 1080p only, so a correctly
        # constrained 720p or 480p selector matched nothing and was
        # reported as a failure for want of a format rather than for
        # anything it did. Any repository selecting on height would
        # have been unable to pass.
        for height in _PROBE_HEIGHTS:
            selector = (
                f"bestvideo[ext=mp4][height<={height}][protocol^=http]"
                f"/best[ext=mp4][height<={height}][protocol^=http]"
            )
            chosen = _selected_formats(selector, None, _mixed_pool())
            assert chosen, (
                f"the pool offers nothing a {height}p selector can match, so "
                "a tier-limited selector cannot be judged on its merits"
            )
            assert not _unmetered(chosen), (
                f"a fully constrained {height}p selector picked "
                f"{_unmetered(chosen)}, so the pool or the check is wrong"
            )

    def test_a_mixed_selection_is_judged_by_its_components(self) -> None:
        # A guard on the checker, not on the repository. A video+audio
        # selection arrives as one merged entry whose own protocol is
        # the parts joined -- "https+m3u8_native" -- which begins with
        # "http" and so passed a check that read only that field. If
        # this stops holding, every mixed selector above is being
        # waved through.
        selector = "bestvideo[protocol^=http]+bestaudio/best[protocol^=http]"
        chosen = _selected_formats(selector, None, _mixed_pool())
        assert chosen, "the pool could not exercise a mixed selector"
        assert _unmetered(chosen), (
            f"{selector!r} selects an HLS audio track alongside an HTTPS "
            "video stream, and the check did not notice; it is reading the "
            "merged entry's protocol instead of its components"
        )

    def test_no_downloading_selector_falls_back_to_an_unmeterable_format(
        self, call_sites: list[CallSite]
    ) -> None:
        # The other half of C7.selectable, and the one a single pool
        # misses: a selector whose first branch is constrained and
        # whose fallback is not passes against the mixed pool, because
        # the fallback is never reached there. With nothing meterable
        # on offer, a correctly constrained selector must match
        # nothing at all.
        for site in _sites_for(call_sites, DOWNLOAD):
            for selector in _format_selectors(site):
                try:
                    chosen = _selected_formats(
                        selector, site.format_sort, _unmeterable_only_pool()
                    )
                except Exception:  # noqa: BLE001
                    # Nothing matched, which is the required outcome.
                    continue
                unmetered = _unmetered(chosen)
                assert not unmetered, (
                    f"{site.name} selector {selector!r} fell back to "
                    f"{unmetered} when no meterable format was offered; a "
                    "branch of it is unconstrained, and refusing the "
                    "download is the required outcome"
                )

    def test_every_in_process_downloading_site_meters_mid_transfer(
        self, profile: dict[str, Any], call_sites: list[CallSite]
    ) -> None:
        # C7.metered. max_filesize is not a ceiling:
        # downloader/http.py consults it once, before the transfer,
        # only when Content-Length is known, and returns False rather
        # than raising.
        #
        # The hooks are driven rather than counted, and driven as a
        # set: downloader/common.py:488-495 calls every hook in order
        # with the same status object, so one of them raising is what
        # stops the transfer. Requiring each hook to raise individually
        # rejects a perfectly good [observer, guard] pair.
        sites = _in_process(_sites_for(call_sites, DOWNLOAD))
        if not sites:
            pytest.skip("no in-process downloading site to check")
        ceiling = profile["max_bytes"]
        for site in sites:
            assert site.options is not None
            hooks = site.options.get("progress_hooks")
            assert hooks, (
                f"{site.name} has no progress_hooks, so nothing observes "
                "the transfer while it runs and max_filesize alone is not "
                "a ceiling"
            )
            for label, status in _oversized_events(ceiling):
                raised = _drive_hooks(hooks, status)
                assert raised is not None, (
                    f"{site.name} ran its progress hooks to completion on "
                    f"{label} and did not stop the transfer"
                )
                assert not isinstance(raised, AssertionError), (
                    f"{site.name}'s progress hooks failed an assertion of "
                    f"their own on {label} rather than stopping the transfer"
                )

    def test_a_metering_hook_lets_a_transfer_within_the_ceiling_through(
        self, profile: dict[str, Any], call_sites: list[CallSite]
    ) -> None:
        # The other side of it. Hooks that raise on everything would
        # pass the test above and break every download.
        sites = _in_process(_sites_for(call_sites, DOWNLOAD))
        if not sites:
            pytest.skip("no in-process downloading site to check")
        ceiling = profile["max_bytes"]
        for site in sites:
            assert site.options is not None
            hooks = site.options.get("progress_hooks") or ()
            within = {
                "status": "downloading",
                "filename": "conformance.part",
                "downloaded_bytes": max(ceiling - 1, 0),
                "total_bytes": ceiling,
            }
            raised = _drive_hooks(hooks, within)
            assert raised is None, (
                f"{site.name} stopped a transfer inside its ceiling: {raised!r}"
            )

    def test_a_hook_that_rewrites_the_event_defeats_the_one_after_it(
        self,
    ) -> None:
        # A guard on the checker, not on the repository. yt-dlp hands
        # one status object to every hook and says at
        # downloader/common.py:489-493 that the sharing is deliberate,
        # so a hook that rewrites the event changes what the next one
        # sees. Driving the hooks with a copy each hid that: the pair
        # below was reported as stopping a transfer it cannot stop. If
        # this stops holding, every hook set above is being judged
        # against a yt-dlp that does not exist.
        ceiling = 1024

        def clear(status: dict[str, Any]) -> None:
            status["downloaded_bytes"] = 0
            status["total_bytes"] = 0

        def guard(status: dict[str, Any]) -> None:
            if (status.get("downloaded_bytes") or 0) > ceiling:
                raise ValueError("over the ceiling")

        for label, event in _oversized_events(ceiling):
            assert _drive_hooks([guard], event) is not None, (
                f"the guard alone did not stop {label}, so the pair below "
                "proves nothing"
            )
            assert _drive_hooks([clear, guard], event) is None, (
                f"a hook that zeroes the event did not defeat the guard "
                f"after it on {label}; the hooks are being driven with a "
                "copy each, and yt-dlp does not"
            )
            assert event["downloaded_bytes"] > ceiling, (
                f"driving the hooks mutated the caller's {label} event, so "
                "one hook set can no longer be compared against another"
            )

    def test_a_subprocess_download_declares_its_unverifiable_bound(
        self, waivers: dict[str, str], call_sites: list[CallSite]
    ) -> None:
        # C7.external. A subprocess takes its options as argv and
        # reports nothing back, so the shared test cannot watch the
        # transfer or measure what landed. Before this existed, such a
        # site made the C7.metered check skip while C7.metered still
        # counted as covered -- so a repository whose every download
        # shelled out had no byte bound anywhere and a green suite.
        # Measured on a CLI-only adapter with no bound at all:
        # 29 passed, 6 skipped.
        #
        # Deliberately not checked instead: --max-filesize in the argv.
        # It is consulted once, pre-transfer, only with a known
        # Content-Length, and returns False rather than raising, so
        # asserting its presence would record conformance that is not
        # there. A waiver records the gap instead.
        external = [
            site.name for site in _as_subprocess(_sites_for(call_sites, DOWNLOAD))
        ]
        if not external:
            pytest.skip("every download is driven in-process")
        assert "C7.external" in waivers, (
            f"{external} download through a subprocess, whose byte bound no "
            "shared test can observe. Waive C7.external with an issue "
            "naming how the bound is enforced, or drive the download "
            "in-process so C7.metered can check it."
        )


class TestTheTimeBoundIsExplicit:
    def test_the_declared_timeout_is_positive(self, profile: dict[str, Any]) -> None:
        assert profile["socket_timeout_seconds"] > 0

    def test_every_call_site_sets_the_declared_timeout(
        self, profile: dict[str, Any], call_sites: list[CallSite]
    ) -> None:
        expected = profile["socket_timeout_seconds"]
        for site in call_sites:
            if site.options is not None:
                # Leaving it unset is not "no timeout" --
                # networking/common.py applies DEFAULT_TIMEOUT = 20 --
                # but it is undeclared, and an undeclared value is one
                # nobody notices changing.
                assert site.options.get("socket_timeout") == expected, (
                    f"{site.name} does not set the declared socket_timeout"
                )
            else:
                value = _flag_value(site.arguments or [], "--socket-timeout")
                assert value == str(expected), (
                    f"{site.name} passes --socket-timeout {value!r}, not "
                    f"the declared {expected}"
                )

    def test_every_in_process_downloading_site_sets_retries_explicitly(
        self, call_sites: list[CallSite]
    ) -> None:
        # Download sites only. retries and fragment_retries govern the
        # media transfer (downloader/http.py:360) and default to 10
        # only in the CLI option parser (options.py:1025); a directly
        # constructed YoutubeDL gets None, which utils/_utils.py:5267
        # turns into 0, so one transient failure fails that download.
        #
        # Extraction is a different budget and is not checked here:
        # extractor_retries defaults to 3 (extractor/common.py:4072),
        # which the YouTube extractor uses in youtube/_base.py:981 and
        # :1287-1289, so an extract-only site that sets nothing still
        # gets four attempts. Requiring retries there would be
        # asserting a defect that is not present. The subprocess sites
        # get the CLI default and are not checked either.
        sites = _in_process(_sites_for(call_sites, DOWNLOAD))
        if not sites:
            pytest.skip("no in-process downloading site to check")
        for site in sites:
            assert site.options is not None
            for key in ("retries", "fragment_retries"):
                assert site.options.get(key) is not None, (
                    f"{site.name} leaves {key} unset, which means 0 on the "
                    "Python API path, not the CLI's 10"
                )

    def test_no_call_site_disables_extraction_retry(
        self, call_sites: list[CallSite]
    ) -> None:
        # The converse of the above, and the only retry requirement
        # that applies to extraction: extractor_retries is 3 unless
        # someone sets it, and setting it to 0 silently removes a retry
        # that was there by default.
        for site in _in_process(call_sites):
            assert site.options is not None
            configured = site.options.get("extractor_retries")
            if configured is None:
                continue
            assert configured > 0, (
                f"{site.name} sets extractor_retries={configured!r}, which "
                "removes the retry extraction has by default"
            )


class TestYtDlpOutputIsCaptured:
    def test_every_in_process_site_supplies_a_logger(
        self, call_sites: list[CallSite]
    ) -> None:
        # quiet does not suppress trouble(); without a logger those
        # lines go to stderr and, in a container, nowhere.
        for site in _in_process(call_sites):
            assert site.options is not None
            assert site.options.get("logger") is not None, (
                f"{site.name} passes no logger, so yt-dlp's warnings land "
                "on stderr instead of in the log"
            )

    def test_no_call_site_silences_warnings(self, call_sites: list[CallSite]) -> None:
        for site in call_sites:
            if site.options is not None:
                assert site.options.get("no_warnings") is not True, (
                    f"{site.name} sets no_warnings, which hides exactly the "
                    "lines that say the toolchain is degraded"
                )
            else:
                assert "--no-warnings" not in (site.arguments or []), (
                    f"{site.name} passes --no-warnings"
                )


class TestErrorTextCouplingIsPinned:
    def test_any_marker_coupling_names_the_version_it_was_checked_against(
        self, profile: dict[str, Any]
    ) -> None:
        verified = profile["markers_verified_against"]
        if verified == "none":
            pytest.skip("profile declares no coupling to yt-dlp's text")
        installed = metadata.version("yt-dlp")
        assert verified == installed, (
            f"error-text matching was verified against yt-dlp {verified} "
            f"but {installed} is installed; re-verify the markers rather "
            "than bumping this string"
        )


class TestThePoTokenHookIsNotHalfApplied:
    def test_the_environment_and_the_profile_agree(
        self, profile: dict[str, Any]
    ) -> None:
        hook = profile["po_token_hook"]
        assert hook in {"absent", "env"}
        if hook == "absent":
            # Catches the half-applied configuration: a token supplied
            # to a deployment whose code has no mount point for it
            # would otherwise be silently ignored.
            assert not os.environ.get("YTDLP_PO_TOKEN"), (
                "YTDLP_PO_TOKEN is set but the profile says the hook is "
                "absent, so the token is being ignored"
            )


# --- ytdlp-conformance:canonical-end ---
