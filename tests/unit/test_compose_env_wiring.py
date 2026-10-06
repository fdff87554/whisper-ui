"""Every setting an operator can set has to actually reach the containers.

compose.yml carries no `env_file`, so a value in `.env` applies only where
compose names it explicitly. A Settings field with no matching compose entry
is therefore unreachable in a deployment: the operator sets it, nothing
reads it, and nothing says so. That is how YOUTUBE_MAX_DOWNLOAD_SIZE shipped
with no way to configure or disable it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from whisper_ui.core.config import Settings

_COMPOSE = Path(__file__).resolve().parents[2] / "compose.yml"

# Settings that are deliberately not passed through compose, with the reason.
# Adding to this needs a reason, which is the point: the default is wired.
# Settings deliberately not reachable from compose, keyed by the issue or
# reason that excuses each. Empty is the healthy state: every setting an
# operator is told about should be settable where they deploy.
_NOT_WIRED: dict[str, str] = {}


def _compose_env_names() -> set[str]:
    # Environment keys in compose are the SCREAMING_SNAKE mapping entries,
    # whether they sit in an x-*-env anchor or inline under a service.
    return {name.lower() for name in re.findall(r"^\s{2,}([A-Z][A-Z0-9_]*):", _COMPOSE.read_text(), re.M)}


class TestEverySettingIsReachable:
    def test_no_setting_is_missing_from_compose(self) -> None:
        missing = sorted(set(Settings.model_fields) - _compose_env_names() - set(_NOT_WIRED))

        assert not missing, (
            "these settings cannot be configured in a deployment: "
            f"{missing}. Add them to the matching x-*-env anchor in "
            "compose.yml and to .env.example, or record why not in _NOT_WIRED."
        )

    @pytest.mark.parametrize("field", sorted(_NOT_WIRED))
    def test_exemptions_still_exist(self, field: str) -> None:
        # Stops the exemption list outliving the setting it excuses.
        assert field in Settings.model_fields


class TestTheDownloadCapIsReachable:
    def test_the_cap_is_passed_to_the_workers(self) -> None:
        # The four worker services share the x-download-env anchor; the cap
        # belongs there next to the duration cap it backstops.
        anchor = re.search(r"x-download-env: &download-env\n((?:\s{2}\S.*\n)+)", _COMPOSE.read_text())
        assert anchor, "x-download-env anchor not found"
        assert "YOUTUBE_MAX_DOWNLOAD_SIZE" in anchor.group(1)
