"""Counts the collected items that carry an ``xfail`` marker.

Loaded by ``check-conformance-drift.sh`` with ``-p``, and writes its
count to the path in ``CONFORMANCE_XFAIL_COUNT_FILE``.

Why a plugin, rather than the script reading pytest's output: the
count then depends on neither how the test path was spelled nor how
verbose pytest happened to be. An earlier version grepped the listing
for lines beginning with the path it had been given. Measured on a
module with 35 ``xfail`` items, it reported all 35 for a plain relative
path and ``0`` for each of ``./tests/...``, an absolute path, and the
same path under ``PYTEST_ADDOPTS=-q`` -- exiting 0 every time.

Why not a test inside the conformance module: a module-level
``pytestmark`` marks that test too, so its failure is reported as an
``xfail`` and the run stays green. A check cannot police the module it
lives in. Hooks defined in a test module are not called either.

Why ``pytest_collection_finish`` and not ``pytest_collection_modifyitems``:
a ``conftest.py`` may add markers from its own
``pytest_collection_modifyitems``, and whether that runs before or
after this one is a question of hook ordering. Measured on a module of
3 items marked from a ``conftest.py``:

    conftest hook       modifyitems   collection_finish
    plain                   3                 3
    @hookimpl(tryfirst)     3                 3
    @hookimpl(trylast)      0                 3
    no marker               0                 0

A plain conftest hook happens to run first -- pluggy calls the
last-registered implementation first, and a ``conftest.py`` is
registered after a ``-p`` plugin -- so the counts agreed until
``trylast=True`` put the markers on after the count was taken.
``pytest_collection_finish`` is called once collection is complete, so
no ordering within ``pytest_collection_modifyitems`` can reach it.

What this still cannot see: a marker added at run time, as from
``pytest_runtest_setup``. A collect-only run never gets there --
measured at ``0`` while the real run reported ``3 xpassed`` -- and
counting those would mean running the suite, which this check
deliberately does not do.

Reading the collected markers is also what makes an aliased import, a
module-level ``pytestmark`` and a marker added from a ``conftest.py``
all count the same.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

COUNT_FILE_ENV = "CONFORMANCE_XFAIL_COUNT_FILE"


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write the number of collected items marked ``xfail``.

    Every item in ``session.items`` belongs to the one path the script
    passed, so there is nothing to filter and no path to compare
    against -- which is the whole reason the spelling of that path can
    no longer change the answer.
    """
    destination = os.environ.get(COUNT_FILE_ENV)
    if not destination:
        return
    marked = [item for item in session.items if item.get_closest_marker("xfail") is not None]
    with Path(destination).open("w", encoding="utf-8") as handle:
        handle.write(f"{len(marked)}\n")
        for item in marked:
            handle.write(f"{item.nodeid}\n")
