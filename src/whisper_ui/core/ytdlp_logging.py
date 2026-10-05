"""Route yt-dlp's own diagnostics into the application log.

``quiet`` and ``no_warnings`` do not silence yt-dlp: ``YoutubeDL.trouble()``
calls ``to_stderr()`` unconditionally, so every extraction failure writes a
raw line to the process's stderr no matter how the options are set. In a
worker that line lands in the container log outside the logging framework,
unprefixed and unstructured -- and it embeds remote-controlled text such as
the video title, which can carry newlines and ANSI escapes and therefore
forge whole log lines. Passing a ``logger`` is the only supported way to
take that channel over.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

# Longer than any real diagnostic needs to be, short enough that a crafted
# title cannot flood the log.
_MAX_MESSAGE_LENGTH = 500


# The two separators Python treats as line breaks that are not in the "C"
# category: U+2028 LINE SEPARATOR is Zl and U+2029 PARAGRAPH SEPARATOR is Zp.
# str.splitlines() breaks on both, so a log consumer that splits lines sees a
# forged line even though neither is a control character. Named explicitly
# rather than excluding all of "Z": that class also holds U+00A0 NO-BREAK
# SPACE and the other spaces, which are ordinary text in a title.
_LINE_SEPARATORS = ("\u2028", "\u2029")


def neutralise(text: str, *, max_length: int = _MAX_MESSAGE_LENGTH) -> str:
    """Collapse control characters so remote text cannot forge log lines.

    Every Unicode "C" category character -- C0/C1 controls, but also format
    characters such as U+202E RIGHT-TO-LEFT OVERRIDE -- becomes a space, as
    do the two line separators in ``_LINE_SEPARATORS`` that sit outside that
    class but still break a line for anything calling ``str.splitlines()``.
    """
    cleaned = "".join(" " if unicodedata.category(ch).startswith("C") or ch in _LINE_SEPARATORS else ch for ch in text)
    if len(cleaned) > max_length:
        return cleaned[:max_length] + "..."
    return cleaned


class YtDlpLogger:
    """Adapter matching yt-dlp's expected logger interface.

    yt-dlp calls ``debug`` for both debug and info output, distinguishing the
    two by a leading ``[debug] `` prefix on the former.
    """

    def __init__(self, target: logging.Logger) -> None:
        self._log = target

    def debug(self, msg: str) -> None:
        if msg.startswith("[debug] "):
            self._log.debug("yt-dlp: %s", neutralise(msg))
        else:
            self._log.info("yt-dlp: %s", neutralise(msg))

    def info(self, msg: str) -> None:
        self._log.info("yt-dlp: %s", neutralise(msg))

    def warning(self, msg: str) -> None:
        self._log.warning("yt-dlp: %s", neutralise(msg))

    def error(self, msg: str) -> None:
        self._log.error("yt-dlp: %s", neutralise(msg))
