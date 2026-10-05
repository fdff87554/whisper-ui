from __future__ import annotations

# String truncation lengths
ERROR_MAX_LENGTH = 1000
ERROR_DISPLAY_LENGTH = 200
MESSAGE_MAX_LENGTH = 500
STDERR_MAX_LENGTH = 500
TIMESTAMP_DISPLAY_LENGTH = 19

# List limits
DEFAULT_JOBS_PER_PAGE = 20

# Timeouts (seconds)
FFMPEG_CONVERT_TIMEOUT = 300
FFPROBE_TIMEOUT = 30
SQLITE_BUSY_TIMEOUT_MS = 5000

# Worker progress-write throttling. A callback that neither changes the
# message nor crosses these thresholds is dropped so the SQLite/Redis
# write rate stays bounded even when a stage (e.g. whisperx transcribe)
# emits fine-grained per-chunk updates. Stage transitions and the final
# progress=1.0 always flush regardless of these limits.
PROGRESS_WRITE_MIN_DELTA = 0.005  # fraction, i.e. 0.5 percentage points
PROGRESS_WRITE_MIN_INTERVAL_SEC = 0.5

# Stale job recovery
# STALE_JOB_TIMEOUT now lives in Settings.stale_job_timeout so it stays
# consistent with the dynamic job_timeout bounds.
STALE_JOB_CHECK_INTERVAL = 60  # seconds

# Batch upload limits. Shared by file uploads, URL submissions, and playlist
# expansion. Sized so a typical playlist fits in one submission; jobs beyond
# the configured worker throughput simply wait in the queue (the liveness-aware
# stale reaper spares queued work and re-arms its state TTL while it waits).
MAX_BATCH_SIZE = 100

# Upper bound on ids per bulk job action (retry / delete / export). The UI
# selects at most a page of jobs at a time, so this is far above any
# legitimate selection — it only stops a hand-crafted request from tying up
# the event loop in the per-job processing loop.
MAX_BULK_ACTION_IDS = 200

# Viewer client-side search becomes O(n) per keystroke; very large transcripts
# (multi-hour multi-speaker recordings) will lock up the browser tab. Above
# this segment count the viewer disables the live filter and tells the user to
# export TXT instead.
VIEWER_SEARCH_SEGMENT_LIMIT = 2000

# Transcript quality gate (pipeline/postprocess.py). A degenerate Whisper
# output — a hallucination loop over silence — shows up as one normalized
# segment text dominating the transcript (observed incidents sat at 98%+).
# 50% of 20+ segments is far beyond anything real speech produces while
# still catching partial collapses; tripping the gate only adds a warning
# and skips LLM correction, so a false positive costs little.
QUALITY_GATE_MIN_SEGMENTS = 20
QUALITY_GATE_REPEAT_RATIO = 0.5

# yt-dlp
YT_DLP_SOCKET_TIMEOUT = 30

# Substrings (lowercase) that mark a *transient* yt-dlp failure: the source is
# throttling or briefly down, not gone. The headline case is X throttling its
# anonymous guest-token endpoint ("Bad guest token"), which yt-dlp re-fetches
# per attempt but does not retry itself. HTTP 429 and 5xx are likewise
# server-side. The HTTP codes are matched in their "http error NNN" form so a
# video id that merely contains "503" cannot trip a false positive.
#
# Shared by the worker download path and submit-time playlist expansion so the
# two cannot disagree about what "unavailable" means; both must test this list
# *before* any permanent-failure markers, because "HTTP Error 503: Service
# Unavailable" and yt-dlp's own "Service temporarily unavailable" both contain
# the substring "unavailable".
#
# Markers are heuristic: they match yt-dlp's wording, and a release is free to
# reword. YT_DLP_MARKERS_VERIFIED_AGAINST below turns that silent decay into a
# failing test on the next version bump.
YT_DLP_TRANSIENT_MARKERS = (
    "guest token",
    "http error 429",
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "service unavailable",
    "temporarily unavailable",
)

# The yt-dlp release the marker lists above were last checked against. Bumping
# yt-dlp without re-reading its error strings is how a heuristic like this
# rots unnoticed, so test_markers_are_bound_to_the_installed_yt_dlp fails until
# someone re-checks and moves this forward in the same commit.
YT_DLP_MARKERS_VERIFIED_AGAINST = "2026.08.19"

# Ceiling on the downloaded video track's height. The pipeline only needs the
# audio; the video track is kept so the viewer can play back the original
# media. Without a ceiling yt-dlp takes the largest mp4 rendition on offer --
# measured at 2160p / 229 MiB for a 3.5 minute clip whose audio track is
# 3.3 MiB. 1080p is past what the viewer's inline player needs and cuts that
# same clip to 77 MiB.
YT_DLP_MAX_HEIGHT = 1080

# yt-dlp's default sort ranks AV1 first and YouTube serves AV1 inside mp4, so
# the [ext=mp4] filters below constrain the container but not the codec. The
# viewer plays the downloaded file in a browser <video> element, where AV1
# hardware decode coverage is far thinner than H.264's. This is a *preference*
# (-S), not a filter: a video offered only in AV1 still downloads at its own
# resolution rather than falling back to a worse progressive rendition. The
# string is the sort from yt-dlp's own "-t mp4" preset.
YT_DLP_FORMAT_SORT = (
    "vcodec:h264",
    "lang",
    "quality",
    "res",
    "fps",
    "hdr:12",
    "acodec:aac",
)

# The height ceiling repeats on every bounded fallback so a video with no
# capped adaptive rendition cannot silently arrive at full resolution. The
# final unbounded "best" is the last resort: it keeps sources that report no
# height at all (some X posts) downloadable rather than failing selection.
YT_DLP_VIDEO_FORMAT = (
    f"bestvideo[ext=mp4][height<={YT_DLP_MAX_HEIGHT}]+bestaudio[ext=m4a]"
    f"/best[ext=mp4][height<={YT_DLP_MAX_HEIGHT}]"
    f"/best[height<={YT_DLP_MAX_HEIGHT}]"
    "/best"
)

# Redis expiry (seconds)
# REDIS_PROCESSING_EXPIRY now lives in Settings.redis_processing_expiry.
REDIS_COMPLETED_EXPIRY = 86400  # 24 hours
REDIS_FAILED_EXPIRY = 86400  # 24 hours

# Pipeline TTLs at a glance (three layers, longest to shortest):
#   PIPELINE_STATE_TTL_SECONDS (here, 7d)
#       Safety net for per-pipeline state (context HSET, generation
#       counter, sub-job sets). Only matters when a worker is killed
#       between bumping the generation counter and writing the
#       finalizer; a successful/failed pipeline always deletes these
#       keys explicitly. Under the liveness-aware stale reaper a job can stay
#       PROCESSING (queued behind a slow worker) far longer than any single
#       stage, so this must outlive the longest plausible BACKLOG WAIT, not
#       just the longest stage — otherwise is_pipeline_dead reads the expired
#       generation counter as "no live work" and reaps a healthy queued job.
#   Settings.redis_processing_expiry (core/config.py, default ~8.5h)
#       TTL on the per-job progress HSET emitted by RedisProgressReporter
#       during a running job, sized to outlive job_timeout_max.
#   progress._DEFAULT_PROCESSING_TTL (worker/progress.py, 2h)
#       Fallback when callers construct a reporter without going through
#       Settings (unit tests, ad-hoc scripts).
PIPELINE_STATE_TTL_SECONDS = 604_800  # 7 days (must outlive max backlog wait)

# Worker queues. Stages are partitioned across these queues by the resource
# they consume so a long-running stage never blocks an unrelated worker from
# picking up the next job.
#   whisper:io  -> network / disk IO (download, preprocess)
#   whisper:gpu -> GPU inference (transcribe_align, diarize)
#   whisper:cpu -> lightweight CPU finalisation (assign_speakers, postprocess)
#   whisper:llm -> optional Ollama LLM correction. Kept separate from whisper:io
#                  so a worker bound to only io+cpu does not pick up (and block
#                  on) a slow LLM; bind whisper:llm to a dedicated worker to
#                  isolate it from the fast io/cpu path.
# The literal "default" queue appears in every compose WORKER_QUEUES value so
# an operator can drop ad-hoc maintenance jobs on every worker without
# learning the resource-class names; no Python constant refers to it.
WORKER_QUEUE_IO = "whisper:io"
WORKER_QUEUE_GPU = "whisper:gpu"
WORKER_QUEUE_CPU = "whisper:cpu"
WORKER_QUEUE_LLM = "whisper:llm"
