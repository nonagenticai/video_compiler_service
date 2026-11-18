"""Per-pod reaper for crash-orphaned compile jobs.

A render is a fire-and-forget ``asyncio`` task that dies with its pod, leaving the
``compile_jobs`` row stuck ``processing`` (or ``interrupted`` once the next pod's
startup reclaim flips it). This service runs a periodic loop that reclaims such rows
across ALL tenants via the SECURITY DEFINER ``reclaim_next_compile_job`` function
(``FOR UPDATE SKIP LOCKED`` so concurrent reapers never grab the same job) and re-runs
them on THIS pod from the persisted ``job_options`` — under a fresh lease + heartbeat.

The reclaim is lease-gated: a ``processing`` row is only reclaimed once its lease has
lapsed (its owning pod stopped heartbeating ⇒ died), so a render actively running on a
live pod is never reclaimed under itself. ``interrupted`` rows (declared orphaned by the
startup reclaim / shutdown drain) are reclaimed immediately. Jobs past the attempt cap
or lifetime bound are failed as poison inside the reclaim function.

The same loop also drives periodic ``prune`` so a long-lived pod does not accumulate
terminal rows (the pre-reaper code pruned only once at startup, relying on pods
restarting every deploy).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge

from src.config import get_settings
from src.database.session import get_db_session
from src.services import compile_job_store as job_store

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Bound per-sweep reclaims so one pod can't monopolise a large backlog and the loop
# yields between sweeps.
_MAX_RECLAIMS_PER_SWEEP = 50

# A sweep that fails EVERY time is invisible by construction: the loop swallows the
# exception, retries next interval, and the only evidence is a repeating traceback nobody
# reads — while orphaned renders are silently never reclaimed. (The realistic cause is not
# a blip: reclaim_next calls a SQL function whose body every migration reinstalls, so a
# partial/pinned upgrade can leave a body referencing a column a later migration adds, and
# then EVERY sweep raises.) Escalate once the failures have persisted long enough that a
# transient DB/pgbouncer blip is excluded: at the default 30s interval, 10 consecutive
# failures ≈ 5 minutes of continuous unavailability, which is well past any restart or
# failover window and squarely a standing fault. Re-log every further 10 failures (≈5min)
# rather than every sweep — a per-sweep critical is the same noise at a louder level.
_ESCALATE_EVERY_N_FAILURES = 10

# The same sweep-health state as the liveness body, on the surface that can actually be
# collected. The health fields ride `/health/live`, whose body the kubelet DISCARDS — they
# are only reachable by a human running `kubectl exec`, which is a debugging aid, not
# observability. `/metrics` is already mounted over the prometheus_client default REGISTRY
# (`src/asgi.py`), so this is the one surface with a wiring path to alerting. (The pod is
# not currently scraped — its manifest carries no `prometheus.io/scrape` annotation and
# both cluster jobs are `action: keep` gated on it — but adding that annotation is a
# one-line gitops change against a metric that exists, whereas alerting on a discarded
# probe body is not possible at all.)
#
# Naming follows the fleet convention set by `w2v_common.observability.metrics`: `w2v_`
# prefix, module-scope registration against the default REGISTRY at import, and NO
# `service` label — the in-cluster scrape stamps `kubernetes_name` on every series.
#
# Gauge AND counter, because they answer different questions and neither substitutes for
# the other: the gauge is the alertable one ("is the reaper reclaiming nothing RIGHT NOW",
# a level that returns to 0 on recovery), while the counter survives the reset and so
# still shows a flapping reaper that never reaches the escalation threshold — a pattern
# the gauge erases every time it recovers.
reaper_consecutive_sweep_failures = Gauge(
    "w2v_reaper_consecutive_sweep_failures",
    "Current unbroken run of failed reaper sweeps on this pod; 0 when the last sweep succeeded.",
)

reaper_sweep_failures_total = Counter(
    "w2v_reaper_sweep_failures_total",
    "Total reaper sweeps that raised, since pod start.",
)

# Separate from the two metrics above on purpose, not folded into them: a raising
# reap_once means orphaned renders are NOT being reclaimed (the reclaim-health
# signal these two exist to answer); a raising _prune means only housekeeping
# (terminal-row eviction / zombie flip) is broken, which is a real fault but a
# strictly less urgent one, and conflating the two would make
# ``consecutive_failures`` lie about reclaim health on a pod where reclaim is
# perfectly healthy and only prune is broken. Silently swallowing a prune failure
# with no counter at all is the same "invisible failure" class of bug this whole
# module exists to close, so it gets its own counter rather than no counter.
reaper_prune_failures_total = Counter(
    "w2v_reaper_prune_failures_total",
    "Total reaper prune sweeps that raised, since pod start.",
)


class ReaperService:
    """Reclaims and re-dispatches crash-orphaned compile jobs on this pod."""

    def __init__(self, app: FastAPI) -> None:
        settings = get_settings()
        self._app = app
        self._interval = settings.reaper_interval_seconds
        self._prune_interval = settings.prune_interval_seconds
        self._lease_seconds = settings.job_lease_seconds
        self._max_attempts = settings.reclaim_max_attempts
        self._max_lifetime = settings.reclaim_max_lifetime_seconds
        self._max_concurrent = settings.max_concurrent_compile_jobs
        # Terminal-row retention = the Idempotency-Key dedup horizon (the row IS the
        # dedup index). NOT minio_url_expiry_seconds: that is the presigned-URL expiry
        # (3600 in prod), and using it deleted a completed job's dedup row one hour
        # after it finished, so a resumable-timeout resend carrying the SAME stable key
        # started a second full PAID render. An expired presign is re-presignable from
        # the deterministic object key; a lost dedup row is not recoverable.
        self._ttl_seconds = settings.compile_job_retention_seconds
        self._work_timeout = settings.job_work_timeout_seconds
        # Use the SAME identity the render path stamps on its rows + renews its lease
        # with (``app.state.instance_id``), NOT a separate hostname. If the reaper claimed
        # under a different id, a same-pod reclaim of a live-but-lease-lapsed render (a
        # multi-beat DB stall) would overwrite ``instance_id`` and permanently orphan that
        # render's heartbeat (its ``renew_lease WHERE instance_id=…`` would match 0 rows),
        # spiralling the row to the poison cap while it is still rendering.
        self._pod_id = app.state.instance_id
        self._task: asyncio.Task[None] | None = None
        self._elapsed_since_prune = 0.0
        # Sweep-health counters. CONSECUTIVE (reset by any successful sweep), because a
        # total is indistinguishable between "one bad afternoon last week" and "reclaiming
        # nothing right now"; only an unbroken run means orphans are piling up.
        self._consecutive_failures = 0
        self._last_failure: str | None = None
        self._last_failure_at: datetime | None = None
        # Tracked separately from the reap-failure trio above — see
        # ``reaper_prune_failures_total``'s comment for why prune failures are not
        # folded into ``_consecutive_failures``.
        self._last_prune_failure: str | None = None
        self._last_prune_failure_at: datetime | None = None

    @property
    def running(self) -> bool:
        """True while the sweep loop task exists and has not finished.

        Read by the liveness probe so "the reaper object is installed" and "the reaper is
        actually sweeping" are distinguishable. ``_loop`` only exits on cancellation (every
        other fault is caught inside it), so a done() task on a non-draining pod means
        ``stop()`` ran — i.e. nothing is reclaiming orphans any more.
        """
        return self._task is not None and not self._task.done()

    @property
    def consecutive_failures(self) -> int:
        """Unbroken run of failed sweeps; 0 whenever the last sweep succeeded."""
        return self._consecutive_failures

    @property
    def last_failure(self) -> str | None:
        """repr() of the most recent sweep exception (never cleared — the last failure is
        still the best clue after a recovery), or None if no sweep has ever failed."""
        return self._last_failure

    @property
    def last_failure_at(self) -> datetime | None:
        return self._last_failure_at

    @property
    def last_prune_failure(self) -> str | None:
        """repr() of the most recent prune exception, or None if prune has never
        failed. Deliberately separate from :attr:`last_failure` — see
        ``reaper_prune_failures_total``'s module-level comment."""
        return self._last_prune_failure

    @property
    def last_prune_failure_at(self) -> datetime | None:
        return self._last_prune_failure_at

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Reaper started (pod=%s, interval=%.0fs, lease=%ds, max_attempts=%d, max_lifetime=%ds)",
            self._pod_id,
            self._interval,
            self._lease_seconds,
            self._max_attempts,
            self._max_lifetime,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Reaper stopped")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._sweep()
            except asyncio.CancelledError:
                break
            except Exception:
                # _sweep() is supposed to handle every fault from reap_once() and
                # _prune() internally so the loop task itself never dies. This catch
                # exists so a FUTURE bug inside _sweep (e.g. an exception net removed,
                # or a new call added outside one) degrades to "reaper stops reclaiming
                # for one interval and a CRITICAL is logged" instead of "the reaper
                # task silently dies forever and every health signal goes quiet" — the
                # exact failure mode a bare comment-enforced invariant produced before.
                logger.critical(
                    "Reaper loop caught an exception that escaped _sweep() — this "
                    "should be unreachable; _sweep() has a bug. The loop is "
                    "continuing, but investigate immediately.",
                    exc_info=True,
                )

    async def _sweep(self) -> None:
        """One sweep plus its health bookkeeping.

        Split out of ``_loop`` so a sweep can be driven without the interval sleep (tests),
        and so the failure accounting lives next to the thing it accounts for. A failing
        sweep is NEVER fatal — a DB blip must not kill the pod — it is only made visible.
        """
        # Advance prune bookkeeping UNCONDITIONALLY, independent of whether reap_once()
        # raises below. A reap_once that fails EVERY sweep (the concrete hazard: a
        # partially-applied migration leaves reclaim_next's SQL body referencing a column
        # a later migration adds, so it raises forever) must not starve _prune forever —
        # prior to this, the increment sat AFTER the reap_once await inside the same try,
        # so a permanent reap failure jumped straight to `except Exception` and the
        # counter never advanced.
        self._elapsed_since_prune += self._interval
        due_for_prune = self._elapsed_since_prune >= self._prune_interval

        try:
            await self.reap_once()
        except asyncio.CancelledError:
            raise  # shutdown, not a sweep failure — must not count or log
        except Exception as exc:  # noqa: BLE001 — the loop must survive any sweep fault
            self._record_sweep_failure(exc)
        else:
            self._consecutive_failures = 0
            reaper_consecutive_sweep_failures.set(0)

        if due_for_prune:
            self._elapsed_since_prune = 0.0
            # _prune() nets its own internal DB call already, but that internal net is
            # an implementation detail of THIS _prune(), not a guarantee _sweep() may
            # rely on — a future edit to _prune() (or a test double standing in for
            # it) could raise, and without a net HERE that would escape _sweep() into
            # _loop() and silently kill the whole reaper task (the exact bug this
            # comment used to hand-wave away by asserting "never raises"). Netted
            # AFTER the reap-failure bookkeeping above so a prune fault can never mask
            # or pre-empt the reap failure's own logging.
            try:
                await self._prune()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — must not kill the loop
                self._record_prune_failure(exc)

    def _record_sweep_failure(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        self._last_failure = repr(exc)  # repr(), not str(): a bare TimeoutError renders ""
        self._last_failure_at = datetime.now(UTC)
        reaper_sweep_failures_total.inc()
        reaper_consecutive_sweep_failures.set(self._consecutive_failures)

        # Traceback ONCE per streak, not once per sweep. The whole point of this change is
        # that a permanently-broken reaper produced an identical traceback every interval
        # forever; emitting the counter alongside that spam would have left the spam. The
        # traceback is information only the FIRST time (nothing about the 40th identical
        # frame stack is new) and again at each escalation (so the CRITICAL an operator
        # actually reads is self-contained and they need not scroll back to the streak's
        # start). Everything in between is a single WARNING line: enough to see the streak
        # is ongoing in a log tail, cheap enough not to bury other services' logs.
        if self._consecutive_failures % _ESCALATE_EVERY_N_FAILURES == 0:
            logger.critical(
                "REAPER STUCK — %d consecutive sweep failures (~%.0fs); orphaned renders are "
                "NOT being reclaimed. Last error: %s",
                self._consecutive_failures,
                self._consecutive_failures * self._interval,
                self._last_failure,
                exc_info=exc,
            )
        elif self._consecutive_failures == 1:
            logger.exception("Reaper sweep failed (consecutive=1)")
        else:
            logger.warning(
                "Reaper sweep failed (consecutive=%d): %s",
                self._consecutive_failures,
                self._last_failure,
            )

    def _record_prune_failure(self, exc: BaseException) -> None:
        """Record + log a ``_prune()`` fault WITHOUT touching the reap-failure
        counters — see ``reaper_prune_failures_total``'s module-level comment for why
        the two are kept apart. Every occurrence logs (not just every-Nth like
        :meth:`_record_sweep_failure`): prune runs once per ``prune_interval``
        (default 30min), orders of magnitude less often than a sweep, so there is no
        equivalent per-interval spam to suppress.
        """
        self._last_prune_failure = repr(exc)
        self._last_prune_failure_at = datetime.now(UTC)
        reaper_prune_failures_total.inc()
        logger.warning("Reaper prune failed: %s", self._last_prune_failure, exc_info=exc)

    async def reap_once(self) -> int:
        """Run one reclaim sweep. Returns the number of jobs re-dispatched."""
        from src.api.routes.compile import (
            _active_job_ids,
            _active_render_count,
            redispatch_reclaimed,
        )

        count = 0
        for _ in range(_MAX_RECLAIMS_PER_SWEEP):
            # Respect the same per-pod concurrency cap the submit path enforces. A
            # Recreate can leave a large backlog of reclaimable orphans; without this
            # gate the reaper would fire ALL of them (up to _MAX_RECLAIMS_PER_SWEEP)
            # as detached renders in one sweep, blowing past max_concurrent_compile_jobs
            # and piling that many scratch dirs into /tmp. Stop when at capacity; the
            # unclaimed rows stay reclaimable on the next sweep.
            if _active_render_count() >= self._max_concurrent:
                logger.info(
                    "Reaper deferring reclaims — at render capacity (%d/%d)",
                    _active_render_count(),
                    self._max_concurrent,
                )
                break
            # Snapshot the live-render set so the SQL claim skips a job THIS pod is
            # already rendering (a same-pod reclaim of a lease-lapsed-but-live render
            # would otherwise bump its attempts + eventually poison-fail it). Captured
            # per-iteration; the post-claim veto in redispatch_reclaimed is the backstop.
            async with get_db_session() as db:
                claimed = await job_store.reclaim_next(
                    db,
                    worker_id=self._pod_id,
                    lease_seconds=self._lease_seconds,
                    max_attempts=self._max_attempts,
                    max_lifetime_seconds=self._max_lifetime,
                    active_job_ids=sorted(_active_job_ids),
                )
            if claimed is None:
                break
            # A poison / over-lifetime job was CONSUMED (hard-failed inside the SQL
            # function) and handed back as the ``job_store.POISON_SKIPPED`` sentinel, NOT
            # a re-dispatchable job. It is terminal now, so skip it and CONTINUE the sweep
            # — a poison job at head-of-queue must not stall the healthy orphans queued
            # behind it (which a `break` here would, for a full reaper interval). The only
            # non-``None`` non-``ClaimedJob`` value is that sentinel; the isinstance form
            # both expresses that and narrows ``claimed`` to ``ClaimedJob`` for ty below.
            if not isinstance(claimed, job_store.ClaimedJob):
                continue
            logger.warning(
                "Reaper re-dispatching orphaned job %s (tenant=%s)",
                claimed.job_id,
                claimed.tenant_id,
            )
            try:
                await redispatch_reclaimed(
                    claimed, lease_seconds=self._lease_seconds, instance_id=self._pod_id
                )
            except Exception:  # one poison claim must not abort the sweep
                # A transient error stamping a terminal status must not abort the
                # remaining reclaims in this sweep. The row stays reclaimable (or already
                # terminal) and the next sweep retries.
                logger.warning(
                    "Reaper re-dispatch failed for job %s", claimed.job_id, exc_info=True
                )
                continue
            count += 1
        return count

    async def _prune(self) -> None:
        """Evict terminal rows past their TTL. RAISES on failure — deliberately.

        ⚠️ This used to swallow its own exception (``except Exception: log.warning``),
        which made the caller's `_record_prune_failure` net DEAD CODE:
        ``reaper_prune_failures_total`` was permanently 0 and ``last_prune_failure``
        permanently ``None``, no matter how broken prune was. That is the exact
        invisible-failure class this module's own header says it exists to close —
        a prune wedged forever means terminal rows are never evicted and the
        idempotency dedup index grows without bound, with nothing on /metrics to
        say so.

        The caller (`_sweep`) nets this and records it. That net is not a
        belt-and-braces duplicate of an internal one any more; it is the ONLY
        handler, and it is the one wired to the counter.
        """
        async with get_db_session() as db:
            pruned = await job_store.prune(
                db,
                ttl_seconds=self._ttl_seconds,
                work_timeout_seconds=self._work_timeout,
            )
        if pruned:
            logger.info("Reaper pruned %d stale job row(s)", pruned)
