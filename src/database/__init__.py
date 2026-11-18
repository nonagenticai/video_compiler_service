"""Durable Postgres layer for video_compiler_service.

The job store + Idempotency-Key dedup index live in Postgres (not per-pod
memory) so a resume after a pod restart dedupes a re-submitted render instead
of firing a second PAID compile. See ``src/services/compile_job_store.py`` for
the state machine and ``models.py`` for the schema.
"""
