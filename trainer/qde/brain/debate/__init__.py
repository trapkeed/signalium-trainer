"""QDE Brain — debate subpackage (Phase C5).

Internal debate engine: the QDE challenges its own beliefs by running a
structured bull-vs-bear debate for every developing opportunity before
allowing execution.

What was already there (untouched)
-------------------------------------
  debate_engine.py      ``DebateEngine`` / ``DebateVerdict``
                        Evaluates bull vs. bear via ThesisManager.sync();
                        logs verdict to working memory.

  conviction_engine.py  ``compute_conviction()`` / ``ConvictionScore``
                        Conviction from Hypothesis reinforcement + evidence
                        balance + maturity + persistence.

  thesis_manager.py     ``ThesisManager`` / ``Thesis``
                        Maintains the two-sided bull/bear argument for every
                        Opportunity; reprojects hypothesis evidence.

What this package adds (Phase C5)
-------------------------------------
  debate_quality.py      ``DebateQualityScore`` / ``measure_quality()``
                         Five-dimension quality score: evidence density,
                         balance, margin clarity, conviction alignment,
                         timeliness.  Pure function; no new storage.

  overconfidence_tracker.py ``OverconfidenceEvent`` / ``OverconfidenceTracker``
                         Detects and records overconfidence events:
                           • LOW_MARGIN — high conviction, contested debate
                           • POOR_EVIDENCE_BALANCE — winning thesis opposed
                             by its own evidence
                           • CONVICTION_CONFIDENCE_SPLIT — high confidence
                             in a poorly-formed setup

  ignored_evidence_tracker.py ``IgnoredEvidenceRecord`` / ``IgnoredEvidenceTracker``
                         Records which contradicting evidence items were
                         present but not weighted enough to lower conviction.
                         Surfaces chronic blind spots via common_ignored_signals().

  debate_store.py        ``DebateRecord`` / ``DebateMemoryBridge``
                         Complete debate snapshot (verdict + conviction +
                         quality + OC event + IE record).  Soft C2 episodic
                         storage; local deque fallback.

  debate_api.py          ``DebateAPI``
                         Three spec-required methods:
                           create_debate(instrument, opportunity_id, now)
                           calculate_conviction(hypothesis, now)
                           store_debate(record, now)

Integration example
-------------------
::

    # At initialization (debate_engine and hypothesis_engine already exist):
    api = DebateAPI(
        debate_engine=brain.debate_engine,
        hypothesis_engine=brain.hypothesis_engine,
        memory_bridge=DebateMemoryBridge(episodic_memory=emem),
    )

    # On each tick before a potential execution:
    record = api.create_debate("EURUSD", opportunity_id, now=now)
    if record is None:
        pass   # opportunity not tracked

    if record.conclusive and not record.has_overconfidence:
        # Safe to proceed — debate is decisive, no overconfidence detected
        api.store_debate(record, now=now)

    # After trade resolves:
    if record.has_overconfidence:
        oc_event_id = record.overconfidence_event["event_id"]
        api.resolve_overconfidence(oc_event_id, outcome="LOSS", now=now)

    # Inspect chronic blind spots:
    blind_spots = api.common_ignored_signals("EURUSD", top_n=5)
"""

from trainer.qde.brain.debate.debate_api import DebateAPI
from trainer.qde.brain.debate.debate_quality import DebateQualityScore, measure_quality
from trainer.qde.brain.debate.debate_store import (
    DebateMemoryBridge,
    DebateRecord,
    make_debate_record,
)
from trainer.qde.brain.debate.ignored_evidence_tracker import (
    IGNORED_EVIDENCE_MIN_CONVICTION,
    IgnoredEvidenceRecord,
    IgnoredEvidenceTracker,
)
from trainer.qde.brain.debate.overconfidence_tracker import (
    HIGH_CONVICTION_THRESHOLD,
    OverconfidenceEvent,
    OverconfidenceTracker,
)

__all__ = [
    # ── Primary API (Phase C5) ────────────────────────────────────────
    "DebateAPI",
    # ── Debate quality ────────────────────────────────────────────────
    "DebateQualityScore",
    "measure_quality",
    # ── Debate record + memory bridge ─────────────────────────────────
    "DebateRecord",
    "DebateMemoryBridge",
    "make_debate_record",
    # ── Overconfidence tracking ───────────────────────────────────────
    "OverconfidenceEvent",
    "OverconfidenceTracker",
    "HIGH_CONVICTION_THRESHOLD",
    # ── Ignored evidence tracking ─────────────────────────────────────
    "IgnoredEvidenceRecord",
    "IgnoredEvidenceTracker",
    "IGNORED_EVIDENCE_MIN_CONVICTION",
]
