"""Phase 21.6 — Memory Search Engine.

Institutional search over the decision memory:
  "Show all incorrect BUY decisions on EURUSD."
  "Show all profitable NO_TRADE recommendations."
  "Show decisions with confidence above 95%."
  "Show all failed breakout trades."
  "Show all successful pullback entries."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import DecisionRecord
from trainer.memory.decision_types import (
    NEGATIVE_OUTCOMES,
    POSITIVE_OUTCOMES,
    DecisionType,
    MarketRegime,
    OutcomeType,
)


@dataclass
class SearchQuery:
    """Structured search query parameters."""

    # Text filter (applied to instrument, tags, reasoning)
    text: str = ""
    # Field filters
    instrument: Optional[str] = None
    decision_type: Optional[str] = None  # DecisionType value
    outcome_type: Optional[str] = None  # OutcomeType value
    regime: Optional[str] = None
    session: Optional[str] = None
    grade: Optional[str] = None  # EXCELLENT | GOOD | ...
    # Numeric ranges
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None
    # Flags
    direction_correct: Optional[bool] = None
    hit_tp: Optional[bool] = None
    hit_sl: Optional[bool] = None
    profitable: Optional[bool] = None  # pnl > 0
    # Result
    limit: int = 100
    sort_by: str = "timestamp"  # timestamp | confidence | pnl | quality

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


@dataclass
class SearchResult:
    """Result of a memory search."""

    query: SearchQuery
    total_found: int
    records: List[DecisionRecord] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "query": self.query.to_dict(),
            "total_found": self.total_found,
            "records": [r.summary() for r in self.records],
            "summary": self.summary,
        }


class MemorySearch:
    """Institutional search engine over decision memory."""

    def search(
        self,
        records: List[DecisionRecord],
        query: SearchQuery,
    ) -> SearchResult:
        """Execute a structured search over decision records."""

        results = list(records)  # start with all

        # ── Field filters ─────────────────────────────────────────────────────
        if query.instrument:
            results = [r for r in results if r.instrument == query.instrument]

        if query.decision_type:
            try:
                dt = DecisionType(query.decision_type)
                results = [r for r in results if r.decision_type == dt]
            except ValueError:
                pass

        if query.session:
            results = [r for r in results if r.snapshot.session == query.session]

        if query.regime:
            results = [r for r in results if r.snapshot.market_regime == query.regime]

        if query.grade:
            results = [r for r in results if r.grade == query.grade]

        # Confidence range
        if query.min_confidence > 0.0:
            results = [r for r in results if r.snapshot.forecast_confidence >= query.min_confidence]
        if query.max_confidence < 1.0:
            results = [r for r in results if r.snapshot.forecast_confidence <= query.max_confidence]

        # ── Outcome-dependent filters ─────────────────────────────────────────
        if query.outcome_type is not None:
            try:
                ot = OutcomeType(query.outcome_type)
                results = [r for r in results if r.has_outcome and r.outcome.outcome_type == ot]
            except ValueError:
                pass

        if query.direction_correct is not None:
            results = [
                r
                for r in results
                if r.has_outcome and r.outcome.actual_direction_correct == query.direction_correct
            ]

        if query.hit_tp is not None:
            results = [r for r in results if r.has_outcome and r.outcome.hit_tp == query.hit_tp]

        if query.hit_sl is not None:
            results = [r for r in results if r.has_outcome and r.outcome.hit_sl == query.hit_sl]

        if query.profitable is not None:
            results = [
                r
                for r in results
                if r.has_outcome and (r.outcome.actual_pnl > 0) == query.profitable
            ]

        if query.min_pnl is not None:
            results = [
                r for r in results if r.has_outcome and r.outcome.actual_pnl >= query.min_pnl
            ]

        if query.max_pnl is not None:
            results = [
                r for r in results if r.has_outcome and r.outcome.actual_pnl <= query.max_pnl
            ]

        # ── Text filter ───────────────────────────────────────────────────────
        if query.text:
            q = query.text.lower()
            filtered = []
            for r in results:
                text_body = (
                    r.instrument.lower()
                    + " "
                    + r.decision_type.value.lower()
                    + " "
                    + " ".join(r.tags).lower()
                    + " "
                    + r.snapshot.market_regime.lower()
                    + " "
                    + r.snapshot.session.lower()
                    + " "
                    + " ".join(r.reasoning.primary_reasons).lower()
                    + " "
                    + r.reasoning.final_rationale.lower()
                )
                if q in text_body:
                    filtered.append(r)
            results = filtered

        # ── Sort ──────────────────────────────────────────────────────────────
        if query.sort_by == "confidence":
            results.sort(key=lambda r: r.snapshot.forecast_confidence, reverse=True)
        elif query.sort_by == "pnl" and all(r.has_outcome for r in results):
            results.sort(key=lambda r: r.outcome.actual_pnl, reverse=True)
        elif query.sort_by == "quality":
            results.sort(key=lambda r: r.score.overall if r.has_score else 0.0, reverse=True)
        else:
            results.sort(key=lambda r: r.timestamp, reverse=True)

        total = len(results)
        results = results[: query.limit]

        logger.debug(f"MemorySearch: {total} records matched, returning {len(results)}")

        return SearchResult(
            query=query,
            total_found=total,
            records=results,
            summary=(
                f"Found {total} decisions matching query. "
                f"Showing {len(results)} (limit={query.limit})."
            ),
        )

    # ── Convenience methods ───────────────────────────────────────────────────

    def incorrect_buys(
        self, records: List[DecisionRecord], instrument: Optional[str] = None
    ) -> SearchResult:
        return self.search(
            records,
            SearchQuery(
                decision_type=DecisionType.BUY.value,
                direction_correct=False,
                instrument=instrument,
            ),
        )

    def profitable_no_trades(self, records: List[DecisionRecord]) -> SearchResult:
        """Decisions where NO_TRADE was taken and market moved favourably (pnl=0 vs opportunity)."""
        return self.search(
            records,
            SearchQuery(
                decision_type=DecisionType.NO_TRADE.value,
                outcome_type=OutcomeType.MISSED_OPPORTUNITY.value,
            ),
        )

    def high_confidence_decisions(
        self, records: List[DecisionRecord], min_conf: float = 0.90
    ) -> SearchResult:
        return self.search(records, SearchQuery(min_confidence=min_conf))

    def decisions_by_regime_and_outcome(
        self,
        records: List[DecisionRecord],
        regime: str,
        positive: bool = True,
    ) -> SearchResult:
        q = SearchQuery(regime=regime, profitable=positive)
        return self.search(records, q)

    def failed_breakouts(self, records: List[DecisionRecord]) -> SearchResult:
        return self.search(
            records,
            SearchQuery(
                regime=MarketRegime.BREAKOUT.value,
                hit_sl=True,
            ),
        )

    def successful_pullbacks(self, records: List[DecisionRecord]) -> SearchResult:
        return self.search(
            records,
            SearchQuery(
                text="pullback",
                hit_tp=True,
                profitable=True,
            ),
        )
