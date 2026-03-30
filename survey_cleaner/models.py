"""Core data models for survey-cleaner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FlagCode(str, Enum):
    SPEEDER = "speeder"
    STRAIGHT_LINER = "straight_liner"
    PATTERN_LINER = "pattern_liner"
    GIBBERISH = "gibberish"
    TOO_SHORT = "too_short"


@dataclass
class FlagReason:
    code: FlagCode
    detail: str
    severity: float  # 0.0 – 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "detail": self.detail,
            "severity": round(self.severity, 3),
        }


@dataclass
class FlaggedResponse:
    row_index: int
    respondent_id: Optional[str]
    reasons: list[FlagReason]

    @property
    def codes(self) -> list[str]:
        return [r.code.value for r in self.reasons]

    @property
    def max_severity(self) -> float:
        return max((r.severity for r in self.reasons), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "respondent_id": self.respondent_id,
            "reasons": [r.to_dict() for r in self.reasons],
            "max_severity": round(self.max_severity, 3),
        }


@dataclass
class CleaningResult:
    total_responses: int
    flagged: list[FlaggedResponse] = field(default_factory=list)
    clean_indices: list[int] = field(default_factory=list)

    # detector-level counts
    speeder_count: int = 0
    straight_liner_count: int = 0
    pattern_liner_count: int = 0
    gibberish_count: int = 0

    # config snapshot
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged_count(self) -> int:
        return len(self.flagged)

    @property
    def clean_count(self) -> int:
        return len(self.clean_indices)

    @property
    def flag_rate(self) -> float:
        if self.total_responses == 0:
            return 0.0
        return self.flagged_count / self.total_responses

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_responses": self.total_responses,
            "flagged_count": self.flagged_count,
            "clean_count": self.clean_count,
            "flag_rate": round(self.flag_rate, 4),
            "speeder_count": self.speeder_count,
            "straight_liner_count": self.straight_liner_count,
            "pattern_liner_count": self.pattern_liner_count,
            "gibberish_count": self.gibberish_count,
            "config": self.config,
            "flagged": [f.to_dict() for f in self.flagged],
        }
