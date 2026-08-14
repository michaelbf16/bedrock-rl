"""Small, dependency-free scripted-policy result types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyVeto:
    """A policy decision that deliberately emitted no executable actions."""

    code: str
    reason: str
    target: tuple[float, float]
    observed: tuple[tuple[int, int, int], ...] = ()

    def to_dict(self):
        return {
            "code": self.code,
            "reason": self.reason,
            "target": list(self.target),
            "observed": [list(cell) for cell in self.observed],
        }
