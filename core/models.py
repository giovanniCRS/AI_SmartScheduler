"""Small domain classes used for readability outside of the raw dicts
that flow through ScheduleState (which LangGraph requires to stay
JSON-serializable, hence why the state itself uses plain dicts)."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


class Shift(IntEnum):
    MORNING = 0
    AFTERNOON = 1
    NIGHT = 2

    @property
    def weight(self) -> int:
        return 2 if self is Shift.NIGHT else 1

    @property
    def label(self) -> str:
        return {Shift.MORNING: "morning", Shift.AFTERNOON: "afternoon",
                Shift.NIGHT: "night"}[self]


@dataclass
class Worker:
    id: int
    role: str = "standard"  # "standard" | "specialized"

    @property
    def is_specialized(self) -> bool:
        return self.role == "specialized"


@dataclass
class WorkerPreferences:
    id: int
    preferred_shifts: List[str] = field(default_factory=list)
    avoid_shifts: List[str] = field(default_factory=list)
    max_consecutive_nights: int = 2
    weekend_preference: str = "neutral"   # "prefer" | "avoid" | "neutral"
    holiday_tolerance: float = 0.5        # 0 = avoid entirely, 1 = fully ok


@dataclass
class Schedule:
    """Assignments as day -> shift -> list of worker ids."""
    assignments: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)

    def workers_on(self, day: int, shift: int) -> List[int]:
        return self.assignments.get(day, {}).get(shift, [])

    def works(self, worker_id: int, day: int, shift: int) -> bool:
        return worker_id in self.workers_on(day, shift)

    def shifts_of(self, worker_id: int) -> List[tuple]:
        """Return [(day, shift), ...] this worker is assigned to."""
        out = []
        for day, shifts in self.assignments.items():
            for shift, worker_ids in shifts.items():
                if worker_id in worker_ids:
                    out.append((day, shift))
        return sorted(out)

    @classmethod
    def from_dict(cls, raw: Dict) -> "Schedule":
        assignments = {
            int(day): {int(shift): list(worker_ids) for shift, worker_ids in shifts.items()}
            for day, shifts in raw.items()
        }
        return cls(assignments=assignments)
