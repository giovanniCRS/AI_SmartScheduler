"""Common interface for every LangGraph node-agent.

Each agent takes the current ScheduleState and returns a *partial* dict
of updates (LangGraph merges it into the state) -- never the full state,
to keep nodes decoupled from fields they don't touch.
"""
from abc import ABC, abstractmethod
from typing import Dict

from core.state import ScheduleState


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def run(self, state: ScheduleState) -> Dict:
        """Return a dict of state fields to update."""
        raise NotImplementedError
