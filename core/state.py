"""Global LangGraph state for SmartScheduler.

This TypedDict is threaded through every node of the graph; LangGraph
merges each node's returned dict into it, which is how "memory across
iterations" is achieved without any extra plumbing.
"""
from typing import Dict, List, Optional, TypedDict


class ScheduleState(TypedDict):
    # --- Input --------------------------------------------------------
    input_file_content: str
    workers: List[Dict]          # [{"id": 0, "role": "standard"}, ...]
    case_type: str                # "A" or "B"

    # --- Phase 1: preferences ------------------------------------------
    raw_preferences: str           # free-text worker preferences
    parsed_preferences: Dict       # structured JSON preferences

    # --- Phase 2: drafting -----------------------------------------------
    generated_code: str            # OR-Tools Python source (as a string)
    code_error: Optional[str]      # syntax / runtime error from last attempt

    # --- Phase 3: verification --------------------------------------------
    schedule_solution: Optional[Dict]  # {day: {shift: [worker_ids]}}
    hard_errors: List[str]             # hard-constraint violations
    fairness_scores: Dict               # {worker_id: satisfaction score}
    least_satisfied: List[int]          # worker ids with the lowest score

    # --- Flow control --------------------------------------------------------
    iteration: int
    max_iterations: int             # default 5 (config.MAX_ITERATIONS)
    is_complete: bool
    history: List[Dict]              # per-iteration debug trail

    # --- Best-valid-attempt tracking (addition beyond the original spec) ---
    # The refinement loop can occasionally regress on a later iteration
    # (e.g. the LLM drops a required line while "fixing" something else).
    # Without this, hitting max_iterations right after a regression would
    # discard an earlier fully-valid schedule in favor of the broken final
    # attempt. These hold the best fully-valid (hard_errors == []) schedule
    # seen so far, so finalize_node can fall back to it if the last
    # attempt isn't as good.
    best_valid_schedule: Optional[Dict]
    best_valid_generated_code: str
    best_valid_fairness_scores: Dict
    best_valid_least_satisfied: List[int]


def new_initial_state(
    input_file_content: str,
    workers: List[Dict],
    case_type: str,
    raw_preferences: str,
    max_iterations: int,
) -> ScheduleState:
    """Build the state LangGraph starts from."""
    return ScheduleState(
        input_file_content=input_file_content,
        workers=workers,
        case_type=case_type,
        raw_preferences=raw_preferences,
        parsed_preferences={},
        generated_code="",
        code_error=None,
        schedule_solution=None,
        hard_errors=[],
        fairness_scores={},
        least_satisfied=[],
        iteration=0,
        max_iterations=max_iterations,
        is_complete=False,
        history=[],
        best_valid_schedule=None,
        best_valid_generated_code="",
        best_valid_fairness_scores={},
        best_valid_least_satisfied=[],
    )
