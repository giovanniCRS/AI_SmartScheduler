"""Phase 3 agent: SYMBOLIC ONLY (no LLM call). Executes the generated
OR-Tools code in a sandbox, validates every hard constraint, and
computes fairness scores. This is the `execute_and_verify_node` of the
LangGraph flow.
"""
import time
from typing import Dict

from agents.base import Agent
from core.state import ScheduleState
from solver.fairness import calculate_fairness
from solver.or_tools_wrapper import run_generated_code
from solver.validator import validate_schedule
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VerificationAgent(Agent):
    name = "verification_agent"

    def run(self, state: ScheduleState) -> Dict:
        start = time.perf_counter()
        iteration = state["iteration"] + 1

        exec_result = run_generated_code(state["generated_code"])
        elapsed = time.perf_counter() - start

        if not exec_result["success"]:
            logger.info(f"[iter {iteration}] execution/solve failed: {exec_result['error']}")
            history_entry = {
                "iteration": iteration,
                "code_error": exec_result["error"],
                "hard_errors": [],
                "fairness_scores": {},
                "elapsed_seconds": elapsed,
            }
            return {
                "iteration": iteration,
                "code_error": exec_result["error"],
                "schedule_solution": None,
                "hard_errors": [],
                "fairness_scores": {},
                "least_satisfied": [],
                "history": state["history"] + [history_entry],
            }

        schedule = exec_result["schedule"]
        hard_errors = validate_schedule(schedule, state["workers"], state["case_type"])

        fairness_scores: Dict = {}
        least_satisfied = []
        update: Dict = {}
        if not hard_errors:
            fairness = calculate_fairness(schedule, state["parsed_preferences"])
            fairness_scores = fairness["scores"]
            least_satisfied = fairness["least_satisfied"]

            if _is_better(fairness_scores, state.get("best_valid_fairness_scores") or {},
                          state.get("best_valid_schedule")):
                update.update({
                    "best_valid_schedule": schedule,
                    "best_valid_generated_code": state["generated_code"],
                    "best_valid_fairness_scores": fairness_scores,
                    "best_valid_least_satisfied": least_satisfied,
                })
                logger.info(f"[iter {iteration}] new best valid schedule (gap={_gap(fairness_scores):.3f})")

        logger.info(
            f"[iter {iteration}] hard_errors={len(hard_errors)} "
            f"least_satisfied={least_satisfied}"
        )
        history_entry = {
            "iteration": iteration,
            "code_error": None,
            "hard_errors": hard_errors,
            "fairness_scores": fairness_scores,
            "elapsed_seconds": elapsed,
        }
        update.update({
            "iteration": iteration,
            "code_error": None,
            "schedule_solution": schedule,
            "hard_errors": hard_errors,
            "fairness_scores": fairness_scores,
            "least_satisfied": least_satisfied,
            "history": state["history"] + [history_entry],
        })
        return update


def _gap(fairness_scores: Dict) -> float:
    if not fairness_scores:
        return 1.0
    return max(fairness_scores.values()) - min(fairness_scores.values())


def _is_better(candidate_scores: Dict, current_best_scores: Dict, current_best_schedule) -> bool:
    """A valid candidate schedule replaces the stored best if there is no
    best yet, or its fairness gap (max-min satisfaction) is smaller."""
    if current_best_schedule is None:
        return True
    return _gap(candidate_scores) < _gap(current_best_scores)
