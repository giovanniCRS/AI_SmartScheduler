"""Builds the LangGraph StateGraph: 4 nodes, conditional routing after
verification, exactly as specified in the brief.

    parse_preferences_node -> draft_schedule_node -> execute_and_verify_node
                                                          |  |   |
                            code_error ------------------+  |   |
                            (back to draft)                  |   |
                            hard_errors / fairness gap -------+   |
                            (to refine, then back to verify)      |
                            iteration >= max OR all-good ---------+--> END
"""
from typing import Literal

import config
from agents.drafting_agent import DraftingAgent
from agents.preference_agent import PreferenceAgent
from agents.refinement_agent import RefinementAgent
from agents.verification_agent import VerificationAgent
from core.state import ScheduleState
from llm.groq_client import GroqClient
from utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - offline / not installed
    END = "__end__"
    StateGraph = None


def fairness_gap(state: ScheduleState) -> float:
    scores = state.get("fairness_scores") or {}
    if not scores:
        return 1.0  # unknown -> treat as maximally unfair, keep refining
    return max(scores.values()) - min(scores.values())


def route_after_verify(state: ScheduleState) -> Literal["draft", "refine", "end"]:
    if state["code_error"]:
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "draft"
    if state["hard_errors"]:
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "refine"
    if fairness_gap(state) > config.FAIRNESS_GAP_THRESHOLD:
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "refine"
    return "end"


def finalize_node(state: ScheduleState) -> dict:
    current_is_valid = state["schedule_solution"] is not None and not state["hard_errors"]
    current_gap = fairness_gap(state) if current_is_valid else 1.0
    has_best = state.get("best_valid_schedule") is not None

    # Prefer the current attempt if it's valid and at least as good as the
    # best one seen so far (covers the common "converged on the last
    # iteration" case exactly as before). Otherwise, if a later iteration
    # regressed or broke the script, fall back to the best valid schedule
    # found earlier rather than discarding it.
    if current_is_valid and (not has_best or current_gap <= _stored_best_gap(state)):
        success = current_gap <= config.FAIRNESS_GAP_THRESHOLD
        if not success and state["iteration"] >= state["max_iterations"]:
            logger.info(
                f"Stopped after reaching max_iterations={state['max_iterations']} "
                "without a fully satisfactory schedule; returning best attempt "
                f"(fairness gap={current_gap:.3f})."
            )
        return {"is_complete": success}

    if has_best:
        logger.info(
            "Final attempt was invalid/worse than an earlier iteration; "
            "falling back to the best valid schedule found "
            f"(gap={_stored_best_gap(state):.3f})."
        )
        success = _stored_best_gap(state) <= config.FAIRNESS_GAP_THRESHOLD
        return {
            "is_complete": success,
            "schedule_solution": state["best_valid_schedule"],
            "generated_code": state["best_valid_generated_code"],
            "fairness_scores": state["best_valid_fairness_scores"],
            "least_satisfied": state["best_valid_least_satisfied"],
            "hard_errors": [],
        }

    logger.info(
        f"Stopped after reaching max_iterations={state['max_iterations']} "
        "without ever producing a hard-constraint-valid schedule."
    )
    return {"is_complete": False}


def _stored_best_gap(state: ScheduleState) -> float:
    return fairness_gap({"fairness_scores": state.get("best_valid_fairness_scores") or {}})


def build_graph(client: GroqClient):
    if StateGraph is None:
        raise ImportError(
            "langgraph is not installed. Run `pip install -r requirements.txt`."
        )

    preference_agent = PreferenceAgent(client)
    drafting_agent = DraftingAgent(client)
    verification_agent = VerificationAgent()
    refinement_agent = RefinementAgent(client)

    graph = StateGraph(ScheduleState)
    graph.add_node("parse_preferences", lambda s: preference_agent.run(s))
    graph.add_node("draft_schedule", lambda s: drafting_agent.run(s))
    graph.add_node("execute_and_verify", lambda s: verification_agent.run(s))
    graph.add_node("refine_schedule", lambda s: refinement_agent.run(s))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("parse_preferences")
    graph.add_edge("parse_preferences", "draft_schedule")
    graph.add_edge("draft_schedule", "execute_and_verify")
    graph.add_edge("refine_schedule", "execute_and_verify")

    graph.add_conditional_edges(
        "execute_and_verify",
        route_after_verify,
        {
            "draft": "draft_schedule",
            "refine": "refine_schedule",
            "end": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()
