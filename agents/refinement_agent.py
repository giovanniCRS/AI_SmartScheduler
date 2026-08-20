"""Phase 4 agent: patches the previous OR-Tools script using specific
hard-constraint violations and/or fairness feedback from the verification
agent, without touching hard constraints (only soft-constraint weights /
bug fixes should change).
"""
from typing import Dict

import config
from agents.base import Agent
from agents.drafting_agent import _extract_code
from core.state import ScheduleState
from llm.groq_client import GroqClient
from llm.prompt_templates import PREAMBLE_LOGIC_MARKER, build_preamble, refinement_prompt
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RefinementAgent(Agent):
    name = "refinement_agent"

    def __init__(self, client: GroqClient):
        self.client = client

    def run(self, state: ScheduleState) -> Dict:
        preamble, previous_logic = _split_preamble(
            state["generated_code"], state["workers"], state["case_type"],
            state["parsed_preferences"],
        )
        prompt = refinement_prompt(
            previous_code=previous_logic,
            hard_errors=state["hard_errors"],
            fairness_scores=state["fairness_scores"],
            least_satisfied=state["least_satisfied"],
            code_error=state["code_error"],
        )
        raw_response = self.client.call(
            prompt,
            max_tokens=config.MAX_TOKENS_REFINEMENT,
            temperature=config.TEMPERATURE,
            reasoning_effort=config.CODE_REASONING_EFFORT,
        )
        logic = _extract_code(raw_response)
        full_code = f"{preamble}\n{PREAMBLE_LOGIC_MARKER}\n{logic}\n"
        logger.info(f"RefinementAgent produced {len(logic.splitlines())} lines of logic")
        return {
            "generated_code": full_code,
            "code_error": None,
        }


def _split_preamble(generated_code: str, workers, case_type, parsed_preferences):
    """Splits a previously-assembled script into (preamble, logic) using
    the marker DraftingAgent/RefinementAgent insert between them. Reusing
    the SAME preamble (rather than rebuilding it) guarantees it stays
    byte-identical across refinement rounds. Falls back to rebuilding it
    fresh if the marker is missing (e.g. legacy state)."""
    if PREAMBLE_LOGIC_MARKER in generated_code:
        preamble, _, logic = generated_code.partition(PREAMBLE_LOGIC_MARKER)
        return preamble.strip(), logic.strip()
    logger.info(
        "RefinementAgent: marker not found in previous generated_code; "
        "rebuilding preamble and treating the whole thing as logic."
    )
    preamble = build_preamble(workers, case_type, parsed_preferences)
    return preamble, generated_code.strip()
