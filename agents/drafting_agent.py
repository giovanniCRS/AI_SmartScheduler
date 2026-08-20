"""Phase 2 agent: generates the OR-Tools CP-SAT script (as a Python
source string) implementing the hard-constraint template plus a
preference-weighted objective.
"""
import re
from typing import Dict

import config
from agents.base import Agent
from core.state import ScheduleState
from llm.groq_client import GroqClient, strip_reasoning_tags
from llm.prompt_templates import PREAMBLE_LOGIC_MARKER, build_preamble, drafting_prompt
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DraftingAgent(Agent):
    name = "drafting_agent"

    def __init__(self, client: GroqClient):
        self.client = client

    def run(self, state: ScheduleState) -> Dict:
        preamble = build_preamble(
            state["workers"], state["case_type"], state["parsed_preferences"]
        )
        prompt = drafting_prompt(
            workers=state["workers"],
            case_type=state["case_type"],
            parsed_preferences=state["parsed_preferences"],
        )
        raw_response = self.client.call(
            prompt,
            max_tokens=config.MAX_TOKENS_DRAFTING,
            temperature=config.DRAFTING_TEMPERATURE,
            reasoning_effort=config.CODE_REASONING_EFFORT,
        )
        logic = _extract_code(raw_response)
        if len(logic.strip()) < 50:
            preview = (raw_response or "")[:500]
            logger.error(
                f"DraftingAgent: LLM returned little/no code (raw response "
                f"preview): {preview!r}"
            )
        full_code = f"{preamble}\n{PREAMBLE_LOGIC_MARKER}\n{logic}\n"
        logger.info(f"DraftingAgent produced {len(logic.splitlines())} lines of logic")
        return {
            "generated_code": full_code,
            "code_error": None,
            "hard_errors": [],
        }


def _extract_code(raw_response: str) -> str:
    """Strip any leftover <think> block and markdown fences if the LLM
    added them despite instructions."""
    text = strip_reasoning_tags(raw_response)
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
