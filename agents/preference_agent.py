"""Phase 1 agent: turns natural-language worker preferences into the
structured JSON used to build the soft-constraint objective later.
"""
import json
import re
from typing import Dict

import config
from agents.base import Agent
from core.state import ScheduleState
from llm.groq_client import GroqClient, strip_reasoning_tags
from llm.prompt_templates import preference_prompt
from utils.exceptions import CodeGenerationError
from utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_PREF = {
    "preferred_shifts": [],
    "avoid_shifts": [],
    "max_consecutive_nights": 2,
    "weekend_preference": "neutral",
    "holiday_tolerance": 0.5,
}


class PreferenceAgent(Agent):
    name = "preference_agent"

    def __init__(self, client: GroqClient):
        self.client = client

    def run(self, state: ScheduleState) -> Dict:
        # Cached: don't re-run phase 1 on refinement iterations (token budget).
        if state.get("parsed_preferences") and state["iteration"] > 0:
            return {}

        worker_ids = [w["id"] for w in state["workers"]]
        prompt = preference_prompt(state["raw_preferences"], worker_ids)
        raw_response = self.client.call(
            prompt,
            max_tokens=config.MAX_TOKENS_PREFERENCES,
            temperature=config.TEMPERATURE,
            response_format={"type": "json_object"},
            reasoning_effort=config.PREFERENCE_REASONING_EFFORT,
        )
        parsed = _parse_json_response(raw_response)
        parsed = _fill_missing_workers(parsed, worker_ids)

        logger.info(f"Parsed preferences for {len(parsed['workers'])} workers")
        return {"parsed_preferences": parsed}


def _parse_json_response(raw_response: str) -> Dict:
    text = strip_reasoning_tags(raw_response)
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # try to salvage the first {...} block
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        preview = (raw_response or "")[:500]
        logger.error(f"PreferenceAgent: unparseable LLM response (first 500 chars): {preview!r}")
        raise CodeGenerationError(
            f"PreferenceAgent: LLM did not return valid JSON: {e}. "
            f"Raw response preview: {preview!r}"
        ) from e


def _fill_missing_workers(parsed: Dict, worker_ids) -> Dict:
    by_id = {w["id"]: w for w in parsed.get("workers", [])}
    complete = []
    for wid in worker_ids:
        entry = by_id.get(wid, {"id": wid})
        merged = {**_DEFAULT_PREF, **entry, "id": wid}
        complete.append(merged)
    return {"workers": complete}
