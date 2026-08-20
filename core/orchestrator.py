"""Ties everything together: parse input -> build initial state -> run
the LangGraph -> persist outputs/schedule_final.json,
outputs/schedule_model.py and outputs/fairness_report.txt.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict

import config
from core.graph import build_graph
from core.state import ScheduleState, new_initial_state
from llm.groq_client import GroqClient
from solver.fairness import format_fairness_report
from utils.exceptions import SchedulerError
from utils.file_parser import parse_input_file
from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_from_file(input_path: str, output_dir: str = config.OUTPUT_DIR) -> ScheduleState:
    content = Path(input_path).read_text(encoding="utf-8")
    parsed_input = parse_input_file(content)

    client = GroqClient()
    graph = build_graph(client)

    initial_state = new_initial_state(
        input_file_content=content,
        workers=parsed_input["workers"],
        case_type=parsed_input["case_type"],
        raw_preferences=parsed_input["raw_preferences"],
        max_iterations=config.MAX_ITERATIONS,
    )

    logger.info(
        f"Starting SmartScheduler: case_type={parsed_input['case_type']} "
        f"workers={len(parsed_input['workers'])} max_iterations={config.MAX_ITERATIONS}"
    )
    start = time.perf_counter()
    final_state: ScheduleState = graph.invoke(initial_state)
    elapsed = time.perf_counter() - start

    logger.info(
        f"Finished after {final_state['iteration']} iteration(s), "
        f"{elapsed:.1f}s, is_complete={final_state['is_complete']}, "
        f"groq_calls={client.total_calls}, groq_tokens~{client.total_tokens_used}"
    )

    _write_outputs(final_state, output_dir)
    return final_state


def _write_outputs(state: ScheduleState, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    schedule_path = os.path.join(output_dir, config.SCHEDULE_JSON_FILENAME)
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_complete": state["is_complete"],
                "iterations": state["iteration"],
                "case_type": state["case_type"],
                "schedule": state["schedule_solution"],
                "hard_errors": state["hard_errors"],
                "fairness_scores": state["fairness_scores"],
                "least_satisfied": state["least_satisfied"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    model_path = os.path.join(output_dir, config.SCHEDULE_MODEL_FILENAME)
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(state["generated_code"] or "# No code was successfully generated.\n")

    report_path = os.path.join(output_dir, config.FAIRNESS_REPORT_FILENAME)
    fairness_payload = {
        "scores": state["fairness_scores"],
        "least_satisfied": state["least_satisfied"],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(format_fairness_report(fairness_payload, state["workers"]))

    logger.info(f"Outputs written to {output_dir}/")
