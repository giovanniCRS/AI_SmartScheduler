"""Ties everything together: parse input -> build initial state -> run
the LangGraph -> persist outputs/schedule_final.json,
outputs/schedule_model.py, outputs/fairness_report.txt and
outputs/schedule_table.png.
"""
import datetime
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

    _render_schedule_table(
        schedule=state["schedule_solution"],
        output_dir=output_dir,
        case_type=state["case_type"],
        is_complete=state["is_complete"],
        iterations=state["iteration"],
    )

    logger.info(f"Outputs written to {output_dir}/")


def _render_schedule_table(
    schedule: Dict, output_dir: str, case_type: str, is_complete: bool, iterations: int
) -> None:
    """Genera outputs/schedule_table.png (giorno x turno -> lavoratori) a
    partire dallo schedule finale scelto dal grafo. Se matplotlib non e'
    installato, salta la generazione senza far fallire il resto del run
    (i tre output principali sono comunque gia' stati scritti sopra)."""
    if not schedule:
        logger.info("Nessuno schedule valido da disegnare: salto la tabella PNG.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info(
            "matplotlib non installato: salto la generazione della tabella "
            "PNG (pip install matplotlib per abilitarla)."
        )
        return

    start_date = datetime.date.fromisoformat(config.HORIZON_START)
    giorni = sorted(schedule.keys(), key=int)
    nomi_turni = ["Mattina", "Pomeriggio", "Notte"]

    etichette_righe, righe = [], []
    for g in giorni:
        data_giorno = start_date + datetime.timedelta(days=int(g))
        etichette_righe.append(f"Giorno {g}\n{data_giorno.strftime('%d/%m/%Y')}")
        riga = []
        for turno in ["0", "1", "2"]:
            lavoratori = schedule[g].get(turno, [])
            riga.append(", ".join(str(w) for w in lavoratori) if lavoratori else "-")
        righe.append(riga)

    fig, ax = plt.subplots(figsize=(10, len(giorni) * 0.4 + 1))
    ax.axis("off")
    stato = "completo" if is_complete else "miglior tentativo"
    ax.set_title(
        f"SmartScheduler — Caso {case_type} — {stato} ({iterations} iterazioni)",
        fontsize=13, fontweight="bold", pad=14,
    )
    tabella = ax.table(
        cellText=righe, rowLabels=etichette_righe, colLabels=nomi_turni,
        loc="center", cellLoc="center",
    )
    tabella.auto_set_font_size(False)
    tabella.set_fontsize(9)
    tabella.scale(1, 1.8)

    output_path = os.path.join(output_dir, "schedule_table.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Tabella immagine salvata in: {output_path}")