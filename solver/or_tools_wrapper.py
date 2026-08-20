"""Executes LLM-generated OR-Tools scripts in a sandboxed subprocess and
parses the JSON result they print to stdout.

Sandboxing rationale: the drafting/refinement agents produce arbitrary
Python. Running it out-of-process with a timeout means a syntax error,
an infinite loop, or a crash in generated code can never take down the
orchestrator process itself.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import config
from utils.exceptions import SolverExecutionError


def run_generated_code(code: str, timeout: int = None) -> Dict:
    """Writes `code` to a temp file and runs it with `python <file>`.

    Returns a dict:
        {
            "success": bool,
            "status": "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN" | "ERROR",
            "schedule": Optional[Dict],  # {day: {shift: [worker_ids]}}
            "objective_value": Optional[float],
            "error": Optional[str],       # populated on success=False
            "stdout": str,
            "stderr": str,
        }
    Never raises for "expected" failure modes (syntax errors, infeasible
    models, timeouts) -- those come back as success=False with `error`
    set, so the verification node can route to refinement. Only truly
    unexpected failures raise SolverExecutionError.
    """
    timeout = timeout or config.SOLVER_TIMEOUT_SECONDS

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / config.TEMP_MODEL_FILENAME
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "status": "TIMEOUT",
                "schedule": None,
                "objective_value": None,
                "error": f"Solver timed out after {timeout}s",
                "stdout": (e.stdout or ""),
                "stderr": (e.stderr or ""),
            }
        except OSError as e:  # pragma: no cover - environment failure
            raise SolverExecutionError(f"Could not launch subprocess: {e}") from e

        if proc.returncode != 0:
            return {
                "success": False,
                "status": "ERROR",
                "schedule": None,
                "objective_value": None,
                "error": _tail(proc.stderr) or "Non-zero exit with no stderr",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        result = _extract_last_json(proc.stdout)
        if result is None:
            return {
                "success": False,
                "status": "ERROR",
                "schedule": None,
                "objective_value": None,
                "error": "Script exited 0 but printed no valid JSON result",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        status = result.get("status", "UNKNOWN")
        if status not in ("OPTIMAL", "FEASIBLE") or result.get("schedule") is None:
            return {
                "success": False,
                "status": status,
                "schedule": None,
                "objective_value": None,
                "error": f"Solver status={status} (no feasible schedule found)",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        return {
            "success": True,
            "status": status,
            "schedule": result["schedule"],
            "objective_value": result.get("objective_value"),
            "error": None,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }


def _extract_last_json(stdout: str):
    """The script may print solver logs before the final
    print(json.dumps(result)); scan from the last line backwards for
    valid JSON."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _tail(text: str, n_lines: int = 25) -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    return "\n".join(lines[-n_lines:])
