"""SmartScheduler entry point.

Usage:
    python main.py --input examples/input_case_a.txt
    python main.py --input examples/input_case_b.txt --output outputs/
"""
import argparse
import sys

import config
from core.orchestrator import run_from_file
from utils.exceptions import SchedulerError
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SmartScheduler end-to-end.")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to the input text file."
    )
    parser.add_argument(
        "--output", "-o", default=config.OUTPUT_DIR, help="Output directory."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        final_state = run_from_file(args.input, args.output)
    except SchedulerError as e:
        logger.error(f"SmartScheduler failed: {e}")
        return 1
    except Exception as e:  # unexpected
        logger.error(f"Unexpected error: {e}")
        raise

    if final_state["is_complete"]:
        print(f"\nSchedule generated successfully in {final_state['iteration']} iteration(s).")
        return 0
    else:
        print(
            f"\nReached max_iterations={final_state['max_iterations']} without a "
            "fully satisfactory schedule. Best attempt saved to outputs/ "
            "(see hard_errors / fairness_scores in schedule_final.json)."
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
