"""Parses the plain-text input file describing the scheduling instance.

Expected format (simple key: value header + free-text preferences block),
matching the "Input" section of the project brief:

    case_type: A
    num_workers: 13
    shifts: morning=8-14, afternoon=14-20, night=20-8
    horizon_start: 2026-12-07
    horizon_days: 31

    preferences:
    Il Lavoratore 0 preferisce i turni del mattino e vorrebbe evitare
    i turni notturni quando possibile.
    Il Lavoratore 1 puo' lavorare nei weekend, ma non in giorni festivi
    consecutivi.
    ...

For case B, `num_standard` and `num_specialized` replace `num_workers`;
worker ids 0..num_standard-1 are standard, the rest are specialized
(mirrors the convention used in the OR-Tools template).
"""
from typing import Dict, List

from utils.exceptions import InputParsingError


def parse_input_file(content: str) -> Dict:
    """Parse raw file content into a dict with workers, case_type and
    raw_preferences, ready to seed ScheduleState."""
    header, _, prefs_block = content.partition("preferences:")
    fields = _parse_header(header)

    case_type = fields.get("case_type", "A").strip().upper()
    if case_type not in ("A", "B"):
        raise InputParsingError(f"case_type must be 'A' or 'B', got {case_type!r}")

    workers: List[Dict] = []
    if case_type == "A":
        num_workers = int(fields.get("num_workers", 13))
        workers = [{"id": i, "role": "standard"} for i in range(num_workers)]
    else:
        num_standard = int(fields.get("num_standard", 13))
        num_specialized = int(fields.get("num_specialized", 7))
        workers = [{"id": i, "role": "standard"} for i in range(num_standard)]
        workers += [
            {"id": i, "role": "specialized"}
            for i in range(num_standard, num_standard + num_specialized)
        ]

    if not workers:
        raise InputParsingError("No workers parsed from input file.")

    raw_preferences = prefs_block.strip()
    if not raw_preferences:
        raise InputParsingError("No 'preferences:' block found in input file.")

    return {
        "case_type": case_type,
        "workers": workers,
        "raw_preferences": raw_preferences,
    }


def _parse_header(header: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()
    return fields
