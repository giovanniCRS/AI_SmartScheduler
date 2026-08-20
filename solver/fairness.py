"""Computes a per-worker satisfaction score from the (already
hard-constraint-valid) schedule and the parsed soft preferences, and
identifies the least-satisfied worker(s) for the refinement loop.

Score is in [0, 1]; 1 = every soft preference respected, 0 = every shift
worked directly contradicts stated preferences.
"""
from typing import Dict, List

import config


def calculate_fairness(schedule: Dict, parsed_preferences: Dict) -> Dict:
    schedule = {int(d): {int(s): list(ws) for s, ws in shifts.items()}
                for d, shifts in schedule.items()}
    prefs_by_id = {p["id"]: p for p in parsed_preferences.get("workers", [])}

    scores: Dict[int, float] = {}
    for worker_id, pref in prefs_by_id.items():
        scores[worker_id] = _score_worker(worker_id, pref, schedule)

    if not scores:
        return {"scores": {}, "least_satisfied": []}

    min_score = min(scores.values())
    least_satisfied = [w for w, s in scores.items() if abs(s - min_score) < 1e-9]
    return {"scores": scores, "least_satisfied": least_satisfied}


def _score_worker(worker_id: int, pref: Dict, schedule: Dict) -> float:
    assigned = [
        (d, s)
        for d, shifts in schedule.items()
        for s, workers in shifts.items()
        if worker_id in workers
    ]
    total = len(assigned)
    if total == 0:
        return 0.5  # neutral: nothing to evaluate against

    preferred = set(pref.get("preferred_shifts", []))
    avoid = set(pref.get("avoid_shifts", []))
    weekend_pref = pref.get("weekend_preference", "neutral")
    holiday_tolerance = float(pref.get("holiday_tolerance", 0.5))
    max_consec_nights = int(pref.get("max_consecutive_nights", 2))

    bonus = 0.0
    penalty = 0.0

    for d, s in assigned:
        label = config.SHIFT_NAMES[s]
        if label in preferred:
            bonus += 1.0
        if label in avoid:
            penalty += 1.0
        if d in config.WEEKEND_DAYS:
            if weekend_pref == "avoid":
                penalty += 0.5
            elif weekend_pref == "prefer":
                bonus += 0.5
        if d in config.HOLIDAYS:
            penalty += (1.0 - holiday_tolerance)

    penalty += _consecutive_night_penalty(worker_id, schedule, max_consec_nights)

    raw = 0.5 + (bonus - penalty) / (2.0 * total)
    return max(0.0, min(1.0, raw))


def _consecutive_night_penalty(worker_id: int, schedule: Dict, max_consec: int) -> float:
    night_days = sorted(
        d for d, shifts in schedule.items()
        if worker_id in shifts.get(config.SHIFT_NIGHT, [])
    )
    if not night_days:
        return 0.0
    penalty = 0.0
    streak = 1
    for i in range(1, len(night_days)):
        if night_days[i] == night_days[i - 1] + 1:
            streak += 1
        else:
            if streak > max_consec:
                penalty += (streak - max_consec)
            streak = 1
    if streak > max_consec:
        penalty += (streak - max_consec)
    return penalty


def format_fairness_report(fairness: Dict, workers: List[Dict]) -> str:
    """Human-readable report for outputs/fairness_report.txt."""
    lines = ["SmartScheduler - Fairness Report", "=" * 40, ""]
    scores = fairness.get("scores", {})
    for w in sorted(scores, key=lambda k: scores[k]):
        lines.append(f"Worker {w}: satisfaction = {scores[w]:.3f}")
    least = fairness.get("least_satisfied", [])
    lines.append("")
    lines.append(f"Least satisfied worker(s): {least}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        lines.append(f"Average satisfaction: {avg:.3f}")
        lines.append(f"Min satisfaction: {min(scores.values()):.3f}")
        lines.append(f"Max satisfaction: {max(scores.values()):.3f}")
    return "\n".join(lines)
