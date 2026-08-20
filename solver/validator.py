"""Symbolic (non-LLM) validation of a generated schedule against every
hard constraint in the brief. Returns a list of specific, human-readable
error strings (worker + day, so the refinement agent can act on them
directly) rather than booleans.
"""
from typing import Dict, List

import config


def validate_schedule(schedule: Dict, workers: List[Dict], case_type: str) -> List[str]:
    """schedule: {day:int -> {shift:int -> [worker_id, ...]}}."""
    errors: List[str] = []
    worker_ids = [w["id"] for w in workers]
    roles = {w["id"]: w["role"] for w in workers}
    horizon = config.HORIZON_DAYS

    schedule = {int(d): {int(s): list(ws) for s, ws in shifts.items()}
                for d, shifts in schedule.items()}

    errors += _check_min_coverage(schedule, roles, case_type, horizon)
    errors += _check_max_one_shift_per_day(schedule, worker_ids, horizon)
    errors += _check_no_night_to_morning(schedule, worker_ids, horizon)
    errors += _check_rest_after_night(schedule, worker_ids, horizon)
    errors += _check_monthly_weighted_load(schedule, worker_ids, horizon)
    errors += _check_weekly_weighted_load(schedule, worker_ids)
    errors += _check_at_least_one_day_off(schedule, worker_ids, horizon)
    return errors


def _works(schedule, w, d, s):
    return w in schedule.get(d, {}).get(s, [])


def _shift_weight(s: int) -> int:
    return config.SHIFT_WEIGHTS[s]


def _check_min_coverage(schedule, roles, case_type, horizon) -> List[str]:
    errors = []
    for d in range(horizon):
        for s in range(3):
            assigned = schedule.get(d, {}).get(s, [])
            if case_type == "A":
                if len(assigned) < 2:
                    errors.append(
                        f"Copertura insufficiente giorno {d} turno "
                        f"{config.SHIFT_NAMES[s]}: {len(assigned)} lavoratori (minimo 2)"
                    )
            else:
                standard = [w for w in assigned if roles.get(w) == "standard"]
                specialized = [w for w in assigned if roles.get(w) == "specialized"]
                if len(standard) < 2:
                    errors.append(
                        f"Copertura standard insufficiente giorno {d} turno "
                        f"{config.SHIFT_NAMES[s]}: {len(standard)} standard (minimo 2)"
                    )
                if len(specialized) < 1:
                    errors.append(
                        f"Copertura specializzata insufficiente giorno {d} turno "
                        f"{config.SHIFT_NAMES[s]}: {len(specialized)} specializzati (minimo 1)"
                    )
    return errors


def _check_max_one_shift_per_day(schedule, worker_ids, horizon) -> List[str]:
    errors = []
    for w in worker_ids:
        for d in range(horizon):
            count = sum(1 for s in range(3) if _works(schedule, w, d, s))
            if count > 1:
                errors.append(
                    f"Lavoratore {w} ha piu' di un turno il giorno {d} ({count} turni)"
                )
    return errors


def _check_no_night_to_morning(schedule, worker_ids, horizon) -> List[str]:
    errors = []
    for w in worker_ids:
        for d in range(horizon - 1):
            if _works(schedule, w, d, config.SHIFT_NIGHT) and _works(
                schedule, w, d + 1, config.SHIFT_MORNING
            ):
                errors.append(
                    f"Lavoratore {w}: turno notte giorno {d} seguito da "
                    f"mattina giorno {d + 1} (vietato)"
                )
    return errors


def _check_rest_after_night(schedule, worker_ids, horizon) -> List[str]:
    errors = []
    for w in worker_ids:
        for d in range(horizon - 2):
            if _works(schedule, w, d, config.SHIFT_NIGHT):
                for offset in (1, 2):
                    day_off_check = d + offset
                    if day_off_check < horizon and any(
                        _works(schedule, w, day_off_check, s) for s in range(3)
                    ):
                        errors.append(
                            f"Lavoratore {w} ha violato il riposo post-notturno: "
                            f"turno notte giorno {d}, ma lavora anche giorno {day_off_check}"
                        )
    return errors


def _check_monthly_weighted_load(schedule, worker_ids, horizon) -> List[str]:
    errors = []
    for w in worker_ids:
        total = sum(
            _shift_weight(s)
            for d in range(horizon)
            for s in range(3)
            if _works(schedule, w, d, s)
        )
        if total != config.MONTHLY_TARGET_WEIGHT:
            errors.append(
                f"Lavoratore {w}: carico mensile pesato = {total} "
                f"(atteso esattamente {config.MONTHLY_TARGET_WEIGHT})"
            )
    return errors


def _check_weekly_weighted_load(schedule, worker_ids) -> List[str]:
    errors = []
    for w in worker_ids:
        for week_idx, (start, end) in enumerate(config.WEEKS, start=1):
            total = sum(
                _shift_weight(s)
                for d in range(start, end + 1)
                for s in range(3)
                if _works(schedule, w, d, s)
            )
            if total > config.MAX_WEEKLY_WEIGHT:
                errors.append(
                    f"Lavoratore {w}: carico settimana S{week_idx} "
                    f"(giorni {start}-{end}) = {total} (massimo {config.MAX_WEEKLY_WEIGHT})"
                )
    return errors


def _check_at_least_one_day_off(schedule, worker_ids, horizon) -> List[str]:
    errors = []
    for w in worker_ids:
        has_day_off = any(
            all(not _works(schedule, w, d, s) for s in range(3))
            for d in range(horizon)
        )
        if not has_day_off:
            errors.append(f"Lavoratore {w}: nessun giorno completamente libero nel mese")
    return errors
