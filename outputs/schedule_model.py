from ortools.sat.python import cp_model
import json

num_workers = 13
case_type = "A"
worker_roles = {0: 'standard', 1: 'standard', 2: 'standard', 3: 'standard', 4: 'standard', 5: 'standard', 6: 'standard', 7: 'standard', 8: 'standard', 9: 'standard', 10: 'standard', 11: 'standard', 12: 'standard'}
weekend_days = [5, 6, 12, 13, 19, 20, 26, 27]
holidays = [1, 18, 19, 25, 30]
preferences = {0: {'preferred_shifts': ['morning'], 'avoid_shifts': ['night'], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 0}, 1: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 1}, 2: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 2}, 3: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'avoid', 'holiday_tolerance': 0.5, 'id': 3}, 4: {'preferred_shifts': ['afternoon'], 'avoid_shifts': [], 'max_consecutive_nights': 1, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 4}, 5: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.0, 'id': 5}, 6: {'preferred_shifts': ['morning'], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 6}, 7: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.0, 'id': 7}, 8: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 8}, 9: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 9}, 10: {'preferred_shifts': ['afternoon'], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'avoid', 'holiday_tolerance': 0.5, 'id': 10}, 11: {'preferred_shifts': [], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 0.5, 'id': 11}, 12: {'preferred_shifts': ['morning'], 'avoid_shifts': [], 'max_consecutive_nights': 2, 'weekend_preference': 'neutral', 'holiday_tolerance': 1.0, 'id': 12}}
# ===== LLM-generated logic below =====
model = cp_model.CpModel()

# Variables: x[w][d][s] = 1 if worker w works day d (0..30) shift s
# (0=morning, 1=afternoon, 2=night)
x = {}
for w in range(num_workers):
    x[w] = {}
    for d in range(31):
        x[w][d] = {}
        for s in range(3):
            x[w][d][s] = model.NewBoolVar(f'x_{w}_{d}_{s}')

# HARD CONSTRAINT 1: at most one shift per day
for w in range(num_workers):
    for d in range(31):
        model.Add(sum(x[w][d][s] for s in range(3)) <= 1)

# HARD CONSTRAINT 2: no back-to-back night -> next-day morning
for w in range(num_workers):
    for d in range(30):
        model.Add(x[w][d][2] + x[w][d + 1][0] <= 1)

# HARD CONSTRAINT 3: 2 full days off after a night shift
for w in range(num_workers):
    for d in range(29):
        for offset in [1, 2]:
            model.Add(
                x[w][d][2] + sum(x[w][d + offset][s] for s in range(3)) <= 1
            )

# HARD CONSTRAINT 4: monthly weighted load == 25
for w in range(num_workers):
    model.Add(
        sum(x[w][d][0] + x[w][d][1] + 2 * x[w][d][2] for d in range(31)) == 25
    )

# HARD CONSTRAINT 5: weekly weighted load <= 6 (fixed Mon-Sun windows)
weeks = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 30)]
for w in range(num_workers):
    for start, end in weeks:
        model.Add(
            sum(
                x[w][d][0] + x[w][d][1] + 2 * x[w][d][2]
                for d in range(start, end + 1)
            )
            <= 6
        )

# HARD CONSTRAINT 6: at least one fully free day in the horizon
for w in range(num_workers):
    day_off_vars = []
    for d in range(31):
        day_off = model.NewBoolVar(f'day_off_{w}_{d}')
        model.Add(sum(x[w][d][s] for s in range(3)) == 0).OnlyEnforceIf(day_off)
        day_off_vars.append(day_off)
    model.Add(sum(day_off_vars) >= 1)

# HARD CONSTRAINT 7: minimum coverage per shift
days = range(31)
shifts = range(3)
if case_type == "A":
    for d in days:
        for s in shifts:
            model.Add(sum(x[w][d][s] for w in range(num_workers)) >= 2)
else:  # Case B: >=2 standard + >=1 specialized
    # worker ids 0..12 are standard, 13..19 are specialized (per worker_roles)
    standard_workers = [w for w in range(num_workers) if worker_roles[w] == "standard"]
    specialized_workers = [w for w in range(num_workers) if worker_roles[w] == "specialized"]
    for d in days:
        for s in shifts:
            # At least 2 standard
            model.Add(sum(x[w][d][s] for w in standard_workers) >= 2)
            # At least 1 specialized
            model.Add(sum(x[w][d][s] for w in specialized_workers) >= 1)

# SOFT CONSTRAINTS: preferences, encoded as penalty terms to minimize.
# Fairness adjustment: Increase weights for least-satisfied workers (1, 2, 3, 9, 10)
# Original weights: Night=10, Afternoon=4, Morning=4, Weekend=3, Holiday=5 (avg)
# New weights for least-satisfied: Night=20, Afternoon=8, Morning=8, Weekend=6, Holiday=10
least_satisfied_workers = {1, 2, 3, 9, 10}

objective_terms = []
for w in range(num_workers):
    pref = preferences[w]
    
    # Determine base weights
    if w in least_satisfied_workers:
        night_weight = 20
        afternoon_weight = 8
        morning_weight = 8
        weekend_weight = 6
        holiday_base_mult = 20 # 10 * (1 - 0.5) * 2 roughly, or just higher base
    else:
        night_weight = 10
        afternoon_weight = 4
        morning_weight = 4
        weekend_weight = 3
        holiday_base_mult = 10

    for d in range(31):
        if "night" in pref.get("avoid_shifts", []):
            objective_terms.append(night_weight * x[w][d][2])
        if "afternoon" in pref.get("avoid_shifts", []):
            objective_terms.append(afternoon_weight * x[w][d][1])
        if "morning" in pref.get("avoid_shifts", []):
            objective_terms.append(morning_weight * x[w][d][0])
    
    if pref.get("weekend_preference") == "avoid":
        for d in weekend_days:
            objective_terms.append(weekend_weight * sum(x[w][d][s] for s in range(3)))
            
    # Holiday cost calculation adjusted for fairness
    # Original: int(round(10 * (1 - pref.get("holiday_tolerance", 0.5))))
    # If tolerance is 0.5, cost is 5.
    # For least satisfied, we want to penalize working holidays more if they avoid them,
    # or generally weight their holiday preference higher.
    # Let's scale the holiday cost by 2 for least satisfied workers.
    holiday_tolerance = pref.get("holiday_tolerance", 0.5)
    base_holiday_cost = int(round(10 * (1 - holiday_tolerance)))
    if w in least_satisfied_workers:
        holiday_cost = base_holiday_cost * 2
    else:
        holiday_cost = base_holiday_cost
        
    for d in holidays:
        objective_terms.append(holiday_cost * sum(x[w][d][s] for s in range(3)))

model.Minimize(sum(objective_terms))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

result = {"status": solver.StatusName(status)}
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    schedule = {}
    for d in range(31):
        schedule[d] = {}
        for s in range(3):
            schedule[d][s] = [w for w in range(num_workers) if solver.Value(x[w][d][s])]
    result["schedule"] = schedule
    result["objective_value"] = solver.ObjectiveValue()
else:
    result["schedule"] = None

print(json.dumps(result))
