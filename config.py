"""
Configuration for SmartScheduler.

NOTE: per project instructions, credentials and model name are hardcoded
here rather than read from environment variables. Replace GROQ_API_KEY
with a real key before running against the live Groq API.
"""

# --- Groq / LLM configuration -----------------------------------------
GROQ_API_KEY = "API_KEY_GROQ"      #Inserire quì l'API di groq
MODEL_NAME = "qwen/qwen3.6-27b"

# Groq free tier: 8000 tokens/minute. The GroqClient enforces this with a
# token-bucket so we never get 429s.
RATE_LIMIT_TOKENS_PER_MINUTE = 8000
RATE_LIMIT_WINDOW_SECONDS = 60

# Per-call generation limits (kept tight to save budget)
# Per-call generation limits (kept tight to save budget). The template
# reproduced by drafting/refinement is ~1000 tokens on its own; these
# give it headroom for the preamble variables (worker/preferences data)
# on top, now that reasoning is disabled for these calls (see
# CODE_REASONING_EFFORT below) -- originally 2000/1500, raised slightly
# after real runs showed the template alone leaves little margin for
# Case B's larger worker/preferences payload.
# Per-call generation limits (kept tight to save budget). Since the
# preamble (num_workers/worker_roles/preferences) is now injected
# deterministically rather than asked of the LLM (see build_preamble in
# prompt_templates.py), the actual generation task -- the constraints +
# objective "continuation" -- is a near-constant ~1025 tokens regardless
# of worker count (measured directly from DRAFTING_LOGIC_TEMPLATE),
# unlike before when it scaled with Case B's larger worker list and
# caused a real truncated-script failure. This restores headroom close
# to the brief's original 2000/1500 targets while staying safe.
MAX_TOKENS_DRAFTING = 2000
MAX_TOKENS_REFINEMENT = 2000
MAX_TOKENS_PREFERENCES = 1200
TEMPERATURE = 0.2
DRAFTING_TEMPERATURE = 0.3

# Retry / backoff for transient Groq errors
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0

# Groq "reasoning" models (Qwen3 family incl. qwen/qwen3.6-27b, GPT-OSS,
# DeepSeek-R1-distill, ...) think before answering; unless told otherwise
# the chain-of-thought can end up in `content` (raw) or leave it empty
# depending on the model, which breaks naive parsing downstream. Keep
# this "hidden" so response.choices[0].message.content is always just
# the clean final answer. Set to None if you switch to a non-reasoning
# model that rejects the parameter.
REASONING_FORMAT = "hidden"  # "hidden" | "parsed" | "raw" | None

# Preference extraction (Phase 1) is a mechanical text -> JSON task, not
# something that benefits from multi-step reasoning. Combining a
# reasoning model with response_format=json_object has a known failure
# mode on Groq (400 json_validate_failed, empty failed_generation) when
# the model spends its budget "thinking" instead of emitting JSON.
# Disabling reasoning for this one call sidesteps it and is faster/
# cheaper too. Qwen3-only parameter; ignored (auto-dropped) otherwise.
PREFERENCE_REASONING_EFFORT = "none"  # "none" | "default" | None

# Drafting/refinement mostly reproduce a fixed template + light
# adaptation, not something that benefits from extended chain-of-thought
# -- and critically, on Qwen3 reasoning tokens are drawn from the SAME
# max_tokens budget as the visible answer even with reasoning_format=
# hidden. In testing, the model spent the entire max_tokens budget on
# hidden reasoning and returned 0 tokens of actual code every time.
# Disabling reasoning for these two calls fixes that.
CODE_REASONING_EFFORT = "none"  # "none" | "default" | None

# --- Scheduling horizon --------------------------------------------------
# 7 Dec 2026 (Monday) -> 6 Jan 2027 inclusive = 31 days, index 0..30
HORIZON_START = "2026-12-07"
HORIZON_DAYS = 31

# Weekday index (0=Mon..6=Sun) of day 0. 7 Dec 2026 is a Monday.
HORIZON_START_WEEKDAY = 0

# Shift indices
SHIFT_MORNING = 0
SHIFT_AFTERNOON = 1
SHIFT_NIGHT = 2
SHIFT_NAMES = {0: "morning", 1: "afternoon", 2: "night"}
SHIFT_WEIGHTS = {0: 1, 1: 1, 2: 2}  # night counts double (8h vs 6h shifts)

# Fixed Monday-Sunday weekly windows over the 31-day horizon (0-indexed, inclusive)
WEEKS = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 30)]
MAX_WEEKLY_WEIGHT = 6
MONTHLY_TARGET_WEIGHT = 25

# Weekend day indices (Sat/Sun) within the horizon, 0-indexed
WEEKEND_DAYS = [5, 6, 12, 13, 19, 20, 26, 27]

# Italian public holidays falling inside the horizon, 0-indexed day numbers
# 08/12 Immacolata, 25/12 Natale, 26/12 S.Stefano, 01/01 Capodanno, 06/01 Epifania
HOLIDAYS = [1, 18, 19, 25, 30]

# --- Use cases ------------------------------------------------------------
CASE_A_WORKERS = 13          # homogeneous
CASE_B_STANDARD_WORKERS = 13
CASE_B_SPECIALIZED_WORKERS = 7  # total 20 for case B

# --- Orchestration ---------------------------------------------------------
MAX_ITERATIONS = 5
FAIRNESS_GAP_THRESHOLD = 0.15  # acceptable gap between least- and best-satisfied worker

# --- Sandbox execution ------------------------------------------------------
SOLVER_TIMEOUT_SECONDS = 120
TEMP_MODEL_FILENAME = "temp_model.py"

# --- Paths -------------------------------------------------------------------
OUTPUT_DIR = "outputs"
SCHEDULE_JSON_FILENAME = "schedule_final.json"
SCHEDULE_MODEL_FILENAME = "schedule_model.py"
FAIRNESS_REPORT_FILENAME = "fairness_report.txt"
LOG_FILE = "smartscheduler.log"
