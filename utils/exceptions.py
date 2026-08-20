"""Custom exceptions used across SmartScheduler."""


class SchedulerError(Exception):
    """Base class for all SmartScheduler errors."""


class CodeGenerationError(SchedulerError):
    """Raised when the drafting/refinement LLM agent produces unusable code
    (syntax error, missing required symbols, etc.)."""


class SolverExecutionError(SchedulerError):
    """Raised when the generated OR-Tools script fails to run, times out,
    or the model is INFEASIBLE/UNKNOWN."""


class ValidationError(SchedulerError):
    """Raised when a schedule fails symbolic hard-constraint validation
    and cannot be repaired within the iteration budget."""


class RateLimitError(SchedulerError):
    """Raised when the Groq token-bucket cannot grant tokens within a
    reasonable wait, or the API itself refuses the call after retries."""


class InputParsingError(SchedulerError):
    """Raised when the input text file cannot be parsed into workers/shifts."""
