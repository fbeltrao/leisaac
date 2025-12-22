from dataclasses import dataclass
import torch


@dataclass
class TerminationStatus:
    """Class for keeping track of termination status."""

    terminated: bool = False
    success: bool = False
    is_timeout: bool = False
    reasons: list[str] | None = None

TERMINATION_PREFIX = "Episode_Termination/"


def detect_termination(done: torch.Tensor, truncated: torch.Tensor, extras: dict) -> TerminationStatus:
    """
    Detect the reason for termination based on the reset flags and extras.
    Limitation: only works for single environment (no vectorization).
    """

    if not truncated.any() and not done.any():
        return TerminationStatus(terminated=False)

    if truncated.any():
        return TerminationStatus(terminated=True, success=False, is_timeout=True, reasons=["timeout"])

    termination_log = extras.get("log", {})
    termination_reasons = {k: v for k, v in termination_log.items() if k.startswith(TERMINATION_PREFIX) and v > 0}
    if termination_reasons:
        reasons = list(reason.replace(TERMINATION_PREFIX, "") for reason in termination_reasons.keys())
        succeeded = all("success" in reason.lower() for reason in reasons)
        return TerminationStatus(terminated=True, success=succeeded, reasons=reasons)

    return TerminationStatus(terminated=True, success=True, reasons=["unknown"])
