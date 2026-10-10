"""Fail-closed account limits, independent of allocation ownership and cleanup authorization."""
import math
import os


def number(value, name, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"invalid {name}")
    return value


def validate_usage(value, allocated=False):
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1:
        raise ValueError("unknown account usage")
    number(value.get("balance_units"), "balance", True)
    number(value.get("rate_units_hour"), "rate", not allocated)
    if type(value.get("assignments")) is not int or value["assignments"] != int(allocated):
        raise ValueError("unexpected billed allocations")
    if not allocated and value["rate_units_hour"] != 0:
        raise ValueError("existing account usage")
    return value


def limits():
    return {"budget": float(os.environ.get("DLLM_BUDGET_UNITS", "nan")),
            "minimum": float(os.environ.get("DLLM_MIN_BALANCE_UNITS", "15")),
            "hours": float(os.environ.get("DLLM_HOURS", "nan")),
            "rate": float(os.environ.get("DLLM_MAX_RATE_UNITS_HOUR", "5.3")),
            "cleanup": float(os.environ.get("DLLM_CLEANUP_SECONDS", "1800")),
            "margin": float(os.environ.get("DLLM_ACCOUNTING_MARGIN_SECONDS", "155"))}


def preflight(usage, policy, allocated=False):
    validate_usage(usage, allocated)
    for name, value in policy.items():
        number(value, name, name == "minimum")
    rate = max(policy["rate"], usage["rate_units_hour"])
    cost = rate * (policy["hours"] + (policy["cleanup"] + policy["margin"]) / 3600)
    if policy["budget"] > usage["balance_units"] - policy["minimum"] or cost > policy["budget"]:
        raise ValueError("budget/duration incompatible with account reserve")
    return rate


def remaining(usage, policy, initial_balance, elapsed, observed_rate):
    validate_usage(usage, True)
    for name, value in policy.items():
        number(value, name, name == "minimum")
    for name, value in (("initial balance", initial_balance), ("elapsed", elapsed), ("observed rate", observed_rate)):
        number(value, name, True)
    rate = max(policy["rate"], observed_rate, usage["rate_units_hour"])
    spent = max(0, initial_balance - usage["balance_units"], rate * elapsed / 3600)
    available = min(usage["balance_units"] - policy["minimum"], policy["budget"] - spent)
    seconds = available / rate * 3600 - policy["cleanup"] - policy["margin"]
    if seconds <= 0:
        raise ValueError("account reserve reached; cleanup required")
    return seconds, rate
