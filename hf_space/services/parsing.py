"""Small parsers for user-supplied UI values."""
from __future__ import annotations

_RF_MIN = 0.0
_RF_MAX = 0.20


def parse_rf(text: str) -> float:
    """
    Parse a risk-free-rate string into a decimal in [0.0, 0.20].

    Accepts:
      "4.56%"   -> 0.0456
      "4.56"    -> 0.0456   (bare number >= 1 is treated as a percent)
      "0.0456"  -> 0.0456   (bare number <  1 is treated as a decimal)
      "   5 % " -> 0.05
    Clamps to [0.0, 0.20]. Raises ValueError on non-numeric input.
    """
    if not isinstance(text, str):
        raise ValueError("parse_rf: expected str")
    s = text.strip()
    if not s:
        raise ValueError("parse_rf: empty")
    had_percent = s.endswith("%")
    if had_percent:
        s = s[:-1].strip()
    try:
        value = float(s)
    except ValueError as e:
        raise ValueError(f"parse_rf: not a number: {text!r}") from e
    if had_percent or abs(value) >= 1.0:
        value = value / 100.0
    return max(_RF_MIN, min(_RF_MAX, value))
