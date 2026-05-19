import json

import numpy as np

from analysis.json_util import dumps_pretty, to_json_serializable


def test_nan_becomes_null() -> None:
    """Execute the test nan becomes null routine."""
    d = {"x": float("nan"), "y": 1.0}
    s = dumps_pretty(d)
    parsed = json.loads(s)
    assert parsed["x"] is None
    assert parsed["y"] == 1.0


def test_numpy_scalar() -> None:
    """Execute the test numpy scalar routine."""
    x = to_json_serializable({"m": np.float64(3.5)})
    assert x == {"m": 3.5}
