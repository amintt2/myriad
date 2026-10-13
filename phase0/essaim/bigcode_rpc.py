"""Bounded JSON values for the three plain pandas/datetime APIs; no pickle or arbitrary dtype constructors."""
from __future__ import annotations

import math
import re
from datetime import datetime


def encode(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is pd.Timestamp:
        if value.tzinfo is not None or value is pd.NaT:
            raise ValueError("unsupported Timestamp")
        return {"type": "timestamp_ns", "value": value.value}
    if type(value) is datetime:
        return {"type": "datetime", "value": value.isoformat()}
    if type(value) in (list, tuple):
        return {"type": "tuple" if type(value) is tuple else "list", "items": [encode(v) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):
        return {"type": "dict", "items": {k: encode(v) for k, v in value.items()}}
    if type(value) is pd.Series:
        dtype = str(value.dtype)
        if dtype not in ("int64", "object", "datetime64[ns]"):
            raise ValueError("unsupported Series dtype")
        return {"type": "series", "dtype": dtype, "name": encode(value.name), "index": encode(value.index),
                "items": [encode(v) for v in value.tolist()],
                "allows_duplicate_labels": value.flags.allows_duplicate_labels}
    if type(value) is pd.DataFrame:
        return {"type": "frame", "columns": encode(value.columns), "index": encode(value.index),
                "series": [encode(value.iloc[:, i]) for i in range(len(value.columns))],
                "allows_duplicate_labels": value.flags.allows_duplicate_labels}
    if type(value) is pd.RangeIndex:
        return {"type": "range", "start": value.start, "stop": value.stop, "step": value.step,
                "name": encode(value.name)}
    if type(value) is pd.Index and str(value.dtype) in ("int64", "object"):
        return {"type": "index", "dtype": str(value.dtype), "name": encode(value.name),
                "items": [encode(v) for v in value.tolist()]}
    raise ValueError("unsupported RPC value")


def decode(value, depth=0):
    import pandas as pd
    if depth > 20:
        raise ValueError("RPC depth limit")
    child = lambda v: decode(v, depth + 1)
    if value is None or type(value) in (bool, int, str):
        if type(value) is str and len(value) > 65536:
            raise ValueError("RPC string limit")
        if type(value) is int and abs(value) > 2**63 - 1:
            raise ValueError("RPC integer limit")
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is not dict or type(value.get("type")) is not str:
        raise ValueError("invalid RPC value")
    kind = value["type"]
    keys = {"datetime": {"value"}, "timestamp_ns": {"value"}, "list": {"items"}, "tuple": {"items"},
            "dict": {"items"}, "series": {"dtype", "name", "index", "items", "allows_duplicate_labels"},
            "frame": {"columns", "index", "series", "allows_duplicate_labels"},
            "range": {"start", "stop", "step", "name"}, "index": {"dtype", "name", "items"}}
    if kind not in keys or set(value) != keys[kind] | {"type"}:
        raise ValueError("invalid RPC fields")
    if kind in ("series", "frame") and type(value["allows_duplicate_labels"]) is not bool:
        raise ValueError("invalid pandas flags")
    if kind == "timestamp_ns":
        number = value["value"]
        if type(number) is not int or not -(2**63) < number < 2**63:
            raise ValueError("invalid nanosecond Timestamp")
        return pd.Timestamp(number, unit="ns")
    if kind == "datetime":
        if (type(value["value"]) is not str or len(value["value"]) > 64
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?(?:[+-]\d{2}:\d{2})?",
                                    value["value"])):
            raise ValueError("invalid datetime")
        return datetime.fromisoformat(value["value"])
    if kind == "range":
        if any(type(value[k]) is not int for k in ("start", "stop", "step")):
            raise ValueError("invalid range")
        if not value["step"] or len(range(value["start"], value["stop"], value["step"])) > 10000:
            raise ValueError("range limit")
        return pd.RangeIndex(value["start"], value["stop"], value["step"], name=child(value["name"]))
    if kind == "frame":
        if type(value["series"]) is not list or len(value["series"]) > 100:
            raise ValueError("frame column limit")
        columns, index = child(value["columns"]), child(value["index"])
        if not isinstance(columns, pd.Index) or not isinstance(index, pd.Index):
            raise ValueError("invalid frame axes")
        series = [child(v) for v in value["series"]]
        if len(series) != len(columns) or any(type(s) is not pd.Series or not s.index.equals(index) for s in series):
            raise ValueError("invalid frame shape")
        result = pd.concat(series, axis=1) if series else pd.DataFrame(index=index)
        result.columns = columns
        result.flags.allows_duplicate_labels = value["allows_duplicate_labels"]
        return result
    items = value["items"]
    if kind == "dict":
        if type(items) is not dict or len(items) > 10000:
            raise ValueError("invalid dictionary")
        return {k: child(v) for k, v in items.items()}
    if type(items) is not list or len(items) > 10000:
        raise ValueError("RPC sequence limit")
    items = [child(v) for v in items]
    if kind in ("tuple", "list"):
        return tuple(items) if kind == "tuple" else items
    dtype = value["dtype"]
    if dtype not in ("int64", "object", "datetime64[ns]") or (kind == "index" and dtype == "datetime64[ns]"):
        raise ValueError("unsupported dtype")
    if dtype == "int64" and any(type(v) is not int for v in items):
        raise ValueError("noninteger int64 values")
    if dtype == "object" and any(v is not None and type(v) not in (str, int, float, bool) for v in items):
        raise ValueError("nonprimitive object values")
    if dtype == "datetime64[ns]" and any(type(v) is not pd.Timestamp or v.tzinfo is not None for v in items):
        raise ValueError("invalid datetime Series")
    name = child(value["name"])
    if name is not None and type(name) not in (str, int):
        raise ValueError("invalid axis name")
    if kind == "index":
        return pd.Index(items, dtype=dtype, name=name)
    index = child(value["index"])
    if not isinstance(index, pd.Index) or len(index) != len(items):
        raise ValueError("invalid Series index")
    result = pd.Series(items, dtype=dtype, index=index, name=name)
    result.flags.allows_duplicate_labels = value["allows_duplicate_labels"]
    return result
