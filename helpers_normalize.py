
# helpers_normalize.py
from __future__ import annotations

from typing import Dict, Any, Iterable, List, Optional
import pandas as pd
from datetime import datetime

# Default KPI keys you commonly request
KPI_KEYS_DEFAULT = [
    "count_in",
    "turnover",
    "conversion_rate",
    "sales_per_visitor",
    "sales_per_sqm",
    "inside",
    "count_out",
]

# Keys to ignore as metrics
META_FIELDS = {
    "id", "parent_id", "parent_data_id", "company_id", "segment_id",
    "custom_location_id", "level", "name", "data_type", "country", "unique_id"
}

def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _extract_metrics(data_node: Dict[str, Any], kpi_keys: Iterable[str]) -> Dict[str, float]:
    """
    Pull KPI keys from a 'data' dict and coerce to float.
    Also capture other numeric fields that appear in the payload.
    """
    out = {}
    for k in kpi_keys:
        if k in data_node:
            out[k] = _coerce_float(data_node.get(k))
    for k, v in data_node.items():
        if k in META_FIELDS or k in out:
            continue
        fv = _coerce_float(v)
        if fv is not None:
            out[k] = fv
    return out

def normalize_vemcount_response(
    resp_json: Dict[str, Any],
    shop_name_map: Optional[Dict[int, str]] = None,
    kpi_keys: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Flatten Vemcount /report-like responses into a tidy DataFrame.

    Ondersteunt o.a.:
      - year/month/day buckets: resp["data"]["date_YYYY-MM-DD"][shop_id]["data"]
      - week buckets: resp["data"]["this_week" | "last_week"][shop_id]["dates"][timestamp]["data"]
        waarbij de echte datum in data["dt"] zit ("YYYY-MM-DD HH:MM:SS").

    Returns columns: ['date','timestamp','shop_id','shop_name', <metrics...>]
    """
    if not isinstance(resp_json, dict) or "data" not in resp_json:
        return pd.DataFrame()

    kpi_keys = list(kpi_keys) if kpi_keys is not None else KPI_KEYS_DEFAULT

    rows: List[Dict[str, Any]] = []
    data_block = resp_json.get("data", {})

    for bucket_key, shops_dict in data_block.items():
        # voorbeeld:
        # - "date_2025-11-24"  -> date_str = "2025-11-24"
        # - "this_week"        -> date_str = None (we pakken dan data["dt"])
        date_str = None
        if isinstance(bucket_key, str) and bucket_key.startswith("date_"):
            date_str = bucket_key.replace("date_", "", 1)

        if not isinstance(shops_dict, dict):
            continue

        for shop_id_str, shop_node in shops_dict.items():
            try:
                shop_id = int(shop_id_str)
            except Exception:
                continue

            shop_name = shop_name_map.get(shop_id) if shop_name_map else None

            if not isinstance(shop_node, dict):
                continue

            # ---------- timestamp-level nodes (met "dates") ----------
            if "dates" in shop_node and isinstance(shop_node["dates"], dict):
                for ts, node in shop_node["dates"].items():
                    data_node = node.get("data", {}) if isinstance(node, dict) else {}
                    metrics = _extract_metrics(data_node, kpi_keys)
                    ts_str = str(ts)

                    # 1) probeer eerst data["dt"] of data["date"] (bv. "2025-11-24 00:00:00")
                    d_iso = None
                    dt_raw = data_node.get("dt") or data_node.get("date")
                    if isinstance(dt_raw, str):
                        try:
                            d_iso = datetime.fromisoformat(dt_raw).date().isoformat()
                        except Exception:
                            d_iso = None

                    # 2) lukt dat niet? probeer de timestamp-key zelf
                    if d_iso is None:
                        try:
                            d_iso = (
                                datetime.fromisoformat(ts_str.replace(" ", "T"))
                                .date()
                                .isoformat()
                            )
                        except Exception:
                            # 3) laatste fallback: date_str uit bucket (kan None zijn)
                            d_iso = date_str

                    row = {
                        "date": d_iso,
                        "timestamp": ts_str,
                        "shop_id": shop_id,
                        "shop_name": shop_name,
                    }
                    row.update(metrics)
                    rows.append(row)

            # ---------- day-level node (zonder "dates") ----------
            if "data" in shop_node and isinstance(shop_node["data"], dict):
                data_node = shop_node["data"]
                metrics = _extract_metrics(data_node, kpi_keys)

                # idem: probeer eerst data["dt"], dan bucket-key
                d_iso = date_str
                dt_raw = data_node.get("dt") or data_node.get("date")
                if isinstance(dt_raw, str):
                    try:
                        d_iso = datetime.fromisoformat(dt_raw).date().isoformat()
                    except Exception:
                        pass

                row = {
                    "date": d_iso,
                    "timestamp": None,
                    "shop_id": shop_id,
                    "shop_name": shop_name,
                }
                row.update(metrics)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # order columns: id cols first, then KPIs
    id_cols = ["date", "timestamp", "shop_id", "shop_name"]
    present_kpis = [k for k in (kpi_keys or []) if k in df.columns]
    extra_kpis = [c for c in df.columns if c not in id_cols and c not in present_kpis]
    df = df.reindex(columns=id_cols + present_kpis + extra_kpis)

    # friendly sort
    df = df.sort_values(
        by=["date", "shop_id", "timestamp"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return df

def attach_shop_names(df: pd.DataFrame, shop_name_map: Dict[int,str]) -> pd.DataFrame:
    """Add/refresh shop_name column from mapping."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["shop_name"] = out["shop_id"].map(shop_name_map)
    return out

def to_wide(df: pd.DataFrame,
            index: List[str] = ["date","shop_id","shop_name"],
            values: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Summarize to one row per date+shop. If values is None, pivot all metric columns.
    For timestamp payloads, sums metrics per date.
    """
    if df is None or df.empty:
        return df
    metric_cols = [c for c in df.columns if c not in {"date","timestamp","shop_id","shop_name"}]
    values = values or metric_cols
    # aggregate by date+shop, sum numeric
    wide = df.drop(columns=["timestamp"]).groupby(index, as_index=False)[values].sum(numeric_only=True)
    return wide
