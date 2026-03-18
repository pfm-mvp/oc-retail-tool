# agent_region_reasoner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class RegionRollup:
    footfall: float
    turnover: float
    conversion_rate: float
    sales_per_visitor: float
    n_stores: int


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_region_rollup(df: pd.DataFrame) -> RegionRollup:
    # verwacht kolommen: shop_id, count_in, turnover, conversion_rate, sales_per_visitor (optioneel)
    footfall = _safe_float(df.get("count_in", pd.Series(dtype=float)).sum(skipna=True))
    turnover = _safe_float(df.get("turnover", pd.Series(dtype=float)).sum(skipna=True))

    conv = _safe_float(df.get("conversion_rate", pd.Series(dtype=float)).mean(skipna=True))

    spv_series = df.get("sales_per_visitor")
    if spv_series is None or spv_series.dropna().empty:
        # fallback: turnover / footfall (als proxy)
        spv = (turnover / footfall) if (footfall and not np.isnan(footfall) and footfall > 0) else float("nan")
    else:
        spv = _safe_float(spv_series.mean(skipna=True))

    n_stores = int(df["shop_id"].nunique()) if "shop_id" in df.columns else 0
    return RegionRollup(footfall=footfall, turnover=turnover, conversion_rate=conv, sales_per_visitor=spv, n_stores=n_stores)


def compute_opportunity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimalistische score:
    - traffic (count_in) + turnover: positief
    - conversie + SPV: negatief (lage conv/spv => meer upside)
    """
    work = df.copy()

    for col in ["count_in", "turnover", "conversion_rate", "sales_per_visitor"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # SPV fallback per row (indien leeg)
    if work["sales_per_visitor"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            work["sales_per_visitor"] = work["turnover"] / work["count_in"]

    agg = work.groupby("shop_id", as_index=False).agg(
        footfall=("count_in", "sum"),
        turnover=("turnover", "sum"),
        conv=("conversion_rate", "mean"),
        spv=("sales_per_visitor", "mean"),
        days=("date", "nunique") if "date" in work.columns else ("shop_id", "size"),
    )

    # z-scores (robust genoeg voor ranking)
    for col in ["footfall", "turnover", "conv", "spv"]:
        x = agg[col].astype(float)
        agg[col + "_z"] = (x - x.mean()) / (x.std(ddof=0) + 1e-9)

    agg["opportunity_score"] = (
        0.45 * agg["footfall_z"] +
        0.20 * agg["turnover_z"] -
        0.20 * agg["conv_z"] -
        0.15 * agg["spv_z"]
    )

    return agg.sort_values("opportunity_score", ascending=False)


def build_llm_opportunities(
    scores: pd.DataFrame,
    shop_name_map: Optional[Dict[int, str]] = None,
    top_n: int = 7,
) -> List[Dict[str, Any]]:
    m = shop_name_map or {}
    top = scores.head(top_n).copy()

    def name_for(sid: int) -> str:
        return m.get(int(sid), f"Shop {sid}")

    out: List[Dict[str, Any]] = []
    for _, r in top.iterrows():
        sid = int(r["shop_id"])
        out.append({
            "location": name_for(sid),
            "shop_id": sid,
            "footfall": _safe_float(r.get("footfall")),
            "turnover": _safe_float(r.get("turnover")),
            "conversion_rate": _safe_float(r.get("conv")),
            "sales_per_visitor": _safe_float(r.get("spv")),
            "opportunity_score": _safe_float(r.get("opportunity_score")),
        })
    return out