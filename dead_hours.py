# dead_hours.py — Dead Hour Detection & Analysis
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def detect_dead_hours(
    store_id: int,
    hourly_data: pd.DataFrame,
    threshold_pct: float = 0.60,
) -> Dict[str, Any]:
    """
    Identify dead hours from hourly footfall data.
    An hour is "dead" when footfall < threshold_pct * average for that hour-of-week.

    Args:
        store_id: Shop ID
        hourly_data: DataFrame with columns [hour, day_of_week, footfall]
        threshold_pct: Below this fraction of average = dead hour (default 60%)

    Returns:
        Structured dict with dead hour blocks, totals, and classification
    """
    if hourly_data is None or hourly_data.empty:
        return {
            "store_id": store_id,
            "dead_hours": [],
            "total_dead_hours_week": 0.0,
            "total_missed_revenue_week": 0.0,
        }

    # Calculate expected = average footfall per hour-of-week
    hourly_stats = (
        hourly_data.groupby(["day_of_week", "hour"])["footfall"]
        .agg(["mean", "count"])
        .reset_index()
    )
    hourly_stats.columns = ["day_of_week", "hour", "avg_footfall", "occurences"]

    # Merge back and flag dead hours
    merged = hourly_data.merge(
        hourly_stats, on=["day_of_week", "hour"], how="left"
    )
    merged["is_dead"] = merged["footfall"] < (merged["avg_footfall"] * threshold_pct)

    # Group consecutive dead hours into blocks
    dead_blocks = []
    store_df = merged.sort_values(["date", "hour"]).copy()

    # Add a block_id per consecutive dead segment
    store_df["dead_block"] = (
        (store_df["is_dead"] != store_df["is_dead"].shift())
        .cumsum()
    )
    store_df["dead_block"] = store_df["dead_block"].where(store_df["is_dead"], -1)

    for block_id, group in store_df[store_df["dead_block"] >= 0].groupby("dead_block"):
        row = group.iloc[0]
        avg_ff = float(row["avg_footfall"])
        actual_ff = float(group["footfall"].mean())
        missed_pct = max(0.0, (avg_ff - actual_ff) / avg_ff) if avg_ff > 0 else 0

        # Determine if structural vs incidental (appears >50% of weeks)
        occurrence_pct = float(row["occurences"]) / max(
            1, len(store_df["date"].unique())
        )

        dead_blocks.append({
            "day_of_week": int(group["day_of_week"].iloc[0]),
            "start_hour": int(group["hour"].min()),
            "end_hour": int(group["hour"].max()),
            "avg_footfall": round(float(actual_ff), 0),
            "expected_footfall": round(float(avg_ff), 0),
            "missed_revenue_est": round(float(avg_ff - actual_ff) * 15, 0),  # €15 avg spend
            "type": "structural" if occurrence_pct >= 0.5 else "incidental",
            "occurrence_pct": round(occurrence_pct * 100, 1),
        })

    total_dead_hours = len(
        store_df[store_df["is_dead"]]
    )
    total_missed_revenue = sum(b["missed_revenue_est"] for b in dead_blocks)

    return {
        "store_id": store_id,
        "dead_hours": dead_blocks,
        "total_dead_hours_week": float(total_dead_hours),
        "total_missed_revenue_week": round(total_missed_revenue, 0),
    }


# ── Action catalog for dead hour recommendations ──────────────────────────

DEAD_HOUR_ACTIONS = [
    {
        "type": "staffing",
        "description": "Heralloceer FTE naar dode uren voor actieve winkelvloer-begroeting",
        "footfall_lift_pct": 8,
        "conversion_lift_pct": 12,
        "effort": "low",
        "cost_est": 0,
    },
    {
        "type": "marketing",
        "description": "Push flash sale via social media / lokale ads in dode uren",
        "footfall_lift_pct": 15,
        "conversion_lift_pct": 8,
        "effort": "medium",
        "cost_est": 200,
    },
    {
        "type": "marketing",
        "description": "Stuur loyalty-push naar klanten in de buurt (binnen 3 km)",
        "footfall_lift_pct": 12,
        "conversion_lift_pct": 10,
        "effort": "low",
        "cost_est": 50,
    },
    {
        "type": "operations",
        "description": "Wijzig etalage / window display om nieuwe bezoekers te trekken",
        "footfall_lift_pct": 10,
        "conversion_lift_pct": 5,
        "effort": "medium",
        "cost_est": 150,
    },
    {
        "type": "pricing",
        "description": "Tijdelijke promotie: 2e artikel 50% korting in dode uren",
        "footfall_lift_pct": 20,
        "conversion_lift_pct": 15,
        "effort": "low",
        "cost_est": 100,
    },
    {
        "type": "staffing",
        "description": "Verplaats pauzes: zet maximale bezetting in piek-uren, minder in dode uren",
        "footfall_lift_pct": 0,
        "conversion_lift_pct": 18,
        "effort": "low",
        "cost_est": 0,
    },
]


def recommend_actions(
    store_id: int,
    health_score: int,
    dead_hours: List[Dict[str, Any]],
    max_actions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate top-N recommendations based on dead hour blocks.
    Higher impact actions prioritized.
    """
    if not dead_hours:
        return []

    # Score each action on expected health score impact
    scored = []
    for action in DEAD_HOUR_ACTIONS:
        total_lift = action["footfall_lift_pct"] + action["conversion_lift_pct"]
        health_impact = max(1, int(total_lift * 0.8))
        scored.append({
            **action,
            "expected_health_score_impact": min(15, health_impact),
        })

    # Sort by impact descending, low effort first as tiebreaker
    scored.sort(key=lambda a: (-a["expected_health_score_impact"], a["effort"] == "low"))
    return scored[:max_actions]
