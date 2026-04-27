# health_explainer.py — Rule-based health score explanation (NL)
from __future__ import annotations
from typing import Dict, Any, List, Optional


DIM_NAMES_NL = {
    "traffic_trend": "Bezoekerstrend",
    "conversion_efficiency": "Conversie-efficiëntie",
    "revenue_productivity": "Omzetproductiviteit",
    "staffing_alignment": "Personeelsinzet",
    "catchment_potential": "Verzorgingsgebied potentieel",
}


def explain_health_score(
    health_result: Dict[str, Any],
    dead_hours_result: Optional[Dict[str, Any]] = None,
    lang: str = "nl",
) -> Dict[str, str]:
    """
    Generate human-readable explanation of health score (rule-based, NL).

    Returns dict with:
      - summary_nl: 2-3 sentence explanation
      - top_driver: dimension with highest weighted deviation
      - top_lever: recommended focus area
      - dead_hour_impact: optional dead hour impact sentence
    """
    score = health_result.get("health_score", 0)
    dims = health_result.get("dimensions", {})
    flagged = health_result.get("flagged_dimensions", [])

    # Find top driver (highest weight × largest deviation from 50)
    top_driver_name = None
    top_driver_score = 50
    top_driver_weight = 0
    max_driver_val = 0

    for dim_name, dim_data in dims.items():
        dim_score = dim_data["score"]
        dim_weight = dim_data["weight"]
        deviation = abs(dim_score - 50)
        driver_val = deviation * dim_weight
        if driver_val > max_driver_val:
            max_driver_val = driver_val
            top_driver_name = dim_name
            top_driver_score = dim_score
            top_driver_weight = dim_weight

    # Find top lever (flagged dimension with highest weight, or lowest score)
    top_lever_name = None
    if flagged:
        top_lever_name = flagged[0]
        for dim_name in flagged:
            if dims[dim_name]["weight"] > dims.get(top_lever_name, {}).get("weight", 0):
                top_lever_name = dim_name
    else:
        # No flags: suggest improvement on lowest scoring dimension
        min_score = 100
        for dim_name, dim_data in dims.items():
            if dim_data["score"] < min_score:
                min_score = dim_data["score"]
                top_lever_name = dim_name

    # Build summary
    nl_name = DIM_NAMES_NL.get(top_driver_name, top_driver_name or "")
    lever_nl = DIM_NAMES_NL.get(top_lever_name, top_lever_name or "")

    if score >= 80:
        summary_nl = f"Gezondheidsscore: {score}/100 — Zeer gezond. "
    elif score >= 60:
        summary_nl = f"Gezondheidsscore: {score}/100 — Redelijk gezond. "
    elif score >= 40:
        summary_nl = f"Gezondheidsscore: {score}/100 — Aandacht nodig. "
    else:
        summary_nl = f"Gezondheidsscore: {score}/100 — Kritiek. "

    if nl_name:
        direction = "bovengemiddeld" if top_driver_score > 50 else "ondergemiddeld"
        summary_nl += (
            f"Belangrijkste driver: {nl_name} ({direction}, "
            f"gewicht {top_driver_weight*100:.0f}%). "
        )

    if lever_nl:
        summary_nl += f"Grootste hefboom: verbeter {lever_nl}."

    # Dead hour impact
    dead_hour_impact = None
    if dead_hours_result and dead_hours_result.get("dead_hours"):
        total_dead = dead_hours_result["total_dead_hours_week"]
        missed_rev = dead_hours_result["total_missed_revenue_week"]
        structural = sum(
            1 for d in dead_hours_result["dead_hours"]
            if d.get("type") == "structural"
        )
        dead_hour_impact = (
            f"Weekelijks {total_dead:.0f} dode uren "
            f"({structural} structurele blokken) met €{missed_rev:,.0f} gederfde omzet."
        )

        if score < 60:
            summary_nl += f" Dode uren verlagen de conversie-efficiëntie."

    return {
        "summary_nl": summary_nl,
        "summary_en": summary_nl,  # Phase 1: same as NL
        "top_driver": top_driver_name,
        "top_lever": top_lever_name,
        "dead_hour_impact": dead_hour_impact,
    }
