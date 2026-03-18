# advisor.py — rule-based recommendations using forecast + historical KPIs
from __future__ import annotations
from datetime import date
from statistics import median
from typing import Dict, Any, List

def compute_temp_anomaly(forecast_temp: float, hist_day_temps: List[float] | None) -> float:
    if not hist_day_temps:
        return 0.0
    avg = sum(hist_day_temps) / len(hist_day_temps)
    return forecast_temp - avg

def advisor_for_store(store_name: str, hist_day: Dict[str, Any], forecast_day: Dict[str, Any], cci_value: float) -> Dict[str, Any]:
    acts_store: List[str] = []
    acts_region: List[str] = []

    temp_anom = compute_temp_anomaly(forecast_day.get("temp", 0.0), hist_day.get("temps", []))
    pop = float(forecast_day.get("pop", 0.0))  # precipitation probability 0..1

    # Weer → staffing & floor actions
    if pop >= 0.60 and hist_day.get("visitors", 0) >= max(1, hist_day.get("visitors_p30", 0)):
        acts_store.append("Verplaats pauzes 30 min vóór verwachte bui; focus op begroeten/fit-advies om conversiedip te voorkomen.")
    if temp_anom >= 4:
        acts_store.append("Verleng piekshift +1u en highlight lichte/outdoor sets bij entree (verwachte +SPV).")
    if forecast_day.get("feels_like", 999) <= 5:
        acts_store.append("Comfort-maatregelen bij entree; promoot cold-weather accessoires.")

    # Dead hour / SPV taken
    if hist_day.get("visitors", 0) >= hist_day.get("visitors_p30", 0) and hist_day.get("spv", 0) < hist_day.get("spv_median", 0):
        acts_store.append("Dead hour: 3-stappen script (actieve begroeting • add-on prompt • kassa-script). Doel: +€0,50 SPV.")

    # Macro (CCI) → boodschap
    if cci_value < 0:
        acts_store.append("Benadruk bundel/waardeproposities vandaag.")
    else:
        acts_store.append("Push premium add-ons tijdens piekuren; richt SPV-doel op +€0,50.")

    # Regionale invalshoek
    if pop >= 0.60:
        acts_region.append(f"{store_name}: regen-risico → mobiele FTE op 16–19u en queue-buster paraat.")
    if abs(temp_anom) >= 4:
        acts_region.append(f"{store_name}: temperatuurafwijking {temp_anom:+.0f}°C → check thematafel/omslag.")

    return {"store_actions": acts_store, "regional_actions": acts_region}

def build_advice(company_name: str,
                 stores_hist_by_weekday: Dict[int, Dict[str, Dict[str, Any]]],
                 forecast: List[Dict[str, Any]],
                 cci_value: float) -> Dict[str, Any]:
    out = {"company": company_name, "cci": cci_value, "days": []}
    for f in forecast:
        wd = date.fromisoformat(f["date"]).weekday()  # 0=Mon
        day_hist = stores_hist_by_weekday.get(wd, {})
        stores_out = []
        for store_name, hist in day_hist.items():
            advice = advisor_for_store(store_name, hist, f, cci_value)
            stores_out.append({"store": store_name, **advice})
        out["days"].append({"date": f["date"], "weather": f, "stores": stores_out})
    return out
