# helpers_shop.py
from __future__ import annotations
from typing import Dict, Any, List
from shop_mapping import SHOP_NAME_MAP as RAW_MAP

def _normalize(raw: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
    norm: Dict[int, Dict[str, Any]] = {}
    for sid, meta in (raw or {}).items():
        if isinstance(meta, dict):
            name = meta.get("name")
            region = meta.get("region", "ALL")
        else:
            # oud formaat: meta is een string met de naam
            name = str(meta)
            region = "ALL"
        if not name:
            # skip kapotte regels
            continue
        norm[int(sid)] = {"name": name, "region": region}
    return norm

SHOP_NAME_MAP_NORM: Dict[int, Dict[str, Any]] = _normalize(RAW_MAP)

# Afgeleide mappen
ID_TO_NAME: Dict[int, str] = {sid: v["name"] for sid, v in SHOP_NAME_MAP_NORM.items()}
NAME_TO_ID: Dict[str, int] = {v["name"]: sid for sid, v in SHOP_NAME_MAP_NORM.items()}
REGIONS: List[str] = sorted({v.get("region", "ALL") for v in SHOP_NAME_MAP_NORM.values()})

def get_ids_by_region(region: str) -> List[int]:
    if region == "ALL":
        return list(ID_TO_NAME.keys())
    return [sid for sid, v in SHOP_NAME_MAP_NORM.items() if v.get("region", "ALL") == region]

def get_region_by_id(shop_id: int) -> str | None:
    return SHOP_NAME_MAP_NORM.get(int(shop_id), {}).get("region")

def get_name_by_id(shop_id: int) -> str | None:
    return ID_TO_NAME.get(int(shop_id))
