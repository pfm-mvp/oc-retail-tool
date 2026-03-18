# helpers_vemcount_api.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, List, Tuple

import requests


@dataclass(frozen=True)
class VemcountApiConfig:
    report_url: str  # e.g. https://<your-fastapi>/get-report


def build_report_params(
    shop_ids: Iterable[int],
    data_outputs: Iterable[str],
    period: str,
    step: str = "day",
    source: str = "shops",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> List[Tuple[str, str]]:
    """
    Important: matches YOUR working backend format:
    - repeated params: data=1&data=2 (no [])
    - period="date" uses form_date_from/form_date_to
    """
    params: List[Tuple[str, str]] = []

    # order is not critical, but keep consistent for debugging
    for sid in shop_ids:
        params.append(("data", str(int(sid))))
    for dout in data_outputs:
        params.append(("data_output", str(dout)))

    params.append(("period", period))
    params.append(("step", step))
    params.append(("source", source))

    if period == "date":
        if date_from is None or date_to is None:
            raise ValueError("period='date' requires date_from and date_to")
        # ✅ your backend expects these keys
        params.append(("form_date_from", str(date_from)))
        params.append(("form_date_to", str(date_to)))

    return params


def fetch_report(
    cfg: VemcountApiConfig,
    shop_ids: Iterable[int],
    data_outputs: Iterable[str],
    period: str,
    step: str = "day",
    source: str = "shops",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    timeout: int = 120,
) -> dict:
    params = build_report_params(
        shop_ids=shop_ids,
        data_outputs=data_outputs,
        period=period,
        step=step,
        source=source,
        date_from=date_from,
        date_to=date_to,
    )
    resp = requests.post(cfg.report_url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
