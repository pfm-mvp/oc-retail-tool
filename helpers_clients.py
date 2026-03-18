# helpers_clients.py

from pathlib import Path
from typing import Any, Dict, List

import json


def load_clients(json_path: str | Path = "clients.json") -> List[Dict[str, Any]]:
    """
    Laadt clients.json met structuur:
    [
      {"company_id": 3363, "name": "Retailer A", "brand": "CASA"},
      {"company_id": 4456, "name": "Retailer B", "brand": "MADAQ"},
      ...
    ]
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"clients.json niet gevonden op pad: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("clients.json moet een lijst van clients bevatten.")

    return data
