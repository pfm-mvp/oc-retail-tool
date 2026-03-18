
# utils_pfmx.py  — Plain keys (no []), repeated params style.
import os
import requests
from typing import List, Tuple, Union, Iterable, Any
import streamlit as st

API_URL = os.getenv("API_URL", "").rstrip("/") or st.secrets.get("API_URL","").rstrip("/")

def inject_css():
    css = """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .stMetric { text-align: left !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def _expand_plain(params: List[Tuple[str, Any]]) -> List[Tuple[str, str]]:
    """Return a flat list of (key, value) where list/tuple values are expanded as repeated keys.
       Also normalizes keys: strips a trailing [] if present.
    """
    out: List[Tuple[str,str]] = []
    for k, v in params:
        key = str(k)
        if key.endswith("[]"):  # normalize to plain
            key = key[:-2]
        if isinstance(v, (list, tuple)):
            for vi in v:
                out.append((key, str(vi)))
        else:
            out.append((key, str(v)))
    return out

def api_get_report(params: List[Tuple[str, Any]], timeout: int = 60, headers: dict | None = None) -> dict:
    """POST to /get-report using *plain keys*, repeated entries, like:
       ("data", 32224), ("data", 31977), ("data_output","count_in"), ...
       This matches your proven working calculators.
    """
    if not API_URL:
        return {"_error": True, "status": 500, "exception": "Missing API_URL", "_url": "", "_method": "POST"}

    expanded = _expand_plain(params)
    try:
        r = requests.post(API_URL, params=expanded, timeout=timeout, headers=headers or {})
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
        try:
            return r.json()
        except Exception:
            return {"_error": True, "status": r.status_code, "_url": r.request.url, "_method":"POST",
                    "exception": "JSON parse failed", "_body": r.text[:1000]}
    except requests.HTTPError as he:
        resp = he.response
        return {"_error": True, "status": getattr(resp, "status_code", 500),
                "_url": getattr(resp.request, "url", API_URL), "_method": "POST",
                "exception": str(he), "_body": getattr(resp, "text", "")[:1000]}
    except Exception as e:
        return {"_error": True, "status": 500, "_url": API_URL, "_method": "POST", "exception": str(e)}

def friendly_error(js: dict, period: str | None = None) -> bool:
    """Show a neat error in Streamlit if the API helper returned an error dict."""
    if isinstance(js, dict) and js.get("_error"):
        msg = f"API call failed — status: {js.get('status')} | _url: {js.get('_url')} | _method: {js.get('_method')} | exception: {js.get('exception')}"
        st.error(msg)
        if period:
            st.caption(f"Tip: controleer verplichte params voor periode **{period}** (bijv. period_step/step/weather).")
        if "_body" in js and js["_body"]:
            with st.expander("Response body (preview)"):
                st.code(js["_body"])
        return True
    return False
