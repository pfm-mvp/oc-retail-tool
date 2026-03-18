# outcome_explainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable
import os
import json
import time
import requests


@dataclass
class ExplainerConfig:
    enable_llm: bool = True
    provider: str = "openrouter"  # openrouter | openai (later)
    model: str = ""  # if empty: auto-pick a free OpenRouter model
    max_tokens: int = 450
    temperature: float = 0.2
    timeout_sec: int = 30

    # env var names
    api_key_env: str = "OPENROUTER_API_KEY"

    # OpenRouter
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_models_endpoint: str = "https://openrouter.ai/api/v1/models"
    openrouter_app_url_env: str = "OPENROUTER_APP_URL"
    openrouter_app_name_env: str = "OPENROUTER_APP_NAME"


class OutcomeExplainer:
    def __init__(self, cfg: Optional[ExplainerConfig] = None):
        self.cfg = cfg or ExplainerConfig()

    def build_cards(
        self,
        outcomes: Dict[str, Any],
        persona: str = "region_manager",
        style: str = "crisp",
        use_llm: Optional[bool] = None,
    ) -> List[Dict[str, str]]:
        use_llm_final = self._should_use_llm(use_llm)
        baseline_cards = self._template_cards(outcomes, persona=persona, style=style)

        if not use_llm_final:
            return baseline_cards

        try:
            rewritten = self._llm_rewrite_cards(outcomes, baseline_cards, persona, style)
            if rewritten and isinstance(rewritten, list):
                return rewritten
        except Exception:
            pass

        return baseline_cards

    def stream_typing(self, text: str, chunk_size: int = 18, delay: float = 0.01) -> Iterable[str]:
        if not text:
            yield ""
            return
        for i in range(0, len(text), chunk_size):
            yield text[: i + chunk_size]
            if delay > 0:
                time.sleep(delay)

    def _should_use_llm(self, use_llm: Optional[bool]) -> bool:
        if use_llm is not None:
            return bool(use_llm)
        if not self.cfg.enable_llm:
            return False
        key = os.getenv(self.cfg.api_key_env, "").strip()
        return bool(key)

    # -----------------------------
    # Deterministic templates (keep your existing version)
    # -----------------------------
    def _template_cards(self, outcomes: Dict[str, Any], persona: str, style: str) -> List[Dict[str, str]]:
        # KEEP YOUR EXISTING TEMPLATE LOGIC HERE
        # (use your current version; unchanged)
        meta = outcomes.get("meta", {}) if isinstance(outcomes, dict) else {}
        scores = outcomes.get("scores", {}) if isinstance(outcomes, dict) else {}

        client = str(meta.get("client", "Client")).strip()
        region = str(meta.get("region", "Region")).strip()
        period_label = str(meta.get("period_label", "selected period")).strip()

        svi = scores.get("region_svi", None)
        status = str(scores.get("status", "")).strip()
        avg_ratio = scores.get("avg_ratio_vs_company", None)

        opps = outcomes.get("opportunities", []) if isinstance(outcomes, dict) else []
        risks = outcomes.get("risks", []) if isinstance(outcomes, dict) else []

        cards: List[Dict[str, str]] = []

        body_lines = []
        if svi is not None:
            body_lines.append(f"Region SVI: **{int(round(float(svi)))} / 100**" + (f" — {status}" if status else ""))
        if avg_ratio is not None:
            body_lines.append(f"Avg ratio vs company: **{float(avg_ratio):.0f}%**")

        if opps:
            body_lines.append("")
            body_lines.append("Top opportunities:")
            for o in opps[:3]:
                nm = o.get("store_name") or o.get("store_id") or "-"
                drv = o.get("driver", "")
                ip = o.get("impact_period_eur", None)
                ia = o.get("impact_annual_eur", None)
                body_lines.append(f"• **{nm}** — {drv} · € {ip:,.0f} (period) · € {ia:,.0f}/yr".replace(",", "."))

        if risks:
            body_lines.append("")
            body_lines.append("Top risks:")
            for r in risks[:2]:
                body_lines.append(f"• {r.get('driver','')} ({r.get('severity','')})")

        cards.append({
            "tag": "Summary",
            "title": f"{client} · {region} · {period_label}",
            "body": "\n".join(body_lines).strip()
        })
        return cards

    # -----------------------------
    # OpenRouter integration
    # -----------------------------
    def _pick_free_openrouter_model(self, api_key: str) -> Optional[str]:
        """
        Fetch models and pick a model with $0 prompt AND $0 completion if available.
        """
        try:
            resp = requests.get(
                self.cfg.openrouter_models_endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.cfg.timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            # Heuristic: pick first free instruct-ish model
            for m in models:
                pricing = m.get("pricing", {}) or {}
                prompt = str(pricing.get("prompt", "")).strip()
                completion = str(pricing.get("completion", "")).strip()
                mid = m.get("id", "")
                name = (m.get("name", "") or "").lower()
                if prompt in ("0", "0.0") and completion in ("0", "0.0"):
                    if any(k in (mid.lower() + " " + name) for k in ["instruct", "chat", "turbo", "it"]):
                        return mid
            # fallback: any free model
            for m in models:
                pricing = m.get("pricing", {}) or {}
                prompt = str(pricing.get("prompt", "")).strip()
                completion = str(pricing.get("completion", "")).strip()
                if prompt in ("0", "0.0") and completion in ("0", "0.0"):
                    return m.get("id")
        except Exception:
            return None
        return None

    def _llm_rewrite_cards(
        self,
        outcomes: Dict[str, Any],
        baseline_cards: List[Dict[str, str]],
        persona: str,
        style: str
    ) -> List[Dict[str, str]]:
        """
        Rewrites baseline cards into clearer 'agentic workload' copy.
        STRICT RULE: no new numbers/facts; only rewrite & restructure.
        """
        api_key = os.getenv(self.cfg.api_key_env, "").strip()
        if not api_key:
            return baseline_cards

        model = (self.cfg.model or "").strip()
        if not model:
            model = self._pick_free_openrouter_model(api_key) or ""

        # If still no model, bail out safely
        if not model:
            return baseline_cards

        app_url = os.getenv(self.cfg.openrouter_app_url_env, "").strip()
        app_name = os.getenv(self.cfg.openrouter_app_name_env, "").strip()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers (recommended by OpenRouter docs)
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        system = (
            "You rewrite analytics cards for a retail region manager.\n"
            "Rules:\n"
            "- Do NOT invent new facts, numbers, stores, or metrics.\n"
            "- You may reorder, summarize, and clarify.\n"
            "- Keep it agentic: what’s happening → impact → evidence → next actions.\n"
            "- Output MUST be valid JSON: a list of {title, body, tag}.\n"
        )

        user_payload = {
            "persona": persona,
            "style": style,
            "cards": baseline_cards,
            "outcomes": outcomes,  # provides context, but still: no new facts allowed
        }

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        url = f"{self.cfg.openrouter_base_url}/chat/completions"
        r = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout_sec)
        r.raise_for_status()
        j = r.json()

        # OpenAI-like shape
        content = j["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # minimal validation
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            cleaned = []
            for x in parsed:
                cleaned.append({
                    "title": str(x.get("title", "")),
                    "body": str(x.get("body", "")),
                    "tag": str(x.get("tag", "")),
                })
            return cleaned

        return baseline_cards