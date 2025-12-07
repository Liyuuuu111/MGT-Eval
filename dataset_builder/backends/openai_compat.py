from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import os
import json

from .base import GenerationResult
from ..config import GenConfig


@dataclass
class OpenAICompatBackend:
    """
    OpenAI-compatible 后端（支持 OpenAI / Azure OpenAI / vLLM OpenAI server / etc.）
    - 默认走 Chat Completions
    - 也可选 completions endpoint（legacy）
    """
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    endpoint: str = "chat"  # chat | completions
    timeout_s: int = 120

    name: str = "openai_compat"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OpenAICompatBackend 需要 api_key（参数或环境变量 OPENAI_API_KEY）")

        # Prefer official client if available
        self._client_mode = None
        self._client = None

        try:
            # new SDK (OpenAI>=1.x)
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
            self._client_mode = "openai_sdk_v1"
        except Exception:
            # fallback to requests
            import requests  # type: ignore
            self._requests = requests
            self._client_mode = "requests"

        if self.endpoint not in ("chat", "completions"):
            raise ValueError("endpoint must be 'chat' or 'completions'")

        if self._client_mode == "requests" and not self.base_url:
            # for requests we need explicit base_url
            raise RuntimeError("requests 模式需要 base_url（例如 http://localhost:8000/v1 ）")

    def get_tokenizer(self) -> Any:
        # API backend usually doesn't provide tokenizer; caller should set tokenizer_strategy=whitespace or hf:xxx
        return None

    def _apply_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not text or not stop:
            return text
        cut = None
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx >= 0:
                cut = idx if cut is None else min(cut, idx)
        return text if cut is None else text[:cut]

    def generate(self, prompt: str, gen: GenConfig, system_prompt: Optional[str] = None) -> GenerationResult:
        kw = gen.openai_kwargs()

        if self._client_mode == "openai_sdk_v1":
            if self.endpoint == "chat":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kw,
                )
                completion = resp.choices[0].message.content or ""
                completion = self._apply_stop(completion, gen.stop)

                text = (prompt + completion) if gen.return_full_text else completion
                meta = {
                    "backend": self.name,
                    "client_mode": self._client_mode,
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "gen_config": gen.to_dict(),
                    "usage": getattr(resp, "usage", None),
                }
                return GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta)

            # legacy completions
            resp = self._client.completions.create(
                model=self.model,
                prompt=prompt,
                **kw,
            )
            completion = resp.choices[0].text or ""
            completion = self._apply_stop(completion, gen.stop)
            text = (prompt + completion) if gen.return_full_text else completion
            meta = {
                "backend": self.name,
                "client_mode": self._client_mode,
                "endpoint": self.endpoint,
                "model": self.model,
                "gen_config": gen.to_dict(),
                "usage": getattr(resp, "usage", None),
            }
            return GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta)

        # requests mode
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        base = self.base_url.rstrip("/")
        if self.endpoint == "chat":
            url = f"{base}/chat/completions"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {"model": self.model, "messages": messages, **kw}
        else:
            url = f"{base}/completions"
            payload = {"model": self.model, "prompt": prompt, **kw}

        r = self._requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        if self.endpoint == "chat":
            completion = data["choices"][0]["message"].get("content", "") or ""
        else:
            completion = data["choices"][0].get("text", "") or ""

        completion = self._apply_stop(completion, gen.stop)
        text = (prompt + completion) if gen.return_full_text else completion

        meta = {
            "backend": self.name,
            "client_mode": self._client_mode,
            "endpoint": self.endpoint,
            "model": self.model,
            "gen_config": gen.to_dict(),
            "usage": data.get("usage", None),
        }
        return GenerationResult(prompt=prompt, completion=completion, text=text, meta=meta)

    def close(self) -> None:
        return
