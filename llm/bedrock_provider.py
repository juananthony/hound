"""AWS Bedrock provider implementation using the Anthropic SDK's Bedrock client."""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from .base_provider import BaseLLMProvider
from .schema_definitions import get_schema_definition

T = TypeVar('T', bound=BaseModel)


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider for Anthropic Claude models."""

    def __init__(
        self,
        config: dict[str, Any],
        model_name: str = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        verbose: bool = False,
        thinking_enabled: bool = False,
        **kwargs,
    ):
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.verbose = verbose
        self.thinking_enabled = thinking_enabled
        self._last_token_usage = None

        bedrock_cfg = config.get("bedrock", {}) if isinstance(config, dict) else {}

        aws_region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or bedrock_cfg.get("region")
        )

        try:
            from anthropic import AnthropicBedrock
        except ImportError:
            raise ImportError(
                "anthropic[bedrock] extras not installed. "
                "Run: pip install 'anthropic[bedrock]'"
            )

        client_kwargs: dict[str, Any] = {}
        if aws_region:
            client_kwargs["aws_region"] = aws_region

        aws_profile = os.environ.get("AWS_PROFILE") or bedrock_cfg.get("profile")
        if aws_profile:
            import botocore.session
            session = botocore.session.Session(profile=aws_profile)
            creds = session.get_credentials()
            if creds:
                resolved = creds.get_frozen_credentials()
                client_kwargs["aws_access_key"] = resolved.access_key
                client_kwargs["aws_secret_key"] = resolved.secret_key
                if resolved.token:
                    client_kwargs["aws_session_token"] = resolved.token

        self.client = AnthropicBedrock(**client_kwargs)

        if self.verbose:
            region_display = aws_region or "(auto-detect)"
            print(f"[Bedrock Provider] region={region_display}, model={self.model_name}")

    def parse(self, *, system: str, user: str, schema: type[T], **kwargs) -> T:
        """Make a structured call returning a Pydantic model instance."""
        schema_info = get_schema_definition(schema)
        full_user_prompt = (
            f"{user}\n\n{schema_info}\n\n"
            "Return ONLY valid JSON, no markdown or explanation."
        )

        if self.verbose:
            request_chars = len(system) + len(full_user_prompt)
            print("\n[Bedrock Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Schema: {getattr(schema, '__name__', str(schema))}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars // 4:,} tokens)")

        last_err = None
        for attempt in range(self.retries):
            try:
                start_time = time.time()
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system,
                    messages=[{"role": "user", "content": full_user_prompt}],
                )

                response_text = response.content[0].text if response.content else ""
                self._record_usage(response)

                if self.verbose:
                    elapsed = time.time() - start_time
                    print(f"[Bedrock Response] Time: {elapsed:.2f}s  Output: {len(response_text):,} chars")

                json_str = self._extract_json(response_text)
                json_data = json.loads(json_str)
                return schema(**json_data)

            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Bedrock parse failed after {self.retries} attempts: {last_err}")

    def raw(self, *, system: str, user: str, **kwargs) -> str:
        """Make a plain text call."""
        last_err = None
        for attempt in range(self.retries):
            try:
                start_time = time.time()
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

                response_text = response.content[0].text if response.content else ""
                self._record_usage(response)

                if self.verbose:
                    elapsed = time.time() - start_time
                    print(f"[Bedrock Response] Time: {elapsed:.2f}s  Output: {len(response_text):,} chars")

                return response_text

            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Bedrock raw call failed after {self.retries} attempts: {last_err}")

    @property
    def provider_name(self) -> str:
        return "Bedrock"

    @property
    def supports_thinking(self) -> bool:
        return self.thinking_enabled and "sonnet" in self.model_name.lower()

    def get_last_token_usage(self) -> dict[str, int] | None:
        return self._last_token_usage

    def _record_usage(self, response) -> None:
        if hasattr(response, "usage"):
            inp = response.usage.input_tokens or 0
            out = response.usage.output_tokens or 0
            self._last_token_usage = {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
            }

    @staticmethod
    def _extract_json(text: str) -> str:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()
