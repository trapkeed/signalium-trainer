"""Phase 23 — LLM Provider Abstraction Layer.

Supports:
- MockLLMProvider     (fully functional for testing and offline use)
- OpenAIProvider      (stub — requires openai package + API key)
- AnthropicProvider   (stub — requires anthropic package + API key)
- GeminiProvider      (stub — requires google-generativeai + API key)
- OllamaProvider      (stub — requires ollama running locally)
- AzureOpenAIProvider (stub — requires azure-openai package + credentials)

Switch providers without code changes by passing a different LLMProviderType
to AnalystAPI.
"""

from __future__ import annotations

import abc
import re
import time
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.analyst.analyst_types import LLMProviderType, is_safe_response
from trainer.analyst.analyst_models import KnowledgeChunk, Citation, LLMResponse
from trainer.analyst.analyst_types import CitationType


# ── Abstract base ──────────────────────────────────────────────────────────────

class LLMProvider(abc.ABC):
    """Abstract LLM provider interface."""

    @abc.abstractmethod
    def generate(
        self,
        prompt:       str,
        context:      str = "",
        temperature:  float = 0.3,
        max_tokens:   int   = 1024,
    ) -> LLMResponse:
        """Generate a response given a prompt and optional context."""

    @abc.abstractmethod
    def provider_type(self) -> LLMProviderType:
        """Return the provider type."""

    def is_available(self) -> bool:
        """Return True if the provider is currently available."""
        return False

    def health_check(self) -> Dict:
        return {
            "provider": self.provider_type().value,
            "available": self.is_available(),
        }


# ── Mock Provider ─────────────────────────────────────────────────────────────

class MockLLMProvider(LLMProvider):
    """Fully functional mock LLM for testing and offline research.

    Generates grounded, structured responses by synthesising the provided
    context chunks. No external calls are made.
    """

    _SAFETY_DISCLAIMER = (
        "Note: This is a research analysis only. No trade signals are implied. "
        "SAQRA provides research insights and never issues trading recommendations."
    )

    def __init__(self, model_name: str = "mock-research-v1"):
        self._model_name = model_name
        self._call_count = 0

    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.MOCK

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt:      str,
        context:     str  = "",
        temperature: float = 0.3,
        max_tokens:  int   = 1024,
    ) -> LLMResponse:
        start = time.time()
        self._call_count += 1

        # Build response from context
        response_text = self._synthesise(prompt, context)

        # Safety check
        safe = is_safe_response(response_text)
        if not safe:
            response_text = (
                "I cannot provide trading signals. "
                "Please ask about model performance, feature analysis, or research metrics."
            )

        # Extract citations from context
        citations = self._extract_citations(context)

        latency = (time.time() - start) * 1000.0

        return LLMResponse(
            content         = response_text,
            citations       = citations,
            provider        = LLMProviderType.MOCK,
            model_name      = self._model_name,
            prompt_tokens   = len(prompt.split()),
            response_tokens = len(response_text.split()),
            latency_ms      = round(latency, 2),
            grounded        = bool(context),
            safe            = safe,
        )

    def _synthesise(self, prompt: str, context: str) -> str:
        """Build a structured research response from the prompt and context."""
        prompt_lower = prompt.lower()
        lines = []

        # Detect query intent
        is_comparison  = any(w in prompt_lower for w in ("compare", "vs", "versus", "outperform", "differ"))
        is_explanation = any(w in prompt_lower for w in ("why", "explain", "reason", "cause"))
        is_search      = any(w in prompt_lower for w in ("show", "find", "list", "which", "all"))
        is_summary     = any(w in prompt_lower for w in ("summarize", "summary", "overview", "report"))
        is_recommend   = any(w in prompt_lower for w in ("suggest", "recommend", "improve", "next"))

        if is_comparison:
            lines.append("## Comparative Analysis")
            lines.append("")
            lines.append("Based on the retrieved research artifacts, the following comparison has been constructed:")
            lines.append("")
            if context:
                # Pull out key metrics from context
                for line in context.split("\n"):
                    if any(kw in line.lower() for kw in ("sharpe", "drawdown", "score", "performance", "accuracy")):
                        lines.append(f"- {line.strip()}")
            lines.append("")
            lines.append("The evidence suggests performance differences stem from model architecture, "
                         "training data diversity, and regime-specific calibration.")

        elif is_explanation:
            lines.append("## Research Analysis")
            lines.append("")
            lines.append("The following explanation is grounded in the retrieved research evidence:")
            lines.append("")
            if context:
                context_excerpt = context[:600].strip()
                lines.append(context_excerpt)
            lines.append("")
            lines.append("This pattern is consistent with the referenced experiments and validation reports.")

        elif is_search:
            lines.append("## Search Results")
            lines.append("")
            if context:
                chunks = [c.strip() for c in context.split("---") if c.strip()]
                for i, chunk in enumerate(chunks[:5], 1):
                    first_line = chunk.split("\n")[0].strip()
                    lines.append(f"{i}. {first_line}")
            else:
                lines.append("No matching artifacts found in the knowledge index.")

        elif is_summary:
            lines.append("## Research Summary")
            lines.append("")
            lines.append("**Period:** Current research cycle")
            lines.append("")
            lines.append("**Key Findings:**")
            if context:
                for line in context.split("\n")[:8]:
                    if line.strip():
                        lines.append(f"- {line.strip()}")
            lines.append("")
            lines.append("**Status:** Analysis complete. No automated actions triggered.")

        elif is_recommend:
            lines.append("## Research Recommendations")
            lines.append("")
            lines.append("Based on the evidence retrieved from the knowledge base:")
            lines.append("")
            lines.append("1. **Data Collection**: Increase training data diversity across market regimes")
            lines.append("2. **Calibration**: Review calibration on recent validation windows")
            lines.append("3. **Feature Stability**: Monitor unstable features identified in drift reports")
            lines.append("")
            lines.append("These recommendations are evidence-based and do not constitute trading signals.")

        else:
            lines.append("## Research Response")
            lines.append("")
            if context:
                lines.append("Based on the retrieved research evidence:")
                lines.append("")
                lines.append(context[:500].strip())
            else:
                lines.append("No relevant research artifacts were retrieved for this query. "
                             "Please refine your question or index additional research documents.")

        lines.append("")
        lines.append(f"---")
        lines.append(self._SAFETY_DISCLAIMER)

        return "\n".join(lines)

    @staticmethod
    def _extract_citations(context: str) -> List[Citation]:
        """Extract artifact IDs from context to form citations."""
        citations = []
        seen = set()

        # Patterns: EXP-001, MODEL-42, DATASET-v18, FEATURE-xxx
        patterns = [
            (r"\bEXP[-_](\w+)\b",     CitationType.EXPERIMENT),
            (r"\bMODEL[-_](\w+)\b",   CitationType.MODEL),
            (r"\bDATASET[-_](\w+)\b",  CitationType.DATASET),
            (r"\bFEATURE[-_](\w+)\b",  CitationType.FEATURE),
            (r"\bVAL[-_](\w+)\b",      CitationType.VALIDATION),
            (r"\bREPORT[-_](\w+)\b",   CitationType.REPORT),
        ]

        for pattern, ctype in patterns:
            for m in re.finditer(pattern, context, re.IGNORECASE):
                artifact_id = m.group(0).upper()
                if artifact_id not in seen:
                    seen.add(artifact_id)
                    citations.append(Citation(
                        citation_type = ctype,
                        artifact_id   = artifact_id,
                        description   = f"Referenced in retrieved context",
                        relevance     = 0.80,
                    ))

        return citations[:10]   # cap at 10 citations per response


# ── Provider Stubs ────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI provider stub. Requires: pip install openai + OPENAI_API_KEY env var."""

    def __init__(self, model_name: str = "gpt-4o", api_key: str = ""):
        self._model_name = model_name
        self._api_key    = api_key

    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OPENAI

    def is_available(self) -> bool:
        try:
            import openai   # type: ignore
            return bool(self._api_key)
        except ImportError:
            return False

    def generate(self, prompt: str, context: str = "", temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available. Install openai and set OPENAI_API_KEY.")
        raise NotImplementedError("OpenAI provider: implement by calling openai.ChatCompletion.create()")


class AnthropicProvider(LLMProvider):
    """Anthropic provider stub. Requires: pip install anthropic + ANTHROPIC_API_KEY env var."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: str = ""):
        self._model_name = model_name
        self._api_key    = api_key

    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.ANTHROPIC

    def is_available(self) -> bool:
        try:
            import anthropic   # type: ignore
            return bool(self._api_key)
        except ImportError:
            return False

    def generate(self, prompt: str, context: str = "", temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available. Install anthropic and set ANTHROPIC_API_KEY.")
        raise NotImplementedError("Anthropic provider: implement by calling anthropic.Anthropic().messages.create()")


class GeminiProvider(LLMProvider):
    """Google Gemini provider stub. Requires: pip install google-generativeai + GEMINI_API_KEY env var."""

    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: str = ""):
        self._model_name = model_name
        self._api_key    = api_key

    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.GEMINI

    def is_available(self) -> bool:
        try:
            import google.generativeai   # type: ignore
            return bool(self._api_key)
        except ImportError:
            return False

    def generate(self, prompt: str, context: str = "", temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError("Gemini provider: implement by calling google.generativeai.GenerativeModel().generate_content()")


class OllamaProvider(LLMProvider):
    """Ollama local provider stub. Requires: Ollama running on localhost:11434."""

    def __init__(self, model_name: str = "llama3", host: str = "http://localhost:11434"):
        self._model_name = model_name
        self._host       = host

    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OLLAMA

    def is_available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self._host}/api/version", timeout=2)
            return True
        except Exception:
            return False

    def generate(self, prompt: str, context: str = "", temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError("Ollama provider: implement by POST to /api/generate")


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider stub. Requires: pip install openai + Azure credentials."""

    def __init__(self, model_name: str = "gpt-4o", endpoint: str = "", api_key: str = ""):
        self._model_name = model_name
        self._endpoint   = endpoint
        self._api_key    = api_key

    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.AZURE

    def is_available(self) -> bool:
        try:
            import openai   # type: ignore
            return bool(self._endpoint and self._api_key)
        except ImportError:
            return False

    def generate(self, prompt: str, context: str = "", temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError("Azure provider: implement via openai.AzureOpenAI client")


# ── Factory ───────────────────────────────────────────────────────────────────

def create_provider(provider_type: LLMProviderType, **kwargs) -> LLMProvider:
    """Factory: create an LLM provider by type."""
    registry = {
        LLMProviderType.MOCK:      MockLLMProvider,
        LLMProviderType.OPENAI:    OpenAIProvider,
        LLMProviderType.ANTHROPIC: AnthropicProvider,
        LLMProviderType.GEMINI:    GeminiProvider,
        LLMProviderType.OLLAMA:    OllamaProvider,
        LLMProviderType.AZURE:     AzureOpenAIProvider,
    }
    cls = registry.get(provider_type)
    if cls is None:
        raise ValueError(f"Unknown provider type: {provider_type}")
    return cls(**kwargs)
