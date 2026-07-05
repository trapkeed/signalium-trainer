"""Phase 23 — Signalium AI Quant Research Analyst (SAQRA)."""

from trainer.analyst.analyst_types import (
    QueryType,
    ReportType,
    Permission,
    Role,
    LLMProviderType,
    CitationType,
    RecommendationType,
    KnowledgeSourceType,
    ROLE_PERMISSIONS,
    PROHIBITED_PATTERNS,
    has_permission,
    is_safe_response,
)

from trainer.analyst.analyst_models import (
    KnowledgeChunk,
    Citation,
    LLMResponse,
    ResearchQuery,
    ChatMessage,
    ChatSession,
    ResearchReport,
    Recommendation,
)

from trainer.analyst.llm_provider import (
    LLMProvider,
    MockLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    AzureOpenAIProvider,
    create_provider,
)

from trainer.analyst.knowledge_index    import KnowledgeIndex
from trainer.analyst.retrieval_engine   import RetrievalEngine
from trainer.analyst.prompt_engine      import PromptEngine
from trainer.analyst.citation_engine    import CitationEngine
from trainer.analyst.research_chat      import ResearchChat
from trainer.analyst.research_search    import ResearchSearch
from trainer.analyst.model_comparator   import ModelComparator
from trainer.analyst.report_generator   import ReportGenerator
from trainer.analyst.recommendation_engine import RecommendationEngine
from trainer.analyst.analyst_database   import AnalystDatabase
from trainer.analyst.analyst_dashboard  import AnalystDashboard
from trainer.analyst.analyst_api        import AnalystAPI

__all__ = [
    # Types
    "QueryType", "ReportType", "Permission", "Role", "LLMProviderType",
    "CitationType", "RecommendationType", "KnowledgeSourceType",
    "ROLE_PERMISSIONS", "PROHIBITED_PATTERNS",
    "has_permission", "is_safe_response",
    # Models
    "KnowledgeChunk", "Citation", "LLMResponse", "ResearchQuery",
    "ChatMessage", "ChatSession", "ResearchReport", "Recommendation",
    # LLM Providers
    "LLMProvider", "MockLLMProvider", "OpenAIProvider", "AnthropicProvider",
    "GeminiProvider", "OllamaProvider", "AzureOpenAIProvider", "create_provider",
    # Core systems
    "KnowledgeIndex", "RetrievalEngine", "PromptEngine", "CitationEngine",
    "ResearchChat", "ResearchSearch", "ModelComparator",
    "ReportGenerator", "RecommendationEngine",
    # Infrastructure
    "AnalystDatabase", "AnalystDashboard",
    # Unified API
    "AnalystAPI",
]
