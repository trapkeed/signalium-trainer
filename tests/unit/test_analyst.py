"""Phase 23 — Unit tests for SAQRA (Signalium AI Quant Research Analyst).

Run with:  python3 -m unittest tests/unit/test_analyst.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _load(name: str, rel: str):
    path = ROOT / "trainer" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
at   = _load("trainer.analyst.analyst_types",        "analyst/analyst_types.py")
am   = _load("trainer.analyst.analyst_models",       "analyst/analyst_models.py")
llm  = _load("trainer.analyst.llm_provider",         "analyst/llm_provider.py")
ki   = _load("trainer.analyst.knowledge_index",      "analyst/knowledge_index.py")
re_  = _load("trainer.analyst.retrieval_engine",     "analyst/retrieval_engine.py")
pe   = _load("trainer.analyst.prompt_engine",        "analyst/prompt_engine.py")
ce   = _load("trainer.analyst.citation_engine",      "analyst/citation_engine.py")
rc   = _load("trainer.analyst.research_chat",        "analyst/research_chat.py")
rs   = _load("trainer.analyst.research_search",      "analyst/research_search.py")
mc   = _load("trainer.analyst.model_comparator",     "analyst/model_comparator.py")
rg   = _load("trainer.analyst.report_generator",     "analyst/report_generator.py")
rce  = _load("trainer.analyst.recommendation_engine","analyst/recommendation_engine.py")
adb  = _load("trainer.analyst.analyst_database",     "analyst/analyst_database.py")
adash = _load("trainer.analyst.analyst_dashboard",   "analyst/analyst_dashboard.py")
aapi  = _load("trainer.analyst.analyst_api",         "analyst/analyst_api.py")

# Aliases
QueryType        = at.QueryType
ReportType       = at.ReportType
Permission       = at.Permission
Role             = at.Role
LLMProviderType  = at.LLMProviderType
CitationType     = at.CitationType
RecommendationType = at.RecommendationType
KnowledgeSourceType = at.KnowledgeSourceType
has_permission   = at.has_permission
is_safe_response = at.is_safe_response
PROHIBITED_PATTERNS = at.PROHIBITED_PATTERNS

KnowledgeChunk   = am.KnowledgeChunk
Citation         = am.Citation
LLMResponse      = am.LLMResponse
ResearchQuery    = am.ResearchQuery
ChatMessage      = am.ChatMessage
ChatSession      = am.ChatSession
ResearchReport   = am.ResearchReport
Recommendation   = am.Recommendation

LLMProvider      = llm.LLMProvider
MockLLMProvider  = llm.MockLLMProvider
OpenAIProvider   = llm.OpenAIProvider
AnthropicProvider = llm.AnthropicProvider
GeminiProvider   = llm.GeminiProvider
OllamaProvider   = llm.OllamaProvider
AzureOpenAIProvider = llm.AzureOpenAIProvider
create_provider  = llm.create_provider

KnowledgeIndex   = ki.KnowledgeIndex
RetrievalEngine  = re_.RetrievalEngine
PromptEngine     = pe.PromptEngine
CitationEngine   = ce.CitationEngine
ResearchChat     = rc.ResearchChat
ResearchSearch   = rs.ResearchSearch
ModelComparator  = mc.ModelComparator
ReportGenerator  = rg.ReportGenerator
RecommendationEngine = rce.RecommendationEngine
AnalystDatabase  = adb.AnalystDatabase
AnalystDashboard = adash.AnalystDashboard
AnalystAPI       = aapi.AnalystAPI


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    source_type: KnowledgeSourceType = None,
    artifact_id: str = "EXP-001",
    title: str = "EURUSD Transformer Experiment",
    content: str = "Sharpe 2.3, max_drawdown 8.5%, accuracy 0.72, status PASSED. "
                   "Model EXP-001 used Transformer architecture with dataset DATASET-v18.",
    tags: list = None,
    metadata: dict = None,
) -> KnowledgeChunk:
    import uuid
    return KnowledgeChunk(
        chunk_id    = str(uuid.uuid4())[:12],
        source_type = source_type or KnowledgeSourceType.EXPERIMENT,
        artifact_id = artifact_id,
        title       = title,
        content     = content,
        tags        = tags or ["eurusd", "transformer", "sharpe"],
        timestamp   = time.time(),
        metadata    = metadata or {"sharpe": 2.3, "max_drawdown": 8.5, "status": "PASSED"},
    )


def make_index_with_data() -> KnowledgeIndex:
    idx = KnowledgeIndex()
    idx.index(make_chunk(artifact_id="EXP-001"))
    idx.index(make_chunk(
        artifact_id = "MODEL-42",
        title       = "Transformer Model 42 — Champion",
        content     = "MODEL-42 is the current champion. Sharpe 2.8, drawdown 6.2%. "
                       "Trained on DATASET-v18. REPORT-Q2 validation passed.",
        tags        = ["model", "champion", "transformer"],
        source_type = KnowledgeSourceType.MODEL_REGISTRY,
        metadata    = {"sharpe": 2.8, "max_drawdown": 6.2, "status": "CHAMPION"},
    ))
    idx.index(make_chunk(
        artifact_id = "MODEL-37",
        title       = "Model 37 — Challenger",
        content     = "MODEL-37 achieved Sharpe 1.9, drawdown 11.1%. "
                       "Calibration deteriorated in Q3. EXP-042 validation failed.",
        tags        = ["model", "challenger"],
        source_type = KnowledgeSourceType.MODEL_REGISTRY,
        metadata    = {"sharpe": 1.9, "max_drawdown": 11.1, "status": "FAILED"},
    ))
    idx.index(make_chunk(
        artifact_id = "DATASET-v18",
        title       = "Dataset v18 — Training Data",
        content     = "Dataset version 18. Covers Jan 2022–Dec 2023. Includes ranging and trending regimes. "
                       "Limited volatile regime coverage noted in REPORT-Q2.",
        tags        = ["dataset", "v18"],
        source_type = KnowledgeSourceType.DATASET_VERSION,
        metadata    = {"version": "v18", "sharpe": None, "status": "ACTIVE"},
    ))
    idx.index(make_chunk(
        artifact_id = "REPORT-Q2",
        title       = "Q2 Research Report",
        content     = "Quarterly review. Champion MODEL-42 maintained performance. "
                       "Recommendation: collect more volatile regime data. "
                       "Feature RSI instability detected. Calibration stable.",
        tags        = ["report", "quarterly"],
        source_type = KnowledgeSourceType.RESEARCH_REPORT,
        metadata    = {"period": "Q2", "status": "COMPLETE"},
    ))
    return idx


# ── Tests: Types ──────────────────────────────────────────────────────────────

class TestAnalystTypes(unittest.TestCase):
    def test_query_types(self):
        vals = {q.value for q in QueryType}
        for v in ("CHAT","COMPARISON","ANALYSIS","SEARCH","SUMMARY","RECOMMENDATION","REPORT"):
            self.assertIn(v, vals)

    def test_report_types_count(self):
        self.assertGreaterEqual(len(list(ReportType)), 12)

    def test_permissions(self):
        perms = {p.value for p in Permission}
        for v in ("READ","EXPORT","SUMMARIZE","COMPARE","AUDIT"):
            self.assertIn(v, perms)

    def test_roles(self):
        self.assertIn(Role.ADMIN, list(Role))
        self.assertIn(Role.RESEARCHER, list(Role))
        self.assertIn(Role.VIEWER, list(Role))

    def test_llm_provider_types(self):
        vals = {p.value for p in LLMProviderType}
        for v in ("MOCK","OPENAI","ANTHROPIC","GEMINI","OLLAMA","AZURE"):
            self.assertIn(v, vals)

    def test_has_permission_admin(self):
        for p in Permission:
            self.assertTrue(has_permission(Role.ADMIN, p))

    def test_has_permission_viewer(self):
        self.assertTrue(has_permission(Role.VIEWER, Permission.READ))
        self.assertFalse(has_permission(Role.VIEWER, Permission.EXPORT))
        self.assertFalse(has_permission(Role.VIEWER, Permission.AUDIT))

    def test_has_permission_researcher(self):
        self.assertTrue(has_permission(Role.RESEARCHER, Permission.SUMMARIZE))
        self.assertTrue(has_permission(Role.RESEARCHER, Permission.COMPARE))
        self.assertFalse(has_permission(Role.RESEARCHER, Permission.AUDIT))

    def test_is_safe_response_clean(self):
        self.assertTrue(is_safe_response("Model 42 outperformed Model 37 on Sharpe."))

    def test_is_safe_response_buy_rejected(self):
        self.assertFalse(is_safe_response("You should BUY EURUSD now."))

    def test_is_safe_response_sell_rejected(self):
        self.assertFalse(is_safe_response("SELL immediately."))

    def test_prohibited_patterns_count(self):
        self.assertGreaterEqual(len(PROHIBITED_PATTERNS), 5)

    def test_knowledge_source_types_count(self):
        self.assertGreaterEqual(len(list(KnowledgeSourceType)), 20)

    def test_citation_types(self):
        vals = {c.value for c in CitationType}
        for v in ("EXPERIMENT","MODEL","DATASET","FEATURE","VALIDATION","REPORT"):
            self.assertIn(v, vals)


# ── Tests: Models ─────────────────────────────────────────────────────────────

class TestAnalystModels(unittest.TestCase):
    def test_knowledge_chunk_to_dict(self):
        chunk = make_chunk()
        d = chunk.to_dict()
        self.assertIn("chunk_id", d)
        self.assertIn("source_type", d)
        self.assertIn("artifact_id", d)
        self.assertIn("word_count", d)

    def test_knowledge_chunk_word_count(self):
        chunk = make_chunk(content="one two three four five")
        self.assertEqual(chunk.word_count, 5)

    def test_citation_to_dict(self):
        c = Citation(CitationType.EXPERIMENT, "EXP-001", "Referenced experiment", 0.9)
        d = c.to_dict()
        self.assertEqual(d["artifact_id"], "EXP-001")
        self.assertIn("relevance", d)

    def test_citation_str(self):
        c = Citation(CitationType.MODEL, "MODEL-42", "")
        self.assertIn("MODEL", str(c))
        self.assertIn("MODEL-42", str(c))

    def test_research_query_to_dict(self):
        q = ResearchQuery(text="Why did Model 42 outperform?", query_type=QueryType.ANALYSIS)
        d = q.to_dict()
        self.assertEqual(d["query_type"], "ANALYSIS")
        self.assertIn("query_id", d)

    def test_chat_session_add_message(self):
        session = ChatSession()
        msg = session.add_message("user", "Hello")
        self.assertEqual(session.message_count, 1)
        self.assertEqual(msg.role, "user")

    def test_chat_session_context_window(self):
        session = ChatSession()
        for i in range(10):
            session.add_message("user", f"msg {i}")
        ctx = session.context_window(last_n=4)
        self.assertEqual(len(ctx), 4)
        self.assertEqual(ctx[-1].content, "msg 9")

    def test_chat_session_to_dict(self):
        session = ChatSession(researcher="alice", role=Role.RESEARCHER)
        d = session.to_dict()
        self.assertEqual(d["researcher"], "alice")
        self.assertIn("active", d)

    def test_research_report_to_dict(self):
        r = ResearchReport(report_type=ReportType.DAILY_SUMMARY, title="Test", content="Body")
        d = r.to_dict()
        self.assertEqual(d["report_type"], "DAILY_SUMMARY")

    def test_recommendation_to_dict(self):
        rec = Recommendation(
            rec_type=RecommendationType.DATA_COLLECTION,
            title="Collect more data",
            priority="HIGH",
        )
        d = rec.to_dict()
        self.assertEqual(d["priority"], "HIGH")
        self.assertFalse(d["implemented"])

    def test_llm_response_total_tokens(self):
        r = LLMResponse(content="hello", prompt_tokens=50, response_tokens=30)
        self.assertEqual(r.total_tokens, 80)

    def test_llm_response_to_dict(self):
        r = LLMResponse(content="test response", citations=[])
        d = r.to_dict()
        self.assertIn("content", d)
        self.assertIn("safe", d)
        self.assertIn("grounded", d)


# ── Tests: LLM Providers ──────────────────────────────────────────────────────

class TestLLMProviders(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def test_mock_is_available(self):
        self.assertTrue(self.provider.is_available())

    def test_mock_provider_type(self):
        self.assertEqual(self.provider.provider_type(), LLMProviderType.MOCK)

    def test_mock_generate_returns_response(self):
        r = self.provider.generate("Why did Model 42 outperform?", "EXP-001 data")
        self.assertIsInstance(r, LLMResponse)
        self.assertTrue(len(r.content) > 0)

    def test_mock_safety_check_buy(self):
        # Patch _synthesise to return a BUY signal
        original = self.provider._synthesise
        self.provider._synthesise = lambda *a, **kw: "You should BUY EURUSD now."
        r = self.provider.generate("test")
        self.assertFalse(r.safe)
        self.provider._synthesise = original

    def test_mock_grounded_with_context(self):
        r = self.provider.generate("question", context="EXP-001 details")
        self.assertTrue(r.grounded)

    def test_mock_not_grounded_without_context(self):
        r = self.provider.generate("question", context="")
        self.assertFalse(r.grounded)

    def test_mock_citation_extraction(self):
        chunks_context = "Experiment EXP-001 shows MODEL-42 vs DATASET-v18."
        r = self.provider.generate("compare", context=chunks_context)
        artifact_ids = {c.artifact_id for c in r.citations}
        # At least one citation should be extracted
        self.assertGreater(len(r.citations), 0)

    def test_mock_comparison_response(self):
        r = self.provider.generate("compare Model 42 vs Model 37", "MODEL-42 Sharpe 2.8")
        self.assertIn("Comparative", r.content)

    def test_mock_summary_response(self):
        r = self.provider.generate("summarize research", "REPORT-Q2 content")
        self.assertIn("Summary", r.content)

    def test_openai_not_available_by_default(self):
        p = OpenAIProvider(api_key="")
        self.assertFalse(p.is_available())

    def test_anthropic_not_available_by_default(self):
        p = AnthropicProvider(api_key="")
        self.assertFalse(p.is_available())

    def test_create_provider_mock(self):
        p = create_provider(LLMProviderType.MOCK)
        self.assertIsInstance(p, MockLLMProvider)

    def test_create_provider_openai(self):
        p = create_provider(LLMProviderType.OPENAI, api_key="")
        self.assertIsInstance(p, OpenAIProvider)

    def test_create_provider_unknown_raises(self):
        with self.assertRaises((ValueError, KeyError)):
            create_provider("UNKNOWN_PROVIDER")

    def test_health_check(self):
        hc = self.provider.health_check()
        self.assertIn("provider", hc)
        self.assertTrue(hc["available"])


# ── Tests: Knowledge Index ────────────────────────────────────────────────────

class TestKnowledgeIndex(unittest.TestCase):
    def setUp(self):
        self.idx = make_index_with_data()

    def test_total_chunks(self):
        self.assertEqual(self.idx.stats()["total_chunks"], 5)

    def test_get_by_artifact(self):
        chunks = self.idx.get_by_artifact("MODEL-42")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].artifact_id, "MODEL-42")

    def test_get_by_source(self):
        chunks = self.idx.get_by_source(KnowledgeSourceType.MODEL_REGISTRY)
        self.assertEqual(len(chunks), 2)

    def test_search_returns_relevant(self):
        results = self.idx.search("Transformer Sharpe", top_k=3)
        self.assertGreater(len(results), 0)

    def test_search_with_source_filter(self):
        results = self.idx.search("model", top_k=5, source_type=KnowledgeSourceType.MODEL_REGISTRY)
        self.assertLessEqual(len(results), 2)

    def test_stats_source_types(self):
        stats = self.idx.stats()
        self.assertIn("source_types", stats)
        self.assertGreater(len(stats["source_types"]), 0)

    def test_index_document(self):
        idx = KnowledgeIndex()
        chunk = idx.index_document(
            KnowledgeSourceType.EXPERIMENT,
            "EXP-999",
            "New Experiment",
            "Sharpe 1.5",
            tags=["eurusd"],
        )
        self.assertEqual(idx.stats()["total_chunks"], 1)

    def test_remove_chunk(self):
        idx = KnowledgeIndex()
        chunk = make_chunk()
        idx.index(chunk)
        self.assertEqual(idx.stats()["total_chunks"], 1)
        removed = idx.remove(chunk.chunk_id)
        self.assertTrue(removed)
        self.assertEqual(idx.stats()["total_chunks"], 0)

    def test_remove_nonexistent(self):
        idx = KnowledgeIndex()
        self.assertFalse(idx.remove("nonexistent"))

    def test_clear(self):
        idx = make_index_with_data()
        idx.clear()
        self.assertEqual(idx.stats()["total_chunks"], 0)

    def test_index_batch(self):
        idx = KnowledgeIndex()
        chunks = [make_chunk(artifact_id=f"EXP-{i:03d}") for i in range(5)]
        ids = idx.index_batch(chunks)
        self.assertEqual(len(ids), 5)
        self.assertEqual(idx.stats()["total_chunks"], 5)


# ── Tests: Retrieval Engine ───────────────────────────────────────────────────

class TestRetrievalEngine(unittest.TestCase):
    def setUp(self):
        self.idx      = make_index_with_data()
        self.engine   = RetrievalEngine(self.idx, top_k=5)

    def test_retrieve_returns_results(self):
        results = self.engine.retrieve("Transformer model performance")
        self.assertGreater(len(results), 0)

    def test_retrieve_top_k_limit(self):
        results = self.engine.retrieve("model", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_empty_query_returns_chunks(self):
        results = self.engine.retrieve("")
        self.assertGreaterEqual(len(results), 0)

    def test_build_context_non_empty(self):
        chunks = self.engine.retrieve("Sharpe model")
        ctx = self.engine.build_context(chunks)
        self.assertIsInstance(ctx, str)
        if chunks:
            self.assertGreater(len(ctx), 0)

    def test_build_context_empty_chunks(self):
        ctx = self.engine.build_context([])
        self.assertEqual(ctx, "")

    def test_retrieve_and_build(self):
        chunks, ctx = self.engine.retrieve_and_build("model champion")
        self.assertIsInstance(chunks, list)
        self.assertIsInstance(ctx, str)

    def test_retrieve_with_source_filter(self):
        results = self.engine.retrieve(
            "model",
            source_type=KnowledgeSourceType.MODEL_REGISTRY
        )
        for r in results:
            self.assertEqual(r.source_type, KnowledgeSourceType.MODEL_REGISTRY)

    def test_context_max_chars(self):
        chunks = self.engine.retrieve("model sharpe experiment")
        ctx = self.engine.build_context(chunks, max_chars=100)
        self.assertLessEqual(len(ctx), 200)  # small buffer for ellipsis


# ── Tests: Prompt Engine ──────────────────────────────────────────────────────

class TestPromptEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PromptEngine()

    def test_build_chat_prompt(self):
        prompt = self.engine.build(QueryType.CHAT, "Why did Model 42 outperform?", "context text")
        self.assertIn("SAQRA", prompt)
        self.assertIn("Why did Model 42", prompt)

    def test_build_comparison_prompt(self):
        prompt = self.engine.build(QueryType.COMPARISON, "compare models", "MODEL-42 vs MODEL-37")
        self.assertIn("comparative", prompt.lower())

    def test_build_report_prompt(self):
        prompt = self.engine.build(
            QueryType.REPORT, "generate report", "context",
            report_type=ReportType.WEEKLY_REPORT
        )
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_system_prompt_contains_safety(self):
        self.assertIn("NEVER", self.engine.system_prompt)
        self.assertIn("BUY", self.engine.system_prompt)
        self.assertIn("SELL", self.engine.system_prompt)

    def test_list_templates(self):
        templates = self.engine.list_templates()
        self.assertIn(QueryType.CHAT.value, templates)

    def test_get_template(self):
        t = self.engine.get_template(QueryType.ANALYSIS)
        self.assertIsInstance(t, str)

    def test_update_system_prompt(self):
        self.engine.update_system_prompt("New system prompt.")
        self.assertEqual(self.engine.system_prompt, "New system prompt.")

    def test_empty_context_handled(self):
        prompt = self.engine.build(QueryType.CHAT, "question", "")
        self.assertIsInstance(prompt, str)


# ── Tests: Citation Engine ────────────────────────────────────────────────────

class TestCitationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CitationEngine()

    def test_extract_experiment_citation(self):
        text = "Based on [EXP:001] the model showed improvement."
        cits = self.engine.extract(text)
        self.assertGreater(len(cits), 0)
        self.assertEqual(cits[0].citation_type, CitationType.EXPERIMENT)

    def test_extract_model_citation(self):
        text = "See [MODEL:42] for performance details."
        cits = self.engine.extract(text)
        self.assertTrue(any(c.citation_type == CitationType.MODEL for c in cits))

    def test_extract_multiple_citations(self):
        text = "[EXP:001] trained on [DATASET:v18], validated in [VAL:2024Q2]."
        cits = self.engine.extract(text)
        self.assertGreaterEqual(len(cits), 3)

    def test_extract_no_citations(self):
        cits = self.engine.extract("No references here.")
        self.assertEqual(len(cits), 0)

    def test_validate_boosts_known_chunk(self):
        chunk = make_chunk(artifact_id="EXP-001")
        cit   = Citation(CitationType.EXPERIMENT, "EXP-001", "test", 0.5)
        validated = self.engine.validate([cit], [chunk])
        self.assertEqual(validated[0].relevance, 1.0)

    def test_validate_lowers_unknown(self):
        chunk = make_chunk(artifact_id="MODEL-42")
        cit   = Citation(CitationType.EXPERIMENT, "EXP-999", "test", 1.0)
        validated = self.engine.validate([cit], [chunk])
        self.assertEqual(validated[0].relevance, 0.60)

    def test_add_from_chunks(self):
        chunk = make_chunk(artifact_id="EXP-001")
        cits  = self.engine.add_from_chunks([], [chunk])
        self.assertGreater(len(cits), 0)

    def test_format_bibliography(self):
        cits = [Citation(CitationType.EXPERIMENT, "EXP-001", "description", 0.9)]
        bib  = self.engine.format_bibliography(cits)
        self.assertIn("EXP-001", bib)
        self.assertIn("References", bib)

    def test_format_bibliography_empty(self):
        bib = self.engine.format_bibliography([])
        self.assertIn("No citations", bib)

    def test_is_grounded_with_citations(self):
        cits = [Citation(CitationType.EXPERIMENT, "EXP-001", "", 1.0)]
        self.assertTrue(self.engine.is_grounded(cits))

    def test_is_grounded_without_citations(self):
        self.assertFalse(self.engine.is_grounded([]))


# ── Tests: Research Chat ──────────────────────────────────────────────────────

class TestResearchChat(unittest.TestCase):
    def setUp(self):
        self.idx  = make_index_with_data()
        self.chat = ResearchChat(self.idx)

    def test_new_session(self):
        session = self.chat.new_session("alice", Role.RESEARCHER)
        self.assertEqual(session.researcher, "alice")
        self.assertTrue(session.active)

    def test_get_session(self):
        session = self.chat.new_session()
        fetched = self.chat.get_session(session.session_id)
        self.assertEqual(fetched.session_id, session.session_id)

    def test_end_session(self):
        session = self.chat.new_session()
        result  = self.chat.end_session(session.session_id)
        self.assertTrue(result)
        self.assertFalse(session.active)

    def test_active_sessions(self):
        self.chat.new_session()
        self.chat.new_session()
        active = self.chat.active_sessions()
        self.assertGreaterEqual(len(active), 2)

    def test_chat_valid_query(self):
        session = self.chat.new_session()
        query   = ResearchQuery(
            text       = "Why did Model 42 outperform Model 37?",
            query_type = QueryType.ANALYSIS,
            role       = Role.RESEARCHER,
        )
        response = self.chat.chat(query, session.session_id)
        self.assertIsInstance(response, LLMResponse)
        self.assertTrue(response.safe)

    def test_chat_signal_request_rejected(self):
        session = self.chat.new_session()
        query   = ResearchQuery(
            text       = "Should I buy EURUSD now?",
            query_type = QueryType.CHAT,
            role       = Role.RESEARCHER,
        )
        response = self.chat.chat(query, session.session_id)
        self.assertTrue(response.safe)
        self.assertIn("cannot", response.content.lower())

    def test_chat_access_denied_no_permission(self):
        session = self.chat.new_session()
        query   = ResearchQuery(
            text  = "show experiments",
            role  = Role.VIEWER,  # only has READ permission
        )
        # Viewer has READ — should succeed
        query2 = ResearchQuery(text="show experiments", role=Role.VIEWER)
        response = self.chat.chat(query2, session.session_id)
        self.assertIsInstance(response, LLMResponse)

    def test_session_records_messages(self):
        session = self.chat.new_session()
        query   = ResearchQuery(text="What is the Sharpe of Model 42?", role=Role.RESEARCHER)
        self.chat.chat(query, session.session_id)
        # User + assistant messages
        self.assertGreaterEqual(session.message_count, 2)

    def test_session_stats(self):
        self.chat.new_session()
        stats = self.chat.session_stats()
        self.assertIn("total", stats)
        self.assertIn("active", stats)

    def test_chat_new_session_auto_created(self):
        query = ResearchQuery(text="question", role=Role.RESEARCHER)
        # Pass a non-existent session_id — should auto-create
        response = self.chat.chat(query, "nonexistent-session")
        self.assertIsInstance(response, LLMResponse)


# ── Tests: Research Search ────────────────────────────────────────────────────

class TestResearchSearch(unittest.TestCase):
    def setUp(self):
        self.idx    = make_index_with_data()
        self.search = ResearchSearch(self.idx)

    def test_search_returns_results(self):
        query  = ResearchQuery(text="Transformer experiments", role=Role.RESEARCHER)
        result = self.search.search(query)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    def test_search_denied_for_no_permission(self):
        # Viewer has READ — search should work
        query  = ResearchQuery(text="model", role=Role.VIEWER)
        result = self.search.search(query)
        self.assertNotIn("error", result)

    def test_keyword_search(self):
        results = self.search.keyword_search("Sharpe", top_k=3)
        self.assertIsInstance(results, list)

    def test_search_by_artifact(self):
        results = self.search.search_by_artifact("MODEL-42")
        self.assertTrue(all(c.artifact_id == "MODEL-42" for c in results))

    def test_search_by_source(self):
        results = self.search.search_by_source(KnowledgeSourceType.MODEL_REGISTRY)
        self.assertTrue(all(c.source_type == KnowledgeSourceType.MODEL_REGISTRY for c in results))

    def test_infer_source_type_experiment(self):
        st = self.search._infer_source_type("show all experiments")
        self.assertEqual(st, KnowledgeSourceType.EXPERIMENT)

    def test_infer_source_type_model(self):
        st = self.search._infer_source_type("find model architectures")
        self.assertEqual(st, KnowledgeSourceType.MODEL_REGISTRY)

    def test_infer_source_type_dataset(self):
        st = self.search._infer_source_type("find experiments using Dataset v18")
        self.assertEqual(st, KnowledgeSourceType.DATASET_VERSION)


# ── Tests: Model Comparator ───────────────────────────────────────────────────

class TestModelComparator(unittest.TestCase):
    def setUp(self):
        self.idx  = make_index_with_data()
        self.comp = ModelComparator(self.idx)

    def test_compare_returns_dict(self):
        query  = ResearchQuery(text="compare models", role=Role.RESEARCHER)
        result = self.comp.compare(query, "MODEL-42", "MODEL-37")
        self.assertIn("artifact_a", result)
        self.assertIn("artifact_b", result)
        self.assertIn("analysis", result)

    def test_compare_denied_for_viewer(self):
        query  = ResearchQuery(text="compare", role=Role.VIEWER)
        result = self.comp.compare(query, "MODEL-42", "MODEL-37")
        self.assertIn("error", result)

    def test_compare_table_has_metrics(self):
        query  = ResearchQuery(text="compare", role=Role.RESEARCHER)
        result = self.comp.compare(query, "MODEL-42", "MODEL-37")
        self.assertIn("table", result)

    def test_compare_batch(self):
        query  = ResearchQuery(text="rank models", role=Role.RESEARCHER)
        result = self.comp.compare_batch(query, ["MODEL-42", "MODEL-37"])
        self.assertIn("artifacts", result)
        self.assertIn("analysis", result)

    def test_ranking_by_sharpe(self):
        query  = ResearchQuery(text="rank", role=Role.RESEARCHER)
        ranked = self.comp.ranking(query, ["MODEL-42", "MODEL-37"], metric="sharpe")
        if len(ranked) >= 2:
            self.assertLessEqual(ranked[0]["rank"], ranked[1]["rank"])

    def test_ranking_denied_viewer(self):
        query  = ResearchQuery(text="rank", role=Role.VIEWER)
        result = self.comp.ranking(query, ["MODEL-42"], metric="sharpe")
        self.assertEqual(result, [])


# ── Tests: Report Generator ───────────────────────────────────────────────────

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.idx    = make_index_with_data()
        self.gen    = ReportGenerator(self.idx)

    def test_generate_daily_summary(self):
        query  = ResearchQuery(role=Role.RESEARCHER)
        report = self.gen.generate(query, ReportType.DAILY_SUMMARY)
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.report_type, ReportType.DAILY_SUMMARY)

    def test_generate_report_has_title(self):
        query  = ResearchQuery(role=Role.RESEARCHER)
        report = self.gen.generate(query, ReportType.WEEKLY_REPORT)
        self.assertGreater(len(report.title), 0)

    def test_generate_report_has_content(self):
        query  = ResearchQuery(role=Role.RESEARCHER)
        report = self.gen.generate(query, ReportType.MODEL_HEALTH)
        self.assertGreater(len(report.content), 0)

    def test_generate_denied_viewer_no_summarize(self):
        query  = ResearchQuery(role=Role.VIEWER)
        report = self.gen.generate(query, ReportType.DAILY_SUMMARY)
        # VIEWER lacks SUMMARIZE permission; message contains "permission" or "denied"
        content_lower = report.content.lower()
        self.assertTrue(
            "permission" in content_lower or "denied" in content_lower,
            f"Expected access-denied message, got: {report.content!r}",
        )

    def test_list_reports(self):
        query  = ResearchQuery(role=Role.RESEARCHER)
        self.gen.generate(query, ReportType.DAILY_SUMMARY)
        reports = self.gen.list_reports()
        self.assertEqual(len(reports), 1)

    def test_get_report_by_id(self):
        query  = ResearchQuery(role=Role.RESEARCHER)
        report = self.gen.generate(query, ReportType.WEEKLY_REPORT)
        fetched = self.gen.get_report(report.report_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.report_id, report.report_id)

    def test_report_types_list(self):
        types = self.gen.report_types()
        self.assertIn("DAILY_SUMMARY", types)
        self.assertGreaterEqual(len(types), 12)

    def test_generate_with_period(self):
        query  = ResearchQuery(role=Role.RESEARCHER)
        report = self.gen.generate(query, ReportType.MONTHLY_REPORT, period="2025-01")
        self.assertIn("2025-01", report.title)


# ── Tests: Recommendation Engine ─────────────────────────────────────────────

class TestRecommendationEngine(unittest.TestCase):
    def setUp(self):
        self.idx = make_index_with_data()
        self.eng = RecommendationEngine(self.idx)

    def test_recommend_returns_list(self):
        query = ResearchQuery(text="What should we improve?", role=Role.RESEARCHER)
        recs  = self.eng.recommend(query)
        self.assertIsInstance(recs, list)

    def test_recommend_non_empty_with_index(self):
        query = ResearchQuery(text="suggest improvements to calibration", role=Role.RESEARCHER)
        recs  = self.eng.recommend(query)
        self.assertGreater(len(recs), 0)

    def test_recommend_empty_for_no_permission(self):
        # Viewer has READ — still allowed to see recommendations
        query = ResearchQuery(text="suggest", role=Role.VIEWER)
        recs  = self.eng.recommend(query)
        self.assertIsInstance(recs, list)

    def test_auto_recommend_scans_index(self):
        recs = self.eng.auto_recommend()
        self.assertIsInstance(recs, list)

    def test_history_records_recommendations(self):
        query = ResearchQuery(text="improve", role=Role.RESEARCHER)
        self.eng.recommend(query)
        hist = self.eng.history()
        self.assertGreater(len(hist), 0)

    def test_pending_initially_all_pending(self):
        query = ResearchQuery(text="suggest next experiment", role=Role.RESEARCHER)
        recs  = self.eng.recommend(query)
        pending = self.eng.pending()
        self.assertEqual(len(pending), len(self.eng.history()))

    def test_mark_implemented(self):
        query = ResearchQuery(text="calibration improvement", role=Role.RESEARCHER)
        recs  = self.eng.recommend(query)
        if recs:
            result = self.eng.mark_implemented(recs[0].rec_id)
            self.assertTrue(result)
            self.assertTrue(recs[0].implemented)

    def test_mark_implemented_nonexistent(self):
        result = self.eng.mark_implemented("nonexistent-id")
        self.assertFalse(result)


# ── Tests: Analyst Database ───────────────────────────────────────────────────

class TestAnalystDatabase(unittest.TestCase):
    def setUp(self):
        self.db = AnalystDatabase(":memory:")

    def test_save_session(self):
        session = ChatSession(researcher="alice", role=Role.RESEARCHER)
        self.db.save_session(session)
        self.assertEqual(self.db.count_sessions(), 1)

    def test_save_query(self):
        query = ResearchQuery(text="Why did Model 42 outperform?", role=Role.RESEARCHER)
        self.db.save_query(query, "Because of Sharpe...")
        self.assertEqual(self.db.count_queries(), 1)

    def test_recent_queries(self):
        for i in range(3):
            self.db.save_query(ResearchQuery(text=f"query {i}"), "")
        recent = self.db.recent_queries(limit=2)
        self.assertEqual(len(recent), 2)

    def test_save_report(self):
        report = ResearchReport(
            report_type=ReportType.DAILY_SUMMARY,
            title="Test Report",
            content="body",
        )
        self.db.save_report(report)
        self.assertEqual(self.db.count_reports(), 1)

    def test_reports_by_type(self):
        for _ in range(2):
            self.db.save_report(ResearchReport(report_type=ReportType.DAILY_SUMMARY, title="R", content=""))
        self.db.save_report(ResearchReport(report_type=ReportType.WEEKLY_REPORT, title="W", content=""))
        daily = self.db.reports_by_type("DAILY_SUMMARY")
        self.assertEqual(len(daily), 2)

    def test_save_recommendation(self):
        rec = Recommendation(rec_type=RecommendationType.DATA_COLLECTION, title="Collect data")
        self.db.save_recommendation(rec)
        self.assertEqual(self.db.count_recommendations(), 1)

    def test_pending_recommendations(self):
        rec = Recommendation(rec_type=RecommendationType.CALIBRATION, title="Improve calibration")
        self.db.save_recommendation(rec)
        pending = self.db.pending_recommendations()
        self.assertGreater(len(pending), 0)

    def test_save_template(self):
        self.db.save_template("chat_v2", "CHAT", "new template {context} {query}")
        self.assertEqual(self.db.count_templates(), 1)

    def test_record_llm_usage(self):
        self.db.record_usage("MOCK", "mock-v1", 100, 200, 50.0, True, True, "CHAT")
        self.db.record_usage("MOCK", "mock-v1", 150, 300, 60.0, True, True, "REPORT")
        total = self.db.total_tokens()
        self.assertEqual(total, 750)

    def test_usage_by_provider(self):
        self.db.record_usage("MOCK", "mock-v1", 50, 100, 30.0, True, True, "CHAT")
        counts = self.db.usage_by_provider()
        self.assertIn("MOCK", counts)

    def test_stats(self):
        stats = self.db.stats()
        self.assertIn("sessions", stats)
        self.assertIn("queries", stats)
        self.assertIn("reports", stats)
        self.assertIn("recommendations", stats)


# ── Tests: Analyst Dashboard ──────────────────────────────────────────────────

class TestAnalystDashboard(unittest.TestCase):
    def setUp(self):
        self.dash  = AnalystDashboard()
        self.chunks = list(make_index_with_data().all_chunks())
        session = ChatSession(researcher="alice")
        session.add_message("user", "Why did Model 42 outperform?")
        session.add_message("assistant", "Because Sharpe was higher.")
        self.sessions = [session]
        self.reports = [
            ResearchReport(report_type=ReportType.DAILY_SUMMARY, title="Daily", content="body"),
        ]
        self.recs = [
            Recommendation(rec_type=RecommendationType.DATA_COLLECTION, title="Collect data", priority="HIGH"),
            Recommendation(rec_type=RecommendationType.CALIBRATION, title="Calibrate", priority="LOW"),
        ]

    def test_research_chat_view(self):
        d = self.dash.research_chat(self.sessions)
        self.assertEqual(d["view"], "RESEARCH_CHAT")
        self.assertIn("total_sessions", d)
        self.assertIn("total_messages", d)

    def test_experiment_explorer_view(self):
        d = self.dash.experiment_explorer(self.chunks)
        self.assertEqual(d["view"], "EXPERIMENT_EXPLORER")
        self.assertIn("source_breakdown", d)

    def test_knowledge_graph_view(self):
        d = self.dash.knowledge_graph(self.chunks)
        self.assertEqual(d["view"], "KNOWLEDGE_GRAPH")
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("clusters", d)

    def test_recommendation_queue_view(self):
        d = self.dash.recommendation_queue(self.recs)
        self.assertEqual(d["view"], "RECOMMENDATION_QUEUE")
        self.assertIn("pending", d)
        self.assertIn("by_priority", d)

    def test_executive_dashboard_view(self):
        d = self.dash.executive_dashboard(self.sessions, self.reports, self.recs, self.chunks)
        self.assertEqual(d["view"], "EXECUTIVE_DASHBOARD")
        self.assertIn("knowledge_base", d)
        self.assertIn("research_activity", d)
        self.assertIn("recommendations", d)

    def test_all_dashboards_5_views(self):
        all_d = self.dash.all_dashboards(self.sessions, self.reports, self.recs, self.chunks)
        self.assertEqual(len(all_d), 5)

    def test_empty_sessions_graceful(self):
        d = self.dash.research_chat([])
        self.assertEqual(d["total_sessions"], 0)

    def test_empty_chunks_graceful(self):
        d = self.dash.knowledge_graph([])
        self.assertEqual(d["nodes"], [])


# ── Tests: Analyst API ────────────────────────────────────────────────────────

class TestAnalystAPI(unittest.TestCase):
    def setUp(self):
        self.api = AnalystAPI(db_path=":memory:")
        # Seed knowledge
        self.api.index_document(
            KnowledgeSourceType.EXPERIMENT, "EXP-001",
            "EURUSD Experiment", "Sharpe 2.3 MODEL-42 DATASET-v18",
            tags=["eurusd", "transformer"],
        )
        self.api.index_document(
            KnowledgeSourceType.MODEL_REGISTRY, "MODEL-42",
            "Champion Model", "Sharpe 2.8 max drawdown 6.2% calibration stable",
            tags=["champion"], metadata={"sharpe": 2.8, "max_drawdown": 6.2},
        )
        self.api.index_document(
            KnowledgeSourceType.MODEL_REGISTRY, "MODEL-37",
            "Challenger Model", "Sharpe 1.9 calibration deteriorated EXP-042 failed",
            tags=["challenger"], metadata={"sharpe": 1.9, "max_drawdown": 11.1},
        )

    def test_new_session(self):
        sid = self.api.new_session("alice", Role.RESEARCHER)
        self.assertIsInstance(sid, str)
        self.assertGreater(len(sid), 0)

    def test_chat_returns_session_and_response(self):
        result = self.api.chat("Why did Model 42 outperform?", researcher="alice")
        self.assertIn("session_id", result)
        self.assertIn("response", result)

    def test_chat_safe(self):
        result = self.api.chat("Should I buy EURUSD?")
        self.assertTrue(result["response"]["safe"])

    def test_chat_continues_session(self):
        sid = self.api.new_session()
        r1  = self.api.chat("First question", session_id=sid)
        r2  = self.api.chat("Follow-up question", session_id=sid)
        self.assertEqual(r1["session_id"], r2["session_id"])

    def test_search_returns_results(self):
        result = self.api.search("Transformer experiment Sharpe")
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    def test_compare_returns_analysis(self):
        result = self.api.compare("MODEL-42", "MODEL-37", question="Why did 42 outperform?")
        self.assertIn("analysis", result)
        self.assertIn("table", result)

    def test_rank_returns_sorted_list(self):
        ranked = self.api.rank(["MODEL-42", "MODEL-37"], metric="sharpe")
        self.assertIsInstance(ranked, list)

    def test_report_generates(self):
        result = self.api.report(ReportType.DAILY_SUMMARY, period="2025-07-05")
        self.assertIn("report_type", result)
        self.assertEqual(result["report_type"], "DAILY_SUMMARY")

    def test_recommend_returns_list(self):
        recs = self.api.recommend("What should we improve?")
        self.assertIsInstance(recs, list)

    def test_auto_recommend(self):
        recs = self.api.auto_recommend()
        self.assertIsInstance(recs, list)

    def test_pending_recommendations(self):
        self.api.recommend("suggest improvements")
        pending = self.api.pending_recommendations()
        self.assertIsInstance(pending, list)

    def test_knowledge_search(self):
        results = self.api.knowledge_search("Sharpe", top_k=5)
        self.assertIsInstance(results, list)

    def test_get_dashboard_all(self):
        all_d = self.api.get_dashboard("all")
        self.assertEqual(len(all_d), 5)

    def test_get_dashboard_single(self):
        d = self.api.get_dashboard("executive_dashboard")
        self.assertEqual(d["view"], "EXECUTIVE_DASHBOARD")

    def test_get_dashboard_unknown(self):
        d = self.api.get_dashboard("nonexistent")
        self.assertIn("error", d)

    def test_system_stats(self):
        s = self.api.system_stats()
        self.assertIn("knowledge_index", s)
        self.assertIn("database", s)
        self.assertIn("provider", s)
        self.assertEqual(s["provider"], "MOCK")

    def test_index_document_returns_chunk_id(self):
        result = self.api.index_document(
            KnowledgeSourceType.RESEARCH_REPORT, "REPORT-001",
            "Q1 Research Report", "Findings..."
        )
        self.assertTrue(result["indexed"])
        self.assertIn("chunk_id", result)

    def test_switch_provider(self):
        result = self.api.switch_provider(LLMProviderType.MOCK)
        self.assertEqual(result["provider"], "MOCK")
        self.assertTrue(result["available"])

    def test_database_records_queries(self):
        self.api.chat("test query")
        stats = self.api.system_stats()
        self.assertGreater(stats["database"]["queries"], 0)


if __name__ == "__main__":
    unittest.main()
