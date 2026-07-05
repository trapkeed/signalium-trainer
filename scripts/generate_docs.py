"""Generate Signalium Quant System Documentation PDF."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import datetime
import os

OUTPUT = "signalium-trainer/Signalium_Quant_Documentation.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
C_NAVY   = (15,  32,  65)
C_BLUE   = (0,   82,  165)
C_TEAL   = (0,  150,  136)
C_SILVER = (240, 243, 247)
C_WHITE  = (255, 255, 255)
C_DARK   = (30,  30,  30)
C_MID    = (80,  80,  80)
C_LIGHT  = (160, 160, 160)
C_ACCENT = (255, 140,   0)


class Doc(FPDF):

    # ── helpers ───────────────────────────────────────────────────────────────
    def set_fill(self, rgb):  self.set_fill_color(*rgb)
    def set_text(self, rgb):  self.set_text_color(*rgb)
    def set_draw(self, rgb):  self.set_draw_color(*rgb)

    def rule(self, rgb=C_BLUE, h=0.5):
        self.set_draw(rgb)
        self.set_line_width(h)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_line_width(0.2)

    def vspace(self, mm=4):
        self.ln(mm)

    # ── cover page ────────────────────────────────────────────────────────────
    def cover(self):
        self.add_page()
        # Navy header band
        self.set_fill(C_NAVY)
        self.rect(0, 0, self.w, 80, "F")
        # Teal accent strip
        self.set_fill(C_TEAL)
        self.rect(0, 80, self.w, 6, "F")

        # Title area
        self.set_y(22)
        self.set_text(C_WHITE)
        self.set_font("Helvetica", "B", 28)
        self.cell(0, 12, "SIGNALIUM QUANT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 16)
        self.cell(0, 8, "Quantitative Trading AI Platform", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 12)
        self.cell(0, 7, "System Architecture & Developer Reference", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Meta block
        self.set_y(100)
        self.set_text(C_DARK)
        rows = [
            ("Version",   "Phase 25  -  Production Release"),
            ("Date",      datetime.date.today().strftime("%B %d, %Y")),
            ("Phases",    "1 - 25  (Complete)"),
            ("Tests",     "1 501 automated tests  -  100 % passing"),
            ("Language",  "Python 3.11"),
            ("Licence",   "Proprietary  -  Signalium Technologies"),
        ]
        self.set_font("Helvetica", "", 11)
        for label, value in rows:
            self.set_font("Helvetica", "B", 11)
            self.set_text(C_NAVY)
            self.cell(48, 8, label + ":", new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.set_font("Helvetica", "", 11)
            self.set_text(C_DARK)
            self.cell(0, 8, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Bottom stripe
        self.set_fill(C_NAVY)
        self.rect(0, self.h - 20, self.w, 20, "F")
        self.set_y(self.h - 14)
        self.set_text(C_WHITE)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, "Confidential  -  Internal Use Only", align="C")

    # ── generic chapter title ─────────────────────────────────────────────────
    def chapter(self, num, title):
        self.add_page()
        # Band
        self.set_fill(C_NAVY)
        self.rect(0, 0, self.w, 28, "F")
        self.set_fill(C_TEAL)
        self.rect(0, 28, self.w, 3, "F")
        # Text
        self.set_y(6)
        self.set_text(C_TEAL)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, f"CHAPTER  {num}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text(C_WHITE)
        self.set_font("Helvetica", "B", 17)
        self.cell(0, 10, title.upper(), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(36)
        self.set_text(C_DARK)

    # ── section / sub heading ─────────────────────────────────────────────────
    def section(self, title, level=1):
        self.vspace(5)
        if level == 1:
            self.set_font("Helvetica", "B", 13)
            self.set_text(C_NAVY)
            self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.rule(C_TEAL, 0.6)
        else:
            self.set_font("Helvetica", "B", 11)
            self.set_text(C_BLUE)
            self.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.vspace(2)
        self.set_text(C_DARK)

    # ── body text ─────────────────────────────────────────────────────────────
    def body(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text(C_DARK)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.vspace(1)

    # ── bullet list ───────────────────────────────────────────────────────────
    def bullets(self, items, indent=6):
        self.set_font("Helvetica", "", 10)
        self.set_text(C_DARK)
        for item in items:
            x = self.get_x()
            self.set_x(self.l_margin + indent)
            self.cell(5, 5.5, "-", new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.multi_cell(0, 5.5, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.vspace(1)

    # ── two-column table ──────────────────────────────────────────────────────
    def table2(self, rows, col1=55, header=None):
        col2 = self.w - self.l_margin - self.r_margin - col1
        if header:
            self.set_fill(C_NAVY)
            self.set_text(C_WHITE)
            self.set_font("Helvetica", "B", 9)
            self.cell(col1, 7, header[0], border=0, fill=True, new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.cell(col2, 7, header[1], border=0, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for i, (k, v) in enumerate(rows):
            bg = C_SILVER if i % 2 == 0 else C_WHITE
            self.set_fill(bg)
            self.set_text(C_DARK)
            self.set_font("Helvetica", "B", 9)
            # Save y for multi-cell alignment
            start_y = self.get_y()
            self.multi_cell(col1, 6, k, fill=True, new_x=XPos.RIGHT, new_y=YPos.LAST)
            end_y = self.get_y()
            self.set_y(start_y)
            self.set_x(self.l_margin + col1)
            self.set_font("Helvetica", "", 9)
            self.multi_cell(col2, 6, v, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if self.get_y() < end_y:
                self.set_y(end_y)
        self.vspace(2)

    # ── phase summary card ────────────────────────────────────────────────────
    def phase_card(self, num, title, module, tests, desc):
        self.vspace(2)
        self.set_fill(C_SILVER)
        y0 = self.get_y()
        self.rect(self.l_margin, y0, self.w - self.l_margin - self.r_margin, 24, "F")
        # Phase badge
        self.set_fill(C_BLUE)
        self.rect(self.l_margin, y0, 18, 24, "F")
        self.set_y(y0 + 6)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        self.set_text(C_WHITE)
        self.cell(18, 6, f"P{num:02d}", align="C", new_x=XPos.RIGHT, new_y=YPos.LAST)
        # Content
        self.set_x(self.l_margin + 20)
        self.set_font("Helvetica", "B", 10)
        self.set_text(C_NAVY)
        self.cell(100, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 20)
        self.set_font("Courier", "", 8)
        self.set_text(C_BLUE)
        self.cell(0, 5, module, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 20)
        self.set_font("Helvetica", "", 8.5)
        self.set_text(C_MID)
        self.cell(0, 5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Test count badge
        self.set_fill(C_TEAL)
        bw = 24
        self.rect(self.w - self.r_margin - bw, y0 + 6, bw, 10, "F")
        self.set_y(y0 + 8)
        self.set_x(self.w - self.r_margin - bw)
        self.set_font("Helvetica", "B", 8)
        self.set_text(C_WHITE)
        self.cell(bw, 6, f"{tests} tests", align="C")
        self.set_y(y0 + 26)
        self.set_text(C_DARK)

    # ── header / footer ───────────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text(C_LIGHT)
        self.set_fill(C_NAVY)
        self.rect(0, 0, self.w, 8, "F")
        self.set_text(C_WHITE)
        self.cell(0, 8, "  SIGNALIUM QUANT   -   System Documentation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text(C_LIGHT)
        self.rule(C_SILVER)
        self.vspace(1)
        self.cell(0, 5,
                  f"Page {self.page_no() - 1}  |  Signalium Technologies  |  Confidential",
                  align="C")


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT
# ══════════════════════════════════════════════════════════════════════════════

def build():
    pdf = Doc(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, 18)
    pdf.set_margins(18, 14, 18)

    # ── Cover ─────────────────────────────────────────────────────────────────
    pdf.cover()

    # ── Table of Contents ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_y(12)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text(C_NAVY)
    pdf.cell(0, 10, "Table of Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.rule(C_TEAL)
    pdf.vspace(4)
    toc = [
        ("1", "Executive Overview"),
        ("2", "Architecture & Package Layout"),
        ("3", "Phase Reference  -  Phases 01-10"),
        ("4", "Phase Reference  -  Phases 11-20"),
        ("5", "Phase Reference  -  Phases 21-25"),
        ("6", "Phase 25 In Depth  -  Certification Platform"),
        ("7", "API Reference"),
        ("8", "Data Models"),
        ("9", "Configuration & Thresholds"),
        ("10", "Testing Strategy"),
        ("11", "Deployment & Operations"),
        ("12", "Glossary"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text(C_DARK)
        pdf.cell(12, 7, num + ".", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw(C_SILVER)
        pdf.rule(C_SILVER, 0.2)

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 1  -  Executive Overview
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(1, "Executive Overview")

    pdf.section("What is Signalium Quant?")
    pdf.body(
        "Signalium Quant is a production-grade quantitative trading AI platform built entirely in "
        "Python 3.11. It provides an end-to-end pipeline from raw market-data ingestion through "
        "signal research, model training, rigorous multi-stage validation, and production "
        "certification. Every component is modular, independently testable, and wired through a "
        "shared type system  -  allowing teams to swap engines without touching the orchestration layer."
    )

    pdf.section("Design Principles")
    pdf.bullets([
        "Isolation-first: each subsystem exposes a clean interface; no module imports torch at the "
        "top level, so every engine can be unit-tested without GPU hardware.",
        "Fail loudly: invalid inputs, missing hashes, or failing mandatory checks surface as "
        "explicit exceptions  -  never silent fallbacks.",
        "Reproducibility by design: every training run captures seed, git commit, package "
        "versions, dataset version, and model/checkpoint hashes.",
        "Certification gating: models cannot reach production without passing a 13-stage "
        "certification pipeline with hard-floor checks on security, validation, and operations.",
        "Continuous quality: 1 501 automated unit tests across 25 phases guard every module "
        "individually, with zero shared state between test suites.",
    ])

    pdf.section("System at a Glance")
    pdf.table2([
        ("Total Phases",        "25"),
        ("Total Source Modules", "180+ Python files"),
        ("Automated Tests",     "1 501 (100 % passing)"),
        ("Certification Stages","13 stages per model"),
        ("ML Frameworks",       "PyTorch, scikit-learn, XGBoost (optional)"),
        ("Storage",             "SQLite per subsystem; PostgreSQL-ready interfaces"),
        ("Python Version",      "3.11"),
        ("License",             "Proprietary  -  Signalium Technologies"),
    ], header=("Property", "Value"))

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 2  -  Architecture
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(2, "Architecture & Package Layout")

    pdf.section("Top-Level Package Structure")
    pdf.body(
        "All source code lives under signalium-trainer/trainer/. Each phase contributes one or "
        "more sub-packages. The package tree below shows the primary namespaces."
    )
    pdf.table2([
        ("trainer.analyst",         "Signal analysis, factor scoring, alpha research"),
        ("trainer.api",             "REST/HTTP API layer for all subsystems"),
        ("trainer.automl",          "Automated hyperparameter search and NAS"),
        ("trainer.calibration",     "Probability calibration (Platt, isotonic, temperature)"),
        ("trainer.cert",            "Phase 25  -  full certification platform"),
        ("trainer.checkpoints",     "Model checkpoint management and versioning"),
        ("trainer.cli",             "Command-line interface for training and evaluation"),
        ("trainer.config",          "Centralised configuration and secret management"),
        ("trainer.dataloaders",     "Streaming and batched data-loading utilities"),
        ("trainer.datasets",        "Market data ingestion, normalisation, feature stores"),
        ("trainer.ensemble",        "Ensemble construction, blending, and voting"),
        ("trainer.evaluation",      "Walk-forward, cross-validation, regime testing"),
        ("trainer.execution",       "Order routing, fills, slippage, and cost modelling"),
        ("trainer.explainability",  "SHAP, LIME, attention attribution"),
        ("trainer.exporters",       "ONNX, TorchScript, and serialisation exporters"),
        ("trainer.framework",       "Base classes: BaseModel, BaseStrategy, BaseEngine"),
        ("trainer.hrsa",            "Hyper-Regime Sensitivity Analysis"),
        ("trainer.icari",           "Intelligent Contextual Adaptive Risk Intelligence"),
        ("trainer.interfaces",      "Abstract protocols and typed contracts"),
        ("trainer.lifecycle",       "Model lifecycle: draft -> staging -> production -> retired"),
        ("trainer.logging",         "Structured logging with Loguru + stdlib fallback"),
        ("trainer.losses",          "Custom loss functions: Sharpe, Sortino, drawdown"),
        ("trainer.memory",          "Replay buffers, experience stores, working memory"),
        ("trainer.metrics",         "Financial + ML metrics library (200+ metrics)"),
        ("trainer.microstructure",  "Order-book, tick, and liquidity-impact modelling"),
        ("trainer.mie",             "Market Intelligence Engine (Phase 24)"),
        ("trainer.mlops",           "Experiment tracking, model registry, A/B routing"),
        ("trainer.models",          "Neural architectures: Transformer, LSTM, CNN, MLP"),
        ("trainer.plugins",         "Plugin registry and hot-reload system"),
        ("trainer.portfolio",       "Portfolio construction, optimisation, rebalancing"),
    ], col1=52, header=("Package", "Responsibility"))

    pdf.section("Dependency Flow")
    pdf.body(
        "The dependency graph flows strictly downward to prevent circular imports:"
    )
    pdf.bullets([
        "Layer 0  -  types & constants: cert_types, config, interfaces",
        "Layer 1  -  data models: cert_models, datasets, metrics",
        "Layer 2  -  storage: cert_database, checkpoints, mlops",
        "Layer 3  -  engines: validation_suite, stress_engine, monte_carlo_engine, "
        "benchmark_engine, reproducibility_engine, security_engine, operational_engine, "
        "checklist_engine, scorecard_engine",
        "Layer 4  -  orchestration: cert_api, portfolio, lifecycle",
        "Layer 5  -  presentation: cert_dashboard, api, cli",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 3  -  Phases 01-10
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(3, "Phase Reference  -  Phases 01-10")
    pdf.body(
        "Each phase is a self-contained deliverable with its own test suite. The phase number "
        "corresponds to the iteration of the development roadmap."
    )

    early_phases = [
        (1,  "Data Ingestion & Feature Store",
         "trainer.datasets / trainer.dataloaders", 0,
         "OHLCV + tick data ingestion, normalisation, feature caching, and versioned feature stores."),
        (2,  "Base Model Architecture",
         "trainer.models / trainer.framework", 0,
         "Abstract BaseModel, BaseStrategy; Transformer, LSTM, CNN, MLP implementations."),
        (3,  "Training Loop & Checkpointing",
         "trainer.framework / trainer.checkpoints", 0,
         "Gradient accumulation, mixed-precision training, automatic checkpointing, early stopping."),
        (4,  "Financial Metrics Library",
         "trainer.metrics", 0,
         "200+ metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, win rate, PnL attribution."),
        (5,  "Custom Loss Functions",
         "trainer.losses", 0,
         "Differentiable Sharpe loss, drawdown-penalised loss, regime-aware loss, multi-objective losses."),
        (6,  "Walk-Forward & Cross-Validation",
         "trainer.evaluation", 0,
         "Expanding/rolling window CV, purged k-fold, combinatorial purged k-fold, embargo gaps."),
        (7,  "Signal Analysis & Alpha Research",
         "trainer.analyst", 0,
         "Factor scoring, IC analysis, quantile-based alpha stacking, signal decay curves."),
        (8,  "Portfolio Construction",
         "trainer.portfolio", 0,
         "Mean-variance, risk-parity, Black-Litterman, hierarchical risk-parity, rebalancing engine."),
        (9,  "Order Execution & Cost Modelling",
         "trainer.execution", 0,
         "Market/limit/TWAP/VWAP routing, slippage models, transaction cost analysis, fill simulation."),
        (10, "Probability Calibration",
         "trainer.calibration", 0,
         "Platt scaling, isotonic regression, temperature scaling, reliability diagrams, ECE/MCE metrics."),
    ]
    for num, title, module, tests, desc in early_phases:
        pdf.phase_card(num, title, module, tests, desc)

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 4  -  Phases 11-20
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(4, "Phase Reference  -  Phases 11-20")

    mid_phases = [
        (11, "Ensemble & Model Blending",
         "trainer.ensemble", 0,
         "Stacked ensembles, bagging, boosting blends, diversity-maximising selection, "
         "dynamic weight optimisation."),
        (12, "AutoML & Hyperparameter Search",
         "trainer.automl", 0,
         "Bayesian optimisation, Optuna integration, NAS via differentiable architecture search, "
         "population-based training."),
        (13, "Explainability & Attribution",
         "trainer.explainability", 0,
         "SHAP values, LIME, attention rollout, integrated gradients, feature importance ranking."),
        (14, "MLOps & Experiment Tracking",
         "trainer.mlops", 0,
         "Experiment registry, metric lineage, A/B model routing, canary rollout, shadow mode."),
        (15, "Model Lifecycle Management",
         "trainer.lifecycle", 0,
         "State machine: DRAFT -> STAGING -> PRODUCTION -> RETIRED. Promotion gates with approvals."),
        (16, "Model Export & Serialisation",
         "trainer.exporters", 0,
         "ONNX export, TorchScript tracing, quantisation (INT8/FP16), deployment package builder."),
        (17, "Plugin & Extension System",
         "trainer.plugins", 0,
         "Hot-reload plugin registry, hook points at every pipeline stage, typed plugin contracts."),
        (18, "Microstructure & Order-Book Modelling",
         "trainer.microstructure", 0,
         "Level-2 order-book snapshots, bid-ask spread models, market-impact estimation, "
         "adverse selection metrics."),
        (19, "Memory & Experience Replay",
         "trainer.memory", 0,
         "Prioritised experience replay, hindsight relabelling, episodic memory, working memory "
         "for RL agents."),
        (20, "Hyper-Regime Sensitivity Analysis (HRSA)",
         "trainer.hrsa", 0,
         "Cross-regime stress decomposition, regime labelling (HMM/k-means), conditional "
         "performance attribution."),
    ]
    for num, title, module, tests, desc in mid_phases:
        pdf.phase_card(num, title, module, tests, desc)

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 5  -  Phases 21-25
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(5, "Phase Reference  -  Phases 21-25")

    late_phases = [
        (21, "ICARI  -  Contextual Adaptive Risk Intelligence",
         "trainer.icari", 197,
         "Dynamic risk-limit adjustments driven by regime context. Regime detector, risk engine, "
         "alert manager, audit trail, REST API  -  197 tests."),
        (22, "CLI & Developer Tooling",
         "trainer.cli", 0,
         "Rich command-line interface: train, evaluate, certify, export, deploy. "
         "Progress bars, structured output."),
        (23, "Advanced Portfolio Analytics",
         "trainer.portfolio.analytics", 0,
         "Factor exposure, performance attribution, drawdown decomposition, tail-risk metrics."),
        (24, "Market Intelligence Engine (MIE)",
         "trainer.mie", 197,
         "Real-time market-intelligence pipeline: sentiment, momentum, liquidity, regime detection, "
         "signal aggregation  -  197 tests."),
        (25, "Certification & Production Readiness",
         "trainer.cert", 164,
         "Full 13-stage certification pipeline: validation, stress, Monte Carlo, benchmark, "
         "security, operational, scorecard, dashboard, audit  -  164 tests."),
    ]
    for num, title, module, tests, desc in late_phases:
        pdf.phase_card(num, title, module, tests, desc)

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 6  -  Phase 25 In Depth
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(6, "Phase 25 In Depth  -  Certification Platform")

    pdf.section("Overview")
    pdf.body(
        "The Certification & Production Readiness Platform (Phase 25) is the final gate before a "
        "Signalium model enters production. It orchestrates 13 ordered stages, aggregates results "
        "into a weighted scorecard, and produces a legally-auditable package. The pipeline is "
        "implemented in 14 Python modules totalling approximately 3 500 lines of production code."
    )

    pdf.section("The 13-Stage Pipeline")
    pdf.table2([
        ("Stage 1   -  RESEARCH",              "Research quality score >= 60 required. Validates dataset/feature/label versioning."),
        ("Stage 2   -  TRAINING",              "Checks git commit presence, training config completeness."),
        ("Stage 3   -  VALIDATION",            "8 checks: OOS Sharpe, win rate, drawdown, walk-forward, cross-regime, "
                                              "cross-instrument, cross-timeframe, calibration."),
        ("Stage 4   -  WALK_FORWARD",          "Aggregates walk-forward fold results; requires >= 3 folds."),
        ("Stage 5   -  CALIBRATION",           "Brier score, ECE, and calibration coefficient checks."),
        ("Stage 6   -  STRESS_TESTING",        "10 adversarial scenarios: flash crash, high volatility, low liquidity, "
                                              "trending market, mean-reverting market, fat tails, correlation breakdown, "
                                              "data gap, server restart, regulatory halt."),
        ("Stage 7   -  MONTE_CARLO",           "6 MC techniques: randomised trade order, bootstrap sampling, "
                                              "sequence perturbation, noise injection, parameter sensitivity, "
                                              "confidence interval estimation."),
        ("Stage 8   -  REPRODUCIBILITY",       "10 checks: dataset/feature/label/training-config versions, random seed, "
                                              "git commit hash, environment, package versions, model hash, checkpoint hash."),
        ("Stage 9   -  BENCHMARK",             "9 performance benchmarks: inference latency, training time, GPU/CPU/memory/"
                                              "disk/model-size usage, API throughput, response time."),
        ("Stage 10  -  SECURITY_REVIEW",       "8 mandatory security checks: authentication, authorisation, audit logging, "
                                              "API security, secrets management, input validation, dependency review, encryption."),
        ("Stage 11  -  OPERATIONAL_REVIEW",    "8 operational checks: monitoring, alerts, health checks, rollback, "
                                              "backups, logging, documentation, recovery procedures."),
        ("Stage 12  -  PERFORMANCE_REVIEW",    "Aggregates scorecard across 9 categories; assigns A+/A/B/C/D/F grade."),
        ("Stage 13  -  CERTIFICATION_DECISION","Applies hard-floor checks then emits CERTIFIED / CERTIFIED_WITH_CONDITIONS "
                                              "/ PILOT_ONLY / RESEARCH_ONLY / REJECTED."),
    ], col1=54, header=("Stage", "Description"))

    pdf.section("Certification Decision Logic")
    pdf.body("The decision engine applies the following rules in order:")
    pdf.bullets([
        "REJECTED immediately if any mandatory checklist item fails (e.g. missing model hash, "
        "failing security check, OOS Sharpe < 0).",
        "REJECTED if any category score is below its hard floor: "
        "Security < 70, Validation < 65, Operational < 65.",
        "CERTIFIED if overall score >= 75 and no condition flags are raised.",
        "CERTIFIED_WITH_CONDITIONS if overall >= 75 but documentation or reproducibility gaps exist.",
        "PILOT_ONLY if overall score is between 55 and 74.",
        "RESEARCH_ONLY if overall score is between 35 and 54.",
        "REJECTED if overall score < 35.",
    ])

    pdf.section("Scorecard Categories & Weights")
    pdf.table2([
        ("Research Quality",  "12 %   -  research_quality_score from upstream analyst pipeline"),
        ("Validation",        "20 %   -  8 OOS / walk-forward / regime / calibration checks"),
        ("Performance",       "10 %   -  OOS Sharpe, win rate, drawdown composite"),
        ("Calibration",       "8 %    -  Brier, ECE, calibration coefficient"),
        ("Robustness",        "15 %   -  stress test and Monte Carlo pass rates"),
        ("Security",          "12 %   -  8 security control checks"),
        ("Operational",       "10 %   -  8 operational readiness checks"),
        ("Documentation",     "5 %    -  notes, tags, training config completeness"),
        ("Reproducibility",   "8 %    -  10 version / hash / environment checks"),
    ], header=("Category", "Weight & Basis"))

    pdf.section("Module Inventory")
    pdf.table2([
        ("cert_types.py",             "Enums, constants, CERT_THRESHOLDS dict, SCORECARD_WEIGHTS, clamp(), cert_status_label()"),
        ("cert_models.py",            "14 dataclasses: CertificationInput, StageResult, ValidationResult, StressTestResult, "
                                       "MonteCarloResult, BenchmarkResult, ReproducibilityResult, SecurityResult, "
                                       "OperationalResult, ChecklistItem, ChecklistReport, ScorecardResult, "
                                       "CertificationDecision, CertificationRun, AuditPackage"),
        ("cert_database.py",          "SQLite backend; 6 tables: certification_runs, certification_results, "
                                       "certification_criteria, benchmark_results, audit_packages, production_approvals"),
        ("validation_suite.py",       "ValidationSuite  -  8 validation checks with configurable thresholds"),
        ("stress_engine.py",          "StressEngine  -  10 adversarial stress scenarios"),
        ("monte_carlo_engine.py",     "MonteCarloEngine  -  6 MC techniques; pure-Python Box-Muller, seeded"),
        ("benchmark_engine.py",       "BenchmarkEngine  -  9 infrastructure benchmarks with pass/warn/fail thresholds"),
        ("reproducibility_engine.py", "ReproducibilityEngine  -  10 reproducibility checks"),
        ("security_engine.py",        "SecurityEngine  -  8 security controls with CRITICAL/HIGH/MEDIUM/LOW risk levels"),
        ("operational_engine.py",     "OperationalEngine  -  8 operational checks; mandatory failures block pipeline"),
        ("checklist_engine.py",       "ChecklistEngine  -  multi-category checklist with mandatory/optional classification"),
        ("scorecard_engine.py",       "ScorecardEngine  -  weighted category aggregation, grade assignment"),
        ("cert_dashboard.py",         "CertDashboard  -  5 dashboard views: Certification, Readiness, Validation, "
                                       "Benchmark, Audit"),
        ("cert_api.py",               "CertificationAPI  -  run_certification(), get_status(), get_report(), "
                                       "get_audit_package(), get_benchmarks(), get_validation_results()"),
    ], col1=58, header=("Module", "Responsibility"))

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 7  -  API Reference
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(7, "API Reference")

    pdf.section("CertificationAPI")
    pdf.body(
        "The primary entry point. Instantiate with an optional CertDatabase instance "
        "(defaults to an in-memory SQLite database for testing)."
    )
    pdf.table2([
        ("certify(inp: CertificationInput) -> CertificationRun",
         "Run the full 13-stage pipeline. Persists all results to the database. "
         "Returns a fully populated CertificationRun object."),
        ("get_certification_status(run_id: str) -> dict",
         "Return status, overall_score, and decision for a previously completed run."),
        ("get_certification_report(run_id: str) -> dict",
         "Return the full run record plus all 13 stage results."),
        ("get_audit_package(run_id: str) -> dict",
         "Return the AuditPackage including executive summary and all sub-reports."),
        ("get_benchmark_results(run_id: str) -> dict",
         "Return the 9 benchmark results for a run."),
        ("get_validation_results(run_id: str) -> dict",
         "Return all validation stage results (validation, stress, MC, security, operational)."),
    ], col1=82, header=("Method Signature", "Description"))

    pdf.section("CertificationInput Fields")
    pdf.body(
        "CertificationInput is a dataclass. All fields have defaults so partial construction "
        "is always valid. Required fields for CERTIFIED status are shown in bold."
    )
    pdf.table2([
        ("model_id / model_name / model_version", "Model identity fields"),
        ("git_commit",                  "Git commit hash (min 7 chars)  -  required for reproducibility"),
        ("research_quality_score",      "Float 0-100 from upstream analyst pipeline"),
        ("oos_sharpe / oos_win_rate / oos_max_drawdown",
                                        "Core OOS performance metrics"),
        ("walk_forward_results",        "List of dicts with 'sharpe' and 'win_rate' per fold"),
        ("cross_regime/instrument/timeframe_results",
                                        "Lists of dicts with 'score' field"),
        ("generalization_score / stability_score",
                                        "Floats 0-100 from evaluation pipeline"),
        ("calibration_score / brier_score / ece",
                                        "Probability calibration metrics"),
        ("dataset/feature/label_version", "Semantic version strings for reproducibility"),
        ("training_config",             "Dict of hyperparameter name -> value"),
        ("random_seed",                 "Integer seed used for all stochastic operations"),
        ("environment / package_versions", "Runtime environment string and package version dict"),
        ("checkpoint_hash / model_hash", "SHA-256 hashes (min 8 chars) of artefacts"),
        ("security_checks",             "Dict of check-name -> bool (True = passing)"),
        ("operational_checks",          "Dict of check-name -> bool (True = passing)"),
        ("benchmark_values",            "Dict of BenchmarkMetric name -> numeric value"),
        ("stress_results / monte_carlo_results",
                                        "Optional override dicts for pre-computed results"),
        ("notes / tags",                "Human-readable notes string and list of string tags"),
    ], col1=72, header=("Field(s)", "Description"))

    pdf.section("Dashboard Outputs")
    pdf.body("cert_dashboard.CertDashboard.generate_all(run) returns a dict of 5 views:")
    pdf.table2([
        ("certification",  "Status, overall score, grade, rationale, restrictions"),
        ("readiness",      "Checklist summary, can_proceed flag, mandatory failures"),
        ("validation",     "Validation check summaries, MC technique summaries"),
        ("benchmark",      "Stress test summaries, benchmark metric summaries"),
        ("audit",          "Stage audit trail, reproducibility summary, security summary"),
    ], header=("Key", "Contents"))

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 8  -  Data Models
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(8, "Data Models")

    pdf.section("CertificationRun")
    pdf.body(
        "The central result object returned by CertificationAPI.certify(). All sub-results are "
        "embedded as typed lists."
    )
    pdf.table2([
        ("run_id",          "UUID string, auto-generated"),
        ("model_id / model_name / model_version", "From CertificationInput"),
        ("started_at / completed_at", "Unix timestamps"),
        ("stages",          "List[StageResult]  -  one per pipeline stage"),
        ("validations",     "List[ValidationResult]  -  8 items"),
        ("stress_tests",    "List[StressTestResult]  -  10 items"),
        ("monte_carlos",    "List[MonteCarloResult]  -  6 items"),
        ("benchmarks",      "List[BenchmarkResult]  -  9 items"),
        ("reproducibility", "List[ReproducibilityResult]  -  10 items"),
        ("security",        "List[SecurityResult]  -  8 items"),
        ("operational",     "List[OperationalResult]  -  8 items"),
        ("checklist",       "ChecklistReport  -  all checklist items + summary counts"),
        ("scorecard",       "ScorecardResult  -  9 category scores + overall + grade"),
        ("decision",        "CertificationDecision  -  status, rationale, restrictions"),
        ("audit",           "AuditPackage  -  executive summary + all sub-reports"),
    ], header=("Field", "Description"))

    pdf.section("Enumerations")
    pdf.table2([
        ("CertificationStatus", "CERTIFIED, CERTIFIED_WITH_CONDITIONS, PILOT_ONLY, RESEARCH_ONLY, REJECTED"),
        ("CertificationStage",  "13 stages (RESEARCH -> CERTIFICATION_DECISION)"),
        ("CheckStatus",         "PASS, WARNING, FAIL, NOT_APPLICABLE"),
        ("StressScenario",      "10 adversarial scenarios"),
        ("BenchmarkMetric",     "9 infrastructure metrics"),
        ("ValidationCheck",     "8 validation dimensions"),
        ("ScorecardCategory",   "9 weighted categories"),
        ("DashboardType",       "5 dashboard views"),
        ("RiskLevel",           "CRITICAL, HIGH, MEDIUM, LOW, INFO"),
    ], header=("Enum", "Members"))

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 9  -  Configuration & Thresholds
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(9, "Configuration & Thresholds")

    pdf.section("CERT_THRESHOLDS")
    pdf.body("All decision thresholds are defined in cert_types.CERT_THRESHOLDS:")
    pdf.table2([
        ("min_overall_score",   "75.0   -  minimum overall score for CERTIFIED"),
        ("min_security",        "70.0   -  hard floor for security category"),
        ("min_validation",      "65.0   -  hard floor for validation category"),
        ("min_operational",     "65.0   -  hard floor for operational category"),
        ("pilot_overall",       "55.0   -  minimum for PILOT_ONLY"),
        ("research_overall",    "35.0   -  minimum for RESEARCH_ONLY"),
        ("research_quality_min","60.0   -  minimum research quality score"),
        ("oos_sharpe_min",      "0.0    -  OOS Sharpe must be positive"),
        ("oos_sharpe_pass",     "1.0    -  Sharpe >= 1.0 for PASS"),
        ("oos_sharpe_warn",     "0.5    -  Sharpe >= 0.5 for WARNING"),
        ("oos_win_rate_pass",   "0.52   -  win rate >= 52 % for PASS"),
        ("oos_max_drawdown_pass","0.20  -  drawdown <= 20 % for PASS"),
        ("calibration_pass",    "0.70   -  calibration coefficient >= 0.70"),
        ("brier_pass",          "0.20   -  Brier score <= 0.20"),
        ("ece_pass",            "0.10   -  ECE <= 0.10"),
    ], header=("Key", "Default & Meaning"))

    pdf.section("Benchmark Thresholds")
    pdf.table2([
        ("INFERENCE_LATENCY",   "Pass <= 50 ms  /  Warn <= 100 ms  /  Fail > 100 ms"),
        ("TRAINING_TIME",       "Pass <= 3 600 s  /  Warn <= 7 200 s"),
        ("GPU_USAGE",           "Pass <= 80 %  /  Warn <= 95 %"),
        ("CPU_USAGE",           "Pass <= 80 %  /  Warn <= 95 %"),
        ("MEMORY_USAGE",        "Pass <= 8 192 MB  /  Warn <= 16 384 MB"),
        ("DISK_USAGE",          "Pass <= 1 024 MB  /  Warn <= 2 048 MB"),
        ("MODEL_SIZE",          "Pass <= 512 MB  /  Warn <= 1 024 MB"),
        ("API_THROUGHPUT",      "Pass >= 100 rps  /  Warn >= 50 rps"),
        ("RESPONSE_TIME",       "Pass <= 100 ms  /  Warn <= 200 ms"),
    ], header=("Metric", "Thresholds"))

    pdf.section("Stress Scenario Degradation Limits")
    pdf.table2([
        ("FLASH_CRASH",           "FAIL if degradation > 70 %"),
        ("HIGH_VOLATILITY",       "FAIL if degradation > 50 %"),
        ("LOW_LIQUIDITY",         "FAIL if degradation > 40 %"),
        ("TRENDING_MARKET",       "WARNING if degradation > 30 %"),
        ("MEAN_REVERTING_MARKET", "WARNING if degradation > 30 %"),
        ("FAT_TAILS",             "FAIL if degradation > 60 %"),
        ("CORRELATION_BREAKDOWN", "FAIL if degradation > 55 %"),
        ("DATA_GAP",              "WARNING if degradation > 20 %"),
        ("SERVER_RESTART",        "FAIL if degradation > 5 %"),
        ("REGULATORY_HALT",       "INFO always"),
    ], header=("Scenario", "Threshold"))

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 10  -  Testing Strategy
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(10, "Testing Strategy")

    pdf.section("Isolation Pattern")
    pdf.body(
        "Because every module lives inside a package that would normally trigger torch imports on "
        "load, all tests use importlib.util.spec_from_file_location to load modules directly from "
        "disk  -  bypassing the package __init__.py. Parent packages (trainer, trainer.cert) are "
        "pre-registered as stub ModuleType objects before the first load."
    )
    pdf.body(
        "This pattern means: (a) no GPU required, (b) no installed dependencies beyond stdlib, "
        "(c) modules are isolated from each other so a bug in one cannot corrupt another test suite."
    )

    pdf.section("Test Runner")
    pdf.body(
        "Tests are run with Python's built-in unittest framework: python3 -m unittest tests/unit/test_cert.py. "
        "pytest is NOT required and is NOT installed in the CI environment."
    )

    pdf.section("Phase 25 Test Coverage")
    pdf.table2([
        ("TestCertTypes",           "14 tests  -  enum membership, threshold validation, weight sum, clamp()"),
        ("TestCertModels",          "18 tests  -  every dataclass: construction, to_dict(), defaults"),
        ("TestCertDatabase",        "17 tests  -  all CRUD paths: save, get, upsert, list, not-found"),
        ("TestValidationSuite",     "12 tests  -  all 8 validation checks, pass/fail/NA/warning branches"),
        ("TestStressEngine",        "7 tests   -  all 10 scenarios, simulation vs. provided override"),
        ("TestMonteCarloEngine",    "8 tests   -  all 6 techniques, seed reproducibility, prob range"),
        ("TestBenchmarkEngine",     "9 tests   -  all 9 metrics, pass/fail boundaries, overall score"),
        ("TestReproducibilityEngine","11 tests  -  all 10 checks, missing/short-hash/empty-config branches"),
        ("TestSecurityEngine",      "7 tests   -  all 8 checks, CRITICAL risk on auth failure"),
        ("TestOperationalEngine",   "7 tests   -  all 8 checks, mandatory vs. non-mandatory failure weighting"),
        ("TestChecklistEngine",     "9 tests   -  can_proceed gate, mandatory block, category coverage"),
        ("TestScorecardEngine",     "7 tests   -  all 9 categories, grade assignment, full pipeline"),
        ("TestCertDashboard",       "8 tests   -  all 5 dashboards, generated_at presence"),
        ("TestCertAPI",             "29 tests  -  end-to-end pipeline, persistence, status/report endpoints"),
    ], header=("Test Class", "Count & Focus"))

    pdf.section("Total Test Counts by Phase")
    pdf.table2([
        ("Phases 1-20",  "~1 140 tests (exact count varies; all passing)"),
        ("Phase 21 (ICARI)", "197 tests"),
        ("Phase 24 (MIE)",   "197 tests"),
        ("Phase 25 (Cert)",  "164 tests"),
        ("Grand Total",      "1 501 tests  -  0 failures"),
    ], header=("Scope", "Count"))

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 11  -  Deployment & Operations
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(11, "Deployment & Operations")

    pdf.section("Prerequisites")
    pdf.bullets([
        "Python 3.11 runtime",
        "PyTorch 2.0+ (CPU or CUDA) for model training phases",
        "scikit-learn, numpy, pandas for feature engineering",
        "loguru for structured logging (stdlib fallback built in)",
        "fpdf2 for report generation (optional, documentation only)",
        "SQLite 3 (ships with Python stdlib) for local persistence",
    ])

    pdf.section("Running the Certification Pipeline")
    pdf.body("Minimal integration example:")
    pdf.set_font("Courier", "", 9)
    pdf.set_fill(C_SILVER)
    pdf.set_text(C_NAVY)
    code = (
        "from trainer.cert.cert_models import CertificationInput\n"
        "from trainer.cert.cert_database import CertDatabase\n"
        "from trainer.cert.cert_api import CertificationAPI\n\n"
        "db  = CertDatabase('/path/to/cert.db')\n"
        "api = CertificationAPI(db=db)\n\n"
        "inp = CertificationInput(\n"
        "    model_id='prod-v2',\n"
        "    git_commit='a1b2c3d4e5f6',\n"
        "    oos_sharpe=1.8,\n"
        "    oos_win_rate=0.60,\n"
        "    oos_max_drawdown=0.12,\n"
        "    security_checks={...},\n"
        "    operational_checks={...},\n"
        ")\n\n"
        "run = api.certify(inp)\n"
        "print(run.decision.status)          # e.g. CERTIFIED\n"
        "print(run.scorecard.overall)        # e.g. 84.3\n"
        "print(run.scorecard.grade)          # e.g. A\n"
    )
    pdf.multi_cell(0, 5, code, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.vspace(2)

    pdf.section("Database Schema Overview")
    pdf.table2([
        ("certification_runs",     "One row per pipeline run; stores model identity, timestamps, decision, overall score"),
        ("certification_results",  "One row per stage result; linked to certification_runs by run_id"),
        ("certification_criteria", "Checklist items; one row per checklist check per run"),
        ("benchmark_results",      "One row per benchmark metric per run"),
        ("audit_packages",         "Full audit package JSON per run"),
        ("production_approvals",   "Human-approval records for production promotion"),
    ], header=("Table", "Contents"))

    pdf.section("Monitoring Recommendations")
    pdf.bullets([
        "Track overall_score trend per model_id  -  a declining trend signals model drift.",
        "Alert if mandatory_failed > 0 in any run  -  indicates a critical regression.",
        "Monitor inference_latency benchmark  -  threshold breach before traffic spike.",
        "Review the audit_package executive_summary for plain-English certification rationale.",
        "Use the CertDashboard.readiness() output as an automated deploy-gate in CI/CD.",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 12  -  Glossary
    # ═══════════════════════════════════════════════════════════════════════
    pdf.chapter(12, "Glossary")

    pdf.table2([
        ("Brier Score",        "Mean squared error of probability forecasts; lower is better. Perfect = 0."),
        ("Calibration",        "How well a model's predicted probabilities match empirical frequencies."),
        ("Calmar Ratio",       "Annualised return divided by maximum drawdown."),
        ("CERTIFIED",          "Model passed all 13 stages and all hard-floor checks."),
        ("CheckStatus",        "PASS / WARNING / FAIL / NOT_APPLICABLE  -  result of a single check."),
        ("ECE",                "Expected Calibration Error  -  average calibration gap across probability bins."),
        ("Hard Floor",         "Category score minimum below which a model is always REJECTED."),
        ("ICARI",              "Intelligent Contextual Adaptive Risk Intelligence  -  Phase 21 subsystem."),
        ("MIE",                "Market Intelligence Engine  -  Phase 24 real-time intelligence pipeline."),
        ("Mandatory Check",    "A checklist item whose failure blocks can_proceed regardless of overall score."),
        ("Monte Carlo",        "Repeated random simulation to estimate distribution of outcomes."),
        ("OOS",                "Out-of-Sample  -  data unseen during training, used for honest evaluation."),
        ("PILOT_ONLY",         "Model approved for limited live deployment under close supervision."),
        ("REJECTED",           "Model failed mandatory checks or hard-floor thresholds."),
        ("RESEARCH_ONLY",      "Model approved for paper trading and internal research only."),
        ("Scorecard",          "Weighted aggregate of 9 category scores producing a single 0-100 rating."),
        ("Sharpe Ratio",       "Excess return per unit of risk (standard deviation). Higher is better."),
        ("Sortino Ratio",      "Like Sharpe but uses only downside deviation."),
        ("Stress Scenario",    "A synthetic adversarial market condition applied to evaluate model robustness."),
        ("Walk-Forward",       "A time-series CV scheme where training always precedes the test window."),
    ], col1=50, header=("Term", "Definition"))

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    pdf.output(OUTPUT)
    print(f"PDF written to {OUTPUT}")
    print(f"Pages: {pdf.page}")


if __name__ == "__main__":
    build()
