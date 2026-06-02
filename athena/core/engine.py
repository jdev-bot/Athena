"""AthenaEngine — unified autonomous strategy generator, evaluator, and promoter.

This is the single integration point that wires together:
  1. Generator (GA / random)          → DNA + strategy code
  2. Evaluator (freqtrade backtest)   → PerformanceMetrics
  3. Scorer (composite)               → ScoreResult
  4. Gates (Walk-forward + MonteCarlo)→ RobustnessResult
  5. Promoter (lifecycle manager)     → DRAFT → BACKTEST_DONE → PROMOTED

Callable from both CLI (orchestrator.py) and API (services/api.py).
"""
import uuid
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timezone

from athena.common.models import (
    StrategyTemplate, StrategyDNA, StrategyRecord, StrategyStatus,
    PerformanceMetrics, ScoreResult, GenerationConfig, WalkForwardResult, MonteCarloResult,
)
from athena.common.config import config
from athena.generator.dna import DNAEncoder
from athena.generator.templates import TEMPLATE_MAP
from athena.generator.ga_engine import GAEngine, Individual
from athena.core.freqtrade_wrapper import FreqtradeWrapper
from athena.evaluator.scorer import Scorer
from athena.evaluator.robustness import WalkForwardValidator, MonteCarloStressTest
from athena.services.models import init_db, get_session, StrategyModel

logger = logging.getLogger(__name__)


# ── unified engine ───────────────────────────────────────────────
class AthenaEngine:
    """Single integration point for generate → evaluate → score → gate → promote."""

    def __init__(self, gen_config: Optional[GenerationConfig] = None):
        self.cfg = gen_config or GenerationConfig()
        self.encoder = DNAEncoder()
        self.scorer = Scorer()
        self.freqtrade = FreqtradeWrapper()
        init_db()

    # ── public high-level flows ──────────────────────────────────────

    def evolve(self, template: StrategyTemplate, run_gates: bool = True,
               parallel_workers: Optional[int] = None) -> List[StrategyRecord]:
        """Run full GA evolution: generate → backtest → score → gates → promote."""
        # Check file descriptor limit — freqtrade backtests open exchange sockets
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 4096:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
            except (ValueError, OSError):
                logger.warning(f"FD limit soft={soft}, hard={hard}. Parallel workers forced to 1.")
                pw = 1
            else:
                pw = parallel_workers or config.PARALLEL_WORKERS
        else:
            pw = parallel_workers or config.PARALLEL_WORKERS
        if pw > 1 and run_gates:
            logger.warning("Gates + parallel workers can exhaust FDs; consider run_gates=False for quick iteration")

        logger.info("=" * 60)
        logger.info(f"AthenaEngine.evolve | template={template.value} pop={self.cfg.population_size} "
                    f"gen={self.cfg.generations} gates={run_gates} workers={pw}")
        logger.info("=" * 60)

        # Detect regime from cached candles
        from athena.market.regime import detect_regime, get_suitable_templates, Regime
        candles = self.freqtrade.load_cached_candles("BTC/USDT:USDT", "1h")
        regime = detect_regime(candles) if candles is not None else Regime.UNKNOWN
        suitable = get_suitable_templates(regime)
        logger.info(f"Market regime: {regime.value} | suitable templates: {suitable}")

        # Determine allowed templates — if requested template isn't suitable, still include it
        # (the user explicitly asked for it) but add suitable ones for diversity
        allowed = {template.value}
        for t in suitable:
            allowed.add(t)
        from athena.common.models import StrategyTemplate
        allowed_templates = [t for t in StrategyTemplate if t.value in allowed]

        ga = GAEngine(
            template=template,
            population_size=self.cfg.population_size,
            generations=self.cfg.generations,
            mutation_rate=self.cfg.mutation_rate,
            crossover_rate=self.cfg.crossover_rate,
            elitism_count=self.cfg.elitism_count,
            allowed_templates=allowed_templates,
            ml_seed_ratio=getattr(self.cfg, 'ml_seed_ratio', 0.0),
        )
        ga.initialize_population()

        def fitness_fn(ind: Individual) -> float:
            """Backtest + score a single individual."""
            record = self._individual_to_record(ind)
            metrics = self._backtest(record)
            record.performance = metrics
            record.score = self.scorer.score(metrics)
            self._persist(record)
            return record.score.raw_score

        # ── Evolve ──
        population = ga.evolve(fitness_fn, parallel_workers=pw)

        # ── Final evaluation + gates ──
        if pw > 1:
            records = self._evaluate_parallel(population, run_gates=run_gates)
        else:
            records = []
            for ind in population:
                record = self._individual_to_record(ind)
                record.performance = self._backtest(record)
                record.score = self.scorer.score(record.performance)
                if run_gates:
                    record = self._run_gates(record)
                self._persist(record)
                records.append(record)

        # ── Auto-promote best that passed gates ──
        records.sort(key=lambda r: r.score.raw_score, reverse=True)
        for rec in records:
            if rec.score.verdict == "promote" and rec.metadata.get("robustness_passed", run_gates is False):
                self._promote(rec)

        logger.info(f"Best: {records[0].id} score={records[0].score.raw_score:.3f} "
                    f"verdict={records[0].score.verdict}")
        return records

    def evaluate_record(self, record: StrategyRecord) -> StrategyRecord:
        """Backtest + score a StrategyRecord without requiring DB persistence first."""
        record.performance = self._backtest(record)
        record.score = self.scorer.score(record.performance)
        return record

    def evaluate(self, strategy_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> StrategyRecord:
        """Re-evaluate a single stored strategy."""
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=strategy_id).first()
        if not row:
            raise ValueError(f"Strategy {strategy_id} not found")

        record = self._row_to_record(row)
        record.performance = self._backtest(record, start_date, end_date)
        record.score = self.scorer.score(record.performance)
        self._persist(record)
        return record

    def promote(self, strategy_id: str, auto_paper: bool = False) -> StrategyRecord:
        """Run gates on a strategy and, if passed, promote to PROMOTED status + export."""
        record = self.evaluate(strategy_id)
        record = self._run_gates(record)
        if record.score.verdict == "promote" and record.metadata.get("robustness_passed", False):
            self._promote(record)
            logger.info(f"Promoted {strategy_id}")
            if auto_paper:
                self._auto_paper(record)
        else:
            logger.info(f"Promotion refused for {strategy_id}: gates={record.metadata.get('robustness_passed')} verdict={record.score.verdict}")
        return record

    def deploy(self, strategy_id: str, target_dir: Optional[Path] = None) -> Path:
        """Export a promoted strategy to a deployable file."""
        session = get_session()
        row = session.query(StrategyModel).filter_by(id=strategy_id).first()
        if not row:
            raise ValueError(f"Strategy {strategy_id} not found")

        record = self._row_to_record(row)
        code = self._generate_code(record)

        out_dir = target_dir or config.GENERATED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{strategy_id}.py"
        path.write_text(code)
        logger.info(f"Deployed {strategy_id} → {path}")
        return path

    def run_search(self, templates: Optional[List[StrategyTemplate]] = None,
                   run_gates: bool = True) -> Dict[str, List[StrategyRecord]]:
        """Multi-template autonomous search — evolve each template and return champions."""
        templates = templates or list(StrategyTemplate)
        results: Dict[str, List[StrategyRecord]] = {}
        for tpl in templates:
            try:
                records = self.evolve(tpl, run_gates=run_gates)
                results[tpl.value] = records
            except Exception as exc:
                logger.error(f"Template {tpl.value} failed: {exc}")
                results[tpl.value] = []
        return results

    # ── parallel evaluation ─────────────────────────────────────────

    def _evaluate_parallel(self, population: List[Individual], run_gates: bool = False) -> List[StrategyRecord]:
        """Evaluate a generation of individuals in parallel using ThreadPoolExecutor."""
        pw = config.PARALLEL_WORKERS
        records = []

        def _eval_one(ind: Individual) -> StrategyRecord:
            record = self._individual_to_record(ind)
            try:
                record.performance = self._backtest(record)
                record.score = self.scorer.score(record.performance)
                if run_gates:
                    record = self._run_gates(record)
                self._persist(record)
            except Exception as exc:
                logger.error(f"Parallel eval failed for {record.id}: {exc}")
                record.score = ScoreResult(raw_score=0.0, verdict="demote")
            return record

        with ThreadPoolExecutor(max_workers=pw) as pool:
            records = list(pool.map(_eval_one, population))
        return records

    # ── internal helpers ───────────────────────────────────────────

    def _generate_code(self, record: StrategyRecord) -> str:
        """Render Freqtrade strategy source from a record."""
        template = TEMPLATE_MAP.get(record.dna.template)
        if not template:
            raise ValueError(f"Unknown template: {record.dna.template}")
        params = self.encoder.to_strategy_params(record.dna.vector, record.dna.template)
        params["class_name"] = "AthenaStrategy"
        params["template_name"] = record.dna.template.value
        params["timeframe"] = getattr(record, "timeframe", self.cfg.timeframe)
        return template.format(**params)

    def _backtest(self, record: StrategyRecord, start: Optional[str] = None, end: Optional[str] = None) -> PerformanceMetrics:
        """Run freqtrade backtest and return structured metrics."""
        code = self._generate_code(record)
        result = self.freqtrade.run_backtest(
            code,
            start_date=start or self.cfg.start_date,
            end_date=end or self.cfg.end_date,
            exchange="binance",
            symbol=record.dna.vector.get("symbol", self.cfg.symbols[0] if self.cfg.symbols else "BTC-USD"),
            timeframe=record.dna.vector.get("timeframe", self.cfg.timeframe),
        )
        if "error" in result:
            logger.warning(f"Backtest error for {record.id}: {result['error']}")
        return PerformanceMetrics(
            total_return=float(result.get("total_return", 0.0)),
            sharpe=float(result.get("sharpe", 0.0)),
            sortino=float(result.get("sortino", 0.0)),
            calmar=float(result.get("calmar", 0.0)),
            win_rate=float(result.get("win_rate", 0.0)),
            max_drawdown=float(result.get("max_drawdown", 0.0)),
            total_trades=int(result.get("total_trades", 0)),
            avg_trade=float(result.get("avg_trade", 0.0)),
            profit_factor=float(result.get("profit_factor", 0.0)),
        )

    def _run_gates(self, record: StrategyRecord) -> StrategyRecord:
        """Run walk-forward + Monte Carlo gates."""
        wf_validator = WalkForwardValidator()
        mc_tester = MonteCarloStressTest()

        try:
            wf = wf_validator.run(record, self.cfg.start_date, self.cfg.end_date)
        except Exception as exc:
            logger.warning(f"WF gate error for {record.id}: {exc}")
            wf = WalkForwardResult(
                in_sample_metrics=PerformanceMetrics(),
                out_sample_metrics=PerformanceMetrics(),
                degradation_ratio=0.0,
                is_robust=False,
            )

        try:
            mc = mc_tester.run(record, self.cfg.start_date, self.cfg.end_date)
        except Exception as exc:
            logger.warning(f"MC gate error for {record.id}: {exc}")
            mc = MonteCarloResult(
                original_sharpe=0.0,
                shuffled_sharpe_mean=0.0,
                shuffled_sharpe_std=0.0,
                p_value=1.0,
                is_significant=False,
            )

        passes = wf.is_robust and mc.is_significant
        record.metadata = record.metadata or {}
        record.metadata["walk_forward"] = wf.model_dump()
        record.metadata["monte_carlo"] = mc.model_dump()
        record.metadata["robustness_passed"] = passes
        return record

    def _promote(self, record: StrategyRecord) -> None:
        """Mark strategy as PROMOTED and export to deployable directory."""
        record.status = StrategyStatus.PROMOTED
        self._persist(record)
        try:
            self.deploy(record.id, target_dir=config.GENERATED_DIR)
        except Exception as exc:
            logger.error(f"Deploy failed for {record.id}: {exc}")

    def _auto_paper(self, record: StrategyRecord) -> None:
        """Best-effort: start a paper trading session for a promoted strategy."""
        try:
            from athena.live.runner import LiveRunner
            runner = LiveRunner(
                strategy_id=record.id,
                mode="paper",
                risk={"max_drawdown": 0.15, "daily_loss_limit": 0.10},
            )
            import asyncio
            asyncio.create_task(runner.start())
            logger.info(f"Auto-paper started for {record.id}")
        except Exception as exc:
            logger.warning(f"Auto-paper failed for {record.id}: {exc}")

    def _persist(self, record: StrategyRecord) -> None:
        """Upsert StrategyRecord to DB."""
        session = get_session()
        existing = session.query(StrategyModel).filter_by(id=record.id).first()
        if existing:
            existing.status = record.status.value
            existing.generation = record.generation
            existing.total_return = record.performance.total_return
            existing.sharpe = record.performance.sharpe
            existing.sortino = record.performance.sortino
            existing.calmar = record.performance.calmar
            existing.win_rate = record.performance.win_rate
            existing.max_drawdown = record.performance.max_drawdown
            existing.total_trades = record.performance.total_trades
            existing.avg_trade = record.performance.avg_trade
            existing.profit_factor = record.performance.profit_factor
            existing.raw_score = record.score.raw_score
            existing.verdict = record.score.verdict
            if record.metadata:
                meta = existing.metadata_json or {}
                meta.update(record.metadata)
                existing.metadata_json = meta
            existing.updated_at = datetime.now(timezone.utc)
        else:
            strat = StrategyModel(
                id=record.id,
                name=record.name,
                template=record.dna.template.value,
                dna=record.dna.vector,
                objective=record.objective,
                status=record.status.value,
                generation=record.generation,
                parent_id=record.parent_id,
                total_return=record.performance.total_return,
                sharpe=record.performance.sharpe,
                sortino=record.performance.sortino,
                calmar=record.performance.calmar,
                win_rate=record.performance.win_rate,
                max_drawdown=record.performance.max_drawdown,
                total_trades=record.performance.total_trades,
                avg_trade=record.performance.avg_trade,
                profit_factor=record.performance.profit_factor,
                raw_score=record.score.raw_score,
                verdict=record.score.verdict,
                metadata_json=record.metadata or {},
            )
            session.add(strat)
        session.commit()

    # ── converters ───────────────────────────────────────────────────

    def _individual_to_record(self, ind: Individual) -> StrategyRecord:
        return StrategyRecord(
            id=ind.id,
            name=f"{ind.template.value}_{ind.id[-6:]}",
            template=ind.template,
            dna=StrategyDNA(template=ind.template, vector=ind.dna),
            generation=ind.generation,
            status=StrategyStatus.GENERATED,
        )

    def _row_to_record(self, row: StrategyModel) -> StrategyRecord:
        return StrategyRecord(
            id=row.id,
            name=row.name,
            template=StrategyTemplate(row.template),
            dna=StrategyDNA(template=StrategyTemplate(row.template), vector=row.dna),
            objective=row.objective,
            status=StrategyStatus(row.status),
            generation=row.generation,
            parent_id=row.parent_id,
            performance=PerformanceMetrics(
                total_return=row.total_return,
                sharpe=row.sharpe,
                sortino=row.sortino,
                calmar=row.calmar,
                win_rate=row.win_rate,
                max_drawdown=row.max_drawdown,
                total_trades=row.total_trades,
                avg_trade=row.avg_trade,
                profit_factor=row.profit_factor,
            ),
            score=ScoreResult(raw_score=row.raw_score, verdict=row.verdict),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
