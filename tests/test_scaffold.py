from ai_trading_research.agents.base_agent import BaseAgent
from ai_trading_research.backtests.runner import BacktestRunner
from ai_trading_research.strategies.base_strategy import BaseStrategy


def test_base_agent_run() -> None:
    agent = BaseAgent(name="researcher")
    response = agent.run("summarize trend")
    assert "researcher" in response


def test_backtest_runner_stub_metrics() -> None:
    strategy = BaseStrategy(name="baseline")
    metrics = BacktestRunner(strategy=strategy).run()
    assert metrics["strategy"] == "baseline"
    assert metrics["return_pct"] == 0.0
