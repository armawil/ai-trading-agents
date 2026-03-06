# AI Trading Research Project

A lightweight Python project scaffold for researching AI-driven trading ideas.

## Project Structure

```text
.
├── src/ai_trading_research/
│   ├── agents/         # LLM/AI agents coordinating analysis and execution logic
│   ├── strategies/     # Trading strategy definitions and signal logic
│   ├── backtests/      # Backtest engine wrappers and experiment runners
│   ├── data/           # Data handling utilities + local data directories
│   │   ├── raw/
│   │   ├── processed/
│   │   └── external/
│   └── utils/          # Shared helpers (config, logging, metrics)
├── tests/              # Unit/integration tests
├── scripts/            # CLI helpers and workflow scripts
└── pyproject.toml      # Project metadata and tooling config
```

## Quick Start

1. Create a virtual environment and activate it.
2. Install in editable mode:

```bash
pip install -e .
```

3. Run the sample smoke test:

```bash
python -m ai_trading_research
```

## Next Steps

- Add concrete agent implementations in `agents/`.
- Define tradable strategy classes in `strategies/`.
- Build a reproducible backtest workflow in `backtests/`.
- Add data ingestion pipelines and metadata in `data/`.
