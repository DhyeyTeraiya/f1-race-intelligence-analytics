# Contributing to F1 Race Intelligence

Thank you for helping improve this project. Contributions are welcome through **issues**, **pull requests**, data-quality reports, modeling improvements, documentation, and dashboard enhancements.

## Contribution flow

1. Open an issue before significant work so the problem or proposal can be discussed.
2. Fork the repository and create a focused branch from `main`.
3. Make a small, reviewable change with tests where practical.
4. Run the local validation suite:

   ```bash
   python -m unittest discover -s tests -p 'test_*.py' -v
   python -m compileall -q src app tests
   ```

5. Open a pull request using the pull-request template.
6. Explain the data source, assumptions, model impact, and limitations.
7. Respond to review feedback; maintainers merge changes after CI and review pass.

## Good contribution areas

- Live-data provider adapters and resilient caching
- Data validation and provenance checks
- Leakage-safe feature engineering and model evaluation
- Calibration, uncertainty estimates, and backtesting
- Pit-stop and tire-strategy modeling
- Streamlit usability and accessibility
- Documentation, tests, and reproducible notebooks

## Data and modeling standards

- Do not commit API keys, credentials, or private data.
- Do not present simulated or derived telemetry as official live telemetry.
- Prefer real, cited data sources and document rate limits.
- Keep train/test splits chronological for forecasting tasks.
- Add or update tests for new parsing, analytics, or strategy behavior.
- Avoid changing benchmark claims without rerunning the evaluation.

## Issue triage

Use a clear title and include reproducible steps, logs, expected behavior, and the data source involved. Issues labeled `good first issue` are suitable for newcomers; issues labeled `help wanted` are open for community implementation.

## Code of conduct

Be respectful, constructive, and specific. Harassment, discrimination, spam, and intentionally misleading data claims are not welcome.
