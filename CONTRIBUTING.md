# Contributing to F1 Race Intelligence

Thank you for helping improve this project. Contributions are welcome through **issues**, **pull requests**, data-quality reports, modeling improvements, documentation, and dashboard enhancements. You do not need to be an F1 expert to contribute: clear bug reports, tests, documentation fixes, and reproducible analysis are valuable.

## Before you start

- Check the existing issues and discussions before opening a duplicate.
- For a substantial feature or model change, open an issue first so the approach can be discussed.
- Never commit API keys, credentials, personal data, or data that you do not have permission to redistribute.
- Review the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

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

Contributors keep authorship of their work. Accepted public issues and pull requests are linked to the contributor's GitHub profile, and contributors may optionally add a public profile link to [CONTRIBUTORS.md](CONTRIBUTORS.md). Maintainers review and merge contributions; opening a pull request does not guarantee acceptance.

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

## Getting recognized

If your contribution is accepted, GitHub will show it in the repository's contributor graph. To be listed in the project's public contributor directory, add your preferred name, contribution area, and profile URL to `CONTRIBUTORS.md` in the same pull request or in a follow-up pull request. This listing is opt-in.

## Code of conduct

Please follow the full [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, constructive, and specific. Harassment, discrimination, spam, and intentionally misleading data claims are not welcome.
