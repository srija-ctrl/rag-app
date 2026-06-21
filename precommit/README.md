# pre-commit (short)


Hooks: `ruff` (format + lint) and a local `pytest` runner.

Install and enable:

```bash
pip install pre-commit ruff pytest
pre-commit install --config precommit/.pre-commit-config.yaml
chmod +x precommit/run_tests.sh
```

Run once:

```bash
pre-commit run --all-files -c precommit/.pre-commit-config.yaml
```
