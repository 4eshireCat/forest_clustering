# Deployment report - forest-clustering 0.9.0

Status: ready for local validation and TestPyPI/PyPI upload.

Checks performed in this workspace:

- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`: 61 passed
- package metadata updated to version 0.9.0
- diagnostics smoke tests cover reports, health checks, plots, stability, comparisons, mixed-data sklearn fallback and AutoTree search plots

Recommended release commands:

```bash
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```
