# Release checklist for forest-clustering 0.8.0

1. Run tests: `python -m pytest -q`.
2. Build artifacts: `python -m build`.
3. Check artifacts: `python -m twine check dist/*`.
4. Install the wheel in a clean environment and smoke-test imports.
5. Upload to TestPyPI: `python -m twine upload --repository testpypi dist/*`.
6. Verify TestPyPI installation.
7. Upload to PyPI: `python -m twine upload dist/*`.

## Release highlights

- Adds `PrototypeSampler` for conservative weighted prototype reduction.
- Adds `SubsampledClusterer` to cluster prototypes and expand labels back to all rows.
- Adds compression diagnostics and plotting helpers.
- Keeps rare buckets by default to reduce the risk of destroying microclusters.
