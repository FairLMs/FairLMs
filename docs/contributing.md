# Contributing

## Development setup

```bash
git clone https://github.com/michaellarionov/FairLMs.git
cd FairLMs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest --cov=fairlms --cov-report=term-missing
```

## Adding a metric

See [Writing a metric](guides/custom-metric.md) for the full contract. In short:

1. Subclass `FairnessMetric`, declare `name`, `bias_type` and `architectures`,
   and keep `__init__` to configuration stored verbatim.
2. Implement `compute(model, data, **legacy) -> MetricResult`, resolving the
   model through `fairlms.metrics.resolve.get_tokenizer_model` rather than
   loading a checkpoint yourself.
3. Add the class to `METRIC_REGISTRY` in `fairlms/metrics/__init__.py` and to
   `__all__`.
4. Run the suite — the contract tests iterate over the registry, so
   registration alone earns the checks.

```bash
pytest -k your_metric_name
```

The internal implementation belongs under `fairlms/definition/` in the
`{architecture}/{bias_type}/…` taxonomy, with a short `main.py` demonstrating
the public API. The wrapper in `fairlms/metrics/` is the stable surface;
`definition/` can change without breaking user code.

## Adding a loader

Subclass `FairnessDataset`, implement `load()`, and export the class from
`fairlms/datasets/__init__.py` — the export list is what the generated
[Loaders](registry/loaders.md) page reads. Prefer Hub download over vendoring;
only CrowS-Pairs and BBQ are bundled, and anything you add under
`fairlms/data/` must also be declared in `[tool.setuptools.package-data]` or it
will not ship in the wheel.

## Adding a diagnostic

See [Writing a diagnostic](guides/custom-diagnostic.md). Register in
`DIAGNOSTIC_REGISTRY`; every component must report one of `ready`, `blocked`,
`not_applicable`, `failed`, and non-ready components must carry `value=None`.

## Docs

```bash
python scripts/gen_registry_docs.py    # regenerate registry tables
mkdocs serve
```

Registry pages are generated from the registries themselves — edit the code, not
the tables. CI runs `gen_registry_docs.py --check` and `mkdocs build --strict`,
so a stale table or a broken link fails the build.

## Releasing

Bump `fairlms/_version.py`, then tag; pushing a `v*` tag triggers the publish
workflow, which verifies that the tag matches the package version.

```bash
git tag -a "v$(python scripts/package_version.py)" -m "Release" && git push origin --tags
```
