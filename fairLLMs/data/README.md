# Packaged data files

Canonical copies of datasets that several metrics share.

- `crows_pairs/crows_pairs_anonymized.csv` — CrowS-Pairs
- `bbq/*.jsonl` — BBQ category files (hard-linked from a legacy leaf when possible)

Loaders in `fairLLMs.datasets` resolve these paths first, then fall back to
legacy locations under `fairLLMs/definition/**/data/` so existing scripts keep
working during the migration.
