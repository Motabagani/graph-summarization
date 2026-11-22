Metrics, tuning choices and tables from our runs (predictions.jsonl files are excluded because of
their size). Layout:

    all/<dataset>/<run>/          main baselines          (metrics.json, tuning.json)
    sensitivity/<dataset>/<run>/  robustness variants
    position_analysis/            position_summary.md, positions.json, positions.png
    diagnostics/                  experiments used to explain differences from published results
    summary.md / summary.csv      all baselines on all datasets   (python baselines/collect_results.py)
    alignment.md                  comparison with published numbers (python baselines/compare_published.py)
