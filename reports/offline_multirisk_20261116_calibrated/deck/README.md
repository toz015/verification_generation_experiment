# Offline multi-risk deck (2026-11-16)

`离线多风险实验_20261116.pptx`: PPT version of `../results.html` (calibrated p_correct main comparison, raw-p control).

Rebuild (cache only, no API calls):

```bash
PYTHONPATH=src .venv/bin/python reports/offline_multirisk_20261116_calibrated/deck/extract.py reports/offline_multirisk_20261116_calibrated/deck/data.json
PPTX_SKILL=<pptx skill dir> NODE_PATH=<dir with pptxgenjs> node reports/offline_multirisk_20261116_calibrated/deck/build.js base.pptx
python reports/offline_multirisk_20261116_calibrated/deck/post.py base.pptx deck.pptx   # python-pptx; adds the charts
```

Charts are added with python-pptx because Keynote does not draw pptxgenjs charts.
