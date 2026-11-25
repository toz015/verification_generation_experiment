# Offline multi-risk decks

Two versions, same scripts:

- `离线多风险实验_原版本.pptx`: Ours selects only among its adaptive candidates
  (`configs/offline_multirisk_20261116_calibrated.json`, report `../results.html`).
- `离线多风险实验_加固定panel版.pptx`: Ours' Validation pool also holds fixed-panel mixtures over its own shortlist
  (`configs/offline_multirisk_20261117_pool_calibrated.json`, report
  `reports/offline_multirisk_20261117_pool_calibrated/results.html`).

Rebuild (cache only, no API calls); `CFG` picks the version, `DATA` the matching data file:

```bash
CFG=configs/offline_multirisk_20261117_pool_calibrated.json PYTHONPATH=src .venv/bin/python reports/offline_multirisk_20261116_calibrated/deck/extract.py data_pool.json
DATA=data_pool.json PPTX_SKILL=<pptx skill dir> NODE_PATH=<dir with pptxgenjs> node reports/offline_multirisk_20261116_calibrated/deck/build.js base.pptx
python reports/offline_multirisk_20261116_calibrated/deck/post.py base.pptx deck.pptx   # python-pptx; adds the charts
```

`make2.sh <orig|pool>` runs build, charts, validation and a Keynote render in the scratch folder it sits in.
Charts are added with python-pptx because Keynote does not draw pptxgenjs charts.
