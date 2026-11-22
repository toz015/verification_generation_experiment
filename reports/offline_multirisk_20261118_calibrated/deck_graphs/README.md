# Graph-first deck (2026-11-22)

`离线多风险实验_结果图.pptx`: release error rate and FPR only, from `results/offline_multirisk_20261118_calibrated/`
(the report at `../results.html`). Per budget f in 0.1 / 0.5 / 1: alpha vs release rate per dataset x generator plus the
mean release rate per method; actual risk vs target alpha; actual cost vs budget at the main alpha. FNR is a one-slide
note; baseline comparisons and constraint overruns follow; Self-consistency is in the appendix.

Rebuild (cache only): `extract2.py data_charts.json` and `extract.py data_orig.json` (CFG = the 20261118 calibrated
config, PYTHONPATH=src), then `make3.sh` (pptxgenjs build, python-pptx charts, Keynote render).
