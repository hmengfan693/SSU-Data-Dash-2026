# List-aware 接入 FLAML

使用与 Off-market 相同的 FLAML 2.7.0 CFO 和七参数搜索空间，优化现有统一 LightGBM 成交挂牌比模型。训练前固定规则见[预先固定方案.md](预先固定方案.md)。本轮不加入新特征。

在项目根目录运行：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_list_aware_flaml.py tests/test_list_aware_v1.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m experiments.list_aware_flaml.run_experiment --output experiments/list_aware_flaml/artifacts/run_2026-11-04_v1 --time-budget 180 --max-trials 24
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.list_aware_flaml.verify_results --run experiments/list_aware_flaml/artifacts/run_2026-11-04_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.list_aware_flaml.predict_saved --run experiments/list_aware_flaml/artifacts/run_2026-11-04_v1 --output /tmp/list_aware_flaml_replay.csv
```

沿用当前 .venv 和 experiments/flaml_tuning/requirements.txt，无需新增依赖。预算为搜索阶段软限制，不含特征重建、外层/最终重训和核验。时间预算导致完成试验数可能不同；精确重放使用保存模型与冻结参数。

输出包含中文报告、同当前最优配对的整体/房型/月度误差、每次试验、参数选择、来源哈希、训练ID、全部缓存、保存模型和隔离的4064行预测。运行入口仅接受第一版统一模型仍为最优的情况；后续最优已变化时需先适配对照，不能重复执行后误报改善。

复现需保留项目源文件、list_aware_v1存档及项目上级data目录；源文件快照在运行存档中保存。原正式提交不覆盖，所有成绩为累计开发回测。
