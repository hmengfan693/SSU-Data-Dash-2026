"""Independently verify diagnostic predictions and write a reviewable report."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.mlp.models import sha, write_json
from .ablations import load_snapshot
from .audit import OUT, RESNET, MLP, REPO


def independent_score(prediction, prices):
    price = np.asarray(prices)
    error = 100 * (np.asarray(prediction) - price) / price
    return dict(n=len(price), mdape=float(np.median(np.abs(error))),
                mape=float(np.mean(np.abs(error))), bias=float(np.median(error)),
                log_mae=float(np.mean(np.abs(np.log(prediction) - np.log(price)))),
                over30=float(np.mean(np.abs(error) > 30) * 100))


def assert_score(actual, expected):
    for k, value in actual.items():
        if not np.isclose(value, expected[k], atol=1e-9, rtol=1e-10):
            raise ValueError(f"Metric differs: {k}: {value} vs {expected[k]}")


def main():
    target = OUT / "ablations"
    manifest = json.loads((target / "manifest.json").read_text())
    records = json.loads((target / "results.json").read_text())
    assert [r["name"] for r in records] == [s["name"] for s in manifest["specs"]]
    specs = list(manifest["specs"])
    extra = target / "resnet_capacity_matched"
    extra_spec = json.loads((extra / "prespecified.json").read_text())
    assert sha(Path(__file__).with_name("capacity_control.py")) == extra_spec["source_sha256"]
    specs.append(extra_spec["spec"])
    records.append(json.loads((extra / "result.json").read_text()))
    cache = OUT / "cache"
    hashes = json.loads((OUT / "cache_hashes.json").read_text())
    assert all(sha(cache / n) == h for n, h in hashes.items())
    assert all(sha(p) == h for p, h in manifest["protected_sha256"].items())
    for name, digest in manifest["source_sha256"].items():
        assert sha(target / name) == digest
        assert sha(Path(__file__).with_name(name)) == digest
    x_valid = pd.read_parquet(cache / "inner_valid_x.parquet")
    x_outer = pd.read_parquet(cache / "outer_x.parquet")
    y_valid = pd.read_parquet(cache / "inner_valid_labels.parquet")
    y_outer = pd.read_parquet(cache / "outer_labels.parquet")
    n_inner = len(pd.read_parquet(cache / "inner_train_labels.parquet"))
    y_train = pd.read_parquet(cache / "train_labels.parquet")
    x_train = pd.read_parquet(cache / "train_x.parquet")
    n_train = len(y_train)
    earlier = pd.read_parquet(cache / "inner_train_labels.parquet")
    assert (earlier.date_sold < pd.Timestamp("2025-12-01")).all()
    assert y_valid.date_sold.ge("2025-12-01").all() and y_valid.date_sold.lt("2026-02-01").all()
    assert y_train.date_sold.lt("2026-02-01").all()
    assert y_outer.date_sold.ge("2026-02-01").all() and y_outer.date_sold.lt("2026-06-01").all()
    assert not set(earlier.ml_num).intersection(y_valid.ml_num)
    assert not set(y_train.ml_num).intersection(y_outer.ml_num)
    original_manifest = json.loads((RESNET / "manifest.json").read_text())
    original_inner = json.loads((RESNET / "selection_A/2026-02/inner_selection/audit.json").read_text())
    original_refit = json.loads((RESNET / "selection_A/2026-02/model/schema.json").read_text())
    data_dir = REPO.parent / "data"
    for name, digest in original_manifest["input_sha256"].items():
        assert sha(data_dir / name) == digest
    comparison, curves, groups, total_batches = [], [], [], 0
    for r in records:
        folder = target / r["name"]
        curve = pd.read_csv(folder / "learning_curve.csv")
        best, best_epoch, stale = np.inf, None, 0
        for row in curve.itertuples():
            assert stale < 10
            if row.inner_mdape < best - 0.0001:
                best, best_epoch, stale = row.inner_mdape, row.epoch, 0
            else:
                stale += 1
        assert best_epoch == r["selected_epochs"]
        assert len(curve) == r["epochs_run_inner"]
        for kind, n, epochs, x, y, filename in [
            ("inner_model", n_inner, len(curve), x_valid, y_valid, "inner_predictions.csv"),
            ("refit_model", n_train, best_epoch, x_outer, y_outer, "outer_predictions.csv")]:
            s = json.loads((folder / kind / "schema.json").read_text())
            e = s["execution"]
            expected_batches = math.ceil(n / s["trainer_params"]["batch_size"]) * epochs
            assert e["gpu_batches"] == expected_batches and e["cpu_batches"] == 0
            assert e["device"] == "mps" and not e["cpu_fallback_enabled"]
            assert all(e[k].startswith("mps") for k in ["parameter_device", "numeric_device", "categorical_device", "target_device"])
            expected_preprocessing = original_inner["preprocessing"] if kind == "inner_model" else original_refit["preprocessing"]
            assert s["preprocessing"] == expected_preprocessing
            assert s["diagnostic_spec"] == next(z for z in specs if z["name"] == r["name"])
            total_batches += expected_batches
            model = load_snapshot(folder / kind)
            frame = pd.read_csv(folder / filename)
            assert np.array_equal(frame.ml_num, y.ml_num)
            assert np.array_equal(frame.sold_price, y.sold_price)
            # CSV float parsing can change the last bit of unrounded prices.
            actual = model.predict(x)
            if kind == "inner_model":
                assert np.allclose(actual, np.round(frame.predicted_price), rtol=0, atol=1)
            else:
                assert np.array_equal(actual, frame.predicted_price)
                train_prediction = model.predict_tensors(*model.preprocessor.transform(x_train))
                assert_score(independent_score(train_prediction, y_train.sold_price), r["refit_train"])
            assert_score(independent_score(frame.predicted_price, frame.sold_price), r["inner" if kind == "inner_model" else "outer"])
        comparison.append(dict(name=r["name"], selected_epochs=best_epoch,
                               inner_mdape=r["inner"]["mdape"], train_mdape=r["refit_train"]["mdape"],
                               outer_mdape=r["outer"]["mdape"], outer_mape=r["outer"]["mape"],
                               outer_bias=r["outer"]["bias"]))
        macro = curve.loc[curve.inner_macro_mdape_n30.idxmin()]
        curves.append(dict(name=r["name"], global_selected_epoch=best_epoch,
                           macro_best_epoch=int(macro.epoch), macro_best=float(macro.inner_macro_mdape_n30),
                           macro_at_global_selection=float(curve.loc[curve.epoch == best_epoch, "inner_macro_mdape_n30"].iloc[0])))
        frame = pd.read_csv(folder / "outer_predictions.csv")
        for typ, g in frame.groupby("house_type_name"):
            groups.append(dict(name=r["name"], group=typ, **independent_score(g.predicted_price, g.sold_price)))
        print(f"Verified {r['name']}: inner and refit snapshots; epochs and GPU batches", flush=True)
    pd.DataFrame(comparison).to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "global_vs_macro_selection.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "verified_outer_by_property_type.csv", index=False)
    write_json(OUT / "independent_verification.json", dict(passed=True, configs=len(records),
        snapshot_replays=2 * len(records), gpu_fits=2 * len(records), gpu_batches=total_batches,
        protected_files=len(manifest["protected_sha256"]), protected_unchanged=True,
        cache_hashes_verified=True, source_hashes_verified=True, epoch_selection_verified=True))
    saved = json.loads((OUT / "saved_model_summary.json").read_text())
    parity = json.loads((OUT / "cpu_gpu_inference_parity.json").read_text())
    coverage = pd.read_csv(OUT / "location_coverage.csv")
    dist = pd.read_csv(OUT / "inner_outer_distribution.csv")
    by_type = pd.DataFrame(groups).pivot(index="group", columns="name", values="mdape")
    lines = ["# ResNet 误差原因诊断", "",
             "检查原模型，并预先固定八组 GPU 受控实验；另外根据主干参数扩大 43 倍的发现，追加一个接近 MLP 主干容量的控制。所有配置均只用 2025 年 12 月—2026 年 1 月内部验证选择轮数，随后全量重训；原八组在全部完成后统一评分，容量对照独立评分。未替换正式模型或提交。", "",
             "## 原模型的泛化差距", "", "| 模型 | 训练集 MdAPE | 后续窗口 MdAPE | 训练 log-MAE | 后续 log-MAE |", "|---|---:|---:|---:|---:|"]
    for name in ["resnet", "mlp"]:
        t, o = saved[name]["train"], saved[name]["outer"]
        lines.append(f"| {name} | {t['mdape']:.2f}% | {o['mdape']:.2f}% | {t['log_mae']:.4f} | {o['log_mae']:.4f} |")
    lines += ["", "训练集结果采用 eval 模式、关闭 Dropout，与后续窗口同口径。训练集表现更好而后续表现更差，支持泛化问题；单靠此现象不能区分过拟合与分布变化。", "",
              "主干容量拆分：原 MLP 非 embedding 参数为 25,089，原 ResNet 为 1,083,905，扩大 43.2 倍；相同 embedding 部分掩盖了主干增加幅度。追加的 64 维、2 块 ResNet 用于检验接近原 MLP 容量时的表现。", "",
              "## 精确位置类别的冷启动", "", "| 房型 | 样本量 | 后续未见位置比例 |", "|---|---:|---:|"]
    for row in coverage.itertuples():
        lines.append(f"| {row.property_type} | {row.n:,} | {row.unseen_pct:.2f}% |")
    lines += ["", "位置由经纬度保留 5 位小数组成。训练期的每个位置都在其词表内；后续未知位置使用固定零向量。公寓与非公寓的覆盖存在巨大差异，但覆盖差异本身只是机制线索，必须结合移除/屏蔽位置的受控实验判断。", "",
              "固定其他输入、仅将精确位置类别改为未知时，原 ResNet 的公寓预测中位数下降 5.50%，原 MLP 下降 0.58%；非公寓没有同样的整体下降。这说明 ResNet 对公寓的精确位置依赖更强，但不能据此把非公寓低估直接归咎于零向量。详见 embedding_zero_intervention.csv（机制探针，不是可部署成绩）。", "",
              "跨四个原始回测窗口，独立屋、半独立屋、自由产权联排共 12 组对比的 ResNet MdAPE 均高于 MLP，见 cross_window_property_type.csv。窗口训练集重叠，因此不能把它们当成 12 个独立统计试验。", "",
              "## 受控重训结果", "", "| 实验 | 选定轮数 | 内层 MdAPE | 重训训练集 MdAPE | 后续 MdAPE | 后续 MAPE |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['name']} | {row['selected_epochs']} | {row['inner_mdape']:.2f}% | {row['train_mdape']:.2f}% | {row['outer_mdape']:.2f}% | {row['outer_mape']:.2f}% |")
    lines += ["", "实验定义：repeat42 为原配置同种子重新运行；seed43 仅换种子；two_blocks 仅减少残差块数 4→2；lower_lr 仅将 lr 0.001→0.0003；no_location 保留词表与初始化，将位置 embedding 固定为零；location_dropout 训练时以 50% 概率将整条位置向量置零，推理不屏蔽；batchnorm 将各残差块与输出头的 LayerNorm 替换为 BatchNorm；mlp_gpu 用原 MLP 网络在同一 GPU 训练循环中运行；capacity_matched 使用 64 维主干、2 块，每块内部 128 维，全连接参数为原 MLP 的约 1.6 倍，属于近似容量控制。其他训练和预处理设置相同。", "",
              "## 各房型：受控实验后续 MdAPE", ""]
    names = list(by_type.columns)
    lines += ["| 房型 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for typ, values in by_type.iterrows():
        lines.append("| " + typ + " | " + " | ".join(f"{values[name]:.2f}%" for name in names) + " |")
    lines += ["", "## 数值后端检查", "",
              f"原 ResNet 同一份权重 CPU 与 GPU 推理平均差 {parity['resnet']['mean_absolute_dollar_difference']:.4f} 元，最大相对差 {parity['resnet']['max_relative_difference']:.2e}；这排除了推理数值误差作为主要解释，但不能单独排除训练轨迹的后端差异。", "",
              "## 验证分布与选择指标", "", "| 阶段 | 房型 | 样本量 | 占比 | 成交价中位数 |", "|---|---|---:|---:|---:|"]
    for row in dist.itertuples():
        if row.property_type in ["Condo Apt", "Detached", "Semi-Detached"]:
            lines.append(f"| {row.partition} | {row.property_type} | {row.n:,} | {row.share_pct:.2f}% | {row.median_price:,.0f} |")
    lines += ["", "冬季内层验证中公寓占比更高，后续窗口非公寓占比上升；分房型价格分布也变化。这是样本构成差异，不能解释为同一套房屋的涨跌。全体 MdAPE 的最优轮数与房型等权指标的最优轮数可能不同，见 global_vs_macro_selection.csv；后者仅作诊断，没有根据后续标签选轮数。", "",
              "## 证据边界与验收", "",
              "- 九组配置各运行一次；同种子重跑和另一个种子是稳定性对照，不能替代完整的多种子实验。",
              "- 前次同时改变了网络深度/宽度、归一化与运行后端，不能把差异全部归因于残差连接。",
              "- 已使用过的后续窗口仍属于累计开发评价；不能把诊断后最好的配置当成独立测试成绩。",
              "- 每个配置均核验内层与重训 snapshot；独立复算误差、早停轮数、GPU batch 数、原产物与输入缓存哈希。",
              "- 原提交与模型保持原样；这些控制模型仅用于诊断，没有生成替代正式提交。", "",
              "作者官方 rtdl 的 ResNet.make_baseline 默认使用 BatchNorm1d；先前版本使用 LayerNorm，是本项目的适配选择。两者的收益需要以受控实验衡量。", "",
              "参考：[表格 ResNet 原论文](https://arxiv.org/abs/2106.11959)、[官方 ResNet 基线代码](https://github.com/yandex-research/rtdl/blob/main/rtdl/modules.py)、[PyTorch 可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。", ""]
    body = "\n".join(lines)
    notes = OUT / "研究结论.md"
    if notes.exists():
        body = body.replace("# ResNet 误差原因诊断\n\n",
                            "# ResNet 误差原因诊断\n\n" + notes.read_text().rstrip() + "\n\n", 1)
    (OUT / "诊断报告.md").write_text(body)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
