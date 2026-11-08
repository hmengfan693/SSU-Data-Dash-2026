"""Reproduce the January-cutoff model and inspect its actual decisions.

All stage scores, SHAP and permutation checks are post-hoc descriptions of
an already selected model. They are not used for tuning or model selection.
"""
import argparse
import gzip
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster

import data
from experiments.plan_a.features import CATEGORICAL, HistoryFeatures
from experiments.plan_a.models import PARAMS, PriceModel
from experiments.plan_a.run_experiment import metrics, sha, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def score(prices, predicted):
    return metrics(pd.DataFrame({"sold_price": np.asarray(prices), "predicted_price": np.round(predicted)}))


def importance(booster):
    table = pd.DataFrame({"feature": booster.feature_name(),
                          "gain": booster.feature_importance("gain"),
                          "split_count": booster.feature_importance("split")})
    table["gain_share_pct"] = 100 * table.gain / table.gain.sum()
    table["split_share_pct"] = 100 * table.split_count / table.split_count.sum()
    return table


def node_summary(node, names, categories):
    feature = names[node["split_feature"]]
    threshold = node["threshold"]
    if node["decision_type"] == "==":
        codes = [int(v) for v in str(threshold).split("||")]
        labels = [categories[feature][i] for i in codes]
        threshold_text = f"in {len(labels)} categories; first 8: {labels[:8]}"
        human = threshold_text
    else:
        threshold_text = str(threshold)
        human = threshold_text
        if feature.endswith("_log_price"):
            human = f"log-price <= {float(threshold):.6f}; price-scale exp(threshold)={np.exp(float(threshold)):.2f} CAD"
    return dict(feature=feature, decision_type=node["decision_type"], threshold=threshold_text,
                human_threshold=human, gain=float(node["split_gain"]),
                training_count=node["internal_count"], default_left=node["default_left"])


def walk_nodes(node, names, categories, tree_id, depth=0):
    if "split_index" not in node:
        return []
    row = dict(tree=tree_id + 1, split_index=node["split_index"], depth=depth,
               **node_summary(node, names, categories))
    return [row, *walk_nodes(node["left_child"], names, categories, tree_id, depth + 1),
            *walk_nodes(node["right_child"], names, categories, tree_id, depth + 1)]


def decision_path(tree, x, row_id, names, categories):
    node, rows = tree["tree_structure"], []
    while "split_index" in node:
        feature = names[node["split_feature"]]
        value = x.iloc[row_id][feature]
        missing = pd.isna(value)
        if node["decision_type"] == "==":
            code = int(x[feature].cat.codes.iloc[row_id])
            left = bool(node["default_left"]) if code < 0 else code in {int(v) for v in str(node["threshold"]).split("||")}
        elif missing or (node["missing_type"] == "Zero" and value == 0):
            left = bool(node["default_left"])
        else:
            left = float(value) <= float(node["threshold"])
        rows.append(dict(**node_summary(node, names, categories),
                         actual_value=None if missing else str(value), branch="left" if left else "right"))
        node = node["left_child"] if left else node["right_child"]
    return dict(tree=tree["tree_index"] + 1, path=rows,
                leaf_index=node["leaf_index"], leaf_output_log_price=node["leaf_value"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent / "data")
    parser.add_argument("--previous", type=Path, default=REPO / "experiments/plan_a/artifacts/run_2026-10-16_v1")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/run_2026-10-17_v1")
    args = parser.parse_args()
    out, previous = args.output.resolve(), args.previous.resolve()
    old = json.loads((previous / "manifest.json").read_text())
    if sha(args.data_dir / "listings.parquet") != old["inputs_sha256"]["listings.parquet"]:
        raise ValueError("Original data changed")
    for filename in ["features.py", "models.py"]:
        if sha(REPO / "experiments/plan_a" / filename) != old["code_sha256"][filename]:
            raise ValueError("Original model or features changed")
    protected_paths = [args.data_dir / "listings.parquet", *[p for p in previous.rglob("*") if p.is_file()]]
    protected = {str(p): sha(p) for p in protected_paths}
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis_source_at_runtime.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    listings = pd.read_parquet(args.data_dir / "listings.parquet")
    for c in ["date_sold", "date_listed"]:
        listings[c] = pd.to_datetime(listings[c])
    train = listings.loc[listings.date_sold < "2026-02-01"].sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    homes = listings.loc[(listings.date_sold >= "2026-02-01") & (listings.date_sold < "2026-06-01")].reset_index(drop=True)
    print(f"Reconstructing features: {len(train):,} train, {len(homes):,} validation", flush=True)
    history = HistoryFeatures(train, "2026-02-01")
    x_train, x_valid = history.transform(train), history.transform(data.hide_answers(homes, "off_market"))
    print("Refitting the frozen 600-tree configuration", flush=True)
    fit_started = time.monotonic()
    model = PriceModel("lightgbm").fit(x_train, train.sold_price)
    fit_seconds = time.monotonic() - fit_started
    train_x, valid_x = model.encode(x_train), model.encode(x_valid)
    booster = model.model.booster_
    raw = booster.predict(valid_x, num_threads=6)
    predicted = np.round(np.exp(raw))
    archive = pd.read_csv(previous / "selection_A/2026-02/lightgbm_predictions.csv")
    if (not np.array_equal(archive.ml_num, homes.ml_num)
            or not np.array_equal(archive.sold_price, homes.sold_price)
            or not np.array_equal(archive.predicted_price, predicted)):
        raise ValueError("Refit does not exactly reproduce the archived validation predictions")
    print("Exact archived reproduction passed", flush=True)
    booster.save_model(str(out / "main_window_lightgbm.txt"))
    model_dump = booster.dump_model()
    with gzip.open(out / "model_dump.json.gz", "wt") as stream:
        json.dump(model_dump, stream)
    names = booster.feature_name()
    all_nodes = pd.DataFrame([row for tree in model_dump["tree_info"]
                             for row in walk_nodes(tree["tree_structure"], names, model.categories, tree["tree_index"])])
    all_nodes.to_csv(out / "split_nodes.csv", index=False)
    all_nodes.loc[all_nodes.depth == 0].to_csv(out / "tree_roots.csv", index=False)
    stages = []
    for iteration in [1, 5, 10, 25, 50, 100, 200, 400, 600]:
        train_raw = booster.predict(train_x, num_iteration=iteration, num_threads=6)
        valid_raw = booster.predict(valid_x, num_iteration=iteration, num_threads=6)
        stages.append(dict(iteration=iteration,
                           train_log_mae=float(np.mean(np.abs(train_raw - np.log(train.sold_price.to_numpy())))),
                           validation_log_mae=float(np.mean(np.abs(valid_raw - np.log(homes.sold_price.to_numpy())))),
                           **score(homes.sold_price, np.exp(valid_raw))))
    pd.DataFrame(stages).to_csv(out / "training_stages.csv", index=False)
    table = importance(booster)
    print("Computing TreeSHAP contributions on all 7,773 validation homes", flush=True)
    contrib_started = time.monotonic()
    contributions = np.asarray(booster.predict(valid_x, pred_contrib=True, num_threads=6))
    additive_error = float(np.max(np.abs(contributions.sum(axis=1) - raw)))
    if additive_error > 1e-8:
        raise ValueError("Contributions do not add back to the model's log prediction")
    contribution_seconds = time.monotonic() - contrib_started
    table["mean_abs_shap_log_price"] = np.abs(contributions[:, :-1]).mean(axis=0)
    table["shap_share_pct"] = 100 * table.mean_abs_shap_log_price / table.mean_abs_shap_log_price.sum()
    table["gain_rank"] = table.gain.rank(ascending=False, method="min").astype(int)
    table["shap_rank"] = table.mean_abs_shap_log_price.rank(ascending=False, method="min").astype(int)
    table.sort_values("gain", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(contributions, columns=[*names, "expected_log_price"]).assign(ml_num=homes.ml_num).to_parquet(out / "validation_contributions.parquet", index=False)
    homes[["ml_num", "sold_price", "house_type_name", "community"]].assign(predicted_price=predicted).to_csv(out / "validation_predictions.csv", index=False)
    baseline = score(homes.sold_price, predicted)
    single_features = list(dict.fromkeys([*table.nlargest(8, "gain").feature,
                                          *table.nlargest(5, "mean_abs_shap_log_price").feature]))
    groups = {f"single:{c}": [c] for c in single_features}
    groups.update({"group:all_area_representations": ["area_low", "area_high", "area_mid", "log_area", "area_is_range"],
                   "group:location_identifiers": ["location_group", "community", "municipality", "latitude", "longitude"],
                   "group:community_type_history": [c for c in names if c.startswith("hist_community_type_")]})
    permutation_rows = []
    for group, columns in groups.items():
        for repeat in range(3):
            # Same permutation across columns preserves relationships within
            # a feature group, but deliberately breaks links to other groups.
            permutation = np.random.default_rng(4200 + repeat).permutation(len(valid_x))
            changed = valid_x.copy()
            for column in columns:
                changed[column] = valid_x[column].iloc[permutation].reset_index(drop=True)
            prices = np.exp(booster.predict(changed, num_threads=6))
            s = score(homes.sold_price, prices)
            permutation_rows.append(dict(group=group, columns="|".join(columns), repeat=repeat,
                                          mdape=s["mdape"], mape=s["mape"],
                                          delta_mdape_pp=s["mdape"] - baseline["mdape"],
                                          delta_mape_pp=s["mape"] - baseline["mape"]))
        print(f"  permutation complete: {group}", flush=True)
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table.to_csv(out / "permutation_repeats.csv", index=False)
    permutation_summary = permutation_table.groupby("group", sort=False).agg(
        delta_mdape_mean_pp=("delta_mdape_pp", "mean"),
        delta_mdape_std_pp=("delta_mdape_pp", "std"),
        delta_mape_mean_pp=("delta_mape_pp", "mean"),
        shuffled_mdape_mean=("mdape", "mean")).reset_index()
    permutation_summary.to_csv(out / "permutation_summary.csv", index=False)
    picks = []
    for quantile in [.1, .5, .9]:
        index = int(np.argmin(np.abs(homes.sold_price.to_numpy() - homes.sold_price.quantile(quantile))))
        picks.append((f"actual_price_quantile_{quantile}", index))
    picks.append(("largest_absolute_percentage_error", int(np.argmax(np.abs(predicted / homes.sold_price.to_numpy() - 1)))))
    example_rows, example_features, paths = [], [], {}
    leaves = booster.predict(valid_x.iloc[[i for _, i in picks]], pred_leaf=True, num_threads=6)
    for pick_id, (rule, index) in enumerate(picks):
        home = homes.iloc[index]
        shap_values = contributions[index, :-1]
        baseline_log = float(contributions[index, -1])
        row = dict(selection_rule=rule, ml_num=home.ml_num, house_type_name=home.house_type_name,
                   community=home.community, house_area=home.house_area,
                   washrooms=int(home.washrooms), sold_price=float(home.sold_price),
                   predicted_price=float(predicted[index]),
                   signed_error_pct=float(100 * (predicted[index] / home.sold_price - 1)),
                   expected_log_price=baseline_log, baseline_price_exp_expected=float(np.exp(baseline_log)),
                   sum_shap=float(shap_values.sum()), predicted_log_price=float(raw[index]),
                   hist_community_type_365_price_scale=float(np.exp(x_valid.iloc[index].hist_community_type_365_log_price)))
        example_rows.append(row)
        for feature_id in np.argsort(-np.abs(shap_values))[:10]:
            feature = names[feature_id]
            value = x_valid.iloc[index][feature]
            phi = float(shap_values[feature_id])
            example_features.append(dict(ml_num=home.ml_num, feature=feature,
                                         actual_value=None if pd.isna(value) else str(value),
                                         shap_log_price=phi, multiplicative_effect_pct=float(100 * np.expm1(phi))))
        first_path = decision_path(model_dump["tree_info"][0], valid_x, index, names, model.categories)
        if first_path["leaf_index"] != int(leaves[pick_id, 0]):
            raise ValueError("Manually traversed tree path disagrees with LightGBM")
        outputs = [float(booster.get_leaf_output(t, int(leaves[pick_id, t]))) for t in range(booster.num_trees())]
        if not np.isclose(sum(outputs), raw[index], atol=1e-8):
            raise ValueError("Leaf outputs do not reconstruct log prediction")
        paths[home.ml_num] = dict(first_tree=first_path, tree_leaf_outputs=outputs,
                                  all_leaf_outputs_sum=float(sum(outputs)), predicted_log_price=float(raw[index]))
    pd.DataFrame(example_rows).to_csv(out / "example_homes.csv", index=False)
    pd.DataFrame(example_features).to_csv(out / "example_feature_contributions.csv", index=False)
    write_json(out / "example_decision_paths.json", paths)
    # The through-May final model is a separate model. Inspect its tree structure
    # without using February-May labels to score that model.
    final_path = previous / "trained_models/lightgbm.txt"
    if sha(final_path) != old["trained_models_sha256"]["lightgbm.txt"]:
        raise ValueError("Final saved model hash changed")
    final = Booster(model_file=str(final_path))
    final_importance = importance(final).sort_values("gain", ascending=False)
    final_importance.to_csv(out / "final_model_feature_importance.csv", index=False)
    stability = []
    for path in sorted(previous.glob("*_A/*/lightgbm_feature_importance.csv")):
        frame = pd.read_csv(path)
        for rank, row in enumerate(frame.head(5).itertuples(), 1):
            stability.append(dict(window=str(path.parent.relative_to(previous)), rank=rank,
                                  feature=row.feature, gain_share_pct=float(100 * row.importance / frame.importance.sum())))
    pd.DataFrame(stability).to_csv(out / "importance_across_windows.csv", index=False)
    missing_summary = {c: float(100 * x_valid[c].isna().mean()) for c in names}
    category_info = {c: dict(training_categories=len(model.categories[c]),
                            validation_unseen_pct=float(100 * valid_x[c].isna().mean())) for c in CATEGORICAL}
    summary = dict(train_n=len(train), validation_n=len(homes), feature_n=len(names),
                   cutoff_exclusive="2026-02-01", validation_end_exclusive="2026-06-01",
                   target="log(sold_price)", params=PARAMS["lightgbm"],
                   num_trees=booster.num_trees(), total_splits=len(all_nodes),
                   total_leaves=sum(t["num_leaves"] for t in model_dump["tree_info"]),
                   fit_seconds=fit_seconds, contribution_seconds=contribution_seconds,
                   archived_predictions_exact=True, shap_additivity_max_abs_error=additive_error,
                   manually_checked_example_paths=len(picks),
                   baseline_scores=baseline,
                   first_root=node_summary(model_dump["tree_info"][0]["tree_structure"], names, model.categories),
                   categories=category_info, missing_pct=missing_summary,
                   gain_top=table.nlargest(10, "gain").to_dict("records"),
                   shap_top=table.nlargest(10, "mean_abs_shap_log_price").to_dict("records"),
                   final_model_gain_top=final_importance.head(10).to_dict("records"),
                   permutation_repeats=3, permutation_seed_base=4200,
                   permutation_is_not_retraining_or_causal_effect=True,
                   stage_scores_not_used_for_tuning=True, total_seconds=time.monotonic() - started,
                   packages={p: importlib.metadata.version(p) for p in ["lightgbm", "numpy", "pandas"]},
                   analysis_source_sha256=sha(Path(__file__)), protected_sha256=protected)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError("Original data or previous artifacts changed")
    write_json(out / "summary.json", summary)
    write_report(out, summary, table, stages, permutation_summary, example_rows, example_features)
    summary["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "summary.json"}
    write_json(out / "summary.json", summary)
    print(f"Complete: {out / '运行过程与预测依据.md'}", flush=True)


def write_report(out, summary, table, stages, permutation, examples, example_features):
    gain = table.sort_values("gain", ascending=False)
    shap = table.sort_values("mean_abs_shap_log_price", ascending=False)
    final_gain = pd.DataFrame(summary["final_model_gain_top"])
    roots = pd.read_csv(out / "tree_roots.csv").feature.value_counts()
    lines = ["# LightGBM 运行过程与预测依据", "",
             f"**正式候选模型（截至 2026 年 5 月训练）的 Gain 第一名是 `{final_gain.iloc[0].feature}`，占 {final_gain.iloc[0].gain_share_pct:.2f}%。** 这版模型与下文截至 1 月的回测模型分别报告。", "",
             "本报告针对既有方案 A 的 Off-market LightGBM；重新训练截至 2026 年 1 月的模型，并逐套复现此前 2—5 月验证预测。", "",
             f"训练 {summary['train_n']:,} 套，验证 {summary['validation_n']:,} 套，{summary['feature_n']} 项输入，{summary['num_trees']} 棵树、{summary['total_splits']:,} 次分裂、{summary['total_leaves']:,} 个叶节点。", "",
             "## 最主要的输入变量", "",
             f"按训练累计分裂收益，第一为 `{gain.iloc[0].feature}`，占 {gain.iloc[0].gain_share_pct:.2f}%；第二为 `{gain.iloc[1].feature}`，占 {gain.iloc[1].gain_share_pct:.2f}%。",
             f"按验证集平均绝对 TreeSHAP 贡献，第一为 `{shap.iloc[0].feature}`，平均绝对贡献 {shap.iloc[0].mean_abs_shap_log_price:.4f} 对数价格单位。", "",
             f"`location_group` 出现在 {int(roots.get('location_group', 0))}/{summary['num_trees']} 棵树的根节点，占 {100 * roots.get('location_group', 0) / summary['num_trees']:.1f}%。它在训练集有 {summary['categories']['location_group']['training_categories']:,} 个类别，而 {summary['categories']['location_group']['validation_unseen_pct']:.2f}% 验证房屋的坐标组在训练中未出现，编码后按缺失类别处理。", "",
             "这两个排名回答不同问题。Gain 是内部训练分裂收益；SHAP 是模型输出在当前验证集上的分摊解释；两者都不是因果贡献，也不是预测准确率的百分比。", "",
             "| 输入变量 | Gain 占比 | 分裂次数 | 验证集平均绝对 SHAP | SHAP 排名 |",
             "|---|---:|---:|---:|---:|"]
    for row in gain.head(12).itertuples():
        lines.append(f"| {row.feature} | {row.gain_share_pct:.2f}% | {row.split_count} | {row.mean_abs_shap_log_price:.4f} | {row.shap_rank} |")
    lines += ["", "## 正式候选模型的 Gain 排名（截至 5 月训练）", "",
              "这一模型用全部 150,000 条记录训练；以下直接读取原始保存模型，无需用 2—5 月标签评价它。", "",
              "| 变量 | Gain 占比 |", "|---|---:|"]
    for row in final_gain.head(8).itertuples():
        lines.append(f"| {row.feature} | {row.gain_share_pct:.2f}% |")
    lines += ["", "## 第一棵树的实际根节点", "", "```json", json.dumps(summary["first_root"], ensure_ascii=False, indent=2), "```", "",
              "完整节点见 split_nodes.csv；所有树根见 tree_roots.csv。示例房屋的实际路径见 example_decision_paths.json，已与 LightGBM 返回的叶节点编号核对。", "",
              "## 逐轮运行数据", "", "同一个已经固定的 600 轮模型，事后查看前若干棵树的累计预测；没有据此调整轮数。误差为取整到元后的价格误差，Log-MAE 则在原始对数输出计算。", "",
              "| 树数 | 训练 Log-MAE | 验证 Log-MAE | 验证 MdAPE | 验证 MAPE |", "|---:|---:|---:|---:|---:|"]
    for row in stages:
        lines.append(f"| {row['iteration']} | {row['train_log_mae']:.4f} | {row['validation_log_mae']:.4f} | {row['mdape']:.2f}% | {row['mape']:.2f}% |")
    lines += ["", "## 打乱输入后的验证变化", "", "每项使用三次确定的随机排列。正数表示打乱后误差增加；标准差只是三次排列的波动，不是置信区间。组内列使用同一排列。", "",
              "| 输入或变量组 | MdAPE 变化均值／百分点 | 三次标准差 | MAPE 变化均值／百分点 |", "|---|---:|---:|---:|"]
    for row in permutation.itertuples():
        lines.append(f"| {row.group} | {row.delta_mdape_mean_pp:+.2f} | {row.delta_mdape_std_pp:.2f} | {row.delta_mape_mean_pp:+.2f} |")
    lines += ["", "相关变量可以互相替代；排列会破坏部分真实特征组合，因此不是删除该变量并重新训练的效果，也不代表改变房屋属性产生的真实价格变化。", "",
              "## 逐套预测与解释", "", "按真实价格第 10/50/90 百分位附近各选一套，再展示绝对百分比误差最大的一套；未按预测是否好看筛选。", "",
              "| ID | 房型 | 面积原值 | 真实价格／CAD | 预测价格／CAD | 有符号误差 |", "|---|---|---|---:|---:|---:|"]
    for row in examples:
        lines.append(f"| {row['ml_num']} | {row['house_type_name']} | {row['house_area']} | {row['sold_price']:,.0f} | {row['predicted_price']:,.0f} | {row['signed_error_pct']:+.2f}% |")
    for row in examples[:3]:
        lines += ["", f"### {row['ml_num']}", "", f"解释基准 exp(expected log-price) 为 {row['baseline_price_exp_expected']:,.0f} CAD；其全部变量贡献相加得到模型的对数预测。", "",
                  "| 变量 | 输入值 | 对数价格贡献 | exp(贡献)−1 |", "|---|---|---:|---:|"]
        for feature in [r for r in example_features if r["ml_num"] == row["ml_num"]][:6]:
            lines.append(f"| {feature['feature']} | {feature['actual_value']} | {feature['shap_log_price']:+.4f} | {feature['multiplicative_effect_pct']:+.2f}% |")
    lines += ["", "SHAP 在对数价格尺度可相加；exp(贡献)−1 为相对这一分摊基准的乘法因子，不能把这些百分比直接相加，也不能解释为装修或搬迁的因果溢价。", "",
              "## 字段定义和模型参数", "",
              "- location_group：经纬度分别四舍五入到小数点后 5 位，再拼接为类别标识。不是行政区，坐标与街区均为合成数据。",
              "- hist_community_type_365_log_price：查询时点之前 365 天，同社区×同房型历史成交对数价格的平滑均值。不是算术均价或中位数。",
              "- 查询时点是 min(挂牌日, 训练截止日)，查询区间左闭右开；同日及之后成交不进入该房屋的统计。",
              "- 社区×房型统计用强度 20 的同房型历史均值平滑；没有可用父级历史时不使用父级权重。",
              "- area_low/area_mid：原始面积范围的下界与中点，单位平方英尺；例如 600—699 的下界为 600，中点为 649.5。",
              "- num_leaves=31、min_child_samples=40、learning_rate=0.05 是控制训练的超参数，不是房屋估价的输入变量。min_child_samples 是实现中的 Hessian 近似约束，不能据此保证每个实际叶子都至少有 40 行。",
              "- 训练目标为 MAE(log(sold_price))，600 轮、L2 正则 5、种子 42；不是分类任务，也不使用挂牌价。", "",
              "## 核验与解释边界", "",
              f"- 主窗口预测与存档逐套完全一致；TreeSHAP 相加复原对数预测的最大绝对误差 {summary['shap_additivity_max_abs_error']:.3g}。",
              "- 四套示例的第一棵树路径与叶编号一致，600 棵树叶值之和复原对数预测。",
              "- 最终截至 5 月训练的模型另存 feature importance，不能与截至 1 月的模型混称；未用 2—5 月标签给最终模型打分。",
              "- 2—5 月此前已用于模型选择，这仍是累计开发集上的事后解释，不是独立测试或未来效果保证。",
              "- 高基数类别的 Gain 排名可能受可选切分数量、相关输入和训练样本影响，应结合验证集解释与排列检查。",
              "- 初始训练和历史特征源码、原始数据与此前产物已核验哈希，未修改。", "",
              "官方接口依据：[Gain/split 与 pred_contrib](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)、[树生长与类别分裂](https://lightgbm.readthedocs.io/en/stable/Features.html)、[参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)。", ""]
    (out / "运行过程与预测依据.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
