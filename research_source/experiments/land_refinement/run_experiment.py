"""Land-type refinements with preserved apartment predictions and frozen selection."""
import argparse
import importlib.metadata
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.features import HistoryFeatures
from experiments.plan_a.models import PARAMS
from experiments.plan_a.run_experiment import metrics
from experiments.property_groups.models import GroupModel, group_labels, routed_predict
from experiments.property_groups.run_experiment import now, sha, split_window, scores_bundle, write_json, breakdown
from experiments.property_groups.selection import DEV_WINDOWS, CHECK_WINDOW
from .features import CONFIG, ComparableFeatures, geometry_features
from .models import LandModel
from .selection import TARGET_TYPES, mix_predictions, route, select

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOG = logging.getLogger(__name__)
NEW_VARIANTS = {"Detached": ("geometry", "comparables"), "Semi-Detached": ("split",),
                "Freehold Townhouse": ("split",)}


def slug(kind):
    return kind.lower().replace("-", "_").replace(" ", "_")


def validate_archive(previous, data_dir):
    manifest = json.loads((previous / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Previous experiment is incomplete")
    for path, digest in manifest["protected_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Previous source/input has changed: {path}")
    for relative, digest in manifest["artifact_sha256"].items():
        if sha(previous / relative) != digest:
            raise ValueError(f"Previous artifact has changed: {relative}")
    for name in ("listings.parquet", "test_off_market.parquet"):
        old_hash = manifest["protected_sha256"][str((REPO.parent / "data" / name).resolve())]
        if sha(data_dir / name) != old_hash:
            raise ValueError("Data differs from archived baseline")
    if manifest["params"] != PARAMS:
        raise ValueError("Training parameters changed from previous baseline")
    for package, version in manifest["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Runtime version changed: {package}")


def checked_rows(frame, homes):
    if frame.ml_num.duplicated().any() or set(frame.ml_num) != set(homes.ml_num):
        raise ValueError("Archived prediction ID coverage differs")
    aligned = homes[["ml_num", "sold_price", "house_type_name"]].merge(
        frame[["ml_num", "sold_price", "house_type_name", "predicted_price"]],
        on="ml_num", validate="one_to_one", suffixes=("", "_archived"))
    if not np.array_equal(aligned.sold_price, aligned.sold_price_archived) or not np.array_equal(
            aligned.house_type_name, aligned.house_type_name_archived):
        raise ValueError("Archived labels/types differ")
    if not np.isfinite(aligned.predicted_price).all() or not (aligned.predicted_price > 0).all():
        raise ValueError("Invalid archived predictions")
    result = homes[["ml_num", "sold_price", "house_type_name", "community"]].copy().reset_index(drop=True)
    result["predicted_price"] = aligned.predicted_price.to_numpy()
    result["month"] = pd.to_datetime(homes.date_sold).dt.strftime("%Y-%m").to_numpy()
    result["group"] = group_labels(homes).to_numpy()
    return result


def add_frame(meta, prices, candidate, protocol, cutoff):
    result = meta.copy()
    result["predicted_price"] = prices
    result["candidate"] = candidate
    result["protocol"] = protocol
    result["cutoff_exclusive"] = cutoff
    return result


def enriched(base, homes, variant, comparable=None):
    if variant == "split":
        return base.reset_index(drop=True)
    extra = geometry_features(homes)
    if variant == "comparables":
        if comparable is None:
            raise ValueError("Missing comparable feature provider")
        extra = pd.concat([extra, comparable.transform(homes)], axis=1)
    return pd.concat([base.reset_index(drop=True), extra], axis=1)


def train_variant(tx, prices, vx, folder, cutoff, variant):
    start = time.monotonic()
    model = LandModel().fit(tx, prices)
    predicted = model.predict(vx)
    model.save_land(folder, cutoff, len(tx), variant)
    replay = GroupModel.load_bundle(folder).predict(vx)
    if not np.array_equal(predicted, replay):
        raise ValueError("Saved new model does not reproduce predictions")
    importance = model.model.booster_.feature_importance(importance_type="gain")
    pd.DataFrame({"feature": model.feature_names, "importance": importance}).sort_values(
        "importance", ascending=False).to_csv(folder / "feature_importance.csv", index=False)
    return predicted, {"train_n": len(tx), "feature_count": len(tx.columns), "features": list(tx.columns),
                       "reload_exact": True, "seconds": time.monotonic() - start}


def evaluate(listings, previous, old_candidates, old_routed, out, cutoff, end, protocol):
    train, homes = split_window(listings, cutoff, end)
    history = HistoryFeatures(train, cutoff)
    x_train = history.transform(train)
    x_valid = history.transform(data.hide_answers(homes, "off_market"))
    folder = out / protocol / cutoff[:7]
    folder.mkdir(parents=True)
    LOG.info("[%s] cutoff=%s train=%s evaluation=%s", protocol, cutoff, len(train), len(homes))
    comparable = ComparableFeatures(train.loc[train.house_type_name == "Detached"], cutoff)
    old_cutoff = old_candidates.loc[old_candidates.cutoff_exclusive == cutoff]
    previous_rows = checked_rows(old_routed.loc[old_routed.cutoff_exclusive == cutoff], homes)
    previous_rows["protocol"] = protocol
    previous_rows["cutoff_exclusive"] = cutoff
    global_rows = checked_rows(old_cutoff.loc[old_cutoff.candidate == "global__lightgbm"], homes)
    results, audit = [], {}
    for kind in TARGET_TYPES:
        tpos = np.flatnonzero(train.house_type_name.eq(kind))
        vpos = np.flatnonzero(homes.house_type_name.eq(kind))
        th, vh = train.iloc[tpos].reset_index(drop=True), homes.iloc[vpos].reset_index(drop=True)
        meta = previous_rows.iloc[vpos].reset_index(drop=True)
        baseline = add_frame(meta, meta.predicted_price.to_numpy(), "previous", protocol, cutoff)
        global_frame = add_frame(meta, global_rows.predicted_price.to_numpy()[vpos], "global", protocol, cutoff)
        results += [baseline, global_frame, add_frame(meta, mix_predictions(baseline, global_frame),
                                                     "half_previous", protocol, cutoff)]
        for variant in NEW_VARIANTS[kind]:
            tx = enriched(x_train.iloc[tpos], th, variant, comparable)
            vx = enriched(x_valid.iloc[vpos], data.hide_answers(vh, "off_market"), variant, comparable)
            predicted, detail = train_variant(tx, th.sold_price, vx, folder / f"{slug(kind)}__{variant}", cutoff, variant)
            row = add_frame(meta, predicted, variant, protocol, cutoff)
            row.to_csv(folder / f"{slug(kind)}__{variant}" / "predictions.csv", index=False)
            detail["scores"] = metrics(row)
            audit[f"{kind}/{variant}"] = detail
            results.append(row)
            results.append(add_frame(meta, mix_predictions(row, global_frame), f"half_{variant}", protocol, cutoff))
            LOG.info("  %s/%s MdAPE=%.3f%% MAPE=%.3f%% (%.1fs)", kind, variant,
                     detail["scores"]["mdape"], detail["scores"]["mape"], detail["seconds"])
    write_json(folder / "audit.json", {"train_n": len(train), "evaluation_n": len(homes),
        "max_training_sale": str(train.date_sold.max()), "cutoff_exclusive": cutoff, "end_exclusive": end,
        "evaluation_labels_used_for_fit_or_early_stopping": False, "models": audit,
        "comparable_config": CONFIG, "history_pool_and_base_features_unchanged": True})
    return pd.concat(results, ignore_index=True), previous_rows


def pooled_full(previous_rows, selected):
    result = previous_rows.copy().reset_index(drop=True)
    mapping = selected.set_index("ml_num")
    mask = result.ml_num.isin(mapping.index)
    result.loc[mask, "predicted_price"] = result.loc[mask, "ml_num"].map(mapping.predicted_price)
    if result.ml_num.duplicated().any() or not np.isfinite(result.predicted_price).all():
        raise ValueError("Invalid pooled full predictions")
    return result


def final_fit(listings, test, previous, out, choices):
    (out / "final_models").mkdir(parents=True, exist_ok=True)
    train = listings.sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    history = HistoryFeatures(train, "2026-06-01")
    x_train, x_test = history.transform(train), history.transform(test)
    comparable = ComparableFeatures(train.loc[train.house_type_name == "Detached"], "2026-06-01")
    details = {}
    for kind, choice in choices.items():
        variant = choice.removeprefix("half_")
        if variant in ("previous", "global"):
            continue
        tpos = np.flatnonzero(train.house_type_name.eq(kind))
        vpos = np.flatnonzero(test.house_type_name.eq(kind))
        tx = enriched(x_train.iloc[tpos], train.iloc[tpos], variant, comparable)
        vx = enriched(x_test.iloc[vpos], test.iloc[vpos], variant, comparable)
        _, details[kind] = train_variant(tx, train.sold_price.iloc[tpos], vx,
            out / "final_models" / slug(kind), "2026-06-01", variant)
        LOG.info("[final] %s: %s train_n=%s", kind, choice, len(tpos))
    write_json(out / "final_models" / "routes.json", {"choices": choices,
        "previous_run": str(previous.resolve()), "cutoff_exclusive": "2026-06-01", "models": details})


def replay(run, data_dir, output):
    run, data_dir, output = Path(run), Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((run / "final_models/routes.json").read_text())
    previous = Path(state["previous_run"])
    validate_archive(previous, data_dir)
    listings, homes = pd.read_parquet(data_dir / "listings.parquet"), pd.read_parquet(data_dir / "test_off_market.parquet")
    history = HistoryFeatures(listings, state["cutoff_exclusive"])
    x = history.transform(homes)
    old_state = json.loads((previous / "final_models/routes.json").read_text())
    models = {identifier: GroupModel.load_bundle(previous / "final_models" / identifier)
              for identifier in set(old_state["choices"].values())}
    price, old_routes = routed_predict(x, homes, old_state["choices"], models)
    old_submission = pd.read_csv(previous / "final_predictions/submission_off_market.csv").set_index("ml_num")
    if not np.array_equal(price, homes.ml_num.map(old_submission.predicted_price).to_numpy()):
        raise ValueError("Previous final model does not reproduce archived predictions")
    global_price = models["global__lightgbm"].predict(x)
    routes = old_routes.astype(object)
    comparable = ComparableFeatures(listings.loc[listings.house_type_name == "Detached"], state["cutoff_exclusive"])
    for kind, choice in state["choices"].items():
        positions = np.flatnonzero(homes.house_type_name.eq(kind))
        variant = choice.removeprefix("half_")
        if variant == "previous":
            specialist = price[positions].copy()
        elif variant == "global":
            specialist = global_price[positions]
        else:
            model = GroupModel.load_bundle(run / "final_models" / slug(kind))
            specialist = model.predict(enriched(x.iloc[positions], homes.iloc[positions], variant, comparable))
        price[positions] = np.round(.5 * specialist + .5 * global_price[positions]) if choice.startswith("half_") else specialist
        routes[positions] = f"{kind}/{choice}"
    result = pd.DataFrame({"ml_num": homes.ml_num, "predicted_price": price})
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    errors = check_submission.check_file(output, homes.ml_num.tolist())
    if errors:
        raise ValueError(errors)
    return result, pd.DataFrame({"ml_num": homes.ml_num, "house_type_name": homes.house_type_name,
                                 "route": routes, "predicted_price": price})


def report(out, choices, details, summaries, comparison):
    lines = ["# 独立屋与半独立/联排组：模型改进结果", "", "日期：2026-10-18（多伦多）。仅 Off-market。", "",
        "## 实验步骤", "", "1. 核对上一轮全部输入、源码、模型和预测哈希及运行库版本，复用已验证的统一/分组基线。",
        "2. 公寓两组预测保持逐套完全一致；仅改进 Detached、Semi-Detached、Freehold Townhouse。",
        "3. 半独立与 Freehold Townhouse 分别训练 LightGBM，其他参数和历史统计口径沿用原方案。",
        "4. 独立屋先增加四个几何比例特征，再在其基础上加入八个相似成交特征，分别消融。",
        "5. 相似成交只查已完成月份之前 365 天的同房型成交；社区内不足五套则回退同房型全市场；按对数室内/土地面积找最多100套候选，再加入房屋样式差异排序，保留最多20套。距离权重不是训练样本时间衰减。",
        "6. 每个新模型和原专用模型分别与统一 LightGBM 做固定 50/50 价格融合；不搜索连续权重。",
        "7. 仅用 2025 年三个开发窗口选择：以原专用/统一模型中开发 MdAPE 较好的作为参考；新方案必须合并 MdAPE 更低、至少两个窗口改善。仅按 MdAPE 排序，完全相同则按候选名排序；MAPE 不参与淘汰或平局判断。",
        "8. 选择写入 selection.json 后冻结，再回测 2026 年2—5月；不依据该段答案改选择或加价校准。",
        "9. 用截至5月数据重训已选新模型，复用旧模型分支；重载后生成隔离提交并独立复核。", "",
        "## 模型选择与后续回测", "", "| 房型 | 选用方案 | 套数 | 原 MdAPE | 新 MdAPE | 原 MAPE | 新 MAPE | 新中位偏差 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    old = {x["house_type_name"]: x for x in summaries["check"]["previous"]["by_type"]}
    new = {x["house_type_name"]: x for x in summaries["check"]["selected"]["by_type"]}
    for kind in TARGET_TYPES:
        s,b=new[kind],old[kind]
        lines.append(f"| {kind} | {choices[kind]} | {s['n']:,} | {b['mdape']:.2f}% | {s['mdape']:.2f}% | {b['mape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% |")
    lines += ["", "previous=上一轮专用模型；global=统一LightGBM；split=拆分房型LightGBM；geometry=独立屋几何特征；comparables=几何＋相似成交；half_=该模型与统一模型各50%。", "",
        "## 原两组与整体结果", "", "| 范围 | 原 MdAPE | 新 MdAPE | 原 MAPE | 新 MAPE |", "|---|---:|---:|---:|---:|"]
    for group in ("detached", "attached_freehold"):
        b=next(x for x in summaries["check"]["previous"]["by_group"] if x["group"]==group)
        s=next(x for x in summaries["check"]["selected"]["by_group"] if x["group"]==group)
        lines.append(f"| {group} | {b['mdape']:.2f}% | {s['mdape']:.2f}% | {b['mape']:.2f}% | {s['mape']:.2f}% |")
    b,s=summaries["check"]["previous"]["overall"],summaries["check"]["selected"]["overall"]
    lines.append(f"| 整体（含固定公寓组） | {b['mdape']:.2f}% | {s['mdape']:.2f}% | {b['mape']:.2f}% | {s['mape']:.2f}% |")
    lines += ["", "## 各候选对照", "", "| 房型 | 候选 | 开发 MdAPE | 后续 MdAPE | 后续 MAPE | 选用 |", "|---|---|---:|---:|---:|---|"]
    for kind in TARGET_TYPES:
        for name,item in details[kind]["candidates"].items():
            row=comparison.loc[(comparison.protocol=="check") & (comparison.house_type_name==kind) & (comparison.candidate==name)].iloc[0]
            lines.append(f"| {kind} | {name} | {item['overall']['mdape']:.2f}% | {row.mdape:.2f}% | {row.mape:.2f}% | {'是' if choices[kind]==name else ''} |")
    lines += ["", "## 解释边界与文件", "", "- 已查看过2026年2—5月误差，也复用了项目开发窗口；本次是累计开发回测，不能称作新的独立测试。",
        "- 开发期改善不保证后续改善；若后续退步，仍报告冻结后的真实结果，不事后更换模型。",
        "- 几何特征单独对照，comparables相对geometry才是增加相似成交的效果；融合与原始预测也分别报告。",
        "- 当前比较L1对数价格目标、固定600轮，不增加训练样本时间衰减，不做直接乘系数校准。",
        "- 基础历史仍查询min(挂牌日,训练截止日)之前；相似成交更保守地排除查询所在整月。整个模型按成交截止日训练，不是逐挂牌日重训仿真。",
        "- 若保留融合/原模型分支，重放依赖上一轮完整存档，manifest记录绝对路径与哈希。",
        "- 6—9月无公开答案，无法报告隐藏测试误差；官方提交文件保持不变，新预测在本运行目录内。",
        "", "逐步复现见本实验README。各候选逐套预测见all_candidate_predictions.csv；所有分组/月份分数见metrics.json；测试、原始文件保护和重载复现见checks.json、verification.json、pytest.log。", ""]
    (out/"改进结果报告.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO.parent/"data")
    parser.add_argument("--previous", type=Path, default=REPO/"experiments/property_groups/artifacts/run_2026-10-17_v1")
    parser.add_argument("--output", type=Path, default=HERE/"artifacts/run_2026-10-18_v1")
    args=parser.parse_args()
    out,previous=args.output.resolve(),args.previous.resolve()
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(out/"run.log")])
    validate_archive(previous,args.data_dir)
    sources=[*HERE.glob("*.py"),* (REPO/"experiments/plan_a").glob("*.py"),*(REPO/"experiments/property_groups").glob("*.py")]
    protected={str(p):sha(p) for p in sources+[args.data_dir/"listings.parquet",args.data_dir/"test_off_market.parquet"]}
    manifest={"started_at_Toronto":now(),"previous_run":str(previous),"previous_manifest_sha256":sha(previous/"manifest.json"),
        "source_and_input_sha256":protected,"params":PARAMS,"comparable_config":CONFIG,
        "new_variants":NEW_VARIANTS,"development_windows":DEV_WINDOWS,"check_window":CHECK_WINDOW,
        "selection_gate":"MdAPE only: lower pooled MdAPE and >=2/3 window MdAPE improvements versus better original/global development anchor; ties by candidate name",
        "status":"running"}
    write_json(out/"manifest.json",manifest)
    snapshot=out/"source_snapshot";snapshot.mkdir()
    for p in sources:shutil.copy2(p,snapshot/str(p.relative_to(REPO)).replace("/","__"))
    listings=pd.read_parquet(args.data_dir/"listings.parquet")
    test=pd.read_parquet(args.data_dir/"test_off_market.parquet")
    for c in ("date_sold","date_listed"):listings[c]=pd.to_datetime(listings[c])
    test["date_listed"]=pd.to_datetime(test.date_listed)
    if listings.ml_num.duplicated().any() or test.ml_num.duplicated().any() or set(listings.ml_num)&set(test.ml_num):raise ValueError("Invalid IDs")
    if not (listings.date_sold<"2026-06-01").all():raise ValueError("Future training labels")
    old_candidates=pd.read_csv(previous/"all_candidate_predictions.csv")
    old_routed=pd.concat([pd.read_csv(previous/f"{p}_routed_predictions.csv") for p in ("development","check")],ignore_index=True)
    dev_parts,dev_previous=[],[]
    for cutoff,end in DEV_WINDOWS:
        rows,old=evaluate(listings,previous,old_candidates,old_routed,out,cutoff,end,"development")
        dev_parts.append(rows);dev_previous.append(old)
    dev=pd.concat(dev_parts,ignore_index=True)
    choices,details=select(dev)
    write_json(out/"selection.json",{"choices":choices,"details":details,"frozen_at_Toronto":now(),"check_labels_used_for_selection":False})
    frozen_sha=sha(out/"selection.json")
    LOG.info("FROZEN CHOICES: %s",choices)
    check,check_previous=evaluate(listings,previous,old_candidates,old_routed,out,*CHECK_WINDOW,"check")
    all_rows=pd.concat([dev,check],ignore_index=True)
    all_rows.to_csv(out/"all_candidate_predictions.csv",index=False)
    comparison=breakdown(all_rows,["protocol","house_type_name","candidate"])
    comparison.to_csv(out/"candidate_scores.csv",index=False)
    summaries={}
    for protocol,rows,old in (("development",dev,pd.concat(dev_previous,ignore_index=True)),("check",check,check_previous)):
        selected=route(rows,choices)
        full=pooled_full(old,selected)
        old.to_csv(out/f"{protocol}_previous_predictions.csv",index=False)
        full.to_csv(out/f"{protocol}_selected_predictions.csv",index=False)
        selected.to_csv(out/f"{protocol}_target_routes.csv",index=False)
        summaries[protocol]={"previous":scores_bundle(old),"selected":scores_bundle(full)}
        apartment=~old.house_type_name.isin(TARGET_TYPES)
        if not np.array_equal(old.loc[apartment,"predicted_price"],full.loc[apartment,"predicted_price"]):raise ValueError("Apartment predictions changed")
    write_json(out/"metrics.json",summaries)
    final_fit(listings,test,previous,out,choices)
    prediction,routes=replay(out,args.data_dir,out/"final_predictions/submission_off_market.csv")
    routes.to_csv(out/"final_predictions/prediction_routes.csv",index=False)
    replayed,_=replay(out,args.data_dir,out/"final_predictions/replay.csv")
    pd.testing.assert_frame_equal(prediction,replayed)
    if sha(out/"selection.json")!=frozen_sha:raise ValueError("Frozen selection changed")
    if not all(sha(p)==digest for p,digest in protected.items()):raise ValueError("Original inputs or sources changed")
    validate_archive(previous,args.data_dir)
    write_json(out/"checks.json",{"source_and_input_hashes_unchanged":True,"previous_run_intact":True,
        "apartment_validation_predictions_unchanged":True,"selection_sha256":frozen_sha,
        "all_new_models_reload_exactly":True,"final_replay_exact":True,"submission_n":len(prediction),"submission_format_errors":[]})
    report(out,choices,details,summaries,comparison)
    manifest.update(status="complete",finished_at_Toronto=now())
    write_json(out/"manifest.json",manifest)
    LOG.info("FINAL previous=%s selected=%s",summaries["check"]["previous"]["overall"],summaries["check"]["selected"]["overall"])


if __name__=="__main__":main()
