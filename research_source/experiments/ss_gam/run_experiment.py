"""Run isolated paper-inspired non-apartment experiments against archived paired baseline."""
import argparse
import importlib.metadata
import json
import logging
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import check_submission
import data
from experiments.plan_a.run_experiment import metrics
from experiments.property_groups.run_experiment import now, sha, split_window, write_json
from experiments.property_groups.selection import DEV_WINDOWS, CHECK_WINDOW
from .models import SSGAM, TYPES, eligible

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINE = REPO / 'experiments/fixed_four_groups/artifacts/run_2026-10-18_v1'
LOG = logging.getLogger(__name__)
PRIMARY = 'extended_ss_gam'


def score_table(rows):
    result = []
    for (protocol, candidate), frame in rows.groupby(['protocol', 'candidate']):
        for dimension in ('overall', 'house_type_name', 'month', 'cutoff_exclusive'):
            groups = [('all', frame)] if dimension == 'overall' else frame.groupby(dimension)
            for label, current in groups:
                result.append({'protocol': protocol, 'candidate': candidate,
                               'dimension': dimension, 'slice': label, **metrics(current)})
    return pd.DataFrame(result)


def baseline_rows(homes, cutoff, protocol):
    folder = BASELINE / protocol / cutoff[:7]
    parts = [pd.read_csv(folder / route / 'predictions.csv') for route in (
        'detached__lightgbm', 'attached_freehold__lightgbm', 'condo_townhouse__lightgbm')]
    archived = pd.concat(parts).set_index('ml_num').loc[homes.ml_num].reset_index()
    for column in ('sold_price', 'house_type_name', 'community'):
        if not np.array_equal(archived[column], homes[column]):
            raise ValueError(f'Archived baseline mismatch: {column}')
    if archived.ml_num.duplicated().any() or not archived.cutoff_exclusive.eq(cutoff).all():
        raise ValueError('Invalid archived baseline IDs/cutoff')
    return archived.assign(candidate='existing_group_lightgbm')


def fit_model(train, hidden, cutoff, folder, extended):
    started = time.monotonic()
    model = SSGAM(extended=extended, knots=64).fit(train, cutoff)
    if not model.optimization['success']:
        raise RuntimeError(f'Maximum likelihood failed: {model.optimization}')
    model.save(folder / 'model.pkl')
    loaded = SSGAM.load(folder / 'model.pkl')
    for kind in ('ss_gam', 'gam'):
        np.testing.assert_array_equal(model.predict(hidden, kind), loaded.predict(hidden, kind))
    pd.DataFrame(model.period_audit).to_csv(folder / 'monthly_gam.csv', index=False)
    pd.DataFrame([{'month': row['month'], **dict(zip(model.state_names, row['state']))}
                  for row in model.states]).to_csv(folder / 'filtered_states.csv', index=False)
    pd.DataFrame({'ml_num': train.ml_num}).to_csv(folder / 'train_ids.csv', index=False)
    write_json(folder / 'audit.json', {
        'cutoff_exclusive': cutoff, 'training_n': len(train), 'max_training_sale': model.max_training_sale,
        'features': model.state_names, 'extended': extended, 'knots': 64,
        'variances': model.variances, 'optimization': model.optimization,
        'burn_in_months': model.burn_in, 'reload_exact': True,
        'seconds': time.monotonic()-started,
        'future_labels_used': False, 'forecast_mean': 'last filtered state and last observed location surface',
        'residual_spatial_correction': 'zero: new houses, diagonal V, no same-period outcomes',
    })
    LOG.info('fit %s extended=%s n=%d %.1fs', cutoff, extended, len(train), time.monotonic()-started)
    return loaded


def report(out, scores, paired, checks):
    check = scores.loc[(scores.protocol == 'check') & (scores.dimension == 'overall')].set_index('candidate')
    delta = check.loc[PRIMARY, 'mdape'] - check.loc['existing_group_lightgbm', 'mdape']
    lines = ['# 非公寓房型：SS+GAM 论文方法尝试', '', f'完成时间：{now()}。范围：Off-market。', '',
        f"固定回测共 {int(check.loc[PRIMARY, 'n']):,} 套。扩展 SS+GAM 的 MdAPE 为 {check.loc[PRIMARY, 'mdape']:.3f}%，"
        f"现有分组 LightGBM 为 {check.loc['existing_group_lightgbm', 'mdape']:.3f}%，差值 {delta:+.3f} 个百分点。",
        '本轮结果仅用于研究。实验预测单独保存，既有提交和模型未替换。', '',
        '## 论文与任务的对应', '',
        '来源：Hill, Rambaldi & Scholz (2021), Higher frequency hedonic property price indices: a state-space approach. '
        'https://doi.org/10.1007/s00181-020-01862-y （2020 年在线发表）。', '',
        '- 实现每期半参数 hedonic GAM：log(price) = 截距 + 房屋特征 + 经度纬度薄板样条。',
        '- 采用论文式 (9)—(12) 的状态空间更新：截距、特征系数、空间载荷随时间变化；创新项使用前期空间曲面。',
        '- 按论文允许的随机游走特例固定 rho=1；训练期最大似然估计四个方差，前 12 个月预热。不是复制论文估计的 rho=0.4。',
        '- 空间样条选 64 个结点、GCV 选平滑强度；这是 Python 低秩薄板实现，并非原文 R/mgcv 的 k=600。',
        '- 原论文允许非对角 V；本实现用每期 GAM 残差方差构造 V=vI，并估计 H=hI。只能区分方差分量的近似作用，未验证两者强识别。',
        '- 原论文研究当期价格指数；这里改为未知未来房屋估值，按月而非按周。随机游走的多步条件均值保持最后状态与最后曲面，不外推增长率。',
        '- 未来无当期成交答案，且采用对角 V，因此式 (13) 的跨房屋残差修正为零。没有计算双重插补 Törnqvist 指数或重复交易指标。',
        '- 四类房型共同训练，保留房型虚拟变量；原文排除联排，本轮明确将其作为外推实验。Condo Townhouse 土地缺失以训练中位数及缺失指示处理，不能解释为实际土地估计。', '',
        '## 两种预先指定输入', '',
        '- core：log(地块宽×深)、土地缺失、3/4/5+ 卧室、2/3+ 浴室及房型虚拟变量。土地从平方英尺换单位只改变截距。',
        '- extended（预先指定的主要候选）：在 core 上加室内面积、面积区间、停车、车库、房屋样式及已有六个文本指标。',
        '- 两种输入各报告 SS+GAM 和末期独立 GAM，后者用于检查时间更新的增量；不根据 2026 年回测选择主要候选或混合权重。',
        '- 坐标直接来自合成数据，不使用真实地图；list_price、成交后字段、第三方估价都不进入预测。', '',
        '## 验证口径', '',
        '- 三个 2025 年四个月开发窗口，加截至 2026-01-31 训练、固定预测 2026 年 2—5 月的回测；四个月期间不更新答案。',
        '- 特征编码和参数仅用截止日前训练数据拟合；未来预测不接收 date_sold。结果按实际成交月汇总仅用于评分。',
        '- 模型按成交截止日训练，估值时点是截止日，不是逐套挂牌当日仿真；挂牌早于截止日的房屋也使用截止日信息。',
        '- 这些窗口已经参与项目开发，属于累计开发评估，不能称为全新的独立测试。',
        '- 对照直接读取既有固定四组实验的逐套预测，逐个 ID 核对原始标签、房型和截止日；表中只统计四种非公寓房型。', '',
        '## 2026 年 2—5 月整体', '', '| 候选 | 套数 | MdAPE | MAPE | 中位偏差 |', '|---|---:|---:|---:|---:|']
    for candidate, row in check.iterrows():
        lines.append(f'| {candidate} | {int(row.n):,} | {row.mdape:.3f}% | {row.mape:.3f}% | {row.median_signed_error:+.3f}% |')
    lines += ['', '## 四种房型：主要候选与现有模型', '',
              '| 房型 | 套数 | 扩展 SS+GAM MdAPE | 现有分组 MdAPE | 差值（百分点） |', '|---|---:|---:|---:|---:|']
    for row in paired.loc[(paired.protocol == 'check') & (paired.dimension == 'house_type_name') & (paired.candidate == PRIMARY)].itertuples():
        lines.append(f'| {row.slice} | {row.n:,} | {row.mdape:.3f}% | {row.baseline_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    lines += ['', '## 跨窗口', '', '| 截止日 | 套数 | 扩展 SS+GAM MdAPE | 现有分组 MdAPE | 差值（百分点） |', '|---|---:|---:|---:|---:|']
    for row in paired.loc[(paired.dimension == 'cutoff_exclusive') & (paired.candidate == PRIMARY)].itertuples():
        lines.append(f'| {row.slice} | {row.n:,} | {row.mdape:.3f}% | {row.baseline_mdape:.3f}% | {row.delta_mdape_pp:+.3f} |')
    lines += ['', '## 如何解读', '',
        ('主要候选在本次固定回测中更差，当前结果不支持用它替换现有分组模型。' if delta >= 0 else
         '主要候选在本次固定回测中更好，但是否跨窗口稳定需结合上表；尚未证明未来隐藏测试收益。'),
        '论文的优势证据针对高频价格变动与指数，不能直接推出四个月未来单套估值会胜过树模型。',
        'core 与 extended 的差异包含信息量变化；SS+GAM 与 GAM 是同输入对照，但时间平滑与最后一期系数不同。',
        '合成社区可能具有非平滑边界；连续空间样条可能无法像社区标签及历史特征那样完整表达位置。',
        '未进行显著性检验；误差较低也不能据此宣称统计显著。', '',
        '## 产物', '',
        f"- 最终四类非公寓预测：{checks['non_apartment_n']:,} 行，non_apartment_predictions.csv。",
        '- experimental_submission_off_market.csv：共 3,974 行；非公寓采用预先指定 extended_ss_gam，公寓保留既有固定四组预测。仅为实验整表，非推荐替换。',
        '- all_predictions.csv：所有窗口全部候选逐套预测；scores.csv 与 paired_comparison.csv：整体、房型、逐月和窗口。',
        '- 每个模型保存 model.pkl、训练 IDs、每月 GAM 方差/平滑强度、滤波系数、MLE 参数和截止日审计。',
        '- manifest.json/source_snapshot：原始数据、既有对照、源码哈希和环境；verification.json：独立复算及保存模型重放。',
        '- 未知的 2026 年 6—9 月成交价格不在数据中，因此不能报告隐藏测试误差。', '']
    (out/'结果报告.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(out/'run.log')])
    listings, test = data.load('listings'), data.load('off_market')
    write_json(out/'protocol.json', {'created': now(), 'primary_candidate': PRIMARY, 'types': TYPES,
        'configs': ['core', 'extended'], 'methods': ['ss_gam', 'gam'], 'rho': 1., 'knots': 64,
        'windows': [*DEV_WINDOWS, CHECK_WINDOW], 'test_labels_used_for_selection': False,
        'promote_to_existing_submission': False})
    frames = []
    for cutoff, end in (*DEV_WINDOWS, CHECK_WINDOW):
        protocol = 'check' if (cutoff, end) == CHECK_WINDOW else 'development'
        train, homes = split_window(listings, cutoff, end)
        train, homes = train.loc[eligible(train)].reset_index(drop=True), homes.loc[eligible(homes)].reset_index(drop=True)
        hidden = data.hide_answers(homes, 'off_market')
        frames.append(baseline_rows(homes, cutoff, protocol))
        for config in ('core', 'extended'):
            folder = out/protocol/cutoff[:7]/config
            model = fit_model(train, hidden, cutoff, folder, config == 'extended')
            for kind in ('ss_gam', 'gam'):
                rows = homes[['ml_num', 'sold_price', 'house_type_name', 'community']].copy()
                rows['month'] = homes.date_sold.dt.strftime('%Y-%m')
                rows['cutoff_exclusive'], rows['protocol'] = cutoff, protocol
                rows['candidate'] = config+'_'+kind
                rows['predicted_price'] = model.predict(hidden, kind)
                frames.append(rows)
                LOG.info('%s %s %s', cutoff, rows.candidate.iloc[0], metrics(rows))
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(out/'all_predictions.csv', index=False)
    scores = score_table(rows)
    scores.to_csv(out/'scores.csv', index=False)
    base = scores.loc[scores.candidate.eq('existing_group_lightgbm')].drop(columns='candidate')
    paired = scores.loc[~scores.candidate.eq('existing_group_lightgbm')].merge(
        base[['protocol','dimension','slice','mdape','mape']], on=['protocol','dimension','slice'],
        suffixes=('', '_baseline'), validate='many_to_one').rename(columns={'mdape_baseline':'baseline_mdape','mape_baseline':'baseline_mape'})
    paired['delta_mdape_pp'] = paired.mdape - paired.baseline_mdape
    paired.to_csv(out/'paired_comparison.csv', index=False)
    # Always emit the predeclared candidate, even if it loses. No post-check selection.
    train = listings.loc[eligible(listings)].sort_values(['date_sold','ml_num']).reset_index(drop=True)
    hidden = data.hide_answers(test.loc[eligible(test)].reset_index(drop=True), 'off_market')
    for config in ('core', 'extended'):
        model = fit_model(train, hidden, '2026-06-01', out/'final_models'/config, config == 'extended')
        if config == 'extended':
            final = hidden[['ml_num','house_type_name']].copy()
            final['predicted_price'] = model.predict(hidden)
            final.to_csv(out/'non_apartment_predictions.csv', index=False)
    reference_file = BASELINE/'final_predictions/submission_off_market.csv'
    submission = pd.read_csv(reference_file).set_index('ml_num').loc[test.ml_num].reset_index()
    replacement = final.set_index('ml_num').predicted_price
    mask = submission.ml_num.isin(replacement.index)
    submission.loc[mask, 'predicted_price'] = submission.loc[mask,'ml_num'].map(replacement)
    submission_file = out/'experimental_submission_off_market.csv'
    submission.to_csv(submission_file, index=False)
    errors = check_submission.check_file(submission_file, test.ml_num.tolist())
    if errors:
        raise ValueError(errors)
    sources = [Path(__file__), HERE/'models.py', HERE/'__init__.py', HERE/'verify_results.py',
               HERE/'README.md', REPO/'tests/test_ss_gam.py', REPO/'experiments/plan_a/features.py',
               REPO/'experiments/property_groups/run_experiment.py', REPO/'experiments/property_groups/selection.py',
               REPO/'experiments/plan_a/run_experiment.py', REPO/'data.py', REPO/'check_submission.py']
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshot/str(source.relative_to(REPO)).replace('/', '__'))
    baselines = [reference_file, BASELINE/'manifest.json']
    for cutoff, _ in (*DEV_WINDOWS, CHECK_WINDOW):
        protocol = 'check' if cutoff == CHECK_WINDOW[0] else 'development'
        baselines.extend((BASELINE/protocol/cutoff[:7]/route/'predictions.csv') for route in (
            'detached__lightgbm', 'attached_freehold__lightgbm', 'condo_townhouse__lightgbm'))
    write_json(out/'manifest.json', {'created': now(), 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn')},
        'sources': {str(p.relative_to(REPO)):sha(p) for p in sources},
        'data': {str(data.find_file(name).resolve()):sha(data.find_file(name)) for name in ('listings','off_market')},
        'baseline_files': {str(p.relative_to(REPO)):sha(p) for p in baselines},
        'paper_url':'https://doi.org/10.1007/s00181-020-01862-y'})
    checks = {'submission_n':len(submission), 'non_apartment_n':len(final), 'format_errors':errors,
              'all_model_reload_exact': True, 'primary_candidate': PRIMARY}
    write_json(out/'checks.json', checks)
    report(out, scores, paired, checks)
    LOG.info('Completed %s', out)


if __name__ == '__main__':
    main()
