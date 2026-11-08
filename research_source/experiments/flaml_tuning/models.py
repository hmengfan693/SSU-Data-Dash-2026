"""Search capacity and regularisation while preserving features and Log-L1."""
# 本文件提供调参公共工具：定义参数范围、创建模型、计算误差、检查时间切分和替换房型预测。
# 这里不生成特征、不拆分数据，也不执行搜索；训练流程由 run_experiment.py 控制。

# NumPy 用于参数近似比较、数组检查和误差统计；底层训练由 OfficialLightGBM 完成。
import numpy as np

from experiments.lightgbm_official.models import OfficialLightGBM

# 旧模型的七项参数，也是搜索的起点：
# n_estimators：树数量；num_leaves：单棵树叶子上限；learning_rate：每棵树的更新步长；
# min_child_samples：每个叶子的最少训练样本；reg_alpha/reg_lambda：L1/L2 正则强度；
# colsample_bytree：每棵树可用特征的采样比例。正则和叶子限制用于约束模型复杂度。
BASE_CONFIG = dict(n_estimators=600, num_leaves=31, learning_rate=.05,
                   min_child_samples=40, reg_alpha=0., reg_lambda=5.,
                   colsample_bytree=1.)
# 搜索范围的普通字典表示，用于写入实验协议；实际抽样规则见下面的 search_space()。
SEARCH_SPEC = dict(n_estimators=[100, 1800], num_leaves=[8, 96],
                   learning_rate=[.01, .15], min_child_samples=[10, 160],
                   reg_alpha=[0., 5.], reg_lambda=[.01, 30.],
                   colsample_bytree=[.65, 1.])


def search_space():
    # 将可读范围转换为 FLAML 分布。lograndint/loguniform 在对数尺度搜索，
    # 让较小值也有充分探索机会；uniform 在线性尺度搜索。
    # 整数分布的上界不包含在内，因此写 1801/97/161 对应最大值 1800/96/160。
    from flaml import tune
    return dict(n_estimators=tune.lograndint(100, 1801),
                num_leaves=tune.lograndint(8, 97),
                learning_rate=tune.loguniform(.01, .15),
                min_child_samples=tune.lograndint(10, 161),
                reg_alpha=tune.uniform(0., 5.), reg_lambda=tune.loguniform(.01, 30.),
                colsample_bytree=tune.uniform(.65, 1.))


def canonical_config(config):
    # 将树数量、叶子数、最少样本数转为 int，其余转为 float，便于训练与保存。
    # 对极接近旧参数的值恢复精确旧值，避免对数空间转换的微小浮点误差影响基线核对。
    # CFO maps log-space points back to floats; preserve the exact baseline.
    result = {key: (int(config[key]) if isinstance(value, int) else float(config[key]))
              for key, value in BASE_CONFIG.items()}
    if all(np.isclose(result[key], value, rtol=1e-13, atol=1e-13)
           for key, value in BASE_CONFIG.items()):
        return dict(BASE_CONFIG)
    return result


def make_model(config, threads=6):
    # 只接受预先声明的七个参数，防止搜索器意外改变特征、目标函数等实验条件。
    # 将 sklearn 风格参数名映射到 LightGBM 原生 API 名称；树数量传给训练轮数。
    # 目标函数仍来自封装：对 log(成交价) 优化 L1；预测时取 exp 并四舍五入回价格。
    if set(config) != set(BASE_CONFIG):
        raise ValueError('Unexpected tuning parameters')
    return OfficialLightGBM(num_boost_round=int(config['n_estimators']),
        num_leaves=int(config['num_leaves']), learning_rate=float(config['learning_rate']),
        min_data_in_leaf=int(config['min_child_samples']), lambda_l1=float(config['reg_alpha']),
        lambda_l2=float(config['reg_lambda']), feature_fraction=float(config['colsample_bytree']),
        num_threads=threads)


def price_metrics(truth, prediction):
    # 真实价与预测价必须是非空、等长的一维正数数组，且没有 NaN/无穷值。
    # 本函数只能检查形状和数值；房屋 ID 的对应关系由调用方先核对。
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if (truth.ndim != 1 or truth.size == 0 or truth.shape != prediction.shape
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()
            or not (truth > 0).all() or not (prediction > 0).all()):
        raise ValueError('Prices must be aligned, positive and finite')
    # 每套房的绝对百分比误差 APE = |预测价 / 真实价 - 1| × 100。
    # MdAPE 取这些误差的中位数（主要选择指标），MAPE 取平均数；都返回百分数。
    error = np.abs(prediction / truth - 1) * 100
    return dict(n=len(truth), mdape=float(np.median(error)), mape=float(np.mean(error)))


def validate_window(train, valid, cutoff, end):
    # 时间切分检查：训练/验证都非空，各自 ID 唯一、彼此不重叠；
    # 训练成交日严格早于 cutoff，验证成交日在 [cutoff, end)，防止未来成交标签进入训练。
    import pandas as pd
    if (train.empty or valid.empty or train.ml_num.duplicated().any()
            or valid.ml_num.duplicated().any() or set(train.ml_num) & set(valid.ml_num)
            or not (pd.to_datetime(train.date_sold) < pd.Timestamp(cutoff)).all()
            or not pd.to_datetime(valid.date_sold).between(pd.Timestamp(cutoff),
                pd.Timestamp(end), inclusive='left').all()):
        raise ValueError('Invalid chronological window or overlapping IDs')


def replace_branch(predictions, homes, values):
    """Require exactly the predeclared branch IDs, preserving the full-table order."""
    # 只替换一个预先声明房型的预测，其他房型沿用原结果，原输入表保持不变。
    # 先检查该批房屋只含一个房型、ID 不重复，且预测数量等于房屋数量。
    result = predictions.copy()
    if (homes.empty or homes.house_type_name.nunique() != 1 or homes.ml_num.duplicated().any()
            or result.ml_num.duplicated().any() or len(values) != len(homes)):
        raise ValueError('Invalid branch prediction alignment')
    # 两表中这个房型的 ID 集合必须完全相同，既不能漏房屋，也不能多替换房屋。
    mask = result.house_type_name.eq(homes.house_type_name.iloc[0])
    if set(result.loc[mask, 'ml_num']) != set(homes.ml_num):
        raise ValueError('Branch ID coverage mismatch')
    # 用全 1 的占位真值复用正数/有限值检查；这里不把返回值作为真实模型成绩。
    # 再通过 ID 映射回填预测，避免依赖两表碰巧具有相同的行顺序。
    price_metrics(np.ones(len(values)), values)
    result.loc[mask, 'predicted_price'] = result.loc[mask, 'ml_num'].map(
        dict(zip(homes.ml_num, values))).to_numpy()
    return result
