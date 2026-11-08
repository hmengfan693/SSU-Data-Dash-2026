"""Large trees; early stopping is permitted only in the earlier inner windows."""
import lightgbm as lgb
import numpy as np

from experiments.flaml_tuning.models import price_metrics
from experiments.lightgbm_official.models import OfficialLightGBM
from experiments.plan_a.features import CATEGORICAL

EXTRA_DEFAULTS = dict(max_depth=-1, subsample=1., max_bin=255)
SEARCH_SPEC = dict(num_leaves=[127, 255, 511], max_depth=[12, 16, 24, -1],
    n_estimators=8000, learning_rate=[.008, .08], min_child_samples=[10, 160],
    reg_alpha=[0., 5.], reg_lambda=[.1, 60.], colsample_bytree=[.7, 1.],
    subsample=[.7, 1.], max_bin=[255, 511], early_stopping_rounds=120)
KEYS = {'n_estimators', 'num_leaves', 'max_depth', 'learning_rate',
        'min_child_samples', 'reg_alpha', 'reg_lambda', 'colsample_bytree',
        'subsample', 'max_bin'}


def normalized(config):
    config = dict(EXTRA_DEFAULTS, **config)
    if set(config) != KEYS:
        raise ValueError('Unknown capacity-search parameter')
    integer = {'n_estimators', 'num_leaves', 'max_depth', 'min_child_samples', 'max_bin'}
    return {key: int(value) if key in integer else float(value) for key, value in config.items()}


def make_model(config, threads=6):
    c = normalized(config)
    if (c['n_estimators'] < 1 or c['num_leaves'] < 2 or c['max_depth'] == 0
            or c['min_child_samples'] < 1 or not 0 < c['subsample'] <= 1
            or not 0 < c['colsample_bytree'] <= 1):
        raise ValueError('Invalid capacity configuration')
    return ComplexLightGBM(num_boost_round=c['n_estimators'],
        num_leaves=c['num_leaves'], max_depth=c['max_depth'], learning_rate=c['learning_rate'],
        min_data_in_leaf=c['min_child_samples'], lambda_l1=c['reg_alpha'],
        lambda_l2=c['reg_lambda'], feature_fraction=c['colsample_bytree'],
        bagging_fraction=c['subsample'], bagging_freq=1 if c['subsample'] < 1 else 0,
        max_bin=c['max_bin'], num_threads=threads)


def search_space():
    from flaml import tune
    return dict(n_estimators=8000, num_leaves=tune.choice([127, 255, 511]),
        max_depth=tune.choice([12, 16, 24, -1]), learning_rate=tune.loguniform(.008, .08),
        min_child_samples=tune.lograndint(10, 161), reg_alpha=tune.uniform(0., 5.),
        reg_lambda=tune.loguniform(.1, 60.), colsample_bytree=tune.uniform(.7, 1.),
        subsample=tune.uniform(.7, 1.), max_bin=tune.choice([255, 511]))


def starting_points():
    base = dict(n_estimators=8000, num_leaves=127, max_depth=12, learning_rate=.03,
        min_child_samples=40, reg_alpha=0., reg_lambda=10., colsample_bytree=.9,
        subsample=.9, max_bin=255)
    return [base, dict(base, num_leaves=255, max_depth=16, learning_rate=.02,
                      reg_lambda=20., subsample=.85, max_bin=511),
        dict(base, num_leaves=511, max_depth=24, learning_rate=.02,
             min_child_samples=20, reg_lambda=30., colsample_bytree=.85,
             subsample=.85, max_bin=511),
        dict(base, max_depth=-1, learning_rate=.01, min_child_samples=20,
             reg_lambda=5., colsample_bytree=1., subsample=1.)]


def original_price_mdape(log_prediction, dataset):
    truth = np.exp(dataset.get_label())
    prediction = np.round(np.exp(log_prediction))
    return 'mdape', price_metrics(truth, prediction)['mdape'], False


class ComplexLightGBM(OfficialLightGBM):
    def fit_inner(self, x, prices, valid_x, valid_prices, patience=120):
        """Only call with validated past-to-future inner windows; never outer labels."""
        # Reuse the existing input contract before creating validation datasets.
        prices = np.asarray(prices, float)
        valid_prices = np.asarray(valid_prices, float)
        price_metrics(prices, prices)
        price_metrics(valid_prices, valid_prices)
        if len(x) != len(prices) or len(valid_x) != len(valid_prices) or list(x) != list(valid_x):
            raise ValueError('Inner data and target alignment mismatch')
        self.feature_names = list(x)
        self.categories = {c: sorted(x[c].dropna().unique().tolist()) for c in CATEGORICAL if c in x}
        train = lgb.Dataset(self.encode(x), label=np.log(prices),
                            categorical_feature=list(self.categories))
        valid = lgb.Dataset(self.encode(valid_x), label=np.log(valid_prices),
                            reference=train, categorical_feature=list(self.categories))
        # Float64 truth in the closure avoids scoring against rounded float32
        # labels held internally by LightGBM's Dataset.
        def evaluate(log_prediction, _):
            return 'mdape', price_metrics(valid_prices, np.round(np.exp(log_prediction)))['mdape'], False
        params = dict(self.params, metric='None')
        self.booster = lgb.train(params, train, num_boost_round=self.num_boost_round,
            valid_sets=[valid], feval=evaluate,
            callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
        self.selected_rounds = int(self.booster.best_iteration)
        return self


def complexity(model):
    """Measure actual fitted depth and leaf count, not configured upper bounds."""
    trees = model.booster.dump_model()['tree_info']
    depths, leaves = [], []
    for tree in trees:
        stack = [(tree['tree_structure'], 0)]
        depth = 0
        count = 0
        while stack:
            node, d = stack.pop()
            depth = max(depth, d)
            if 'leaf_value' in node:
                count += 1
            else:
                stack.extend([(node['left_child'], d+1), (node['right_child'], d+1)])
        depths.append(depth)
        leaves.append(count)
    return dict(trees=len(trees), max_actual_depth=max(depths),
        median_actual_depth=float(np.median(depths)), total_leaves=sum(leaves),
        mean_leaves_per_tree=float(np.mean(leaves)))
