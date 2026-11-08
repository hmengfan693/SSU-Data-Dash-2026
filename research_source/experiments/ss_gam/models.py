"""Monthly spatial GAM + time-varying hedonic coefficients, with causal forecasts.

Adapted from DOI 10.1007/s00181-020-01862-y, equations 8--12.
This is an adaptation for unseen houses, not a reproduction of the paper's index.
"""
from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from experiments.plan_a.features import property_features

TYPES = ("Detached", "Semi-Detached", "Freehold Townhouse", "Condo Townhouse")
EXTRA = ("log_area", "area_is_range", "parking_total", "garage",
         "condition_full_renovation", "condition_update", "condition_needs_work",
         "condition_ready", "feature_basement", "feature_view")


def eligible(homes):
    return homes.house_type_name.isin(TYPES)


class HedonicFeatures:
    def __init__(self, extended=False):
        self.extended = extended

    def raw(self, homes):
        h = homes.reset_index(drop=True)
        land = pd.to_numeric(h.lot_front) * pd.to_numeric(h.lot_depth)
        x = pd.DataFrame({"log_land": np.log(land.where(land > 0)),
                          "land_missing": (~(land > 0)).astype(float)})
        for value in (3, 4, 5):
            x[f"bed_{value}"] = (h.bedrooms.ge(5) if value == 5 else h.bedrooms.eq(value)).astype(float)
        for value in (2, 3):
            x[f"bath_{value}"] = (h.washrooms.ge(3) if value == 3 else h.washrooms.eq(value)).astype(float)
        # Detached is the reference type; townhouse land is never invented.
        for kind in TYPES[1:]:
            x[kind] = h.house_type_name.eq(kind).astype(float)
        if self.extended:
            base = property_features(h)
            for name in EXTRA:
                x[name] = base[name].to_numpy()
            for kind in self.styles[1:]:
                x[f"style_{kind}"] = h.house_style.fillna("__MISSING__").eq(kind).astype(float)
        return x

    def fit(self, homes):
        self.styles = sorted(homes.house_style.fillna("__MISSING__").unique())
        x = self.raw(homes)
        self.names = x.columns.tolist()
        self.median = x.median().fillna(0).to_numpy()
        values = np.where(np.isfinite(x), x, self.median)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        return self

    def transform(self, homes):
        x = self.raw(homes)[self.names].to_numpy(float)
        x = np.where(np.isfinite(x), x, self.median)
        return np.column_stack([np.ones(len(x)), (x - self.mean) / self.scale])


def kernel(a, b):
    r = cdist(a, b)
    return r * r * np.log(np.maximum(r, 1e-12))


class SpatialGAM:
    """Reduced-rank thin-plate spline with affine null space and GCV smoothing."""
    def __init__(self, knots=64):
        self.knots_n = knots

    def basis(self, coordinates):
        c = (np.asarray(coordinates) - self.center) / self.scale
        return np.column_stack([c, kernel(c, self.knots) @ self.rotation]) - self.basis_mean

    def fit(self, z, coordinates, y):
        coordinates = np.asarray(coordinates)
        self.center = coordinates.mean(axis=0)
        self.scale = np.maximum(coordinates.std(axis=0), 1e-6)
        c = (coordinates - self.center) / self.scale
        unique = np.unique(c, axis=0)
        if len(unique) < 5:
            raise ValueError("At least five unique coordinates needed")
        rng = np.random.default_rng(42)
        self.knots = unique[rng.choice(len(unique), min(self.knots_n, len(unique)), replace=False)]
        projection = null_space(np.column_stack([np.ones(len(self.knots)), self.knots]).T)
        k = projection.T @ kernel(self.knots, self.knots) @ projection
        eig, vectors = np.linalg.eigh((k + k.T) / 2)
        keep = eig > max(eig.max() * 1e-10, 1e-10)
        self.rotation = projection @ (vectors[:, keep] / np.sqrt(eig[keep]))
        self.basis_mean = np.zeros(2 + keep.sum())
        b = self.basis(coordinates)
        self.basis_mean = b.mean(axis=0)
        b -= self.basis_mean
        design = np.column_stack([z, b])
        gram, rhs = design.T @ design, design.T @ y
        n, p = len(y), z.shape[1]
        best = None
        for smoothing in (0.01, 0.1, 1., 10., 100., 1000.):
            penalty = np.r_[np.repeat(1e-7, p + 2), np.repeat(smoothing, b.shape[1] - 2)]
            penalized = gram + np.diag(penalty)
            coef = np.linalg.solve(penalized, rhs)
            residual = y - design @ coef
            df = np.trace(np.linalg.solve(penalized, gram))
            gcv = np.mean(residual ** 2) / max(1 - df / n, 0.05) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, coef, residual, df, smoothing)
        _, coef, residual, df, self.smoothing = best
        self.hedonic_coef, self.spatial_coef = coef[:p], coef[p:]
        # Diagonal spatial uncertainty approximation; no fabricated cross-house covariances.
        self.v = max(float(residual @ residual / max(n - df, 1)), 1e-6)
        self.df = float(df)
        return self

    def location(self, coordinates):
        return self.basis(coordinates) @ self.spatial_coef

    def predict(self, z, coordinates):
        return z @ self.hedonic_coef + self.location(coordinates)


@dataclass
class Period:
    month: str
    n: int
    xx: np.ndarray
    xy: np.ndarray
    prior_xx: np.ndarray
    prior_xy: np.ndarray
    cross: np.ndarray
    yy: float
    v: float


def period_stats(month, x, previous_x, y, v):
    return Period(str(month), len(y), x.T @ x, x.T @ y,
                  previous_x.T @ previous_x, previous_x.T @ y,
                  x.T @ previous_x, float(y @ y), float(v))


def filter_period(a, p, q, h, period):
    """Woodbury evaluation of paper eqs 11--12, using X_previous in innovation.

    R = H + V = (h + period.v) I. Never build an N by N matrix.
    """
    prior_p = p + np.diag(q)
    precision = np.linalg.inv(prior_p)
    r = h + period.v
    posterior_p = np.linalg.inv(precision + period.xx / r)
    innovation_rhs = period.xy - period.cross @ a
    updated = a + posterior_p @ innovation_rhs / r
    innovation_sq = period.yy - 2 * a @ period.prior_xy + a @ period.prior_xx @ a
    logdet = period.n * np.log(r) + np.linalg.slogdet(prior_p)[1] + np.linalg.slogdet(precision + period.xx / r)[1]
    quadratic = innovation_sq / r - innovation_rhs @ posterior_p @ innovation_rhs / r ** 2
    nll = .5 * (period.n * np.log(2 * np.pi) + logdet + quadratic)
    return updated, (posterior_p + posterior_p.T) / 2, float(nll)


class SSGAM:
    def __init__(self, extended=False, knots=64):
        self.extended, self.knots = extended, knots

    def fit(self, homes, cutoff):
        cutoff = pd.Timestamp(cutoff)
        if not eligible(homes).all() or not (homes.date_sold < cutoff).all():
            raise ValueError("Training contains apartment or sale at/after cutoff")
        if homes.ml_num.duplicated().any() or not (homes.sold_price > 0).all():
            raise ValueError("Invalid training IDs/prices")
        self.cutoff = cutoff
        self.train_ids = homes.ml_num.tolist()
        self.max_training_sale = str(homes.date_sold.max())
        self.features = HedonicFeatures(self.extended).fit(homes)
        z = self.features.transform(homes)
        y = np.log(homes.sold_price.to_numpy())
        coordinates = homes[["longitude", "latitude"]].to_numpy()
        months = homes.date_sold.dt.to_period("M")
        expected = pd.period_range(months.min(), cutoff.to_period("M") - 1, freq="M")
        if set(months.unique()) != set(expected):
            raise ValueError("Missing training month; explicit empty-period handling required")
        periods, self.gams, self.period_audit = [], {}, []
        previous = None
        for month in expected:
            idx = np.flatnonzero(months.eq(month))
            zz, cc, yy = z[idx], coordinates[idx], y[idx]
            gam = SpatialGAM(self.knots).fit(zz, cc, yy)
            x = np.column_stack([zz, gam.location(cc)])
            previous_x = np.column_stack([zz, (previous or gam).location(cc)])
            periods.append(period_stats(month, x, previous_x, yy, gam.v))
            self.gams[str(month)] = gam
            self.period_audit.append({"month": str(month), "n": len(idx), "v": gam.v,
                                      "smoothing": gam.smoothing, "effective_df": gam.df})
            previous = gam
        self.last_month = expected[-1]
        # First month supplies initialization only. Burn-in is excluded from likelihood.
        first = self.gams[str(expected[0])]
        self.initial_a = np.r_[first.hedonic_coef, 1.]
        self.initial_p = np.eye(len(self.initial_a)) * .05
        self.burn_in = min(12, max(1, len(periods) // 3))

        def run(log_variances, record=False):
            variance = np.exp(log_variances)
            h, q_mu, q_beta, q_gamma = variance
            q = np.r_[q_mu, np.repeat(q_beta, len(self.initial_a) - 2), q_gamma]
            a, p, total, states = self.initial_a.copy(), self.initial_p.copy(), 0., []
            states.append({"month": periods[0].month, "state": a.tolist()})
            for i, period in enumerate(periods[1:], start=1):
                a, p, nll = filter_period(a, p, q, h, period)
                if not np.isfinite(nll) or not np.isfinite(a).all():
                    return 1e100 if not record else None
                if i >= self.burn_in:
                    total += nll
                if record:
                    states.append({"month": period.month, "state": a.tolist()})
            return (a, p, states, q) if record else total

        # rho=1 is a permitted paper specification. Random walk means retain the
        # final location loading for unseen future months; do not fade geography.
        result = minimize(run, np.log([.002, .0001, .00001, .001]), method="L-BFGS-B",
                          bounds=[(-16, -1), (-18, -3), (-20, -5), (-18, -1)],
                          options={"maxiter": 100, "ftol": 1e-8})
        self.state, self.covariance, self.states, self.q = run(result.x, record=True)
        self.variances = dict(zip(("measurement", "trend", "hedonic", "location"), np.exp(result.x).tolist()))
        self.optimization = {"success": bool(result.success), "message": str(result.message),
                             "iterations": int(result.nit), "nll": float(result.fun), "rho": 1.}
        self.state_names = ["intercept", *self.features.names, "location_loading"]
        return self

    def predict(self, homes, kind="ss_gam"):
        # Explicit whitelist transform: sold_price/date_sold/list_price cannot affect forecasts.
        if not eligible(homes).all():
            raise ValueError("Apartment/unknown type passed to non-apartment model")
        z = self.features.transform(homes)
        coordinates = homes[["longitude", "latitude"]].to_numpy()
        gam = self.gams[str(self.last_month)]
        if kind == "gam":
            log_price = gam.predict(z, coordinates)
        elif kind == "ss_gam":
            log_price = np.column_stack([z, gam.location(coordinates)]) @ self.state
        else:
            raise ValueError(kind)
        if not np.isfinite(log_price).all() or (np.abs(log_price) > 40).any():
            raise ValueError("Unstable log-price forecast")
        # exp(log-price) is a conditional median; do not apply mean bias correction.
        return np.exp(log_price)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)
