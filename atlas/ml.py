"""Forecasting (gradient boosting vs seasonal-naive baseline) and anomaly detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

LAGS = [1, 2, 3, 24, 48, 168]


def _features(s: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    for l in LAGS:
        X[f"lag{l}"] = s.shift(l)
    X["hour"] = s.index.hour
    X["dow"] = s.index.dayofweek
    X["roll24"] = s.shift(1).rolling(24).mean()
    return X


def _mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-6)))


class Forecaster:
    """Direct recursive forecaster on total cluster CPU demand."""

    def __init__(self):
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        self.series: pd.Series | None = None
        self.metrics: dict = {}

    def fit(self, series: pd.Series, holdout_h: int = 24 * 7) -> dict:
        self.series = series.asfreq("h")
        X = _features(self.series).dropna()
        y = self.series.loc[X.index]
        Xtr, ytr, Xte, yte = X[:-holdout_h], y[:-holdout_h], X[-holdout_h:], y[-holdout_h:]
        self.model.fit(Xtr, ytr)
        pred = self.model.predict(Xte)
        naive = self.series.shift(168).loc[Xte.index]
        self.metrics = {"gbm_mape": round(_mape(yte, pred), 4),
                        "seasonal_naive_mape": round(_mape(yte, naive), 4),
                        "holdout_hours": holdout_h}
        self.model.fit(X, y)  # refit on everything
        return self.metrics

    def forecast(self, horizon_h: int = 24 * 7) -> pd.Series:
        if (cached := getattr(self, "_cache", {}).get(horizon_h)) is not None:
            return cached
        s = self.series.copy()
        for _ in range(horizon_h):
            nxt = s.index[-1] + pd.Timedelta(hours=1)
            s.loc[nxt] = np.nan
            x = _features(s.iloc[-200:]).iloc[[-1]]
            s.loc[nxt] = float(self.model.predict(x)[0])
        self._cache = {**getattr(self, "_cache", {}), horizon_h: s.iloc[-horizon_h:]}
        return self._cache[horizon_h]


def detect_anomalies(svc_df: pd.DataFrame, train_until_h: int = 24 * 30) -> pd.DataFrame:
    """IsolationForest per service on (demand, ratio-to-same-hour-last-week)."""
    out = []
    for svc, g in svc_df.groupby("service"):
        g = g.sort_values("ts").reset_index(drop=True)
        g["wow"] = g.cpu_cores / g.cpu_cores.shift(168)
        g["z"] = (g.cpu_cores - g.cpu_cores.rolling(168).mean()) / g.cpu_cores.rolling(168).std()
        feats = pd.DataFrame({"wow": g["wow"].fillna(1.0), "z": g["z"].fillna(0.0)})
        clf = IsolationForest(contamination=0.01, random_state=0).fit(feats.iloc[168:train_until_h])
        g["anomaly"] = clf.predict(feats) == -1
        g["score"] = -clf.score_samples(feats)
        g.loc[: 168, "anomaly"] = False
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summarize_anomalies(an: pd.DataFrame, since_h: int = 24 * 14) -> list[dict]:
    recent = an[an.ts >= an.ts.max() - pd.Timedelta(hours=since_h)]
    res = []
    for svc, g in recent.groupby("service"):
        rate = g.anomaly.mean()
        if rate > 0.05:
            first = g[g.anomaly].ts.min()
            res.append({"service": svc, "anomalous_hour_rate": round(float(rate), 3),
                        "first_seen": str(first),
                        "wow_ratio_median": round(float(g.wow.median()), 2)})
    return sorted(res, key=lambda r: -r["anomalous_hour_rate"])
