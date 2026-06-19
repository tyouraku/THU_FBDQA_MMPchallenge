from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb


LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
SYM_IDS = list(range(10))
WINDOWS = [5, 10, 20, 60, 100]
UPPER_LIMIT = 0.1
LOWER_LIMIT = -0.1
LIMIT_ATOL = 5e-4
MIN_DIRECTION_MARGIN = 0.2
THRESHOLDS = {
    "label_5": 0.70,
    "label_10": 0.69,
    "label_20": 0.60,
    "label_40": 0.56,
    "label_60": 0.55,
}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(df: pd.DataFrame, sym: int, session: str) -> pd.DataFrame:
    n = len(df)
    idx = np.arange(n, dtype=np.float32)
    tick_pos = idx / max(n - 1, 1)

    feat_cols: dict[str, np.ndarray] = {}
    feat_cols["tick_pos"] = tick_pos
    feat_cols["tick_pos2"] = tick_pos * tick_pos
    feat_cols["session_pm"] = np.full(n, 1.0 if session == "pm" else 0.0, dtype=np.float32)

    for sid in SYM_IDS:
        feat_cols[f"sym_{sid}"] = np.full(n, 1.0 if sid == sym else 0.0, dtype=np.float32)

    mid = df["n_midprice"].astype(np.float32)
    close = df["n_close"].astype(np.float32)
    amount = df["amount_delta"].astype(np.float32)
    amount_log = np.log1p(amount.clip(lower=0.0))
    spread = (df["n_ask1"] - df["n_bid1"]).astype(np.float32)
    book_mid = ((df["n_ask1"] + df["n_bid1"]) / 2.0).astype(np.float32)

    bsize_cols = [f"n_bsize{i}" for i in range(1, 6)]
    asize_cols = [f"n_asize{i}" for i in range(1, 6)]
    bid_cols = [f"n_bid{i}" for i in range(1, 6)]
    ask_cols = [f"n_ask{i}" for i in range(1, 6)]

    bid_stack = np.column_stack([df[c].astype(np.float32).to_numpy() for c in bid_cols])
    ask_stack = np.column_stack([df[c].astype(np.float32).to_numpy() for c in ask_cols])
    bsize_stack = np.column_stack([df[c].astype(np.float32).to_numpy() for c in bsize_cols])
    asize_stack = np.column_stack([df[c].astype(np.float32).to_numpy() for c in asize_cols])

    feat_cols["mid"] = mid.to_numpy()
    feat_cols["close"] = close.to_numpy()
    feat_cols["close_mid_gap"] = (close - mid).to_numpy()
    feat_cols["abs_mid"] = mid.abs().to_numpy()
    feat_cols["book_mid"] = book_mid.to_numpy()
    feat_cols["spread"] = spread.to_numpy()
    feat_cols["rel_spread"] = _safe_div(spread, mid.abs() + 1e-6).to_numpy()
    dist_upper = (0.1 - mid).to_numpy()
    dist_lower = (mid + 0.1).to_numpy()
    feat_cols["dist_upper_limit"] = dist_upper
    feat_cols["dist_lower_limit"] = dist_lower
    feat_cols["limit_proximity"] = np.minimum(dist_upper, dist_lower)
    feat_cols["amount_log"] = amount_log.to_numpy()
    feat_cols["amount_delta_1"] = amount.diff().fillna(0.0).to_numpy(dtype=np.float32)
    feat_cols["mid_delta_1"] = mid.diff().fillna(0.0).to_numpy(dtype=np.float32)
    feat_cols["mid_delta_5"] = (mid - mid.shift(5)).fillna(0.0).to_numpy(dtype=np.float32)
    feat_cols["close_delta_1"] = close.diff().fillna(0.0).to_numpy(dtype=np.float32)

    depth_bid = bsize_stack.sum(axis=1)
    depth_ask = asize_stack.sum(axis=1)
    feat_cols["depth_bid_sum"] = depth_bid.astype(np.float32)
    feat_cols["depth_ask_sum"] = depth_ask.astype(np.float32)
    feat_cols["depth_sum"] = (depth_bid + depth_ask).astype(np.float32)
    feat_cols["depth_imbalance_1"] = _safe_div(
        df["n_bsize1"].astype(np.float32) - df["n_asize1"].astype(np.float32),
        df["n_bsize1"].astype(np.float32) + df["n_asize1"].astype(np.float32),
    ).to_numpy()
    feat_cols["depth_imbalance_5"] = _safe_div(
        pd.Series(depth_bid - depth_ask, index=df.index),
        pd.Series(depth_bid + depth_ask, index=df.index),
    ).to_numpy()

    mid_np = mid.to_numpy()
    for i in range(5):
        feat_cols[f"bid_gap_{i+1}"] = (mid_np - bid_stack[:, i]).astype(np.float32)
        feat_cols[f"ask_gap_{i+1}"] = (ask_stack[:, i] - mid_np).astype(np.float32)
        feat_cols[f"bid_size_log_{i+1}"] = np.log(np.clip(bsize_stack[:, i], 1e-12, None)).astype(np.float32)
        feat_cols[f"ask_size_log_{i+1}"] = np.log(np.clip(asize_stack[:, i], 1e-12, None)).astype(np.float32)
        feat_cols[f"spread_level_{i+1}"] = (ask_stack[:, i] - bid_stack[:, i]).astype(np.float32)

    source_map = {
        "mid": mid,
        "mid_delta_1": pd.Series(feat_cols["mid_delta_1"], index=df.index),
        "close": close,
        "amount_log": amount_log,
        "spread": spread,
        "depth_imbalance_5": pd.Series(feat_cols["depth_imbalance_5"], index=df.index),
    }
    for name, series in source_map.items():
        series = pd.Series(series, index=df.index, dtype=np.float32)
        for w in WINDOWS:
            roll = series.rolling(window=w, min_periods=w)
            feat_cols[f"{name}_mean_{w}"] = roll.mean().to_numpy(dtype=np.float32)
            feat_cols[f"{name}_std_{w}"] = roll.std().to_numpy(dtype=np.float32)
            feat_cols[f"{name}_min_{w}"] = roll.min().to_numpy(dtype=np.float32)
            feat_cols[f"{name}_max_{w}"] = roll.max().to_numpy(dtype=np.float32)
            feat_cols[f"{name}_trend_{w}"] = ((series - series.shift(w - 1)) / max(w - 1, 1)).to_numpy(dtype=np.float32)

    feat = pd.DataFrame(feat_cols, index=df.index)
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.fillna(0.0, inplace=True)
    return feat.astype(np.float32)


class Predictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.models = {}
        for label in LABEL_COLS:
            booster = xgb.Booster()
            booster.load_model(os.path.join(base_dir, f"{label}.json"))
            self.models[label] = booster

    def _infer_session(self, df: pd.DataFrame) -> str:
        first_time = str(df["time"].iloc[0])
        return "pm" if first_time >= "12:00:00" else "am"

    def _limit_override(self, df: pd.DataFrame) -> int | None:
        last_mid = float(df["n_midprice"].iloc[-1])
        last_close = float(df["n_close"].iloc[-1])
        last_ask1 = float(df["n_ask1"].iloc[-1])
        last_bid1 = float(df["n_bid1"].iloc[-1])

        upper_probe = max(last_mid, last_close, last_ask1)
        lower_probe = min(last_mid, last_close, last_bid1)
        hit_upper = np.isclose(upper_probe, UPPER_LIMIT, atol=LIMIT_ATOL)
        hit_lower = np.isclose(lower_probe, LOWER_LIMIT, atol=LIMIT_ATOL)
        if hit_upper and hit_lower:
            return 1
        if hit_upper:
            return 0
        if hit_lower:
            return 2
        return None

    def _apply_threshold(self, proba: np.ndarray, label: str) -> int:
        row = proba[0]
        pred = int(np.argmax(row))
        sorted_proba = np.sort(row)
        top = float(sorted_proba[-1])
        second = float(sorted_proba[-2]) if len(sorted_proba) > 1 else 0.0
        if top < THRESHOLDS[label]:
            return 1
        if pred in (0, 2) and top - second < MIN_DIRECTION_MARGIN:
            return 1
        return pred

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        preds: List[List[int]] = []
        for df in x:
            override = self._limit_override(df)
            if override is not None:
                preds.append([override] * len(LABEL_COLS))
                continue
            sym = int(df["sym"].iloc[0])
            session = self._infer_session(df)
            feat = build_features(df.copy(), sym=sym, session=session)
            row = feat.iloc[[-1]]
            dmat = xgb.DMatrix(row)
            one = []
            for label in LABEL_COLS:
                proba = self.models[label].predict(dmat)
                one.append(self._apply_threshold(proba, label))
            preds.append(one)
        return preds
