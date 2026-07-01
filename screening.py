from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


ROOT = Path(r"D:\college\FBDQA")
DATA_DIR = ROOT / "FBDQA2026S_MMP_Challenge" / "FBDQA2021A_MMP_Challenge" / "data"
OUT_DIR = ROOT / "mmp_plan_workspace" / "step1_screening"
FIG_DIR = OUT_DIR / "figures"
DETAIL_DIR = FIG_DIR / "details"
HEAT_DIR = FIG_DIR / "heatmaps"
KLINE_DIR = FIG_DIR / "daily_kline"
REPORT_DIR = OUT_DIR / "reports"

FILE_RE = re.compile(r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv$")
JUMP_THRESHOLD = 0.0035


def ensure_dirs() -> None:
    for p in [OUT_DIR, FIG_DIR, DETAIL_DIR, HEAT_DIR, KLINE_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy, text, size=16, fill=(20, 20, 20), bold=False):
    draw.text(xy, text, font=font(size=size, bold=bold), fill=fill)


def longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = cur = 0
    for flag in mask:
        if flag:
            cur += 1
        else:
            best = max(best, cur)
            cur = 0
    return max(best, cur)


def longest_equal_run(values: np.ndarray) -> int:
    if values.size <= 1:
        return int(values.size)
    eq = values[1:] == values[:-1]
    return longest_true_run(eq) + 1 if eq.size else int(values.size)


def parse_file(path: Path) -> dict:
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"bad file name: {path.name}")
    sym = int(m.group(1))
    date = int(m.group(2))
    session = m.group(3)

    df = pd.read_csv(path, usecols=["date", "time", "sym", "amount_delta", "n_midprice", "n_close"])
    mid = df["n_midprice"].to_numpy(dtype=float)
    close = df["n_close"].to_numpy(dtype=float)
    amt = df["amount_delta"].to_numpy(dtype=float)
    diffs = np.abs(np.diff(mid))
    zero_mask = mid == 0
    equal_run = longest_equal_run(mid)
    zero_run = longest_true_run(zero_mask)
    zero_ratio = float(zero_mask.mean())
    flat_step_ratio = float((diffs == 0).mean()) if diffs.size else 0.0
    jump_count = int((diffs > JUMP_THRESHOLD).sum()) if diffs.size else 0
    max_abs_diff = float(diffs.max()) if diffs.size else 0.0
    max_mid = float(np.max(mid))
    min_mid = float(np.min(mid))
    mid_range = float(max_mid - min_mid)
    mid_mean = float(np.mean(mid))
    mid_std = float(np.std(mid))
    amount_zero_ratio = float((amt == 0).mean())
    amount_mean = float(np.mean(amt))
    unique_count = int(np.unique(mid).size)

    near_boundary = abs(max_mid) >= 0.09 or abs(min_mid) >= 0.09 or abs(mid_mean) >= 0.08
    limit_flag = bool((equal_run >= 1000 and near_boundary) or (unique_count <= 3 and near_boundary))
    halt_flag = bool((equal_run >= 1000 and not near_boundary) or (unique_count <= 3 and not near_boundary))
    drop_flag = bool(zero_ratio > 0.20)
    jump_flag = bool(jump_count >= 2 or max_abs_diff > JUMP_THRESHOLD)

    notes = []
    if drop_flag:
        notes.append("zero_ratio>20%")
    if limit_flag:
        notes.append("limit-like")
    if halt_flag:
        notes.append("halt-like")
    if jump_flag:
        notes.append("jump-like")
    if not notes:
        notes.append("clean")

    return {
        "file": path.name,
        "sym": sym,
        "date": date,
        "session": session,
        "n_rows": int(len(df)),
        "zero_ratio": zero_ratio,
        "flat_step_ratio": flat_step_ratio,
        "jump_count": jump_count,
        "max_abs_diff": max_abs_diff,
        "equal_run": int(equal_run),
        "zero_run": int(zero_run),
        "mid_min": min_mid,
        "mid_max": max_mid,
        "mid_range": mid_range,
        "mid_mean": mid_mean,
        "mid_std": mid_std,
        "amount_zero_ratio": amount_zero_ratio,
        "amount_mean": amount_mean,
        "unique_count": unique_count,
        "drop_flag": drop_flag,
        "limit_flag": limit_flag,
        "halt_flag": halt_flag,
        "jump_flag": jump_flag,
        "notes": ";".join(notes),
        "_mid": mid,
        "_close": close,
        "_amt": amt,
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_heatmap(matrix: np.ndarray, row_labels, col_labels, title: str, out_path: Path, vmin=None, vmax=None):
    rows, cols = matrix.shape
    cell_w = 18 if cols <= 90 else max(8, int(1200 / max(cols, 1)))
    cell_h = 22
    left = 120
    top = 70
    bottom = 50
    right = 40
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom + 50
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(22, bold=True)
    draw.text((left, 20), title, font=title_font, fill=(20, 20, 20))
    draw.text((20, top - 20), "sym", font=font(14, bold=True), fill=(30, 30, 30))

    finite = matrix[np.isfinite(matrix)]
    if vmin is None:
        vmin = float(np.min(finite)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    def color_for(v):
        if not np.isfinite(v):
            return (235, 235, 235)
        t = (v - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        r = int(245 * (1 - t) + 33 * t)
        g = int(247 * (1 - t) + 113 * t)
        b = int(250 * (1 - t) + 181 * t)
        return (r, g, b)

    # axes labels
    for i, lab in enumerate(row_labels):
        y = top + i * cell_h + 4
        draw.text((20, y), str(lab), font=font(13), fill=(60, 60, 60))
    for j, lab in enumerate(col_labels):
        if j % 5 != 0 and j != cols - 1:
            continue
        x = left + j * cell_w + 1
        draw.text((x, top + rows * cell_h + 4), str(lab), font=font(11), fill=(80, 80, 80))

    for i in range(rows):
        for j in range(cols):
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], fill=color_for(matrix[i, j]), outline=(255, 255, 255))

    # legend
    leg_x = left
    leg_y = top + rows * cell_h + 24
    for k in range(180):
        t = k / 179
        v = vmin + (vmax - vmin) * t
        draw.line([(leg_x + k, leg_y), (leg_x + k, leg_y + 16)], fill=color_for(v), width=1)
    draw.rectangle([leg_x, leg_y, leg_x + 179, leg_y + 16], outline=(120, 120, 120))
    draw.text((leg_x + 190, leg_y - 2), f"{vmin:.4g}", font=font(11), fill=(60, 60, 60))
    draw.text((leg_x + 320, leg_y - 2), f"{vmax:.4g}", font=font(11), fill=(60, 60, 60))

    img.save(out_path)


def save_kline_chart(sym: int, daily_rows: list[dict], out_path: Path):
    if HAS_MATPLOTLIB:
        save_kline_chart_matplotlib(sym, daily_rows, out_path)
        return

    save_kline_chart_pillow(sym, daily_rows, out_path)


def save_kline_chart_pillow(sym: int, daily_rows: list[dict], out_path: Path):
    days = [r["date"] for r in daily_rows]
    valid = [r for r in daily_rows if all(np.isfinite([r["open"], r["high"], r["low"], r["close"]]))]
    if not valid:
        return
    low = min(r["low"] for r in valid)
    high = max(r["high"] for r in valid)
    if math.isclose(low, high):
        high = low + 1e-6
    width = 1800
    height = 820
    left = 80
    top = 70
    bottom = 90
    right = 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 20), f"sym {sym} daily kline (rebased from n_close, start=100)", font=font(22, bold=True), fill=(20, 20, 20))
    draw.text((left, 46), f"days: {min(days)}-{max(days)} | red=up, green=down, gray=flat | y-axis=rebased price", font=font(12), fill=(90, 90, 90))

    # grid
    for i in range(6):
        y = top + int(plot_h * i / 5)
        draw.line([(left, y), (left + plot_w, y)], fill=(235, 235, 235), width=1)
        val = high - (high - low) * i / 5
        draw.text((6, y - 7), f"{val:.4f}", font=font(11), fill=(80, 80, 80))
    for d in range(0, 80, 10):
        x = left + int(plot_w * d / max(78, 1))
        draw.line([(x, top), (x, top + plot_h)], fill=(245, 245, 245), width=1)
        draw.text((x - 6, top + plot_h + 6), str(d), font=font(11), fill=(80, 80, 80))
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(150, 150, 150))

    step = plot_w / max(78, 1)
    body_w = max(4, int(step * 0.6))
    for r in daily_rows:
        if not all(np.isfinite([r["open"], r["high"], r["low"], r["close"]])):
            continue
        x = left + int((r["date"] / 78) * plot_w)
        def y(v):
            return top + int((high - v) / (high - low) * plot_h)
        y_open = y(r["open"])
        y_close = y(r["close"])
        y_high = y(r["high"])
        y_low = y(r["low"])
        if r["close"] > r["open"]:
            color = (200, 45, 45)
        elif r["close"] < r["open"]:
            color = (30, 140, 60)
        else:
            color = (120, 120, 120)
        draw.line([(x, y_high), (x, y_low)], fill=(60, 60, 60), width=1)
        top_body = min(y_open, y_close)
        bot_body = max(y_open, y_close)
        if top_body == bot_body:
            draw.line([(x - body_w // 2, top_body), (x + body_w // 2, bot_body)], fill=color, width=2)
        else:
            draw.rectangle([x - body_w // 2, top_body, x + body_w // 2, bot_body], fill=color, outline=color)
    img.save(out_path)


def save_kline_chart_matplotlib(sym: int, daily_rows: list[dict], out_path: Path):
    valid = [r for r in daily_rows if all(np.isfinite([r["open"], r["high"], r["low"], r["close"]]))]
    if not valid:
        return
    xs = [r["date"] for r in valid]
    opens = np.array([r["open"] for r in valid], dtype=float)
    highs = np.array([r["high"] for r in valid], dtype=float)
    lows = np.array([r["low"] for r in valid], dtype=float)
    closes = np.array([r["close"] for r in valid], dtype=float)

    fig, ax = plt.subplots(figsize=(18, 8.2), dpi=120)
    fig.suptitle(f"sym {sym} daily kline (rebased from n_close, start=100)", x=0.06, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.945, "days: 0-78 | red=up, green=down, gray=flat | y-axis=rebased price", fontsize=10, color="#666666")
    ax.grid(True, color="#ececec", linewidth=0.8)
    ax.set_xlim(0, 79)
    ax.set_xticks(range(0, 80, 10))
    ax.set_ylabel("rebased price")
    width = 0.55
    for x, o, h, l, c in zip(xs, opens, highs, lows, closes):
        color = "#c92a2a" if c > o else "#2b8a3e" if c < o else "#7a7a7a"
        ax.vlines(x, l, h, color="#444444", linewidth=0.6, zorder=1)
        body_bottom = min(o, c)
        body_h = abs(c - o)
        if body_h < 1e-8:
            ax.hlines(o, x - width / 2, x + width / 2, color=color, linewidth=2.0, zorder=2)
        else:
            rect = plt.Rectangle((x - width / 2, body_bottom), width, body_h, facecolor=color, edgecolor=color, linewidth=0.0, zorder=2)
            ax.add_patch(rect)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_detail_chart(record: dict, out_path: Path):
    if HAS_MATPLOTLIB:
        save_detail_chart_matplotlib(record, out_path)
        return

    save_detail_chart_pillow(record, out_path)


def save_detail_chart_pillow(record: dict, out_path: Path):
    mid = record["_mid"]
    amt = record["_amt"]
    amt_vis = np.log1p(np.abs(amt))
    n = len(mid)
    if n == 0:
        return
    diffs = np.abs(np.diff(mid))
    jump_idx = np.where(diffs > JUMP_THRESHOLD)[0] + 1
    width = 1600
    height = 860
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), f"{record['file']}", font=font(20, bold=True), fill=(20, 20, 20))
    meta = (
        f"zero_ratio={record['zero_ratio']:.3f} | jump_count={record['jump_count']} | "
        f"equal_run={record['equal_run']} | notes={record['notes']}"
    )
    draw.text((30, 48), meta, font=font(12), fill=(90, 90, 90))

    panels = [
        (70, 100, width - 40, 400, mid, "n_midprice (normalized)"),
        (70, 470, width - 40, 790, amt_vis, "log1p(amount_delta)"),
    ]
    for x0, y0, x1, y1, arr, label in panels:
        draw.rectangle([x0, y0, x1, y1], outline=(160, 160, 160))
        draw.text((x0, y0 - 20), label, font=font(13, bold=True), fill=(40, 40, 40))
        if arr.size == 0:
            continue
        a_min = float(np.min(arr))
        a_max = float(np.max(arr))
        if math.isclose(a_min, a_max):
            a_max = a_min + 1e-6
        plot_w = x1 - x0
        plot_h = y1 - y0
        for i in range(6):
            yy = y0 + int(plot_h * i / 5)
            draw.line([(x0, yy), (x1, yy)], fill=(245, 245, 245), width=1)
            v = a_max - (a_max - a_min) * i / 5
            draw.text((10, yy - 7), f"{v:.4f}", font=font(10), fill=(100, 100, 100))
        pts = []
        for i, v in enumerate(arr):
            x = x0 + int(i / max(n - 1, 1) * plot_w)
            y = y0 + int((a_max - v) / (a_max - a_min) * plot_h)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=(50, 90, 170), width=2)
        if label == "n_midprice":
            for idx in jump_idx[:200]:
                x = x0 + int(idx / max(n - 1, 1) * plot_w)
                y = y0 + int((a_max - arr[idx]) / (a_max - a_min) * plot_h)
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(220, 50, 50))
    img.save(out_path)


def save_detail_chart_matplotlib(record: dict, out_path: Path):
    mid = record["_mid"]
    amt = record["_amt"]
    amt_vis = np.log1p(np.abs(amt))
    n = len(mid)
    if n == 0:
        return
    diffs = np.abs(np.diff(mid))
    jump_idx = np.where(diffs > JUMP_THRESHOLD)[0] + 1

    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), dpi=120, sharex=True)
    fig.suptitle(record["file"], x=0.02, y=0.985, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.95,
        f"zero_ratio={record['zero_ratio']:.3f} | jump_count={record['jump_count']} | "
        f"equal_run={record['equal_run']} | notes={record['notes']}",
        fontsize=10,
        color="#666666",
    )

    axes[0].plot(mid, color="#355cba", linewidth=1.0)
    if jump_idx.size:
        axes[0].scatter(jump_idx, mid[jump_idx], color="#d94841", s=12, zorder=3)
    axes[0].set_title("n_midprice (normalized)", loc="left", fontsize=11, fontweight="bold")
    axes[0].grid(True, color="#efefef")
    axes[0].set_ylabel("value")

    axes[1].plot(amt_vis, color="#355cba", linewidth=1.0)
    axes[1].set_title("log1p(amount_delta)", loc="left", fontsize=11, fontweight="bold")
    axes[1].grid(True, color="#efefef")
    axes[1].set_xlim(0, n - 1)
    axes[1].set_ylabel("value")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    files = sorted(DATA_DIR.glob("snapshot_sym*_date*_*.csv"))
    if not files:
        raise FileNotFoundError(DATA_DIR)

    records = [parse_file(p) for p in files]
    file_df = pd.DataFrame([{k: v for k, v in rec.items() if not k.startswith("_")} for rec in records])
    save_csv(file_df, REPORT_DIR / "file_metrics.csv")

    # file-level summary
    summary_lines = []
    summary_lines.append(f"total_files: {len(file_df)}")
    summary_lines.append(f"drop_files: {int(file_df['drop_flag'].sum())}")
    summary_lines.append(f"limit_like_files: {int(file_df['limit_flag'].sum())}")
    summary_lines.append(f"halt_like_files: {int(file_df['halt_flag'].sum())}")
    summary_lines.append(f"jump_like_files: {int(file_df['jump_flag'].sum())}")
    summary_lines.append(f"zero_ratio_gt_20pct: {int((file_df['zero_ratio'] > 0.20).sum())}")
    summary_lines.append("")

    # coverage by symbol/date/session
    file_df["key"] = file_df["sym"].astype(str) + "_" + file_df["date"].astype(str) + "_" + file_df["session"]
    coverage = defaultdict(set)
    for _, row in file_df.iterrows():
        coverage[int(row["sym"])].add(int(row["date"]))
    for sym in sorted(coverage):
        missing = sorted(set(range(79)) - coverage[sym])
        summary_lines.append(f"sym {sym} missing_dates: {missing}")
    (REPORT_DIR / "screening_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    md_lines = [
        "# Step 1 Screening Summary",
        "",
        "- daily kline is now rebased from n_close with start price = 100, so the y-axis is pseudo-price rather than raw return.",
        f"- total files: {len(file_df)}",
        f"- drop files: {int(file_df['drop_flag'].sum())}",
        f"- limit-like files: {int(file_df['limit_flag'].sum())}",
        f"- halt-like files: {int(file_df['halt_flag'].sum())}",
        f"- jump-like files: {int(file_df['jump_flag'].sum())}",
        f"- zero ratio > 20%: {int((file_df['zero_ratio'] > 0.20).sum())}",
        "",
        "## Missing Dates by Symbol",
    ]
    for sym in sorted(coverage):
        missing = sorted(set(range(79)) - coverage[sym])
        md_lines.append(f"- sym {sym}: {missing}")
    (REPORT_DIR / "screening_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # heatmaps by session
    for session in ["am", "pm"]:
        sub = file_df[file_df["session"] == session].copy()
        for field, title in [
            ("zero_ratio", f"{session} zero_ratio heatmap"),
            ("jump_count", f"{session} jump_count heatmap"),
            ("flat_step_ratio", f"{session} flat_step_ratio heatmap"),
            ("equal_run", f"{session} equal_run heatmap"),
        ]:
            matrix = np.full((10, 79), np.nan, dtype=float)
            for _, row in sub.iterrows():
                matrix[int(row["sym"]), int(row["date"])] = float(row[field])
            build_heatmap(
                matrix,
                row_labels=list(range(10)),
                col_labels=list(range(79)),
                title=title,
                out_path=HEAT_DIR / f"{session}_{field}.png",
            )

    # daily kline summaries
    grouped = defaultdict(dict)
    for rec in records:
        grouped[(rec["sym"], rec["date"])][rec["session"]] = rec

    daily_rows_out = []
    for sym in range(10):
        daily_rows = []
        base_price = 100.0
        for date in range(79):
            sessions = grouped.get((sym, date), {})
            mids = []
            closes = []
            for sess in ["am", "pm"]:
                if sess in sessions:
                    mids.extend(list(sessions[sess]["_mid"]))
                    closes.extend(list(sessions[sess]["_close"]))
            if mids:
                arr = np.asarray(mids, dtype=float)
                close_arr = np.asarray(closes, dtype=float)
                raw_open = float(close_arr[0])
                raw_high = float(np.max(close_arr))
                raw_low = float(np.min(close_arr))
                raw_close = float(close_arr[-1])
                day_open = base_price * (1.0 + raw_open)
                day_high = base_price * (1.0 + raw_high)
                day_low = base_price * (1.0 + raw_low)
                day_close = base_price * (1.0 + raw_close)
                daily_rows.append(
                    {
                        "sym": sym,
                        "date": date,
                        "open": day_open,
                        "high": day_high,
                        "low": day_low,
                        "close": day_close,
                        "range": float(day_high - day_low),
                        "raw_open_return": raw_open,
                        "raw_high_return": raw_high,
                        "raw_low_return": raw_low,
                        "raw_close_return": raw_close,
                        "mid_open_return": float(arr[0]),
                        "mid_high_return": float(np.max(arr)),
                        "mid_low_return": float(np.min(arr)),
                        "mid_close_return": float(arr[-1]),
                        "n_rows": int(arr.size),
                        "has_am": int("am" in sessions),
                        "has_pm": int("pm" in sessions),
                        "drop_flag": int(any(sessions[s]["drop_flag"] for s in sessions)),
                        "limit_flag": int(any(sessions[s]["limit_flag"] for s in sessions)),
                        "halt_flag": int(any(sessions[s]["halt_flag"] for s in sessions)),
                        "jump_flag": int(any(sessions[s]["jump_flag"] for s in sessions)),
                    }
                )
                base_price = day_close
            else:
                daily_rows.append(
                    {
                        "sym": sym,
                        "date": date,
                        "open": np.nan,
                        "high": np.nan,
                        "low": np.nan,
                        "close": np.nan,
                        "range": np.nan,
                        "raw_open_return": np.nan,
                        "raw_high_return": np.nan,
                        "raw_low_return": np.nan,
                        "raw_close_return": np.nan,
                        "mid_open_return": np.nan,
                        "mid_high_return": np.nan,
                        "mid_low_return": np.nan,
                        "mid_close_return": np.nan,
                        "n_rows": 0,
                        "has_am": 0,
                        "has_pm": 0,
                        "drop_flag": 0,
                        "limit_flag": 0,
                        "halt_flag": 0,
                        "jump_flag": 0,
                    }
                )
        save_kline_chart(sym, daily_rows, KLINE_DIR / f"sym{sym:02d}_daily_kline.png")
        daily_rows_out.extend(daily_rows)

    save_csv(pd.DataFrame(daily_rows_out), REPORT_DIR / "daily_summary.csv")

    # detail charts for the most suspicious files
    score = (
        file_df["drop_flag"].astype(int) * 1000
        + file_df["limit_flag"].astype(int) * 800
        + file_df["halt_flag"].astype(int) * 700
        + file_df["jump_flag"].astype(int) * 300
        + (file_df["zero_ratio"] * 100).round(2)
        + file_df["equal_run"].clip(upper=2000) / 10.0
    )
    top = file_df.assign(score=score).sort_values("score", ascending=False).head(40)
    for _, row in top.iterrows():
        rec = next(r for r in records if r["file"] == row["file"])
        save_detail_chart(rec, DETAIL_DIR / f"{Path(row['file']).stem}.png")

    print(f"written: {OUT_DIR}")


if __name__ == "__main__":
    main()
