"""gta6-forecast-entry: GTA VI Twitch CCU Forecast (Python 3.13, standard library only)

Comparable launch curve matching, similarity weighting, weighted quantile forecasting,
and robust spread flooring for hourly Twitch CCU on release day.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Tuned hyperparameters (optimized via cross-validation on practice cases)
BANDWIDTH = 1.0        # log-space bandwidth for hour-1 similarity kernel
PLATFORM_BOOST = 1.35  # boost for console / matching platforms
RECENCY_BOOST = 0.03   # annual recency weight increment
SPREAD_FLOOR = 0.10    # minimum volatility parameter per sqrt(h-1)
EMPIRICAL_BLEND = 1.0  # weight on empirical pool quantiles vs baseline prior


def round2(x: float) -> float | int:
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def extract_curve(comp: dict) -> list[float] | None:
    """Extract release-day hourly viewers v(1)..v(24) from comparable data.
    Returns None if hour 1 is missing/non-positive or too many gaps.
    """
    day_start_str = comp.get("day_start")
    if not day_start_str:
        return None
    try:
        day_dt = parse_iso(day_start_str)
    except Exception:
        return None

    hourly = comp.get("hourly", [])
    hourly_map = {}
    for pt in hourly:
        from_str = pt.get("from")
        v = pt.get("viewers")
        if from_str and v is not None:
            try:
                dt = parse_iso(from_str)
                hourly_map[dt] = float(v)
            except Exception:
                continue

    v_list = []
    missing_count = 0
    for h in range(1, 25):
        dt = day_dt + timedelta(hours=h - 1)
        v = hourly_map.get(dt)
        if v is not None and v > 0:
            v_list.append(v)
        else:
            v_list.append(None)
            missing_count += 1

    if missing_count > 3 or v_list[0] is None or v_list[0] <= 0:
        return None

    # Bridge isolated missing hours by log-space interpolation
    for h_idx in range(1, 23):
        if v_list[h_idx] is None:
            prev_v = v_list[h_idx - 1]
            next_v = v_list[h_idx + 1]
            if prev_v is not None and prev_v > 0 and next_v is not None and next_v > 0:
                v_list[h_idx] = math.exp(0.5 * (math.log(prev_v) + math.log(next_v)))

    # Recheck for any remaining missing values
    if any(v is None or v <= 0 for v in v_list):
        return None

    return v_list


def weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    """Compute weighted quantile of 1D values given weights."""
    if not values:
        return 0.0
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    vals = [p[0] for p in paired]
    wts = [p[1] for p in paired]
    total_w = sum(wts)
    if total_w <= 0:
        return vals[len(vals) // 2]
    
    cum_w = 0.0
    cum_weights = []
    for w in wts:
        cum_w += w / total_w
        cum_weights.append(cum_w)

    target = max(0.0, min(1.0, quantile))
    for i, cw in enumerate(cum_weights):
        if cw >= target:
            if i == 0:
                return vals[0]
            prev_w = cum_weights[i - 1]
            if cw == prev_w:
                return vals[i]
            frac = (target - prev_w) / (cw - prev_w)
            return vals[i - 1] + frac * (vals[i] - vals[i - 1])
    return vals[-1]


def forecast_with_pool(p1: float, comparables: list[dict], target_game_name: str) -> list[dict]:
    """Forecast hours 2..24 given anchor p1 and pool of comparables."""
    pool_data = [] # list of (weight, ratios)
    
    log_p1 = math.log(p1)

    for comp in comparables:
        c_name = comp.get("name", "")
        if c_name == target_game_name:
            continue
        
        v_list = extract_curve(comp)
        if v_list is None:
            continue

        cv1 = v_list[0]
        log_cv1 = math.log(cv1)
        
        # Similarity weight (Gaussian kernel in log space)
        dist = abs(log_cv1 - log_p1)
        w_sim = math.exp(- (dist ** 2) / (2 * BANDWIDTH ** 2))

        # Platform boost (console match)
        platforms = comp.get("platforms", "")
        if isinstance(platforms, str) and ("PS5" in platforms or "Xbox" in platforms or "Console" in platforms):
            w_sim *= PLATFORM_BOOST

        # Recency boost from day_start year
        day_start = comp.get("day_start", "")
        try:
            year = int(day_start[:4])
            recency = 1.0 + RECENCY_BOOST * max(0, year - 2018)
            w_sim *= recency
        except Exception:
            pass

        if w_sim < 1e-4:
            continue

        # Log ratio curve r(h) = v(h) / v(1)
        log_ratios = [math.log(v / cv1) for v in v_list]
        pool_data.append((w_sim, log_ratios))

    hours_output = []

    for h_idx in range(1, 24): # h = 2 to 24 (index 1 to 23)
        h = h_idx + 1
        sq_h = math.sqrt(h - 1)

        if pool_data:
            vals = [item[1][h_idx] for item in pool_data]
            wts = [item[0] for item in pool_data]

            emp_qs = [weighted_quantile(vals, wts, tau) for tau in LEVELS]
        else:
            emp_qs = [0.0] * 5

        # Baseline prior quantiles (persistence + spread)
        base_qs = [z * 0.15 * sq_h for z in Z]
        floor_qs = [z * SPREAD_FLOOR * sq_h for z in Z]

        row_qs = []
        for i, tau in enumerate(LEVELS):
            # Blend empirical with prior
            eq = EMPIRICAL_BLEND * emp_qs[i] + (1.0 - EMPIRICAL_BLEND) * base_qs[i]
            # Ensure spread floor (tails at least as wide as floor_qs)
            if i == 0: # 0.05 quantile (lower tail)
                q_val = min(eq, floor_qs[i])
            elif i == 4: # 0.95 quantile (upper tail)
                q_val = max(eq, floor_qs[i])
            else:
                q_val = eq

            realized_v = p1 * math.exp(q_val)
            # Bounds: > 0 and <= 1e9
            realized_v = max(1.0, min(1e9, realized_v))
            row_qs.append(realized_v)

        # Ensure non-decreasing
        for i in range(1, 5):
            if row_qs[i] < row_qs[i - 1]:
                row_qs[i] = row_qs[i - 1]

        rounded_q = [round2(q) for q in row_qs]
        hours_output.append({"hour": h, "q": rounded_q})

    return hours_output


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    target_game = case.get("game", {}).get("name", "")
    comparables = case.get("comparables", [])
    
    hours = forecast_with_pool(p1, comparables, target_game)
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
