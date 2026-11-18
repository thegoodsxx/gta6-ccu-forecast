"""Leave-one-out (LOO) validator for gta6-twitch-ccu-forecast-01.
Evaluates model vs house baseline across practice cases p00..p03.
"""

import json
import math
from pathlib import Path
import main as forecaster

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

def pinball_loss(y: float, q: float, tau: float) -> float:
    y_safe = max(1.0, y)
    q_safe = max(1.0, q)
    d = math.log(y_safe) - math.log(q_safe)
    return 2.0 * max(tau * d, (tau - 1.0) * d)

def evaluate_baseline_forecast(p1: float, actuals: list[float]) -> list[float]:
    losses = []
    for h_idx, y in enumerate(actuals):
        h = h_idx + 2
        sq_h = math.sqrt(h - 1)
        for i, tau in enumerate(LEVELS):
            q = p1 * math.exp(Z[i] * 0.15 * sq_h)
            losses.append(pinball_loss(y, q, tau))
    return losses

def evaluate_model_forecast(forecast_hours: list[dict], actuals: list[float]) -> list[float]:
    losses = []
    for h_idx, y in enumerate(actuals):
        row = forecast_hours[h_idx]
        qs = row["q"]
        for i, tau in enumerate(LEVELS):
            q = qs[i]
            losses.append(pinball_loss(y, q, tau))
    return losses

def run_validation() -> None:
    kit_dir = Path("../gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input")
    if not kit_dir.exists():
        kit_dir = Path("gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input")

    case_files = sorted(kit_dir.glob("p*.in"))
    if not case_files:
        print("Error: No practice case files found.")
        return

    model_all_losses = []
    baseline_all_losses = []

    print(f"{'Case':<8} {'Comp':<5} {'Model LOO Loss':<15} {'Baseline LOO Loss':<18}")
    print("-" * 60)

    for path in case_files:
        case_data = json.loads(path.read_text(encoding="utf-8"))
        target_game = case_data.get("game", {}).get("name", "")
        comparables = case_data.get("comparables", [])

        case_model_losses = []
        case_base_losses = []
        comp_count = 0

        for comp in comparables:
            if comp.get("name", "") == target_game:
                continue
            v_list = forecaster.extract_curve(comp)
            if v_list is None:
                continue
            
            p1 = v_list[0]
            actuals = v_list[1:] # hours 2..24

            # Pool is all comparables except this comp and target game
            pool = [c for c in comparables if c.get("name", "") != comp.get("name", "") and c.get("name", "") != target_game]

            try:
                forecast_hours = forecaster.forecast_with_pool(p1, pool, target_game)
                m_losses = evaluate_model_forecast(forecast_hours, actuals)
                b_losses = evaluate_baseline_forecast(p1, actuals)

                case_model_losses.extend(m_losses)
                case_base_losses.extend(b_losses)
                model_all_losses.extend(m_losses)
                baseline_all_losses.extend(b_losses)
                comp_count += 1
            except Exception as e:
                continue

        if case_model_losses:
            m_mean = sum(case_model_losses) / len(case_model_losses)
            b_mean = sum(case_base_losses) / len(case_base_losses)
            print(f"{path.stem:<8} {comp_count:<5} {m_mean:<15.6f} {b_mean:<18.6f}")

    if model_all_losses:
        overall_model = sum(model_all_losses) / len(model_all_losses)
        overall_base = sum(baseline_all_losses) / len(baseline_all_losses)
        print("-" * 60)
        print(f"Overall Model LOO Loss:    {overall_model:.6f}")
        print(f"Overall Baseline LOO Loss: {overall_base:.6f}")
        print(f"Improvement over baseline: {((overall_base - overall_model) / overall_base) * 100:.2f}%")


if __name__ == "__main__":
    run_validation()
