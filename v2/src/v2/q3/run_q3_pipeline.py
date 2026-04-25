from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ensure_output_dirs, get_paths


SIMULATION_YEARS = 10
PURCHASE_COST = 300.0
MEDIUM_COST = 3.0
MAJOR_COST = 12.0
THRESHOLD = 37.0


def _read_table(tables_dir: Path, prefix: str) -> pd.DataFrame:
    matches = sorted(tables_dir.glob(f"{prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(f"Cannot find table starting with {prefix} under {tables_dir}")
    return pd.read_csv(matches[0], encoding="utf-8-sig")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _policy_rows(params: pd.DataFrame) -> pd.DataFrame:
    medium_interval = float(pd.to_numeric(params["maintenance_interval_median"], errors="coerce").median())
    if not np.isfinite(medium_interval):
        medium_interval = 60.0
    major_interval = max(180.0, medium_interval * 4)
    rows = [
        {
            "policy_name": "current_policy",
            "policy_type": "fixed_current_rule",
            "theta_M": np.nan,
            "theta_B": np.nan,
            "I_M_min": medium_interval,
            "I_B_min": major_interval,
            "description": "按第二问提取的当前维护间隔近似复现当前策略。",
        },
        {
            "policy_name": "conservative_policy",
            "policy_type": "threshold_rule",
            "theta_M": 60.0,
            "theta_B": 45.0,
            "I_M_min": 30.0,
            "I_B_min": 120.0,
            "description": "提前维护，降低功能风险但维护成本较高。",
        },
        {
            "policy_name": "delayed_policy",
            "policy_type": "threshold_rule",
            "theta_M": 45.0,
            "theta_B": 37.0,
            "I_M_min": 60.0,
            "I_B_min": 240.0,
            "description": "延迟维护，减少维护次数但失效风险较高。",
        },
    ]
    for theta_m in [45.0, 50.0, 55.0, 60.0]:
        for theta_b in [37.0, 40.0, 43.0]:
            for i_m in [30.0, 45.0, 60.0]:
                for i_b in [120.0, 180.0, 240.0]:
                    if theta_b > theta_m:
                        continue
                    rows.append(
                        {
                            "policy_name": f"grid_M{int(theta_m)}_B{int(theta_b)}_IM{int(i_m)}_IB{int(i_b)}",
                            "policy_type": "grid_candidate",
                            "theta_M": theta_m,
                            "theta_B": theta_b,
                            "I_M_min": i_m,
                            "I_B_min": i_b,
                            "description": "网格搜索候选策略。",
                        }
                    )
    return pd.DataFrame(rows)


def _apply_maintenance(row: pd.Series, maintenance_type: str, x_state: float, hmax_t: float) -> float:
    if maintenance_type == "medium":
        rho = row.get("medium_recovery_ratio_used", np.nan)
        gain = row.get("medium_plateau_gain_used", np.nan)
    else:
        rho = row.get("major_recovery_ratio_used", np.nan)
        gain = row.get("major_plateau_gain_used", np.nan)
    if pd.notna(rho) and np.isfinite(float(rho)):
        recovered = x_state + max(0.0, min(1.0, float(rho))) * max(0.0, hmax_t - x_state)
    else:
        recovered = x_state + (float(gain) if pd.notna(gain) else 0.0)
    return max(x_state, min(recovered, hmax_t))


def _simulate_device(
    param: pd.Series,
    policy: pd.Series,
    purchase_cost: float,
    medium_cost: float,
    major_cost: float,
    simulation_years: int = SIMULATION_YEARS,
) -> dict[str, object]:
    days = int(simulation_years * 365)
    x_state = float(param["initial_x_state"])
    hmax_initial = float(param["h_max_initial"])
    hmax_trend = float(param.get("hmax_trend_used", 0.0))
    decay = float(param.get("cycle_decay_rate_used", -0.1))
    last_medium_day = -10**9
    last_major_day = -10**9
    n_medium = 0
    n_major = 0
    n_replace = 0
    days_below_37 = 0
    rolling_values = [float(param["current_state_level"])] * 365
    rolling_sum = sum(rolling_values)
    min_rolling = math.inf
    status = "completed"

    for day in range(days):
        hmax_t = max(0.0, hmax_initial + hmax_trend * day)
        current_level = x_state
        maintenance_type = ""
        if str(policy["policy_type"]) == "fixed_current_rule":
            if day - last_major_day >= float(policy["I_B_min"]):
                maintenance_type = "major"
            elif day - last_medium_day >= float(policy["I_M_min"]):
                maintenance_type = "medium"
        else:
            if (
                current_level <= float(policy["theta_B"])
                and day - last_major_day >= float(policy["I_B_min"])
            ):
                maintenance_type = "major"
            elif (
                current_level <= float(policy["theta_M"])
                and day - last_medium_day >= float(policy["I_M_min"])
            ):
                maintenance_type = "medium"

        if maintenance_type:
            x_state = _apply_maintenance(param, maintenance_type, x_state, hmax_t)
            if maintenance_type == "medium":
                n_medium += 1
                last_medium_day = day
            else:
                n_major += 1
                last_major_day = day
                last_medium_day = day
            if x_state < THRESHOLD:
                n_replace += 1
                x_state = hmax_initial
                hmax_t = hmax_initial
                last_medium_day = day
                last_major_day = day
        else:
            x_state = x_state + decay

        x_state = min(x_state, hmax_t)
        rolling_sum += x_state
        rolling_values.append(x_state)
        if len(rolling_values) > 365:
            rolling_sum -= rolling_values.pop(0)
        rolling = rolling_sum / len(rolling_values)
        min_rolling = min(min_rolling, rolling)
        if rolling < THRESHOLD:
            days_below_37 += 1
            if hmax_t < THRESHOLD:
                n_replace += 1
                x_state = hmax_initial
                rolling_values = [x_state] * 365
                rolling_sum = sum(rolling_values)

    total_cost = purchase_cost * n_replace + medium_cost * n_medium + major_cost * n_major
    return {
        "policy_name": policy["policy_name"],
        "device_id": param["device_id"],
        "theta_M": policy["theta_M"],
        "theta_B": policy["theta_B"],
        "I_M_min": policy["I_M_min"],
        "I_B_min": policy["I_B_min"],
        "n_medium": n_medium,
        "n_major": n_major,
        "n_replace": n_replace,
        "total_cost": total_cost,
        "simulation_years": simulation_years,
        "annual_cost": total_cost / simulation_years,
        "days_below_37": days_below_37,
        "min_rolling365": min_rolling if np.isfinite(min_rolling) else np.nan,
        "status": status,
    }


def _simulate_policies(
    params: pd.DataFrame,
    policies: pd.DataFrame,
    purchase_cost: float = PURCHASE_COST,
    medium_cost: float = MEDIUM_COST,
    major_cost: float = MAJOR_COST,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, policy in policies.iterrows():
        for _, param in params.iterrows():
            rows.append(_simulate_device(param, policy, purchase_cost, medium_cost, major_cost))
    return pd.DataFrame(rows)


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("policy_name", as_index=False).agg(
        annual_cost=("annual_cost", "mean"),
        total_cost=("total_cost", "sum"),
        n_medium=("n_medium", "sum"),
        n_major=("n_major", "sum"),
        n_replace=("n_replace", "sum"),
        days_below_37=("days_below_37", "sum"),
        min_rolling365=("min_rolling365", "min"),
    )
    grouped["rank_by_annual_cost"] = grouped["annual_cost"].rank(method="dense", ascending=True).astype(int)
    return grouped.sort_values(["rank_by_annual_cost", "policy_name"]).reset_index(drop=True)


def _select_four_policies(results: pd.DataFrame, policies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _summarize(results)
    best_grid = summary[summary["policy_name"].astype(str).str.startswith("grid_")].head(1)
    if best_grid.empty:
        best_name = "delayed_policy"
    else:
        best_name = str(best_grid["policy_name"].iloc[0])
    selected_names = ["current_policy", "conservative_policy", "delayed_policy", best_name]
    selected_results = results[results["policy_name"].isin(selected_names)].copy()
    selected_policies = policies[policies["policy_name"].isin(selected_names)].copy()
    selected_policies.loc[selected_policies["policy_name"] == best_name, "policy_name"] = "optimized_policy"
    selected_results.loc[selected_results["policy_name"] == best_name, "policy_name"] = "optimized_policy"
    return selected_policies, selected_results


def _cost_sensitivity(params: pd.DataFrame, policies: pd.DataFrame, base_best: str) -> pd.DataFrame:
    scenarios = [
        ("purchase_cost_0.8", PURCHASE_COST * 0.8, MEDIUM_COST, MAJOR_COST),
        ("purchase_cost_1.2", PURCHASE_COST * 1.2, MEDIUM_COST, MAJOR_COST),
        ("medium_cost_0.8", PURCHASE_COST, MEDIUM_COST * 0.8, MAJOR_COST),
        ("medium_cost_1.2", PURCHASE_COST, MEDIUM_COST * 1.2, MAJOR_COST),
        ("major_cost_0.8", PURCHASE_COST, MEDIUM_COST, MAJOR_COST * 0.8),
        ("major_cost_1.2", PURCHASE_COST, MEDIUM_COST, MAJOR_COST * 1.2),
    ]
    rows: list[dict[str, object]] = []
    for scenario, purchase, medium, major in scenarios:
        results = _simulate_policies(params, policies, purchase, medium, major)
        summary = _summarize(results)
        best = summary.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "purchase_cost": purchase,
                "medium_cost": medium,
                "major_cost": major,
                "best_policy_name": best["policy_name"],
                "best_annual_cost": best["annual_cost"],
                "policy_changed": str(best["policy_name"]) != base_best,
                "interpretation": "成本扰动下最优策略发生变化。" if str(best["policy_name"]) != base_best else "成本扰动下最优策略保持稳定。",
            }
        )
    return pd.DataFrame(rows)


def _write_summary(path: Path, comparison: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    best = comparison.sort_values("annual_cost").iloc[0]
    content = f"""# 第三问分析总结

## 1. 问题定位

第三问不重新发明模型，而是把第二问状态转移模型作为仿真器，比较不同维护策略下的长期成本和功能风险。

## 2. 策略设置

本次只实现四类策略：当前策略、保守策略、延迟策略和网格搜索得到的优化策略。策略变量为 `theta_M`、`theta_B`、`I_M_min` 和 `I_B_min`。

若状态低于 `theta_B` 且满足大维护最小间隔，则执行大维护；否则若状态低于 `theta_M` 且满足中维护最小间隔，则执行中维护；若维护后仍无法恢复到阈值，则更换。

## 3. 成本函数

年均成本定义为：

```latex
C_{{avg}}(\\pi)=\\frac{{300N_R+3N_M+12N_B}}{{T}}
```

其中 `N_R` 为更换次数，`N_M` 为中维护次数，`N_B` 为大维护次数，`T` 为模拟年数。

## 4. 主要结果

当前最低年均成本策略为 `{best['policy_name']}`，平均年成本为 {float(best['annual_cost']):.3f}。

第三问结果只用于策略优化，不反向修改第二问表06主寿命结果，也不把维护负担作为第二问硬终止。

## 5. 成本敏感性

成本敏感性只做单因素扰动：购买成本、中维护成本和大维护成本分别取 0.8 与 1.2 倍。敏感性表用于说明推荐策略对成本波动是否稳定。

## 6. 论文口径

本文没有进行大量形式化检验，而是围绕建模链条设置必要验证：第一问通过 Spearman 相关性和维护前后非参数检验说明关键因素与维护恢复作用存在；第二问通过历史回测和简化消融说明事件驱动退化模型优于不显式考虑维护恢复的基准；第三问将第二问模型作为仿真器，通过策略对比与成本敏感性分析验证维护方案的经济性和稳健性。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    paths = get_paths()
    ensure_output_dirs(paths)
    params = _read_table(paths.q2_tables_dir, "表01")
    policies = _policy_rows(params)
    all_results = _simulate_policies(params, policies)
    all_summary = _summarize(all_results)
    base_best = str(all_summary[all_summary["policy_name"].astype(str).str.startswith("grid_")]["policy_name"].iloc[0])
    selected_policies, selected_results = _select_four_policies(all_results, policies)
    comparison = _summarize(selected_results)
    selected_best = str(comparison.sort_values(["annual_cost", "policy_name"]).iloc[0]["policy_name"])
    sensitivity = _cost_sensitivity(params, selected_policies, selected_best)

    _write_csv(selected_policies, paths.q3_tables_dir / "表01_策略参数表.csv")
    _write_csv(selected_results, paths.q3_tables_dir / "表02_策略仿真结果.csv")
    _write_csv(comparison, paths.q3_tables_dir / "表03_策略对比汇总表.csv")
    _write_csv(sensitivity, paths.q3_tables_dir / "表04_成本敏感性分析.csv")
    _write_summary(paths.q3_markdown_dir / "第三问分析总结.md", comparison, sensitivity)
    print("v2 q3 pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"v2 q3 pipeline failed: {exc}", file=sys.stderr)
        raise
