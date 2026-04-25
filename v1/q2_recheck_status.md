# Q2 重新检查状态说明

更新时间：2026-04-25

本文档只用于当前工作复盘，不作为论文正文。当前先不继续第三问，核心任务是把第二问从“已经跑出结果”拉回到“实现是否符合题意、参数是否可信、结果是否可解释”重新检查。

## 0. 当前总判断

Q1 的新增产物整体已经比较完整，尤其是缺失分类、rho 恢复比例、维护规律表和检查表都已经生成。

Q2 不能直接作为最终结论使用。当前主要问题不是“没有结果”，而是结果口径还不够可信：恢复比例模型的实现、Hmax 的时间含义、寿命恢复测试逻辑、`current_rolling365` 字段、2022 投用寿命口径和回测解释都需要重新检查。

一句话：

> 现在 Q2 的方向可以保留，但当前寿命预测结果先不要写进最终论文结论。

## 1. 我这轮实际检查了什么

本轮只做状态核查和文档记录，没有继续写第三问，也没有修改 Q2 代码。

已检查文件：

```text
src/run_pipeline.py
src/modeling/run_q2.py
strategy_2.md
template_1.md
data/q_1/tables/q1_maintenance_recovery_ratio.csv
data/q_1/tables/q1_check_rho_estimation_summary.csv
data/q_2/tables/q2_backtest_metrics.csv
data/q_2/tables/q2_life_prediction_results.csv
data/q_2/tables/q2_check_maintenance_schedule_summary.csv
```

注意：当前本地工作区和你看到的 GitHub 远端可能不同步。本地 `src/run_pipeline.py` 已经包含：

```python
from modeling.run_q2 import run_q2
...
run_q2(paths)
```

如果 GitHub 上仍然没有调用 Q2，说明远端版本落后或尚未同步。这一点后面要单独确认。

## 2. 已经做成的部分

### 2.1 Q1 缺失分类和 rho 产物已经存在

本地 Q1 表中已经有：

```text
q1_check_gap_type_summary.csv
q1_check_rho_estimation_summary.csv
q1_maintenance_recovery_ratio.csv
q1_table_05_device_core_metrics.csv
```

`q1_maintenance_recovery_ratio.csv` 本地不是空表，共 18 行，字段包括：

```text
device_id
maintenance_type
n_eligible_events
rho_mean
rho_median
rho_q25
rho_q75
rho_std
rho_clipped_ratio
```

`q1_check_rho_estimation_summary.csv` 本地共 19 行，包含：

```text
n_rho_raw_negative
n_rho_raw_above_1
rho_clipped_ratio
rho_clipped_min
rho_clipped_max
```

当前检查结果显示 `rho_clipped` 位于 `[0,1]`，Q1 rho 的基础输出是有的。

### 2.2 Q2 表和图大多已经生成

本地 `data/q_2/tables` 中已经有：

```text
q2_model_parameters.csv
q2_life_prediction_results.csv
q2_backtest_metrics.csv
q2_check_maintenance_schedule_summary.csv
q2_future_simulation_path.csv
q2_recovery_test_results.csv
q2_device_parameters.csv
q2_lifetime_prediction.csv
```

也就是说，Q2 不是完全没跑过，而是“跑出来的东西需要重判”。

### 2.3 未来维护 schedule 检查表本地不是空表

本地 `q2_check_maintenance_schedule_summary.csv` 共 510 行。

按设备汇总后：

| 设备 | 年份数 | 年最大大维护次数 | 年最小大维护次数 | 降级次数 |
|---|---:|---:|---:|---:|
| a1 | 51 | 1 | 1 | 0 |
| a2 | 51 | 2 | 0 | 0 |
| a3 | 51 | 2 | 0 | 0 |
| a4 | 51 | 0 | 0 | 0 |
| a5 | 51 | 1 | 0 | 0 |
| a6 | 51 | 2 | 0 | 0 |
| a7 | 51 | 2 | 0 | 0 |
| a8 | 51 | 0 | 0 | 0 |
| a9 | 51 | 1 | 0 | 0 |
| a10 | 51 | 2 | 1 | 0 |

这说明本地当前表能证明“大维护不超过 4 次/年”，也允许某些设备某些年份为 0 次。

但它还不够好，因为现在字段较少，只有：

```text
device_id
year
n_major_before_cap
n_major_after_cap
n_major_downgraded_to_medium
cap_rule
```

后面最好补上：

```text
n_medium
future_maintenance_count_per_year
first_10_future_maintenance_dates
major_cap_violated_before_fix
major_cap_violated_after_fix
```

这样才能更直观看出每台设备自己的未来维护是否合理。

## 3. 当前确认存在的问题

### 3.1 `current_rolling365` 是明确 bug

代码里 `_build_q2_life_prediction_results()` 仍然是从 future 表每台设备最后一行取：

```python
future.sort_values("date").groupby("device_id").tail(1)
```

然后重命名为：

```text
current_rolling365
```

这不是当前滚动 365 天均值，而是未来模拟最后一天的滚动 365 天均值。

本地真实历史滚动值示例：

| 设备 | 历史最后日期 | 历史最后日中位数 | 历史 current rolling365 |
|---|---|---:|---:|
| a1 | 2026-04-10 | 92.66 | 90.18 |
| a2 | 2026-04-10 | 41.70 | 67.21 |
| a8 | 2026-04-10 | 61.20 | 80.71 |
| a10 | 2026-04-11 | 49.97 | 71.50 |

但 `q2_life_prediction_results.csv` 里 a10 的 `current_rolling365` 是 25.75，这明显是未来末端值，不是当前值。

结论：

> 当前 `q2_life_prediction_results.csv` 中的 `current_rolling365` 不能引用，必须修成历史真实滚动值，并新增 `final_simulated_rolling365`。

### 3.2 恢复比例维护日更新可能把状态拉低

当前代码里恢复比例维护更新仍是：

```python
after = before + rho * (h_target - before)
```

如果 `h_target < before`，维护后状态会下降。这不符合“维护能清除沉积物、使透水率增加”的主设定。

应该改为：

```python
gap = max(0.0, h_target - before)
after = before + rho * gap
```

维护损伤和长期老化应体现在 `Hmax` 下降，而不是某次维护日直接把状态往下拉。

### 3.3 Hmax 当前仍按维护次数推进，需要重判时间含义

当前模拟中有类似逻辑：

```python
h_max = h_max_initial + b_h_use * maintenance_index
```

这等价于：

$$
H_{i,k}^{max}=H_{i,0}^{max}+b_{H,i}^{use}k
$$

问题是：如果第一问的 $b_H$ 是按事件序号拟合，这样写是内部一致的；但题意里的老化和性能下降更接近随日历时间发生，而不是只在维护次数增加时发生。

下一步要判断：

1. 第一问的 $b_H$ 到底是按事件序号还是按真实日期估计；
2. 若是事件序号斜率，是否应换算为日斜率：

$$
b_{H,i}^{day}=\frac{b_{H,i}^{event}}{I_i}
$$

3. Q2 是否应改成：

$$
H_i^{max}(t)=H_{i,0}^{max}+b_{H,i}^{day}(t-t_0)
$$

### 3.4 寿命恢复测试目前不是当前固定维护规律下的主判据

当前 `_recovery_test` 的逻辑是：触发 rolling365 低于 37 后，立即强制大维护一次，然后未来 365 天只线性衰减，没有继续按当前维护 schedule 执行。

这更像“补救性大维护能力测试”，不是“按照当前固定维护规律还能不能恢复”。

更合理的寿命判定应拆成两层：

1. 当前固定维护规律下，触发后未来一年滚动年均能不能回到 37 以上；
2. 额外做强制大维护救援测试，判断如果立即大维护能否救回。

所以当前寿命判据需要重写，不应直接把强制大维护测试当作主判据。

### 3.5 回测结果显示恢复比例模型当前并不优于固定增益

当前 `q2_backtest_metrics.csv`：

| 模型 | MAE | RMSE | MAPE | rolling365 MAE |
|---|---:|---:|---:|---:|
| baseline_season_trend | 12.19 | 14.96 | 0.207 | 2.086 |
| fixed_gain_baseline | 9.12 | 11.15 | 0.143 | 1.299 |
| recovery_ratio_main | 10.68 | 12.89 | 0.182 | 1.455 |

当前结果里 `fixed_gain_baseline` 的误差低于 `recovery_ratio_main`。

这不能证明恢复比例思想错了，但说明当前恢复比例实现、Hmax 估计或 rho 参数还没调好。论文里不能写“恢复比例模型预测效果优于固定增益模型”。

目前只能写：

> 恢复比例模型机制解释更合理，但当前回测误差未优于固定增益模型，因此需要进一步检查 Hmax、rho 和维护日状态更新；固定增益模型保留为强对照模型。

### 3.6 strategy_2.md 内部口径仍有旧固定增益内容

`strategy_2.md` 前半部分仍有固定增益主模型公式，例如：

```text
X(t+) = min(X(t-) + G*, Hmax)
```

文件末尾虽然追加了“恢复比例主模型”的增量说明，但前后并没有彻底统一。这个会让读者搞不清到底哪个是主模型。

下一步应整体重写或清理 `strategy_2.md`，明确：

```text
recovery_ratio_main = 主模型
fixed_gain_baseline = 回测对照
```

同时在模型检验部分承认当前 fixed gain 回测更好，恢复比例需要继续校准。

### 3.7 template_1.md 也不能直接作为最终论文引用

`template_1.md` 已经加入了 Q2 内容，但它引用了当前 Q2 寿命结果，并写到了 a10 当前滚动年均约 25.75。

由于 `current_rolling365` 已确认取错，这部分必须暂时视为待修。

结论：

> `template_1.md` 可以作为结构草稿，但 Q2 结果段落目前不能作为最终论文结论。

### 3.8 2022 年投用信息还没有进入 Q2 输出

当前 Q2 输出主要给的是从预测起点开始的剩余寿命，没有明确区分：

```text
commission_date
prediction_start_date
age_at_prediction_start_years
remaining_life_years
predicted_total_life_years
```

题目说设备 2022 年 4 月投用，因此论文中必须区分：

$$
\text{剩余寿命}=T_i-t_{\text{prediction start}}
$$

$$
\text{总寿命}=T_i-t_{\text{commission}}
$$

否则很容易把剩余寿命写成总寿命。

### 3.9 回测是否严格 out-of-sample 还未确认

当前回测使用验证期真实维护事件日程，这可以用于“已知维护日程下的历史后段拟合检验”。

但如果 Q2 参数是由全量 Q1 输出得到，就可能包含验证期信息。严格 out-of-sample 应该在训练期重新估计 Q1/Q2 参数，再预测验证期。

所以当前 `q2_backtest_metrics.csv` 最多叫“历史后段近似回测”，不能写成严格样本外预测验证。

## 4. 现在不应该继续做什么

现在不要继续第三问。

原因是第二问还没有稳定。第三问会依赖 Q2 的寿命预测路径、维护恢复逻辑和维护规律。如果 Q2 里的 `current_rolling365`、Hmax、恢复比例更新和寿命判据还没修，第三问优化会建立在不稳基础上。

也不要急着把当前 Q2 寿命结果写进最终论文。

尤其不要写：

```text
恢复比例模型预测效果优于固定增益模型
```

也不要直接引用：

```text
a10 current_rolling365 = 25.75
```

## 5. 下一步必须按这个顺序做

### Step 1：修正并扩展 future schedule 检查

目标是确认每台设备自己的未来维护是否合理。

建议新增或扩展表：

```text
q2_check_future_schedule_reasonableness.csv
```

字段：

```text
device_id
I_i
major_ratio
future_rule_type
year
future_maintenance_count_per_year
future_medium_count_per_year
future_major_count_per_year
n_major_downgraded_by_cap
first_10_future_maintenance_dates
```

判断：

1. 是否逐设备使用自己的维护间隔和大维护比例；
2. 大维护是否满足每年 0 至 4 次；
3. 维护间隔是否接近历史中位间隔；
4. 是否存在某台设备未来维护过密或过稀。

### Step 2：重判 Hmax

需要输出：

```text
device_id
initial_state
h_max_initial
gap_hmax_initial
recent_post_recovery_levels
global_h_recent
b_H_raw
b_H_use
I_i
b_H_day_candidate
```

重点判断：

1. `h_max_initial` 是否明显高于当前状态；
2. $b_H$ 当前到底是每次维护事件斜率还是每日斜率；
3. 是否改成随日历时间下降的 $Hmax(t)$。

### Step 3：修维护日恢复比例更新

把：

```python
after = before + rho * (h_target - before)
```

改成：

```python
gap = max(0.0, h_target - before)
after = before + rho * gap
```

并输出维护日前后检查表：

```text
device_id
date
maintenance_type
x_before
h_max
gap
rho
x_after
delta
```

检查是否还有维护日 `delta < 0`。

### Step 4：修 `current_rolling365`

从历史真实日表最后一个可计算 rolling365 得到：

```text
current_real_rolling365
```

同时保留：

```text
final_sim_rolling365
first_date_rolling365_below_37
```

不要再把未来模拟最后一天叫 `current_rolling365`。

### Step 5：重写寿命判定

主判据应先按当前固定维护规律判断：

$$
M_i^{current}(t)=
\max_{s\in[t,t+365]}
\bar P_{i,365}^{current}(s)
$$

若触发后当前规律下未来一年仍不能回到 37 以上，则当前维护规律下寿命终止。

强制大维护只作为救援测试：

$$
M_i^{major}(t)
$$

它用于说明“立即做一次大维护能不能救回来”，不改变主预测路径。

### Step 6：加入 2022 投用寿命口径

Q2 寿命表中加入：

```text
commission_date
prediction_start_date
age_at_prediction_start_years
remaining_life_years
predicted_total_life_years
```

其中 `commission_date` 暂按题面 2022-04-01。

### Step 7：重新回测并决定主口径

修完上述 bug 后，再比较：

```text
baseline_season_trend
fixed_gain_baseline
recovery_ratio_main
recovery_ratio_time_hmax
```

指标：

```text
MAE
RMSE
post_maintenance_7d_MAE
rolling365_MAE
path_sanity_check
```

如果恢复比例仍明显差于固定增益，就不要硬说恢复比例数值更优。可以把固定增益作为短期预测主口径，把恢复比例作为机制模型或稳健性模型。

## 6. 当前文件可用性判断

| 文件 | 当前状态 | 是否可直接用于论文 |
|---|---|---|
| `q1_check_gap_type_summary.csv` | 已生成 | 可用 |
| `q1_check_rho_estimation_summary.csv` | 已生成 | 可用 |
| `q1_maintenance_recovery_ratio.csv` | 本地非空 | 可用，但建议确认远端 raw 展示 |
| `q1_table_05_device_core_metrics.csv` | 已含 rho 字段 | 可用 |
| `q2_model_parameters.csv` | 已生成 | 暂作诊断，不作最终 |
| `q2_backtest_metrics.csv` | 已生成 | 可用于指出 fixed gain 当前更优，但不能作为严格样本外结论 |
| `q2_life_prediction_results.csv` | 已生成 | 暂不可用，`current_rolling365` 有 bug |
| `q2_check_maintenance_schedule_summary.csv` | 本地非空 | 可作初查，但字段需扩展 |
| `strategy_2.md` | 新旧口径混杂 | 需重写 |
| `template_1.md` | 结构已补 Q2 | Q2 结果段需等修复后再改 |

## 7. 当前最重要的结论

Q2 现在不是“完全没做”，而是“已经跑出了结果，但还不能信到可以写最终论文结论”。

优先修：

```text
1. current_rolling365 取值
2. 维护日 rho 更新公式
3. Hmax 按事件还是按时间下降
4. 当前固定维护规律下的寿命判定
5. 2022 投用后的剩余寿命/总寿命口径
6. strategy_2.md 和 template_1.md 的文档统一
```

第三问先不要推进。等 Q2 的维护路径、状态更新、寿命判据和回测口径站稳，再进入第三问优化。

## 8. update2 补充问题记录

这一节把你针对 `update2` 提出的新问题并入当前状态板。结论不变：Q1 增量基本可用，Q2 结果输出和模型验证口径仍需重判。

### 8.1 update2 的最大硬 bug 仍是 `current_rolling365`

`q2_life_prediction_results.csv` 中：

```text
a10 current_rolling365 = 25.7455
a8  current_rolling365 = 40.8407
```

按字段语义，`current_rolling365` 应该表示预测起点最近 365 天真实均值。但代码实际从未来模拟末尾取：

```python
future.sort_values("date").groupby("device_id").tail(1)
```

因此它现在其实是：

```text
simulation_end_rolling365
```

不是：

```text
current_rolling365
```

必须改成从历史 `daily_median` 最近 365 天计算：

```python
usable = daily[
    daily["daily_quality"].isin(["high_quality", "low_quality"])
    & daily["daily_median"].notna()
].copy()
```

然后逐设备取最近 365 个有效日均值。后续输出应同时保留：

```text
current_rolling365
simulation_end_rolling365
first_date_rolling365_below_37
predicted_life_end_date
```

当前 `q2_life_prediction_results.csv` 不能直接作为寿命结论表引用。

### 8.2 recovery_ratio_main 当前回测不如 fixed_gain_baseline

当前 `q2_backtest_metrics.csv` 显示：

| 模型 | MAE | RMSE | rolling365_MAE |
|---|---:|---:|---:|
| fixed_gain_baseline | 9.116 | 11.147 | 1.299 |
| recovery_ratio_main | 10.685 | 12.888 | 1.455 |

这说明恢复比例模型目前在经验回测上没有优于固定增益模型。不能写：

```text
恢复比例模型预测效果更好
```

只能写：

> 恢复比例模型机制解释更合理，但当前回测误差高于固定增益模型，需要进一步检查 Hmax、rho 估计和维护日状态更新；固定增益模型保留为强 baseline。

`q2_ablation_metrics.csv` 也支持这个判断。本地结果为：

| 模型 | MAE | RMSE | rolling365_MAE |
|---|---:|---:|---:|
| M0_baseline | 12.190 | 14.964 | 2.086 |
| M1_decay | 28.979 | 34.641 | 4.925 |
| M2_decay_gain | 10.042 | 12.817 | 1.701 |
| M3_decay_gain_shrinkage | 10.697 | 12.905 | 1.455 |
| M4_full_hmax | 10.685 | 12.888 | 1.455 |

加入 shrinkage 和 Hmax 后，MAE 没有明显改善。这个不是说恢复比例思路错，而是说明当前参数和状态更新口径还没有校准好。

### 8.3 Hmax 尺度需要专门检查

当前 `q2_model_parameters.csv` 中很多设备的 `H0_max` 在 116 左右，a10 为 132 左右。如果 `H0_max` 偏高，恢复空间会很大，而实际恢复量由：

$$
\rho_{i,u}(H_{i,0}^{max}-X_{i,current})
$$

决定。若 $\rho$ 偏小或当前状态与 Hmax 差距估计不合理，就会导致维护恢复不足或恢复过强。

下一轮应新增：

```text
q2_check_hmax_scale.csv
```

建议字段：

```text
device_id
H0_max
recent_post_level_median
current_state
H0_minus_current_state
rho_medium_use
rho_major_use
median_actual_gain
model_implied_medium_gain_at_current
model_implied_major_gain_at_current
```

重点比较：

```text
历史实际 gain
vs
rho_use * (H0_max - current_state)
```

如果 implied gain 明显小于历史实际 gain，说明恢复比例模型维护恢复不足；如果明显大于历史实际 gain，说明模型恢复过强。

### 8.4 strategy_2.md 需要从“追加补丁”改成“整体统一”

当前 `strategy_2.md` 前面仍保留固定增益主模型：

```text
X(t+) = min(X(t-) + G, Hmax)
```

后面又追加恢复比例模型优先的增量说明。这种“旧正文 + 新补丁”的写法不适合作为最终策略文档。

下一轮需要整体清理成：

```markdown
## 主模型：恢复比例状态转移

X(t+) = X(t-) + rho_use * max(0, Hmax(t) - X(t-))

## 对照模型：固定增益模型

X(t+) = min(X(t-) + G_use, Hmax(t))
```

固定增益只能作为：

```text
fixed_gain_baseline
```

不能继续作为第二问主模型正文。

### 8.5 CSV / Markdown 换行问题本地初查

你提到 GitHub raw 视图中部分 CSV/Markdown 显示为 `Total lines: 1`。我本地检查如下：

| 文件 | 本地行数 | 本地字符数 | 初判 |
|---|---:|---:|---|
| `data/q_2/tables/q2_life_prediction_results.csv` | 12 | 860 | 本地不是单行 |
| `data/q_2/tables/q2_model_parameters.csv` | 12 | 2147 | 本地不是单行 |
| `data/q_1/tables/q1_maintenance_recovery_ratio.csv` | 20 | 2112 | 本地不是单行 |
| `data/q_2/markdown/q2_modeling_summary.md` | 14 | 303 | 本地不是单行，但内容偏短 |

因此，本地当前文件没有被压成一行。若 GitHub raw 仍显示一行，需要进一步检查：

1. 远端是否还是旧版本；
2. 提交时是否发生换行转换；
3. 写文件函数是否在某些路径下生成了不同版本；
4. Markdown 内容是否虽然有换行但过短，需要扩写。

### 8.6 Q1 rho 基本可用，但 Hmax 仍可能是 Q2 症结

Q1 目前已经输出事件级 rho 和设备级 rho 汇总，`rho_clipped` 在 `[0,1]` 内。Q1 的 rho 表不是主要问题。

更可能的问题是：

```text
Hmax 尺度
Hmax 未来下降方式
rho_use 与 Hmax 的组合
维护日状态更新公式
```

所以 Q1 rho 可以暂时作为已跑通，但 Q2 不能直接使用当前寿命结论。

## 9. update3 修复清单

下一轮建议命名为 `update3`，任务不是做第三问，而是修复 Q2 可解释性和可信度。

### A. 修复结果表 bug

1. 修复 `q2_life_prediction_results.csv` 中 `current_rolling365` 的计算。
2. 新增 `simulation_end_rolling365`，避免把未来末尾值误写成当前值。
3. 重新生成 `q2_life_prediction_results.csv`。

### B. 修复输出格式

1. 检查所有 CSV 和 Markdown 是否有正常换行。
2. 如果远端文件被写成一行，检查写文件函数和提交换行设置。
3. 本地用 Python `text.count("\n")` 或 PowerShell 行数检查验证。

### C. 模型验证口径修正

1. `q2_backtest_metrics.csv` 已显示 `fixed_gain_baseline` 优于 `recovery_ratio_main`。
2. 不再写“恢复比例模型预测效果更好”。
3. Markdown 里明确：恢复比例是机制模型，固定增益是经验强 baseline。
4. 若修复后 fixed gain 仍回测更优，则寿命预测应同时输出两套结果：

```text
predicted_life_end_date_recovery_ratio
predicted_life_end_date_fixed_gain
selected_model
```

`selected_model` 可选择回测更优或更保守的模型，并说明理由。

### D. 检查 Hmax 尺度

1. 新增 `q2_check_hmax_scale.csv`。
2. 比较历史实际 gain 与模型 implied gain。
3. 若恢复比例模型维护后恢复不足，优先检查 `H_max_est`、`H0_max`、`rho_use` 和状态更新公式。

### E. 清理 strategy_2.md

1. 删除旧固定增益主模型段落。
2. 固定增益只作为 baseline。
3. 恢复比例模型作为机制模型。
4. 大维护 0 至 4 次/年、小维护背景维护、rolling365 和恢复测试口径保留。

### F. 暂停第三问

第三问继续暂停。只有在 Q2 的以下问题修复后，才进入第三问：

```text
current_rolling365 正确
Hmax 尺度可信
维护日 rho 更新不反向拉低状态
寿命判定符合当前固定维护规律
回测口径可解释
strategy_2.md 与结果表一致
```
