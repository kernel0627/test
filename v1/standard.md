# 项目工作规范

## 1. 适用范围

本文件用于规定本项目在数据清洗、EDA、建模与论文整理之前的统一工作规范。

本项目围绕 `task_and_requirements` 中的过滤设备监测赛题展开。

当前阶段划分如下：
- 第 1 阶段：数据清洗
- 第 2 阶段：EDA 与描述性统计分析
- 建模与优化阶段暂缓，待清洗结果与 EDA 产物稳定后再开展

## 2. 环境规范

- 所有可执行脚本必须使用 Python `.py` 文件。
- 除非用户明确要求，否则不使用 notebook。
- 所有项目脚本必须放在 `src` 目录下。
- 所有 Python 执行必须使用 `torch_env` 环境。
- 推荐执行方式：

```powershell
conda run -n torch_env python .\src\<module>\<script_name>.py
```

- 源代码文件统一使用 UTF-8 编码。
- 文件名统一使用英文小写 snake_case 命名。

## 3. 读写边界

- 只读输入源：
  - `D:\mathmodeling\test\task_and_requirements\` 目录下的全部官方赛题文件
  - 包括题面 markdown 文件与两个官方 Excel 附件
- 不允许覆盖或手动修改原始赛题文件。
- 所有生成结果必须写入 `D:\mathmodeling\test\data`。
- 所有可复用代码必须写入 `D:\mathmodeling\test\src`。

## 4. 数据目录规范

本项目后续统一使用如下标准目录结构：

```text
data/
  01_cleaned/
  02_eda/
    figures/
    tables/
    markdown/
  90_meta/
    logs/
    checks/
```

目录含义如下：
- `01_cleaned`：清洗后、标准化后的数据集
- `02_eda/figures`：EDA 阶段生成的图像
- `02_eda/tables`：EDA 阶段生成的统计表、汇总表、缺失分析表、季节性分析表等
- `02_eda/markdown`：EDA 阶段生成的文字总结、解释说明、观察记录
- `90_meta/logs`：运行日志、脚本日志、执行记录
- `90_meta/checks`：数据校验结果、结构检查、行数检查、质量检查

补充说明：
- `data` 下已有的旧目录，如 `figs`、`tables`、`markdown`，不作为后续新产物的首选写入位置。
- 后续新生成的文件必须优先写入上述标准目录。
- 如果当前工作明确属于“第一问最终交付”，则统一收口到：
  - `data/q_1/cleaned/`
  - `data/q_1/tables/`
  - `data/q_1/figures/`
  - `data/q_1/markdown/`

## 5. `src` 代码目录规范

开始写代码后，统一采用如下结构：

```text
src/
  common/
  cleaning/
  eda/
```

各目录职责如下：
- `common`：公共工具、路径配置、常量、绘图辅助函数
- `cleaning`：原始数据读取、结构对齐、清洗、合并
- `eda`：数据概览、统计分析、可视化、诊断分析

## 6. 命名规范

### 6.1 通用命名规则

- 统一使用英文小写 snake_case。
- 文件名中不允许出现空格。
- 生成文件名中不使用中文。
- 不允许使用 `final`、`final2`、`new`、`temp`、`test1` 这类低信息量命名。
- 文件名应尽量体现“阶段 + 对象 + 粒度 + 内容”。

### 6.2 文件命名模板

统一采用：

```text
<stage>_<subject>_<grain>_<content>.<ext>
```

命名示例：
- `cleaned_filter_hourly_long.parquet`
- `cleaned_filter_hourly_with_maintenance.parquet`
- `eda_all_devices_daily_summary.csv`
- `eda_missingness_by_device.csv`
- `eda_monthly_seasonality_all_devices.png`
- `eda_device_a1_trend_with_maintenance.png`
- `eda_overview_findings.md`
- `check_cleaned_row_count.json`

### 6.3 建议使用的阶段前缀

- `cleaned`：清洗数据产物
- `eda`：EDA 结果
- `check`：校验结果
- `log`：日志文件

### 6.4 设备编号规范

- 读取原始 Excel 时，工作表名称可以保留为 `A_1` 到 `A_10`。
- 在生成的数据表中，标准化设备编号建议使用 `a1` 到 `a10` 或 `A1` 到 `A10`。
- 一旦第一份清洗后数据确定了标准设备编号格式，后续所有脚本和产物必须保持一致。

## 7. 数据产物规范

### 7.1 清洗阶段输出

数据清洗阶段应优先产出：
- 1 张标准化后的长表观测数据
- 1 张标准化后的维护事件表
- 视 EDA 需要生成 1 张观测数据与维护事件合并后的表

推荐输出文件：
- `data/01_cleaned/cleaned_filter_hourly_long.parquet`
- `data/01_cleaned/cleaned_filter_hourly_long.csv`
- `data/01_cleaned/cleaned_maintenance_events.csv`
- `data/01_cleaned/cleaned_filter_hourly_with_maintenance.parquet`

### 7.2 EDA 阶段输出

EDA 阶段至少应生成：
- 数据概况表
- 缺失值分析表
- 各设备描述性统计表
- 时间趋势图
- 季节性分析图
- 维护影响分析图与分析表
- 文字总结 markdown

## 8. 分析规范

- 每一张重要图表都应回答一个明确问题。
- EDA 必须能够通过 `src/eda` 中的脚本复现。
- 数据清洗逻辑必须能够通过 `src/cleaning` 中的脚本复现。
- 避免在 Excel 中做不可追溯的手动操作。
- 优先使用结构化输出格式，如 CSV、Parquet、JSON、PNG、Markdown。

当前阶段的优先顺序如下：
1. 确认数据结构与时间覆盖范围。
2. 检查缺失情况与异常值情况。
3. 标准化时间列、设备编号、维护类型标签。
4. 按设备和不同时间粒度输出描述性统计结果。
5. 分析趋势、季节性和维护导致的跳变现象。

## 9. 校验规范

每次清洗后至少检查以下内容：
- 清洗前后行数
- 缺失值数量
- 每台设备内部是否存在重复时间戳
- 每台设备的时间范围
- 每台设备、各维护类型的事件数量
- 观测数据与维护事件合并后的匹配情况

校验结果统一输出到：
- `data/90_meta/checks/`

## 10. 文档规范

- 重要发现应写成 Markdown 文字总结。
- Markdown 文件统一放入 `data/02_eda/markdown/`。
- Markdown 内容建议明确区分：
  - 观察
  - 证据
  - 解释
  - 下一步动作

## 11. 图表规范

- 图像文件名必须能体现图像的分析用途。
- 图表标题、坐标轴、图例必须可读。
- 即使图中使用中文，图像文件名仍必须保持英文 snake_case。
- 多设备图必须清晰区分不同设备。
- 时序图必须忠实保留时间轴信息。
- 画图不能默认直接使用任意库的默认样式，必须主动统一字体、字号、线宽、颜色、图例和留白。
- 论文最终静态图优先按“论文级样式”输出，交互探索图单独输出，不混用定位。

## 11.1 绘图工具原则

- 静态论文图优先使用适合高质量导出的方案，目标是稳定输出 `png/svg/pdf`。
- 交互探索图优先使用支持缩放、悬停和图例开关的方案，目标是输出 `html`。
- 工具选择原则：
  - 论文最终图：优先使用 `matplotlib` 风格的静态图方案
  - 交互探索图：优先使用 `pyecharts`
- 如果运行环境导致 `matplotlib` 静态导出不稳定，允许使用其他稳定方案生成论文级静态图，但必须保持：
  - 中文标题、中文横纵轴、中文图例
  - 统一颜色方案
  - 统一尺寸和版式
  - 高清导出

## 11.2 绘图输出建议

- 静态图建议输出到：
  - `data/02_eda/figures/static/`
- 交互图建议输出到：
  - `data/02_eda/figures/html/`
- 维护相关颜色必须固定：
  - 中维护：固定一种颜色
  - 大维护：固定一种颜色
  - 维护日缺失：灰色或特殊标记
  - 随机缺失：浅灰色
- 最终交付时，图像产物原则上只保留 `png`；`html/svg` 仅可作为中间生成过程使用，完成转换后应删除。

## 12. 当前回合约束

- 当前优先任务是规范定义与工作区组织。
- 在标准与输出目录固定之前，不开始编写分析代码。
- 本文件完成后的下一步工作顺序是：
  - 创建标准数据目录
  - 在 `src/cleaning` 中开始数据清洗脚本
  - 在 `src/eda` 中开始 EDA 脚本

## 13. 变更控制

- 如果后续需要调整目录结构或命名规则，必须先更新本文件。
- 所有新产物都必须能够追溯到其生成脚本。
- 不允许脚本在相同文件名下静默改变既有产物的含义。
