# 频率表示研究项目文档

## 项目概述

本项目研究偏微分方程（PDE）解族频率轨迹在不同表示空间中的平滑性、低维性和插值性能。

### 核心研究问题
1. 在哪个表示空间中，频率轨迹 w → u_w 更平滑？
2. 在哪个表示空间中，频率轨迹维度更低？
3. 哪个表示空间在缺失频率插值方面表现更好？
4. 这些结论在不同演示族之间是否一致？

## 当前实现状态

### 已完成模块

#### 1. 数据生成模块
- **phase_family**: `u_w(x,y) = A(x,y) * cos(w * tau(x,y) + phi(x,y))`
  - 支持平滑随机振幅场A(x,y)、传播时间场tau(x,y)、相位偏移场phi(x,y)
  - 固定A、tau、phi，在频率网格上扫描w
- **screened_poisson**: `(Δ - α²) u_α = s(x,y)`
  - 在周期2D域上通过FFT求解
  - 固定源场s(x,y)，在参数α上扫描

#### 2. 表示模块
- **raw**: 原始表示 - 将2D场展平为1D向量
- **fourier**: 傅里叶表示 - 使用FFT系数
- **amplitude_phase**: 振幅相位表示 - 通过希尔伯特变换提取振幅和相位
- **pca**: PCA表示 - 在训练数据上拟合PCA并投影

#### 3. 度量模块
- **local_smoothness**: 局部平滑度 - S(δ) = mean ‖r(w+δ) - r(w)‖ / |δ|
- **curvature**: 曲率 - C(w_i) = ‖r(w_{i+1}) - 2r(w_i) + r(w_{i-1})‖
- **intrinsic_rank**: 内在秩 - 使用PCA解释方差比
- **interpolation_error**: 插值误差 - 隐藏中间频率，在表示空间中插值，与真实值比较

#### 4. 实验框架
- **run_benchmark**: 主基准测试编排函数
  - 根据配置生成数据集
  - 应用所有表示（在训练数据上拟合）
  - 计算所有度量（按轨迹和按划分）
  - 保存结果（配置快照、pickle文件、摘要）

### 配置文件结构

```
configs/
├── experiments/          # 实验配置
│   └── full_grid_small.yaml  # 小规模全网格实验
└── demos/               # 演示族配置
    ├── phase_family.yaml
    └── screened_poisson.yaml
```

### 输出结构

```
outputs/
├── datasets/            # 原始轨迹数据
├── representations/     # 缓存表示变换
├── metrics/            # 度量表格（按轨迹、聚合）
├── figures/            # 可视化
├── tables/             # 格式化结果表格
└── summaries/          # Markdown摘要
```

## 技术规格

### 数据格式
- 一个样本 = 一个频率轨迹: {u_w1, u_w2, ..., u_wM}
- 字段形状: [M, H, W]，其中：
  - M: 频率点数
  - H: 垂直分辨率
  - W: 水平分辨率
- 当前范围: 仅2D场，无几何结构，无完整生成建模

### 表示模块接口
所有表示模块必须实现：
- `fit(train_data, config)`: 在训练数据上拟合（如果需要）
- `transform(trajectory)`: 将轨迹转换为表示空间
- `inverse_transform(repr_trajectory)`: 从表示空间重构（如果可能）
- `metadata()`: 返回表示元数据

### 度量模块接口
所有度量模块必须实现：
- `metric_function(trajectory, frequencies)`: 计算度量值
- 返回标量值或数组（按点度量）

## 当前开发阶段

### 阶段1：基础实现 ✅
- 创建项目骨架
- 实现基本模块结构
- 建立配置驱动框架

### 阶段2：数据集生成 ✅
- 实现 phase_family 数据生成
  - 生成形状 [num_trajectories, num_frequencies, resolution, resolution] 的数据集
  - 保存预览图到 `outputs/datasets/phase_family_preview/`
- 实现 screened_poisson 数据生成
  - 支持 `alpha_grid` 和 `frequency_grid` 配置格式
  - 支持 `gaussian_blobs` 源类型（随机高斯斑点）
  - 保存预览图到 `outputs/datasets/screened_poisson_preview/`

### 阶段3：基础表示和度量 ✅
- 实现 raw 表示
  - 将2D场展平为1D向量
  - 支持逆变换重构
- 实现 fourier 表示
  - 使用FFT系数
  - 支持保留全部系数或top-k选择
- 实现 local_smoothness 度量
  - S(δ) = mean ‖r(w+δ) - r(w)‖ / |δ|
- 实现 curvature 度量
  - C(w_i) = ‖r(w_{i+1}) - 2r(w_i) + r(w_{i-1})‖

### 阶段4：小型基准测试 ✅
- 创建测试配置 `configs/experiments/small_test.yaml`
  - 10条轨迹（70%训练，10%验证，20%测试）
  - raw 和 fourier 表示
  - local_smoothness 和 curvature 度量
- 运行成功，结果保存到 `outputs/small_test/`
- 结果摘要：
  - **phase_family**:
    - raw: smoothness=20.57±0.18, curvature=3.44±0.09
    - fourier: smoothness=1316±11.26, curvature=219.9±5.79
  - **screened_poisson**:
    - raw: smoothness=11.83±5.08, curvature=0.14±0.06
    - fourier: smoothness=757.1±325, curvature=9.11±4.03

### 阶段5：完整实现 ✅
- 实现所有表示（amplitude_phase, pca）✅
- 实现所有度量（intrinsic_rank, interpolation_error）✅
- 为所有度量和表示添加中文文档说明物理意义和优劣 ✅
- 验证和测试所有模块功能 ✅
- 运行完整基准测试（待进行）

### 阶段6：分析和报告
- 生成比较图表
- 撰写结果摘要
- 提供下一步建议

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行基准测试
```bash
python -m src.main --config configs/experiments/full_grid_small.yaml
```

### 运行小型测试
```bash
python -m src.main --config configs/experiments/test_small.yaml
```

## 开发原则

1. **尊重上下文文件** - 在实现新功能前阅读 context/*.md
2. **配置驱动** - 避免硬编码参数，添加配置选项
3. **模块化可重用代码** - 优先扩展现有模块，避免一次性脚本
4. **可重现输出** - 始终保存配置、随机种子和文件清单
5. **最小范围** - 不引入深度学习组件或改变研究方向而不更新上下文文件

## 实验规则

1. 始终从最小配置开始
2. 不添加新演示或度量而不更新上下文文件
3. 为每次运行保存配置快照、随机种子和输出清单
4. 在轨迹级别使用固定的训练/验证/测试划分
5. 每次基准测试运行只改变一个主要因素

## 文件清单

### 核心源代码
```
src/
├── data_gen/
│   ├── phase_family.py          # 相位族数据生成
│   └── screened_poisson.py      # 屏蔽泊松数据生成
├── representations/
│   ├── raw_repr.py              # 原始表示
│   ├── fourier_repr.py          # 傅里叶表示
│   ├── amplitude_phase_repr.py  # 振幅相位表示
│   └── pca_repr.py              # PCA表示
├── metrics/
│   ├── smoothness.py            # 平滑度度量
│   ├── curvature.py             # 曲率度量
│   ├── intrinsic_rank.py        # 内在秩度量
│   └── interpolation_error.py   # 插值误差度量
├── experiments/
│   └── run_benchmark.py         # 基准测试编排
└── main.py                      # 主入口点
```

### 配置文件
```
configs/
├── experiments/
│   ├── full_grid_small.yaml     # 小规模全网格实验
│   └── test_small.yaml          # 小型测试配置
└── demos/
    ├── phase_family.yaml        # 相位族配置
    └── screened_poisson.yaml    # 屏蔽泊松配置
```

### 上下文文档
```
context/
├── PROJECT_OVERVIEW.md          # 项目概述
├── RESEARCH_QUESTION.md         # 研究问题
├── DEMO_SPEC.md                # 演示规格
├── REPRESENTATIONS.md          # 表示规范
├── METRICS.md                  # 度量规范
├── EXPERIMENT_RULES.md         # 实验规则
├── AGENT_WORKFLOW.md           # 工作流程
└── REPORT_TEMPLATE.md          # 报告模板
```

## 基准测试结果

### 实验设置
- 配置：`configs/experiments/full_grid_medium.yaml`
- 轨迹数量：30条（70%训练，10%验证，20%测试）
- 演示族：phase_family, screened_poisson
- 表示空间：raw, fourier, amplitude_phase, pca（全部4种）
- 度量：local_smoothness, curvature, intrinsic_rank, interpolation_error（全部4种）

### 关键发现
1. **PCA表示表现最优**：在两个演示族上都提供最平滑、最低维、最易插值的表示
   - phase_family: 内在秩=1.0（完美低维），平滑度=20.09，曲率=3.372，插值误差=1.67
   - screened_poisson: 平滑度=8.998（最低），曲率=0.1066（最低），插值误差=0.09093（最低）

2. **Raw表示作为强基线**：性能接近PCA，内在秩稍高但其他度量相近

3. **Fourier表示表现最差**：平滑度和曲率极高，插值误差大，不适合频率轨迹建模

4. **Amplitude-phase表示中等**：内在秩较高，降维效果有限

### 研究问题初步答案
1. **更平滑的表示空间**：PCA和raw表示最平滑
2. **更低维的表示空间**：PCA表示维度最低（内在秩接近1）
3. **更好的插值性能**：PCA和raw表示插值误差最小
4. **跨演示族一致性**：结论基本一致，PCA在两个演示族上都表现最优

详细分析见：`outputs/full_grid_medium/analysis.md`

### 高频实验结果验证
为验证用户关于"原始图像对于频率不应该平滑"的疑问，进行了高频实验：

#### 实验设置
- 配置：`configs/experiments/high_freq_test.yaml`
- 轨迹数量：20条，分辨率128×128
- phase_family频率：20.0-200.0（10倍于原始范围）
- screened_poisson α参数：5.0-100.0对数尺度（10倍于原始范围）

#### 关键发现
1. **phase_family高频效应显著**：
   - raw表示平滑度增加50%（20.56→31.68）
   - raw表示曲率激增100倍（3.44→364.4）
   - raw表示插值误差增加100倍（1.70→182.9）
   - 验证了高频下raw表示确实更不平滑

2. **screened_poisson相反趋势**：
   - raw表示平滑度降低99%（10.17→0.0296）
   - 大α使解更局部化，变化更简单

3. **表示性能排名**：
   - phase_family高频下：所有表示性能恶化，但PCA相对最优
   - screened_poisson高频下：所有表示性能改善，PCA最优
   - Fourier表示始终最差

#### 结论
- 原始实验中的平滑度较低是因为频率范围较低（2.0-20.0）
- 高频验证了用户的直觉：原始图像对频率变化确实更敏感
- 但PDE类型对频率尺度的影响不同：phase_family（振荡频率）vs screened_poisson（衰减参数）

详细分析见：`outputs/high_freq_test/analysis.md`

## 下一步计划

### ✅ 已完成任务
1. **phase_family 数据生成** - 已实现，包含预览图，测试通过
2. **screened_poisson 数据生成** - 已实现，解决配置兼容性问题，测试通过
3. **raw 和 fourier 表示 + smoothness 和 curvature 度量** - 已实现并集成
4. **10条轨迹的小型基准测试** - 已成功运行，流水线完全通畅
5. **amplitude_phase 和 pca 表示** - 已实现并测试通过
6. **intrinsic_rank 和 interpolation_error 度量** - 已完善实现并添加中文文档
7. **所有度量和表示的中文文档** - 已添加物理意义和优劣说明（low better/high better）
8. **Git仓库维护** - 创建.gitignore，及时commit & push，避免大文件
9. **完整基准测试运行和结果分析** - 已完成30条轨迹的完整测试，生成详细分析报告
10. **高频实验验证** - 已验证高频下raw表示平滑度恶化，曲率激增，支持用户的第一性原理直觉

### 中期任务
1. **复杂PDE实验** - 更新context文件，实现亥姆霍兹方程和波动方程，验证表示在更真实PDE上的性能
2. 增加轨迹数量和分辨率验证结论的稳定性
3. 实现结果可视化（图表生成）
4. 尝试其他表示（如小波变换、自动编码器）
5. 基于最优表示构建生成模型进行外推预测

### 长期任务
1. 扩展到3D场和几何结构
2. 集成生成建模组件
3. 应用于真实PDE问题

---

*文档更新时间：2026-03-18*
*项目版本：0.5.0*