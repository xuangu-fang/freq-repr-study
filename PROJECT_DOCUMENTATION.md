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
- **helmholtz**: `∇²u + k²u = f(x,y)`
  - 在周期2D域上通过FFT求解，添加小虚部ε避免共振奇点
  - 固定源场f(x,y)，在波数k上扫描
- **wave_equation**: `∂²u/∂t² = c²∇²u + f(x,y)`（频域求解）
  - 在频域中简化为亥姆霍兹方程：`∇²U + (ω²/c²)U = F(x,y)`
  - 参数：频率ω，波速c，波数k = ω/c

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
│   ├── screened_poisson.py      # 屏蔽泊松数据生成
│   ├── helmholtz.py             # 亥姆霍兹方程数据生成
│   └── wave_equation.py         # 波动方程数据生成（频域求解）
├── representations/
│   ├── raw_repr.py              # 原始表示
│   ├── fourier_repr.py          # 傅里叶表示
│   ├── amplitude_phase_repr.py  # 振幅相位表示
│   ├── pca_repr.py              # PCA表示
│   ├── real_imag_repr.py        # 实部-虚部表示（复数场）
│   ├── autoencoder_repr.py      # 自编码器表示（实部-虚部输入）
│   ├── amplitude_phase_autoencoder_repr.py  # 振幅-相位自编码器
│   └── enhanced_real_imag_autoencoder_repr.py  # 增强实部-虚部自编码器
├── metrics/
│   ├── smoothness.py            # 平滑度度量
│   ├── curvature.py             # 曲率度量
│   ├── intrinsic_rank.py        # 内在秩度量
│   └── interpolation_error.py   # 插值误差度量
├── experiments/
│   └── run_benchmark.py         # 基准测试编排
└── main.py                      # 主入口点
```

### 测试代码
```
tests/
├── test_helmholtz.py                 # 亥姆霍兹方程测试
├── test_helmholtz_complex.py         # 复数场亥姆霍兹测试
├── test_metrics.py                   # 度量模块测试
├── test_phase_family_dataset.py      # 相位族数据集测试
├── test_representations.py           # 表示模块测试
├── test_screened_poisson_config.py   # 屏蔽泊松配置测试
├── test_screened_poisson_dataset.py  # 屏蔽泊松数据集测试
└── test_wave_equation.py             # 波动方程测试
```

### 配置文件
```
configs/
├── experiments/
│   ├── full_grid_small.yaml     # 小规模全网格实验
│   ├── full_grid_medium.yaml    # 中等规模完整实验
│   ├── high_freq_test.yaml      # 高频实验
│   ├── helmholtz_test.yaml      # 亥姆霍兹方程实验
│   ├── helmholtz_complex_test.yaml  # 亥姆霍兹方程复数场测试
│   ├── helmholtz_complex_analysis_v2.yaml  # 亥姆霍兹复数场分析
│   ├── autoencoder_comparison.yaml  # 自编码器对比实验
│   ├── amplitude_phase_ae_no_unwrap.yaml  # 无解缠绕振幅-相位自编码器测试
│   ├── helmholtz_high_freq.yaml  # 亥姆霍兹高频实验
│   └── wave_equation_test.yaml  # 波动方程实验
└── demos/
    ├── phase_family.yaml        # 相位族配置
    ├── phase_family_high_freq.yaml  # 高频相位族配置
    ├── screened_poisson.yaml    # 屏蔽泊松配置
    ├── screened_poisson_high_freq.yaml  # 高频屏蔽泊松配置
    ├── helmholtz.yaml           # 亥姆霍兹方程配置（原始）
    ├── helmholtz_low_freq.yaml  # 亥姆霍兹方程低频配置（复数场）
    ├── helmholtz_high_freq.yaml # 亥姆霍兹方程高频配置
    ├── wave_equation.yaml       # 波动方程配置
    └── wave_equation_varying_c.yaml  # 波动方程变波速配置
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

### 亥姆霍兹方程实验
为扩展研究到更真实的PDE，实现了亥姆霍兹方程：

#### 实现细节
- **文件**：`src/data_gen/helmholtz.py`
- **方程**：∇²u + k²u = f(x,y) 在周期2D域上
- **求解方法**：FFT频域求解，添加小虚部ε避免共振奇点
- **参数**：波数k（类似频率参数）
- **源类型**：高斯斑点或平滑随机场

#### 实验设置
- 配置：`configs/experiments/helmholtz_test.yaml`
- 轨迹数量：15条，分辨率64×64
- 波数范围：1.0-20.0（线性）

#### 关键发现
1. **表示排名完全一致**：PCA > Raw > Amplitude-phase > Fourier 在亥姆霍兹方程上仍然成立
2. **亥姆霍兹方程特性**：作为线性PDE，产生相对平滑的频率轨迹
   - raw表示平滑度：1.291（远低于phase_family的20.56）
3. **跨PDE验证**：表示性能排名在三个不同PDE上完全一致，证明了框架的泛化能力

详细分析见：`outputs/helmholtz_test/analysis.md`

### 波动方程实验
实现波动方程频域求解，验证与亥姆霍兹方程的数学等价性：

#### 实现细节
- **文件**：`src/data_gen/wave_equation.py`
- **方程**：∂²u/∂t² = c²∇²u + f(x,y) 在周期2D域上
- **求解方法**：频域求解，简化为亥姆霍兹方程 ∇²U + (ω²/c²)U = F(x,y)
- **参数**：频率ω，波速c，波数k = ω/c

#### 实验设置
- 配置：`configs/experiments/wave_equation_test.yaml`
- 轨迹数量：15条，分辨率64×64
- 频率范围：1.0-20.0（线性），波速c=1.0

#### 重要发现
1. **数学等价验证**：波动方程（频域）与亥姆霍兹方程等价，实验结果完全一致
2. **参数映射**：当c=1.0时，频率ω = 波数k，两实验配置相同
3. **表示排名稳健性**：排名在四个不同PDE上完全一致
4. **物理扩展价值**：波动方程框架支持波速变化、时域求解等扩展

详细分析见：`outputs/wave_equation_test/analysis.md`

### 复数场与学习基表示实验（2026-03-19）

为研究复数场PDE的频率轨迹表示问题，实现了亥姆霍兹方程复数场支持和学习基表示方法。

#### 实现细节

1. **亥姆霍兹方程复数场支持**：
   - **修复问题**：原高频范围(20-100)下变化不明显，解只取实部丢失相位信息
   - **解决方案**：
     - 低频配置：`configs/demos/helmholtz_low_freq.yaml` (k=0.5-5.0)
     - 复数场支持：`src/data_gen/helmholtz.py` 添加`return_complex`配置选项
     - 表示模块更新：所有表示模块支持复数场输入
     - 可视化工具：`src/report/complex_trajectory_gif.py` 支持复数场GIF生成

2. **学习基表示方法**：
   - **Autoencoder表示**：卷积自编码器学习非线性表示
     - 输入设计：实部-虚部双通道图像
     - 架构：2层卷积编码器/解码器，潜空间维度32
     - 训练：20 epochs, batch size 8, learning rate 1e-3

   - **Amplitude-phase autoencoder变体**：
     - 输入：振幅和相位作为两个通道
     - 支持相位解缠绕沿频率维度

   - **Enhanced real-imag autoencoder变体**：
     - 深度残差架构（4层卷积，残差连接，批归一化）
     - 理论上更强的表示能力

3. **实验配置**：
   - 复数场实验：`configs/experiments/helmholtz_complex_analysis_v2.yaml`
   - 自编码器对比实验：`configs/experiments/autoencoder_comparison.yaml`
   - 轨迹数量：20条（60%训练，20%验证，20%测试）
   - 分辨率：64×64
   - 频率范围：k = 0.5-5.0（低频，保证明显变化）

#### 关键发现

1. **亥姆霍兹方程复数场实验结果**：

| 表示方法 | 平滑度 ↓ | 曲率 ↓ | 内在秩 ↓ | 插值误差 ↓ |
|----------|-----------|---------|-----------|-------------|
| **Autoencoder (学习基)** | **10.04 ± 5.652** | **1.772 ± 1.035** | **2.538 ± 0.627** | **1.248 ± 0.714** |
| PCA (线性) | 66.63 ± 25.61 | 12.40 ± 4.802 | 2.785 ± 0.501 | 8.222 ± 3.151 |
| Real-imag (实部-虚部) | 105.3 ± 22.35 | 21.27 ± 5.054 | 4.127 ± 1.499 | 12.96 ± 2.759 |
| Amplitude-phase (解缠绕) | 687.7 ± 71.98 | 143.8 ± 15.36 | 4.968 ± 0.313 | 73.87 ± 7.439 |
| Amplitude-phase (无解缠绕) | 997.6 ± 90.98 | 251.4 ± 23.91 | 27.79 ± 1.37 | 128.2 ± 12.18 |

**结论**：
- 学习基表示(Autoencoder)显著优于所有传统表示
- 相位解缠绕对传统振幅-相位表示至关重要
- 实部-虚部表示优于振幅-相位表示

2. **自编码器设计对比实验结果**：

| 自编码器变体 | 平滑度 ↓ | 曲率 ↓ | 内在秩 ↓ | 插值误差 ↓ | 训练损失 ↓ |
|--------------|-----------|---------|-----------|-------------|-------------|
| **Autoencoder (原始，实部-虚部)** | **7.914 ± 4.062** | **1.354 ± 0.706** | **2.549 ± 0.820** | **1.004 ± 0.536** | **0.0945** |
| Amplitude-phase autoencoder (有解缠绕) | 41.39 ± 5.94 | 7.872 ± 1.209 | 2.923 ± 0.475 | 4.269 ± 0.597 | 1.1854 |
| Amplitude-phase autoencoder (无解缠绕) | 72.35 ± 13.11 | 17.69 ± 3.344 | 7.884 ± 1.146 | 9.529 ± 1.718 | 1.1850 |
| Enhanced real-imag autoencoder | 159.5 ± 59.26 | 28.77 ± 12.09 | 1.108 ± 0.115 | 22.38 ± 8.951 | 0.1775 |

**关键发现**：
- **原始自编码器设计（实部-虚部输入）仍然最佳**
- **振幅-相位表示不适合学习基方法**：即使使用神经网络，性能仍显著较差
- **简单性优于复杂性**：简单2层卷积架构优于复杂残差架构
- **学习基方法能大幅减轻相位缠绕问题**：传统无解缠绕平滑度997.6 → 学习基无解缠绕72.35（13.8倍改进）
- **相位解缠绕对学习基方法仍然重要**：学习基无解缠绕平滑度72.35 → 有解缠绕41.39（1.75倍改进）

#### 研究问题答案（复数场特定问题）

1. **Q5**：对于复数场PDE，**实部-虚部表示更适合学习基方法**（平滑度7.914 vs 41.39，5.2倍优势）
2. **Q6**：相位解缠绕对传统表示至关重要，对学习基表示仍有帮助但影响较小
3. **Q7**：振幅-相位表示物理可解释性更好，但表示质量较差
4. **Q8**：振幅和相位随频率变化比实部和虚部更复杂（非线性）
5. **Q9**：学习基表示显著优于传统表示，验证了神经网络的有效性
6. **Q10**：存在明显的权衡：振幅-相位可解释性好但性能差，学习基性能最好但可解释性差

详细分析见：`reports/autoencoder_design_analysis.md`

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
11. **亥姆霍兹方程实现** - 已实现并测试通过，验证了表示性能排名在真实PDE上的泛化性
12. **波动方程实现** - 已实现（频域求解），验证了与亥姆霍兹方程的数学等价性
13. **复数场亥姆霍兹方程支持** - 已修复高频变化不明显问题，添加复数场生成和低频配置
14. **复数场可视化工具** - 已实现complex_trajectory_gif.py支持复数场GIF生成
15. **实部-虚部表示模块** - 已实现RealImagRepresentation支持复数场分解
16. **自编码器学习基表示** - 已实现AutoencoderRepresentation（实部-虚部输入）
17. **振幅-相位自编码器变体** - 已实现AmplitudePhaseAutoencoderRepresentation
18. **增强实部-虚部自编码器** - 已实现EnhancedRealImagAutoencoderRepresentation（残差架构）
19. **复数场表征学习对比实验** - 已运行并分析，验证学习基表示优势
20. **自编码器输入设计对比实验** - 已运行并分析，确认实部-虚部输入为最佳设计

### 中期任务（已部分完成）
1. **波动方程波速变化实验** - ✅ 已有配置，待运行
2. **高频亥姆霍兹实验** - ✅ 已有配置，待运行
3. **时域波动方程求解** - 实现有限差分时域求解，研究时域特性
4. **结果可视化增强** - 生成更多比较图表和可视化，包括复数场和自编码器对比
5. **其他PDE扩展** - 热方程、薛定谔方程等
6. **学习基表示优化** - 探索其他网络架构（U-Net、Vision Transformer等）
7. **潜空间分析** - 分析不同自编码器的潜空间结构差异
8. **扩展到其他复数场PDE** - 验证结论在其他复数场PDE中的普适性

### 长期任务
1. 扩展到3D场和几何结构
2. 集成生成建模组件
3. 应用于真实PDE问题

---

*文档更新时间：2026-03-19*
*项目版本：0.7.0*