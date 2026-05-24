# PDF文献整理与项目公式汇总

## 1. 当前 PDF 文献整理

### 1.1 预测类

1. [基于集成模型的污水处理厂出水总氮预测方法_姚怡帆.pdf](/D:/part3data/文献/基于集成模型的污水处理厂出水总氮预测方法_姚怡帆.pdf)
   重点：`TN` 预测、`Stacking` 集成、`RMSE/MAE/R2` 评价、灰色关联筛特征。
   价值：适合直接支撑项目里的“出水总氮预判”模块。

2. [基于深度学习的污水处理厂出水总磷预测方法_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水总磷预测方法_安昱宁.pdf)
   重点：`TP` 预测、`LSTM` 与 `Informer` 对比、`MSE/RMSE/MAE/R2`。
   价值：适合支撑“总磷前馈预测 + 加药优化”。

3. [基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf)
   重点：完整硕士论文，包含数据清洗、标准化、`TP/TN` 预测、投药优化。
   价值：这是当前目录里公式最全的一份材料，建议作为主公式来源。

4. [Research on prediction algorithm of effluent quality and development of integrated control system for waste-water treatment.pdf](/D:/part3data/文献/Research%20on%20prediction%20algorithm%20of%20effluent%20quality%20and%20development%20of%20integrated%20control%20system%20for%20waste-water%20treatment.pdf)
   重点：`CNN + LSTM + GRU + QR-RF`，同时做预测和 ICS 集成控制。
   价值：适合支撑“预测与控制一体化系统”。

5. [Integrated-real-time-intelligent-control-for-wastewater-treatment-plants-Data-driven-modeling.pdf](/D:/part3data/文献/Integrated-real-time-intelligent-control-for-wastewater-treatment-plants-Data-driven-modeling.pdf)
   重点：`Water Research 2025`，`ASM2D + 高阶导数特征提取 + 数据驱动模型 + 实时动态控制`。
   价值：这是目前目录里最新、也最接近“全流程实时智能控制”的顶刊论文之一，适合支撑“少变量、低成本、实时更新”的控制策略设计。

### 1.2 加药控制类

1. [典型城镇污水处理厂碳源智能投加控制生产性试验_吴宇行.pdf](/D:/part3data/文献/典型城镇污水处理厂碳源智能投加控制生产性试验_吴宇行.pdf)
   重点：碳源智能投加、生产性试验、前后加药点模型、反馈修正。
   价值：非常适合解决“响应滞后、加药不准、药耗高”。

2. [基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf](/D:/part3data/文献/基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf)
   重点：`Self-Attention + MPC + PSO`。
   价值：适合做你项目里“预测驱动加药优化”的高级版本。

3. [污水厂化学强化除磷投药量精准自动控制_李振华.pdf](/D:/part3data/文献/污水厂化学强化除磷投药量精准自动控制_李振华.pdf)
   重点：`CEPRM` 半经验除磷投药模型。
   价值：非常适合作为“加药机理约束”来源，避免全黑箱。

### 1.3 曝气与节能控制类

1. [基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf](/D:/part3data/文献/基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf)
   重点：工况感知、软测量、`PSO` 求最优 `DO` 设定值、经济指标 `EC`。
   价值：适合构建“智能曝气优化器”。

2. [曝气精确控制实现污水处理厂节能降耗的应用_荆玉姝.pdf](/D:/part3data/文献/曝气精确控制实现污水处理厂节能降耗的应用_荆玉姝.pdf)
   重点：全规模厂实际运行、按需曝气、`DO` 设定值控制、节能降耗。
   价值：适合做工程落地与效果证明。

3. [A~2O工艺污水处理厂优化脱氮及节能效果分析_顾升波.pdf](/D:/part3data/文献/A~2O工艺污水处理厂优化脱氮及节能效果分析_顾升波.pdf)
   重点：分段 `DO` 设定、全规模调试、供气节省与脱氮提升。
   价值：适合作为曝气控制的经验规则边界。

### 1.4 耦合控制与工艺优化类

1. [Bardenpho工艺内回流与碳源投加耦合控制动态模拟_孙月娣.pdf](/D:/part3data/文献/Bardenpho工艺内回流与碳源投加耦合控制动态模拟_孙月娣.pdf)
   重点：内回流与碳源投加耦合控制，设定值控制。
   价值：适合扩展成“曝气 + 回流 + 加药”协同优化。

### 1.5 新补充的工程与机理支撑文献

1. [Calibration of a complex activated sludge model for the full-scale.pdf](/D:/part3data/文献/Calibration%20of%20a%20complex%20activated%20sludge%20model%20for%20the%20full-scale.pdf)
   重点：`BioWin` 复杂活性污泥模型、全规模校准、参数敏感性分析。
   价值：适合作为“机理模型校准/数字孪生基础”文献。它的价值不在直接控制，而在告诉你哪些参数最值得优先校准。

2. [Aeration Control with Gain Scheduling in a Full-scale.pdf](/D:/part3data/文献/Aeration%20Control%20with%20Gain%20Scheduling%20in%20a%20Full-scale.pdf)
   重点：全规模污水厂 `gain scheduling PI` 曝气控制、`NH4+` 反馈控制、节能对比。
   价值：适合作为“传统控制基线”。如果你后面要证明 AI/预测控制优于常规控制，这篇很有对照价值。

3. [Statistical monitoring and dynamic simulation of a wastewater treatment plant.pdf](/D:/part3data/文献/Statistical%20monitoring%20and%20dynamic%20simulation%20of%20a%20wastewater%20treatment%20plant.pdf)
   重点：`PCA + multiple regression + MBBR动态仿真`，目标是给 `MPC` 提供可用输入。
   价值：这是“统计预测走向控制”的桥梁型论文，特别适合解释为什么即使在线监测不全，也能先做 `COD/TP` 预测再上控制。

4. [A dynamic physicochemical model for chemical phosphorus removal.pdf](/D:/part3data/文献/A%20dynamic%20physicochemical%20model%20for%20chemical%20phosphorus%20removal.pdf)
   重点：动态理化除磷模型、`HFO` 沉淀、磷酸盐吸附、共沉淀、`pH` 动力学。
   价值：这是你做“化学除磷精准加药”的机理核心文献，适合给 AI 加药模型加上化学约束，避免完全黑箱。

## 2. 项目最需要的公式

下面只保留对当前项目最有用的公式，不追求把文献里的所有数学细节都搬进来。

### 2.1 数据清洗与预处理

#### 1. 3σ 异常值剔除
来源：[基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf)

```text
sigma = sqrt( sum((xi - x_bar)^2) / (n - 1) )
if |xi - x_bar| > 3 * sigma:
    xi is outlier
```

用途：
1. 清洗进水、DO、MLSS、药剂投加量等异常监测值。
2. 适合放进项目数据治理模块。

#### 2. Min-Max 归一化
来源同上。

```text
X_new = (X - X_min) / (X_max - X_min)
```

用途：
1. 适合深度学习模型输入。
2. 当不同变量量纲差异大时特别重要。

### 2.2 预测模型评价指标

#### 3. 均方误差 MSE
来源：[基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf)

```text
MSE = (1 / n) * sum((yi - yhat_i)^2)
```

#### 4. 均方根误差 RMSE

```text
RMSE = sqrt((1 / n) * sum((yi - yhat_i)^2))
```

#### 5. 平均绝对误差 MAE

```text
MAE = (1 / n) * sum(|yi - yhat_i|)
```

#### 6. 决定系数 R2
来源：[基于集成模型的污水处理厂出水总氮预测方法_姚怡帆.pdf](/D:/part3data/文献/基于集成模型的污水处理厂出水总氮预测方法_姚怡帆.pdf)

```text
R2 = 1 - sum((yi - yhat_i)^2) / sum((yi - y_bar)^2)
```

用途：
1. 这些是你比赛报告里必须出现的预测指标。
2. 建议对 `TP/TN/COD/NH3-N` 都分别统计。

### 2.3 预测建模框架公式

#### 7. 一般预测映射
来源：[基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf)

```text
f[x1, ..., xn, xt] = y
```

含义：
1. `x1...xn` 为进水、过程状态、历史工况等特征。
2. `xt` 可视作投药量或控制量。
3. `y` 为目标出水指标。

用途：
1. 适合统一描述你的代理模型。
2. 可用于 `TP/TN/COD/NH3-N` 的单目标或多目标预测。

#### 8. 自注意力计算
来源：[基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf](/D:/part3data/文献/基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf)

```text
Q = H * WQ
K = H * WK
V = H * WV
Attention(Q, K, V) = SoftMax(Q * K^T / sqrt(d)) * V
```

用途：
1. 如果你后续要把加药预测模型升级成 `Self-Attention` 或 `Transformer/Informer`，这就是核心结构。
2. 也能支撑“模型可解释性”部分。

### 2.4 加药控制公式

#### 9. 前加药点理论频率模型
来源：[典型城镇污水处理厂碳源智能投加控制生产性试验_吴宇行.pdf](/D:/part3data/文献/典型城镇污水处理厂碳源智能投加控制生产性试验_吴宇行.pdf)

按原文提取整理，其核心关系为：

```text
H1 ∝ (DO当量硝酸盐负荷 + 回流硝酸盐负荷 - 进水SCOD) * 泵频率换算系数 * 流量修正系数
```

原文符号包括：
`DO1`, `NOl`, `NO3`, `r`, `R`, `beta1`, `beta2`, `beta3`, `SCOD`, `Flow/2000`

用途：
1. 适合前缺氧区碳源投加。
2. 本质是先算“需要去除的硝酸盐负荷”，再映射到泵频率。

#### 10. 后加药点理论频率模型
来源同上。

原文提取式：

```text
H2 = (NOe + beta1 * DO2 - b) * beta4 * beta5 * Flow / 2000
```

其中：
1. `NOe` 为好氧区出水硝酸盐浓度。
2. `DO2` 为好氧区出水溶解氧浓度。
3. `b` 为硝酸盐设定值，文中取 `12.0 mg/L`。

用途：
1. 适合后缺氧区补充碳源投加。
2. 当 `NOe + beta1 * DO2` 低于设定值 `b` 时，可不投或少投。

#### 11. 单泵实际运行频率
来源同上。

```text
A = H * P * alpha
```

其中：
1. `H` 为模型计算频率。
2. `P` 为反馈调节参数。
3. `alpha` 为修正系数，正常运行一般取 `1.0`。

用途：
1. 很适合你项目里把“模型预测值”映射成“PLC/变频泵输出值”。

#### 12. 化学强化除磷 CEPRM 半经验模型
来源：[污水厂化学强化除磷投药量精准自动控制_李振华.pdf](/D:/part3data/文献/污水厂化学强化除磷投药量精准自动控制_李振华.pdf)

根据 PDF 提取和上下文，原文将其定义为反比例函数型半经验模型：

```text
Y = a / (1 + b * X) + c
```

其中：
1. `Y` 为投药后余磷浓度与投药前原水磷浓度之比。
2. `X` 为金属离子物质的量浓度，单位 `mmol Me_i+/L`。
3. `a`, `b`, `c` 为经验常数。

说明：
1. 该式是依据 PDF 中文本提取结果和原文“反比例函数”描述整理出的可用形式。
2. 工程上可用于“金属盐投加量 -> 余磷水平”的快速估计。

用途：
1. 适合化学除磷自动投药。
2. 适合在你的项目里加入机理/经验约束。

#### 13. 加药优化判据
来源：[基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf)

```text
if m + n < 0.5 mg/L:
    dosing scheme is acceptable
```

其中：
1. `m` 为模型预测的出水总磷均值。
2. `n` 为模型在测试集上的 `MAE`。
3. `0.5 mg/L` 对应一级 A 排放标准边界。

用途：
1. 非常适合你的“在满足达标前提下尽量减药”。
2. 文中按 `5%` 步长递减投药量，最终得到最多可节约 `15%` 药剂。

### 2.5 曝气控制与优化公式

#### 14. 粒子群优化 PSO 更新公式
来源：[基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf](/D:/part3data/文献/基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf) 和 [基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf](/D:/part3data/文献/基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf)

```text
v_i^(k+1) = w * v_i^k + c1 * r1 * (pbest_i - x_i^k) + c2 * r2 * (gbest - x_i^k)
x_i^(k+1) = x_i^k + v_i^(k+1)
```

用途：
1. 适合对 `DO设定值`、`加药量`、`回流比` 做连续变量寻优。
2. 这是你后续写“优化控制器”的标准表达。

#### 15. 智能曝气经济指标 EC
来源：[基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf](/D:/part3data/文献/基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf)

原文核心形式为：

```text
EC = 0.04/T * ∫(Qa(t) + Qr(t) + Qw(t))dt
   + 24/T * ∫Σ(0.4032 * KLa_i^2 + 7.8408 * KLa_i)dt
```

其中：
1. `Qa` 为内回流流量。
2. `Qr` 为外回流流量。
3. `Qw` 为排泥量。
4. `KLa` 为氧转移速率，表征曝气强度。

用途：
1. 适合把“节能降耗”写成一个明确的优化目标。
2. 这个式子非常适合你的曝气优化模块。

#### 16. 曝气工艺约束与经验设定值
来源：[A~2O工艺污水处理厂优化脱氮及节能效果分析_顾升波.pdf](/D:/part3data/文献/A~2O工艺污水处理厂优化脱氮及节能效果分析_顾升波.pdf) 和 [Bardenpho工艺内回流与碳源投加耦合控制动态模拟_孙月娣.pdf](/D:/part3data/文献/Bardenpho工艺内回流与碳源投加耦合控制动态模拟_孙月娣.pdf)

建议直接作为运行边界：

```text
A2O前段DO: 0.1–0.3 mg/L
A2O中段DO: 0.5–1.0 mg/L
A2O后段DO: 2.0–4.0 mg/L

Bardenpho后置缺氧区NO3--N设定值: 4 mg/L
Bardenpho缺氧区NO3--N设定值: 2 mg/L
```

用途：
1. 适合直接作为控制器的约束边界或初始规则。
2. 可以避免优化器给出明显不合理的设定值。

#### 17. 二阶导数近似公式
来源：[Integrated-real-time-intelligent-control-for-wastewater-treatment-plants-Data-driven-modeling.pdf](/D:/part3data/文献/Integrated-real-time-intelligent-control-for-wastewater-treatment-plants-Data-driven-modeling.pdf)

原文通过差分近似关键底物浓度关于时间的二阶导数，可整理为：

```text
d(dS/dt)/dt |_(ti) ≈ [(S(ti) - S(ti-1)) / (ti - ti-1) - (S(ti-1) - S(ti-2)) / (ti-1 - ti-2)] / (ti - ti-1)
```

用途：
1. 从 `NO3--N`、`BOD`、`NH3-N` 的时间序列中提取动态反应特征。
2. 适合在“少传感器、低成本”的前提下构造数据驱动控制特征。

#### 18. 高斯特征参数函数
来源同上。

```text
d(dS/dt)/dt ≈ A * exp(-(t - μ)^2 / (2δ^2))
```

其中：
1. `A` 为振幅。
2. `μ` 为均值位置。
3. `δ^2` 为标准差参数。

用途：
1. 用高斯函数拟合关键底物的二阶导数曲线。
2. 把复杂生化动力学压缩成少量可更新的特征参数，适合实时控制。

#### 19. 运行优化变量与动态更新规则
来源同上。

这篇论文没有把总目标函数写成单一编号式，但明确了控制变量和更新机制：

```text
control variables: NRR + KLa
dynamic update interval: every 3 hours
```

用途：
1. `NRR` 用于调节硝化液回流。
2. `KLa` 用于调节氧供给强度。
3. 每 `3 h` 更新一次模型参数，适合解决污水厂工况变化快、传统模型响应慢的问题。

#### 20. 活性污泥模型校准的敏感性分析指标
来源：[Calibration of a complex activated sludge model for the full-scale.pdf](/D:/part3data/文献/Calibration%20of%20a%20complex%20activated%20sludge%20model%20for%20the%20full-scale.pdf)

该文重点不是给单个控制公式，而是给出校准时应重点关注的敏感参数判别指标：

```text
Si,j : normalized sensitivity coefficient
dj_msqr : mean square sensitivity measure
```

用途：
1. 用于对机理模型参数做敏感性排序。
2. 文中指出稳态和动态校准中最敏感的参数高度一致，可先做稳态敏感性分析，再辅助动态校准。

#### 21. Gain Scheduling 曝气控制规则
来源：[Aeration Control with Gain Scheduling in a Full-scale.pdf](/D:/part3data/文献/Aeration%20Control%20with%20Gain%20Scheduling%20in%20a%20Full-scale.pdf)

这篇论文更重要的是控制策略本身：

```text
scheduling variables: NH4+ concentration and/or DO concentration
controller type: gain scheduling PI
best strategy: schedule controller output limit
```

用途：
1. 作为你项目里传统曝气控制基线。
2. 说明“仅固定 DO 设定值”不如“随氨氮负荷动态调度控制器”。

#### 22. 统计预测 + 动态仿真桥接公式思路
来源：[Statistical monitoring and dynamic simulation of a wastewater treatment plant.pdf](/D:/part3data/文献/Statistical%20monitoring%20and%20dynamic%20simulation%20of%20a%20wastewater%20treatment%20plant.pdf)

该文核心是桥接式控制思路：

```text
PCA -> significant variables
multiple regression -> predicted influent COD/TP
predicted COD/TP -> dynamic simulation input
dynamic simulation -> aeration and coagulant dosing support
```

用途：
1. 在没有高质量在线 `COD/TP` 仪表时，先用统计预测补足输入。
2. 再把预测值送入仿真模型，用于 `MPC` 或其他优化控制。

#### 23. 化学除磷机理路径
来源：[A dynamic physicochemical model for chemical phosphorus removal.pdf](/D:/part3data/文献/A%20dynamic%20physicochemical%20model%20for%20chemical%20phosphorus%20removal.pdf)

该文强调化学除磷至少包含以下机理过程：

```text
1. HFO precipitation
2. phosphate adsorption onto HFO
3. co-precipitation
4. chemical equilibrium and pH coupling
```

用途：
1. 适合补足你的加药模块机理解释。
2. 适合在项目里说明“为什么投药量不仅与 TP 有关，还与 pH 和沉淀/吸附过程相关”。

## 3. 建议你在项目里直接采用的公式组合

### 3.1 预测层

建议保留：
1. `3σ` 异常值剔除。
2. `Min-Max` 归一化。
3. `MSE/RMSE/MAE/R2` 评估指标。
4. `f[x1,...,xn,xt]=y` 作为统一代理模型表达。

### 3.2 加药层

建议保留：
1. `CEPRM` 作为化学除磷经验约束。
2. `A = H * P * alpha` 作为泵频率输出层。
3. `m + MAE < 标准值` 作为是否还能继续减药的判断条件。

### 3.3 曝气层

建议保留：
1. `PSO` 更新公式。
2. `EC` 经济指标。
3. `DO` 分段经验边界。

### 3.4 协同控制层

如果做比赛版系统，建议最终统一为：

```text
Min J = w1 * 出水超标风险 + w2 * 曝气能耗 + w3 * 药剂消耗 + w4 * 调节波动惩罚
```

这不是单篇 PDF 里的原式照搬，而是结合当前目录文献后，最适合你项目落地的综合表达。

## 4. 结论

当前文献目录里的 PDF 已经足够支撑项目的核心方法设计：
1. `预测` 有 `TN/TP` 模型和完整评价指标。
2. `加药` 有碳源投加、精确加药 MPC、化学除磷经验模型。
3. `曝气` 有智能曝气优化和全规模节能案例。
4. `协同控制` 有内回流、碳源、曝气联动控制思路。
5. `最新顶刊补充` 有 `Water Research 2025` 的高阶导数特征提取、`NRR + KLa` 实时优化和 `3 h` 动态更新机制。

如果后续继续完善，优先建议：
1. 把这些公式再整理成 `项目公式汇总` Excel 工作表。
2. 再补一版“报告可直接引用的公式说明”和“符号表”。
