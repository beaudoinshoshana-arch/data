# PDF逐篇全量公式手册

说明：本手册覆盖 `D:\part3data\文献` 下当前 `16` 篇 PDF。与现有精选版不同，这里按“逐篇整理”的口径，尽量收录每篇的主要公式、控制规则、评价指标和工程阈值。对 OCR 断裂的公式，已在备注中明确标记 `OCR修正`。

## 1. 基于集成模型的污水处理厂出水总氮预测方法

| 项目 | 内容 |
| --- | --- |
| 年份 | 2023 |
| 期刊/来源 | 工业水处理 |
| 论文定位 | 预测 |
| 对应问题 | 响应滞后---需要预判；精准度低—用量问题 |
| PDF路径 | [基于集成模型的污水处理厂出水总氮预测方法_姚怡帆.pdf](/D:/part3data/文献/基于集成模型的污水处理厂出水总氮预测方法_姚怡帆.pdf) |
| 最适合借鉴模块 | 出水总氮预判模块 |
| 提取难点 | 灰色关联度与距离公式 OCR 有部分断裂，按标准写法校正。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 式(1) | 预处理 | Xscale = (X - Xmin) / (Xmax - Xmin) | Xscale = (X - Xmin) / (Xmax - Xmin) | X为原始特征值；Xmin/Xmax为样本极值 | 将所有特征压缩到[0,1]，避免量纲差异影响TN预测 | 响应滞后；精准度低 | p.189，式(1) | 高 |  |
| 式(2) | 特征筛选 | ζ[y(i),x(j,i)] 由灰色关联度定义 | ζ = [min_j min_i \|y(i)-x(j,i)\| + ρ max_j max_i \|y(i)-x(j,i)\|] / [\|y(i)-x(j,i)\| + ρ max_j max_i \|y(i)-x(j,i)\|] | ζ为灰色关联度；ρ为分辨系数 | 筛选与出水TN最相关的输入特征 | 响应滞后；精准度低 | p.189-190，式(2) | 中 | OCR修正，按灰色关联标准公式还原 |
| 式(3) | 预测映射 | y(t) = f[x′1(t-1), x′2(t-1), …, x′k(t-1)] | y(t) = f(x′(t-1)) | x′为经GRA筛选后的特征向量；y(t)为t时刻出水TN | 建立t-1时刻工况到t时刻TN的非线性映射 | 响应滞后 | p.190，式(3) | 高 |  |
| 式(6) | 评价指标 | R2 = 1 - Σ(yi - yhat_i)^2 / Σ(yi - ȳ)^2 | R2 = 1 - sum((yi - yhat_i)^2) / sum((yi - ybar)^2) | yi为实测值；yhat_i为预测值；ybar为均值 | 评价TN预测拟合优度 | 精准度低 | p.191-192，式(6) | 高 |  |
| 式(7) | 评价指标 | RMSE = sqrt(1/n * Σ(yi - yhat_i)^2) | RMSE = sqrt((1/n) * sum((yi - yhat_i)^2)) | 同上 | 评价预测误差幅度 | 精准度低 | p.191-192，式(7) | 高 |  |
| 式(8) | 评价指标 | MAE = 1/n * Σ\|yi - yhat_i\| | MAE = (1/n) * sum(abs(yi - yhat_i)) | 同上 | 评价平均绝对误差 | 精准度低 | p.191-192，式(8) | 高 |  |

### 逐篇结论

这篇论文主要解决“出水TN能否提前预判”的问题，最适合借鉴到你的总氮预测模块。它包含可直接落地的数据标准化、特征筛选和评价指标公式，但不直接给出工艺控制阈值。

## 2. 基于深度学习的污水处理厂出水总磷预测方法

| 项目 | 内容 |
| --- | --- |
| 年份 | 2024 |
| 期刊/来源 | 工业水处理 |
| 论文定位 | 预测 |
| 对应问题 | 响应滞后---需要预判；精准度低—用量问题 |
| PDF路径 | [基于深度学习的污水处理厂出水总磷预测方法_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水总磷预测方法_安昱宁.pdf) |
| 最适合借鉴模块 | 总磷前馈预测模块 |
| 提取难点 | 短文以评价指标为主，控制公式较少。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 式(1) | 特征筛选 | r = Σ[(xi - X̄)(yi - Ȳ)] / sqrt(Σ(xi - X̄)^2 * Σ(yi - Ȳ)^2) | Pearson r = cov(x,y)/(σx*σy) | xi为特征值；yi为标签值；X̄/Ȳ为均值 | 筛选与出水TP相关的输入特征 | 精准度低 | p.145，式(1) | 高 |  |
| 规则 | 特征筛选 | 选择皮尔逊相关系数 r > 0.2 的特征作为模型输入 | keep feature if r > 0.2 | r为Pearson相关系数 | 降低维度并提升TP预测稳定性 | 精准度低 | p.145，表1前后 | 高 | 关键建模阈值 |
| 式(3) | 评价指标 | MSE = 1/n * Σ(yi - yi,pred)^2 | MSE = (1/n) * sum((yi - yhat_i)^2) | yi为真实值；yi,pred为预测值 | 评价TP预测误差 | 精准度低 | p.147，式(3) | 高 |  |
| 式(4) | 评价指标 | RMSE = sqrt(1/n * Σ(yi - yi,pred)^2) | RMSE = sqrt((1/n) * sum((yi - yhat_i)^2)) | 同上 | 评价TP预测误差尺度 | 精准度低 | p.147，式(4) | 高 |  |
| 式(5) | 评价指标 | MAE = 1/n * Σ\|yi - yi,pred\| | MAE = (1/n) * sum(abs(yi - yhat_i)) | 同上 | 评价TP预测平均绝对误差 | 精准度低 | p.147，式(5) | 高 |  |
| 式(6) | 评价指标 | R2 = 1 - Σ(yi - yi,pred)^2 / Σ(yi - ȳ)^2 | R2 = 1 - sum((yi - yhat_i)^2)/sum((yi - ybar)^2) | 同上 | 评价TP预测拟合优度 | 精准度低 | p.147，式(6) | 高 |  |

### 逐篇结论

这篇论文主要解决“出水总磷能否提前预判”的问题，最适合借鉴到总磷前馈预测和加药前馈信号生成。它给出了清晰的特征筛选公式和预测评价指标，但没有直接给出控制器更新式。

## 3. 基于自注意力机制的污水处理厂精确加药模型预测控制

| 项目 | 内容 |
| --- | --- |
| 年份 | 2023 |
| 期刊/来源 | 环境工程 |
| 论文定位 | 加药控制 |
| 对应问题 | 响应滞后---需要预判；精准度低—用量问题；药耗能耗高—用量问题 |
| PDF路径 | [基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf](/D:/part3data/文献/基于自注意力机制的污水处理厂精确加药模型预测控制_裴力锋.pdf) |
| 最适合借鉴模块 | 预测驱动加药优化模块 |
| 提取难点 | MER 损失函数中的绝对值符号 OCR 可能丢失，按“平均误差率”保守记录。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 式(1) | 模型结构 | X_model(ti) = f(X(ti)) + X_PE | Xmodel = f(X) + position_embedding | X_PE为位置向量 | 将时间位置信息编码进输入序列 | 响应滞后 | p.84-85，式(1) | 高 |  |
| 式(2) | 损失函数 | J_MER = (1/m) * ΣΣ (y - Y) / Y | JMER ≈ mean(relative_error) | y为模型输出；Y为真实值；m/n为样本与目标变量数量 | 作为自注意力模型训练的平均误差率损失 | 精准度低 | p.85，式(2) | 中 | OCR修正，绝对值符号可能缺失 |
| 式(3) | 评价指标 | J_MSE = (1/n) * Σ(yi - Yi)^2 | JMSE = (1/n) * sum((yi - Yi)^2) | yi为输出；Yi为实际值 | 作为模型比较的均方误差指标 | 精准度低 | p.85，式(3) | 高 |  |
| 式(4)-(5) | 控制/优化 | vi = ωvi + c1r1(pbesti - pi) + c2r2(gbest - pi); pi = pi + vi | PSO velocity-position update | ω为惯性权重；c1/c2为加速系数；pbest/gbest为个体和全局最优 | 对加药量进行自动寻优 | 精准度低；药耗能耗高 | p.86-87，式(4)-(5) | 高 |  |
| 式(6)-(7) | 模型结构 | Q = HWQ, K = HWK, V = HWV; Attention(Q,K,V) = SoftMax(QK^T/sqrt(d))V | Self-Attention(Q,K,V) = softmax(QK^T/sqrt(d))V | H为序列输入；Q/K/V为查询键值矩阵 | 计算不同历史时刻对当前预测的权重 | 响应滞后 | p.87-88，式(6)-(7) | 高 |  |
| 式(8) | 可解释性 | f̂_xS = (1/n) * Σ f̂(xS, xC(i)) | PDP(xS) = mean_i fhat(xS, xC(i)) | xS为关注特征；xC为其余特征 | 分析加药量与出水TN的边际关系 | 精准度低 | p.88，式(8) | 高 |  |
| 规则 | 工艺设定值 | 加药量范围 5–25 L/h；以1个HRT数据为主输入；出水TN达标线 6 mg/L | dose∈[5,25] L/h, target TN <= 6 mg/L | HRT约1 h | 限定优化搜索空间并约束出水达标 | 响应滞后；精准度低；药耗能耗高 | p.84-89，材料与方法/应用结果 | 高 | 实际运行约束 |
| 结果规则 | 工程效果 | 2022年1-2月优化后月均加药量分别降低28.72%和21.78% | dosage_reduction ≈ 22%-29% under TN compliance | 对比基准为原厂月平均加药量 | 作为加药优化可行性依据 | 药耗能耗高 | p.89-92，结果与结论 | 高 | 工程效果，不是训练公式 |

### 逐篇结论

这篇论文同时解决“响应滞后、加药不准、药耗高”三类问题，最适合借鉴到你的预测驱动加药优化模块。它包含可直接落地的搜索算法、出水约束和实际投加范围。

## 4. 典型城镇污水处理厂碳源智能投加控制生产性试验

| 项目 | 内容 |
| --- | --- |
| 年份 | 2022 |
| 期刊/来源 | 环境工程 |
| 论文定位 | 加药控制 |
| 对应问题 | 响应滞后---需要预判；精准度低—用量问题；药耗能耗高—用量问题 |
| PDF路径 | [典型城镇污水处理厂碳源智能投加控制生产性试验_吴宇行.pdf](/D:/part3data/文献/典型城镇污水处理厂碳源智能投加控制生产性试验_吴宇行.pdf) |
| 最适合借鉴模块 | 碳源智能投加与泵频率输出模块 |
| 提取难点 | 前加药点模型 OCR 有断裂，已按上下文关系式重组。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 式(1) | 控制/优化 | H1 = β1*DO1 + NOl*r/(1+r+R) + NO3*(R/(1+r+R))*β2 - SCOD ] * β3 * Flow/2000 | H1 = [β1·DO1 + NOl·r/(1+r+R) + NO3·R/(1+r+R)·β2 - SCOD]·β3·Flow/2000 | H1为前置加药泵频率；NOl/NO3为硝酸盐浓度；r/R为回流倍数；β1~β3为换算系数 | 计算前加药点碳源投加泵理论频率 | 响应滞后；精准度低 | §1.3.1，式(1) | 中 | OCR修正，原式括号断裂 |
| 式(2) | 控制/优化 | H2 = (NOe + β1*DO2 - b) * β4 * β5 * Flow / 2000 | H2 = (NOe + β1·DO2 - b)·β4·β5·Flow/2000 | H2为后置加药泵频率；NOe为好氧区出水硝酸盐；DO2为DO；b为硝酸盐设定值 | 计算后加药点碳源投加泵频率 | 响应滞后；精准度低 | §1.3.2，式(2) | 高 |  |
| 式(3) | 控制/优化 | A = H * P * α | A = H·P·alpha | A为单泵最终频率；H为模型计算频率；P为反馈参数；α为修正系数 | 将理论频率映射到实际计量泵输出 | 精准度低；药耗能耗高 | §1.4，式(3) | 高 |  |
| 规则 | 工艺设定值 | 后加药点硝酸盐设定值 b = 12.0 mg/L，低于设定值则后加药点不投加 | if NOe + β1·DO2 < 12 mg/L then no rear dosing | b为后缺氧段硝酸盐控制上限 | 避免不必要的碳源投加 | 药耗能耗高 | §1.3.2 | 高 | 关键阈值 |
| 规则 | 控制规则 | 系统基于在线NO3、NH4、ORP、DO与SCOD需求计算最佳加药量 | online sensors -> nitrate load -> SCOD demand -> pump frequency | 在线仪表包括硝酸盐、氨氮、ORP、DO | 建立从监测到投加的闭环控制逻辑 | 响应滞后；精准度低 | §1.2-1.3 | 高 | 工程控制逻辑 |

### 逐篇结论

这篇论文最直接解决“碳源投加滞后和过量投加”问题，最适合借鉴到碳源智能投加与泵频率输出层。它给出了前后加药点频率模型和实际泵频率换算式。

## 5. 基于感知-决策-评估的污水处理智能曝气方法

| 项目 | 内容 |
| --- | --- |
| 年份 | 2022 |
| 期刊/来源 | 工业水处理 |
| 论文定位 | 曝气 |
| 对应问题 | 响应滞后---需要预判；药耗能耗高—用量问题 |
| PDF路径 | [基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf](/D:/part3data/文献/基于感知-决策-评估的污水处理智能曝气方法_袁沐坤.pdf) |
| 最适合借鉴模块 | 智能曝气优化器 |
| 提取难点 | PSO 适应度与TN公式 OCR 有部分截断，按文意保守还原。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 式(13) | 软测量模型 | EQ(BPNN) = F3(Q, X) | EQ = F3(Q, X) | Q为入水工况；X为辅助状态变量 | 构建出水水质软测量模型 | 响应滞后 | p.69，式(13) | 中 | 公式仅展示BPNN软测量关系 |
| 式(14)-(16) | 控制/优化 | V_i^(k+1)=ωV_i^k+c1r1(P_i^k-X_i^k)+c2r2(P_g^k-X_i^k); F(X_i^k)=EC if COST_i^k<COST_inf else ∞; X_i^(k+1)=X_i^k+V_i^(k+1) | PSO update with infeasible-solution penalty | ω为动量因子；P_i/P_g为个体/群体最优；EC为经济指标 | 搜索当前工况下最优DO设定值 | 药耗能耗高；响应滞后 | p.69-70，式(14)-(16) | 中 | OCR修正，适应度惩罚项按原文语义还原 |
| 式(18) | 目标函数 | EC = 0.04/T * ∫(Qa+Qr+Qw)dt + 24/T * ∫Σ(0.4032*KLa_i^2 + 7.8408*KLa_i)dt | same | Qa/Qr/Qw为流量项；KLa为氧转移速率；T=7d | 量化曝气及回流相关经济成本 | 药耗能耗高 | p.70，式(18) | 高 |  |
| 式(19) | 出水约束 | TSS = 0.75(XS + XI + XB,H + XB,A + XP) | same | XS/XI等为活性污泥模型状态变量 | 约束悬浮物达标 | 响应滞后；药耗能耗高 | p.70，式(19) | 高 |  |
| 式(20) | 出水约束 | COD = SS + SI + XS + XI + XB,H + XB,A + XP | same | SS/SI等为可溶/颗粒有机物状态 | 约束COD达标 | 响应滞后；药耗能耗高 | p.70，式(20) | 高 |  |
| 式(21) | 出水约束 | BOD5 = 0.25[SS + XS + (1-fp)(XP + XI)] | same | fp为颗粒惰性比例 | 约束BOD5达标 | 响应滞后；药耗能耗高 | p.70，式(21) | 高 |  |
| 式(22) | 出水约束 | TN = SNO + SNK,j (后续展开在OCR中不完整) | TN ≈ SNO + SNH + SND + XNH + iXB(XB,H + …) | SNO/SNH/SND/XNH等为含氮组分 | 约束总氮达标 | 响应滞后；药耗能耗高 | p.70，式(22) | 低 | OCR修正，原式后半段不完整 |
| 规则 | 工艺设定值 | Benchmark 约束：TSS<30 g/m3，COD<100 g/m3，BOD5<10 g/m3，TN<18 g/m3 | same | 出水限制值 | 作为PSO求解的约束条件 | 药耗能耗高；响应滞后 | p.70，式(19)-(22)前 | 高 | 关键约束边界 |

### 逐篇结论

这篇论文主要解决“曝气优化如何同时兼顾出水达标和能耗”的问题，最适合借鉴到智能曝气优化器。它同时给出了目标函数、约束公式和寻优策略。

## 6. 污水厂化学强化除磷投药量精准自动控制

| 项目 | 内容 |
| --- | --- |
| 年份 | 2020 |
| 期刊/来源 | 工业水处理 |
| 论文定位 | 加药控制 |
| 对应问题 | 精准度低—用量问题；药耗能耗高—用量问题 |
| PDF路径 | [污水厂化学强化除磷投药量精准自动控制_李振华.pdf](/D:/part3data/文献/污水厂化学强化除磷投药量精准自动控制_李振华.pdf) |
| 最适合借鉴模块 | 化学除磷经验约束模块 |
| 提取难点 | 公式在文本提取中分裂明显，但反比例函数结构清晰。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 公式(1) | 经验模型 | Y = a/(1 + bX) + c | Y = a/(1 + bX) + c | Y为投药后余磷/投药前原水磷；X为金属离子浓度 mmol Me_i+/L；a,b,c为经验常数 | 估计金属盐投加量与余磷水平的函数关系 | 精准度低；药耗能耗高 | p.56，公式(1) | 中 | OCR修正，按原文“反比例函数”描述还原 |
| 规则 | 变量定义 | 将滤池出水磷浓度与二沉池出水磷浓度之比作为 Y，将金属离子物质的量浓度作为 X | Y = P_out / P_in; X = c_Me | P_out为滤池出水磷浓度；P_in为二沉池出水磷浓度 | 把在线监测值映射到CEPRM模型输入输出 | 精准度低 | §3.1 | 高 |  |
| 规则 | 拟合准则 | 4种金属盐拟合相关度均大于0.9，残差平方和均小于0.01 | R > 0.9 and SSE < 0.01 | R为拟合相关度；SSE为残差平方和 | 判断CEPRM模型是否能用于自动投药 | 精准度低；药耗能耗高 | 摘要/§3.2-3.3 | 高 | 用于模型验收 |

### 逐篇结论

这篇论文最适合解决“化学除磷投药不准”问题，最适合借鉴到化学除磷经验约束和自动投药换算模块。它提供了一个非常适合工程实现的半经验模型。

## 7. A2O工艺污水处理厂优化脱氮及节能效果分析

| 项目 | 内容 |
| --- | --- |
| 年份 | 2015 |
| 期刊/来源 | 中国给水排水 |
| 论文定位 | 曝气 |
| 对应问题 | 药耗能耗高—用量问题；响应滞后---需要预判 |
| PDF路径 | [A~2O工艺污水处理厂优化脱氮及节能效果分析_顾升波.pdf](/D:/part3data/文献/A~2O工艺污水处理厂优化脱氮及节能效果分析_顾升波.pdf) |
| 最适合借鉴模块 | A2O分段DO控制边界 |
| 提取难点 | 工程论文以生产性调试和运行规则为主，没有复杂数学推导。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 工艺设定值 | 好氧区前端DO 0.1–0.3 mg/L；中端DO 0.5–1.0 mg/L；后端DO 2.0–4.0 mg/L | front DO=0.1-0.3; middle DO=0.5-1.0; rear DO=2.0-4.0 mg/L | 对应A2O好氧区前中后段 | 作为分区曝气控制边界 | 药耗能耗高 | 摘要/表1 | 高 | 全规模生产性调试 |
| 规则 | 工程效果 | 降低1#或1#、2#廊道曝气量时，脱氮效果提高9.8%和18.5%，供气量节省32.1%和34.1% | nitrogen_removal +9.8%/+18.5%; air_supply -32.1%/-34.1% | 基准为现行工况 | 证明低DO分区控制的节能和脱氮效果 | 药耗能耗高 | 摘要/结论 | 高 | 工程结果 |
| 规则 | 运行规模 | 全规模A2O池处理规模 5×10^4 m3/d，持续试验200 d | full-scale capacity = 5e4 m3/d; trial = 200 d | 全规模运行背景 | 说明规则来自长期实厂调试而非仿真 | 响应滞后；药耗能耗高 | 摘要/§1 | 高 |  |

### 逐篇结论

这篇论文虽然没有复杂数学公式，但给出了非常重要的全规模分段DO设定值和节能效果，最适合用作A2O工艺的运行边界和经验初始化值。

## 8. Bardenpho工艺内回流与碳源投加耦合控制动态模拟

| 项目 | 内容 |
| --- | --- |
| 年份 | 2017 |
| 期刊/来源 | 中国给水排水 |
| 论文定位 | 耦合控制 |
| 对应问题 | 精准度低—用量问题；药耗能耗高—用量问题 |
| PDF路径 | [Bardenpho工艺内回流与碳源投加耦合控制动态模拟_孙月娣.pdf](/D:/part3data/文献/Bardenpho工艺内回流与碳源投加耦合控制动态模拟_孙月娣.pdf) |
| 最适合借鉴模块 | 内回流+碳源耦合控制模块 |
| 提取难点 | 以控制策略和设定值为主，没有显式编号数学公式。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 工艺设定值 | 后置缺氧区硝酸盐氮浓度设定值 4 mg/L | NO3_post_anoxic_setpoint = 4 mg/L | 后置缺氧区末端在线硝酸盐仪表 | 控制碳源投加量 | 精准度低；药耗能耗高 | 摘要/控制策略4 | 高 |  |
| 规则 | 工艺设定值 | 缺氧区硝酸盐氮设定值浓度 2 mg/L | NO3_anoxic_setpoint = 2 mg/L | 缺氧区出水在线硝酸盐仪表 | 控制内回流量 | 精准度低；药耗能耗高 | 摘要/控制策略4 | 高 |  |
| 规则 | 工艺约束 | 内回流泵流量范围 0–18×10^4 m3/d | Q_internal_reflux ∈ [0, 18×10^4] m3/d | 内回流泵流量 | 限定耦合控制搜索范围 | 药耗能耗高 | 控制策略说明 | 高 |  |
| 规则 | 工艺约束 | 外加碳源为100%甲醇，相当于1.188 kgCOD/L，最大投加量不超过6 m3/d | methanol_COD_equiv = 1.188 kgCOD/L; carbon_dose_max = 6 m3/d | 甲醇投加浓度和上限 | 限定碳源投加边界 | 药耗能耗高 | §2/试验条件 | 高 |  |

### 逐篇结论

这篇论文最适合借鉴到“内回流 + 碳源投加”的协同优化模块。虽然没有显式编号公式，但给出了非常清晰的双控制环设定值和操作边界。

## 9. 曝气精确控制实现污水处理厂节能降耗的应用

| 项目 | 内容 |
| --- | --- |
| 年份 | 2022 |
| 期刊/来源 | 环境工程 |
| 论文定位 | 曝气 |
| 对应问题 | 药耗能耗高—用量问题；响应滞后---需要预判 |
| PDF路径 | [曝气精确控制实现污水处理厂节能降耗的应用_荆玉姝.pdf](/D:/part3data/文献/曝气精确控制实现污水处理厂节能降耗的应用_荆玉姝.pdf) |
| 最适合借鉴模块 | 精确曝气执行层与工程效果验证 |
| 提取难点 | 工程应用论文，核心是控制规则与运行效果，不是数学推导。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 控制规则 | 以空气流量为主要控制对象，以DO及进水COD、氨氮、水温等作为辅助控制参数 | airflow_main_control + {DO, COD, NH4-N, temperature} auxiliary control | 空气流量、DO、进水负荷、温度等 | 建立按需供气的多变量曝气控制逻辑 | 响应滞后；药耗能耗高 | §1-§2 | 高 |  |
| 规则 | 工艺设定值 | 曝气池 DO 控制在设定值目标范围 ±0.3 mg/L 内 | \|DO_actual - DO_set\| <= 0.3 mg/L | DO_actual为实测值；DO_set为目标值 | 作为精确曝气控制精度标准 | 药耗能耗高 | 摘要/§2 | 高 | 关键执行指标 |
| 规则 | 控制规则 | 根据所需空气流量、实际空气流量和DO浓度调整空气阀门及鼓风机工况，并可根据出水NH4-N变化自动调整空气供给量 | required_airflow + actual_airflow + DO + effluent NH4-N -> valve/blower adjustment | 阀门开度、鼓风机工况、出水NH4-N | 解决传统DO控制滞后问题 | 响应滞后；药耗能耗高 | §1-§2 | 高 |  |
| 结果规则 | 工程效果 | 曝气能耗节省24.8%，乙酸钠药耗降低15.1% | energy -24.8%; acetate -15.1% | 对比原控制方式 | 作为工程可行性与经济效益证明 | 药耗能耗高 | 摘要/结论 | 高 | 工程结果 |

### 逐篇结论

这篇论文最适合用来支撑你的“精确曝气执行层”和工程效果论证。它强调了空气流量主控和多变量辅助控制，比单纯DO闭环更贴近实厂。

## 10. Research on prediction algorithm of effluent quality and development of integrated control system for waste-water treatment

| 项目 | 内容 |
| --- | --- |
| 年份 | 2025 |
| 期刊/来源 | Scientific Reports |
| 论文定位 | 预测+系统 |
| 对应问题 | 响应滞后---需要预判；药耗能耗高—用量问题；精准度低—用量问题 |
| PDF路径 | [Research on prediction algorithm of effluent quality and development of integrated control system for waste-water treatment.pdf](<D:/part3data/文献/Research on prediction algorithm of effluent quality and development of integrated control system for waste-water treatment.pdf> ) |
| 最适合借鉴模块 | 预测与控制一体化架构 |
| 提取难点 | 公式多且为英文，已按编号完整整理。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eq.(1) | 模型结构 | C1 = σ(W1 * Xc + b1) | same | Xc为输入时间序列矩阵；W1/b1为卷积参数 | 第一层CNN卷积提取空间特征 | 响应滞后 | Eq.(1), p.8 | 高 |  |
| Eq.(2) | 模型结构 | P1 = MaxPool(C1) | same | C1为第一层特征图 | 第一层池化降维 | 响应滞后 | Eq.(2), p.8 | 高 |  |
| Eq.(3) | 模型结构 | C2 = σ(W2 * P1 + b2) | same | P1为池化输出 | 第二层CNN卷积 | 响应滞后 | Eq.(3), p.8 | 高 |  |
| Eq.(4) | 模型结构 | P2 = MaxPool(C2) | same | C2为第二层特征图 | 第二层池化 | 响应滞后 | Eq.(4), p.8 | 高 |  |
| Eq.(5) | 模型结构 | zCNN = Flatten(P2) | same | P2为多维特征张量 | 把卷积特征压平为一维向量 | 响应滞后 | Eq.(5), p.8 | 高 |  |
| Eq.(6) | 模型结构 | ŷ = Wfc · zCNN + bfc | same | Wfc/bfc为全连接层参数 | CNN分支输出出水质量预测值 | 响应滞后 | Eq.(6), p.8 | 高 |  |
| Eq.(7)-(11) | 模型结构 | ft = σ(Wf xt + Uf ht-1 + bf); it = σ(Wi xt + Ui ht-1 + bi); ot = σ(Wo xt + Uo ht-1 + bo); ct = ft⊙ct-1 + it⊙tanh(Wc xt + Uc ht-1 + bc); ht = ot⊙tanh(ct) | standard LSTM gate equations | f/i/o为门控；c为单元状态；h为隐藏状态 | LSTM分支捕捉时间依赖 | 响应滞后 | Eq.(7)-(11), p.8-9 | 高 |  |
| Eq.(12) | 模型结构 | ŷ = Wfc · hT + bfc | same | hT为LSTM最后隐藏状态 | LSTM分支输出预测值 | 响应滞后 | Eq.(12), p.9 | 高 |  |
| Eq.(13)-(16) | 模型结构 | zt = σ(Wz xt + Uz ht-1 + bz); rt = σ(Wr xt + Ur ht-1 + br); h~t = tanh(Wh xt + Uh(rt⊙ht-1)+bh); ht = (1-zt)⊙ht-1 + zt⊙h~t | standard GRU gate equations | z/r为更新门和重置门 | GRU分支建模时间依赖 | 响应滞后 | Eq.(13)-(16), p.9 | 高 |  |
| Eq.(17) | 模型结构 | ŷ = Wfc · hT + bfc | same | hT为GRU最终状态 | GRU分支输出预测值 | 响应滞后 | Eq.(17), p.9 | 高 |  |
| Eq.(18) | 不确定性建模 | Qτ(Y\|X) = Inf{q ∈ R : F(q\|X) ≥ τ} | conditional_quantile = inf { q \| F(q\|X) >= tau } | τ为分位数 | 定义条件分位数，用于不确定性预测 | 响应滞后；精准度低 | Eq.(18), p.10 | 高 |  |
| Eq.(19) | 目标函数 | min_β Σ ρτ(yi - Xi^T β) | quantile_regression objective | β为回归系数；ρτ为分位数损失 | 拟合不同分位数回归模型 | 精准度低 | Eq.(19), p.10 | 高 |  |
| Eq.(20) | 损失函数 | ρτ(u) = { τu, u>=0; (1-τ)(-u), u<0 } | same | u为预测残差 | 对高估和低估施加不同惩罚 | 精准度低 | Eq.(20), p.10 | 高 |  |
| Eq.(21) | 不确定性建模 | F(q\|zCombined) = (1/\|Rl\|) * Σ I(yi <= q) | leaf empirical CDF | Rl为叶节点样本集合 | QR-RF 叶节点经验分布估计 | 精准度低 | Eq.(21), p.10 | 高 |  |
| Eq.(22) | 不确定性建模 | Q̂τ(Y\|zcombined) = (1/M) * Σ Q̂τ^(m)(Y\|zcombined) | aggregate quantile prediction across trees | M为树数量 | 融合多棵树的分位数预测 | 精准度低 | Eq.(22), p.10 | 高 |  |
| Eq.(23) | 模型结构 | ΔVar = Var(YRp) - \|Rleft\|/\|Rp\| Var(YRleft) - \|Rright\|/\|Rp\| Var(YRright) | variance_reduction_split_score | Rp为父节点；Rleft/Rright为子节点 | QR-RF 分裂准则 | 精准度低 | Eq.(23), p.10 | 高 |  |
| Eq.(24) | 集成结构 | zcombined = [ zCNN, zLSTM, zGRU ] | feature_concat = concat(zCNN, zLSTM, zGRU) | 三个基学习器特征向量拼接 | 作为QR-RF元学习器输入 | 响应滞后；精准度低 | Eq.(24), p.10 | 高 |  |
| 规则 | 系统输出 | 点预测取 Q̂0.50；90%置信区间取 [Q̂0.05, Q̂0.95]；ICS 结合预测结果优化药剂、曝气与污泥回流 | median + uncertainty interval + ICS control | Q̂0.50/Q̂0.05/Q̂0.95为分位点输出 | 把预测和控制一体化，并输出不确定性边界 | 响应滞后；药耗能耗高；精准度低 | p.10-11，系统输出说明 | 高 | 可直接借鉴到系统层 |

### 逐篇结论

这篇论文最适合支撑“预测 + 不确定性 + 集成控制系统”这一整套架构。它并不直接给曝气或加药的单变量公式，但给了非常完整的代理模型和不确定性输出框架。

## 11. 基于深度学习的污水处理厂出水水质预测建模研究

| 项目 | 内容 |
| --- | --- |
| 年份 | 2024 |
| 期刊/来源 | 青岛理工大学硕士学位论文 |
| 论文定位 | 预测+优化 |
| 对应问题 | 响应滞后---需要预判；精准度低—用量问题；药耗能耗高—用量问题 |
| PDF路径 | [基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf](/D:/part3data/文献/基于深度学习的污水处理厂出水水质预测建模研究_安昱宁.pdf) |
| 最适合借鉴模块 | 预测代理模型总补全源 |
| 提取难点 | 公式最多，已按模块合并整理，避免重复抄录。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 式(2.1)-(2.2) | 模型结构 | z(1)=W(1)X+b(1), a(1)=f1(z(1)); a(l)=fl(W(l)a(l-1)+b(l)) | BPNN forward propagation | W/b为权重偏置；a(l)为第l层输出 | 描述BPNN前向传播 | 响应滞后；精准度低 | 第2章 p.11 | 高 |  |
| 式(2.3)-(2.7) | 模型结构 | ∂z/∂w, ∂z/∂b, δ(l)=f′(z(l))⊙((W(l+1))^Tδ(l+1)), ∂L/∂W=δ(a)T, ∂L/∂b=δ | BPNN backpropagation gradients | δ为误差项 | 描述BPNN反向传播更新规则 | 精准度低 | 第2章 p.11-12 | 高 |  |
| 式(2.8) | 模型结构 | Zp = Wp ⊗ X + bp, Yp = f(Zp) | CNN convolution block | ⊗为互相关运算 | 描述CNN卷积层计算 | 响应滞后 | 第2章 p.14 | 高 |  |
| 式(2.9)-(2.12) | 模型结构 | Z(l,p)=W(l,p)⊗X(l-1)+b(l,p); ∂L/∂W=δ⊗X; ∂L/∂b=Σδ; δ=f′(z)⊙up(δ_next) | CNN backprop and pooling error | δ为卷积层误差项；up为上采样函数 | 描述CNN训练过程 | 精准度低 | 第2章 p.14-15 | 高 |  |
| 式(2.15) | 模型结构 | F(xT) = Σ fk · xT-K+k | causal convolution | fk为卷积核；K为卷积核大小；T为时间步 | 描述TCN因果卷积 | 响应滞后 | 第2章 p.18 | 高 |  |
| 式(2.16) | 模型结构 | Fd(xT) = Σ fk · xT-(K+k)d | dilated causal convolution | d为膨胀因子 | 扩大感受野以处理长时序 | 响应滞后 | 第2章 p.18 | 高 |  |
| 式(2.27) | 模型结构 | Q = Wq X, K = Wk X, V = Wv X | projection to Q/K/V | Wq/Wk/Wv为投影矩阵 | Informer输入投影 | 响应滞后 | 第2章 p.24 | 高 |  |
| 式(2.28) | 模型结构 | Y = V softmax(K^T Q / sqrt(Dk)) | scaled dot-product attention | Dk为键向量维度 | Informer标准注意力 | 响应滞后 | 第2章 p.24 | 高 |  |
| 式(2.29) | 模型结构 | Attention(Q,K,V) = V softmax(K^T Q̄ / sqrt(Dk)) | ProbSparse attention | Q̄为概率稀疏化后的查询向量 | 降低长序列注意力复杂度 | 响应滞后 | 第2章 p.24-25 | 高 |  |
| 式(2.30) | 模型结构 | X(t)^(j+1) = MaxPool(ELU(Conv1d([X(t)^j]AB))) | Informer attention distilling block | AB为自注意力块 | 在编码器层进行特征压缩和蒸馏 | 响应滞后 | 第2章 p.25 | 高 |  |
| 式(2.31) | 评价指标 | MSE = (1/n) * Σ(yi - yhat_i)^2 | same | yi真实值；yhat_i预测值 | 模型训练与评估 | 精准度低 | 第2章 p.26 | 高 | 与式(2.32)-(2.34)成套出现 |
| 式(2.32) | 评价指标 | RMSE = sqrt((1/n) * Σ(yi - yhat_i)^2) | same | 同上 | 模型评估 | 精准度低 | 第2章 p.26 | 高 |  |
| 式(2.33) | 评价指标 | MAE = (1/n) * Σ\|yi - yhat_i\| | same | 同上 | 模型评估 | 精准度低 | 第2章 p.26 | 高 |  |
| 式(2.34) | 评价指标 | R2 = 1 - Σ(yi - yhat_i)^2 / Σ(yi - ȳ)^2 | same | 同上 | 模型评估 | 精准度低 | 第2章 p.26 | 高 |  |
| 式(3.2) | 预处理 | sigma = sqrt(sum((xi - x̄)^2)/(n-1)); if \|xi - x̄\| > 3*sigma then remove xi | 3σ outlier rule | x̄为均值；sigma为样本标准差 | 剔除污水厂报表中的异常值 | 精准度低 | 第3章 p.54左右 | 高 |  |
| 式(3.3) | 预处理 | Xnew = (X - Xmin) / (Xmax - Xmin) | same | Xnew为归一化值 | 归一化输入特征 | 精准度低 | 第3章 p.54左右 | 高 |  |
| 式(5.1) | 优化映射 | f[x1,…,xn,xt] = y | y = f(features, dose) | xt为投药量；y为出水TP | 把出水TP预测模型作为投药优化代理模型 | 响应滞后；药耗能耗高 | 第5章 p.80-81 | 高 |  |
| 规则 | 控制判据 | 若预测均值 m 与 MAE n 满足 m + n < 0.5 mg/L，则该投药方案满足一级A标准 | accept if m + MAE < 0.5 mg/L | m为预测均值；n为MAE | 在达标前提下判断是否还能继续减药 | 药耗能耗高；精准度低 | 第5章 p.81 | 高 | 关键优化判据 |
| 规则 | 优化结果 | 按5%步长递减聚合氯化硫酸铁投加量，最多可在TP不超标前提下降低15% | max feasible dose reduction ≈ 15% | 以0.5 mg/L为TP达标边界 | 给出实际可采纳的减药幅度 | 药耗能耗高 | 第5章 p.82-83 | 高 | 工程结论，可直接引用 |

### 逐篇结论

这篇硕士论文是当前目录里公式最全的资料，最适合做总补全源。它同时覆盖了预测、评价、预处理、注意力机制和投药优化判据，适合直接转写到项目方法部分。

## 12. Integrated real-time intelligent control for wastewater treatment plants: Data-driven modeling for enhanced prediction and regulatory strategies

| 项目 | 内容 |
| --- | --- |
| 年份 | 2025 |
| 期刊/来源 | Water Research |
| 论文定位 | 预测+实时控制 |
| 对应问题 | 响应滞后---需要预判；药耗能耗高—用量问题；精准度低—用量问题 |
| PDF路径 | [Integrated-real-time-intelligent-control-for-wastewater-treatment-plants-Data-driven-modeling.pdf](/D:/part3data/文献/Integrated-real-time-intelligent-control-for-wastewater-treatment-plants-Data-driven-modeling.pdf) |
| 最适合借鉴模块 | 实时智能控制主框架与低成本动态特征提取模块 |
| 提取难点 | PDF 英文排版较紧，部分导数公式和表格文本提取有断裂，但关键编号式可以识别。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eq.(1) | 特征提取 | d(dS/dt)/dt\|_(ti) 由相邻时刻差分近似 | d2S_dt2(ti) ≈ {[(S(ti)-S(ti-1))/(ti-ti-1)] - [(S(ti-1)-S(ti-2))/(ti-1-ti-2)]} / (ti-ti-1) | S 为底物浓度；ti 为离散时刻 | 从 `NO3--N`、`BOD`、`NH3-N` 时间序列提取动态变化特征 | 响应滞后；精准度低 | Eq.(1), p.4 | 中 | OCR修正，按原文差分式还原 |
| Eq.(2) | 特征提取 | d(dS/dt)/dt ≈ A e^(-(t-μ)^2 / 2δ^2) | d2S_dt2 ≈ A * exp(-(t - mu)^2 / (2*delta^2)) | A 为振幅；μ 为均值；δ^2 为标准差参数 | 用高斯函数拟合关键底物二阶导数，提取稳定特征参数 | 响应滞后；精准度低 | Eq.(2), p.4 | 高 |  |
| 规则 | 机理/建模 | 通过 Gaussian fitting 得到 A、μ、δ，再构造影响因素与特征参数之间的多项式函数关系 | influent/control vars -> {A, μ, δ} -> dynamic model | 影响因素包括 `SNo3`、`SBOD`、`SNH4`、`KLa` 等 | 把复杂活性污泥动力学压缩成低维特征参数函数 | 响应滞后；精准度低 | §2.2.2-2.2.3, p.4-5 | 高 | 关键方法流程 |
| 规则 | 控制变量 | 动态控制通过调节 `NRR` 与 `KLa` 完成 | controls = {NRR, KLa} | NRR 为 nitrification reflux rate；KLa 为氧传递系数 | 在保证出水达标下同时调节回流和供氧 | 响应滞后；药耗能耗高 | §3.4.1, p.10-11 | 高 | 关键控制变量 |
| 规则 | 优化策略 | 在出水达标条件下，平衡最低能耗与出水质量 | min energy subject to effluent compliance | 关注 NH3-N、TN、BOD 与能耗 | 实现“达标 + 节能”的实时智能调控 | 药耗能耗高；精准度低 | §3.4.1, p.10 | 高 | 目标函数以文字定义，非单一编号式 |
| 规则 | 动态更新 | 模型参数每 3 h 更新一次 | update_interval = 3 h | 动态更新特征参数函数 | 提升系统对进水波动和工况变化的适应性 | 响应滞后 | 摘要/§3.4.2, p.11-12 | 高 | 关键工程机制 |
| 规则 | 评价指标 | 相关系数超过 0.8 | corrcoef > 0.8 | 针对 NH3-N、TN、BOD 预测趋势 | 证明数据驱动模型具备较好的预测能力 | 精准度低 | 摘要, p.1 | 高 | 工程效果 |
| 规则 | 计算效率 | ASM2D 参数校准时间 939.75 s，数据驱动模型校准 87.52 s | calibration_time_reduction ≈ 90.7% | 对比 ASM2D 与新模型 | 证明所提方法更适合实时控制部署 | 响应滞后；药耗能耗高 | Table 2, p.3-4 | 高 | 关键部署指标 |
| 规则 | 节能效果 | 稳态优化节能约 18.27%，动态调控节能约 24.3% | steady_saving ≈ 18.27%; dynamic_saving ≈ 24.3% | 对比基本工况 | 证明动态调控优于稳态调控 | 药耗能耗高 | §3.4.1-3.4.2, p.10-12 | 高 | 工程效果 |
| 规则 | 适用范围 | 适用于包含 AO 系统的 BNR 工艺，不适用于除磷控制 | applicable_to = AO-based BNR; not for phosphorus removal | 关注生物脱氮过程 | 明确模型边界，防止误用 | 精准度低 | §3.4.2, p.11 | 高 | 模型边界说明 |

### 逐篇结论

这篇论文最适合支撑你的“实时智能控制主框架”。它没有像传统深度学习论文那样堆很多网络结构公式，而是把重点放在 `高阶导数特征提取 + 高斯特征参数函数 + NRR/KLa 实时优化 + 3h 动态更新` 上，非常适合解决污水厂“响应滞后、控制成本高、实时性不足”的问题。

## 13. Calibration of a complex activated sludge model for the full-scale wastewater treatment plant

| 项目 | 内容 |
| --- | --- |
| 年份 | 2011 |
| 期刊/来源 | Bioprocess and Biosystems Engineering |
| 论文定位 | 机理校准 |
| 对应问题 | 精准度低—用量问题；响应滞后---需要预判 |
| PDF路径 | [Calibration of a complex activated sludge model for the full-scale.pdf](/D:/part3data/文献/Calibration%20of%20a%20complex%20activated%20sludge%20model%20for%20the%20full-scale.pdf) |
| 最适合借鉴模块 | 机理模型校准与数字孪生基础 |
| 提取难点 | 这篇以参数校准和敏感性分析为主，控制公式少，更多是校准指标与参数排名。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 敏感性分析 | Si,j: normalized sensitivity coefficient | Si,j = normalized sensitivity coefficient | 用于衡量模型输出对参数变化的敏感程度 | 用于筛选最值得优先校准的参数 | 精准度低 | 摘要/符号表 | 高 | 论文核心指标 |
| 规则 | 敏感性分析 | dj_msqr: mean square sensitivity measure | dj_msqr = mean square sensitivity measure | 用于对敏感参数做综合排序 | 比较稳态与动态校准中参数的重要性 | 精准度低 | 摘要/结论/Table 4 | 高 | 论文核心指标 |
| 规则 | 校准结论 | 稳态条件下17个参数敏感，动态条件下19个参数敏感 | sensitive_params = 17 (steady) / 19 (dynamic) | 敏感参数主要与 OHO、PAO 生长和衰减相关 | 指导机理模型校准优先级 | 精准度低；响应滞后 | 摘要/结论 | 高 | 校准策略结论 |
| 规则 | 校准方法 | 稳态敏感性分析可显著支持动态校准 | steady-state sensitivity -> support dynamic calibration | 先稳态后动态 | 降低复杂活性污泥模型校准难度 | 精准度低 | 结论 | 高 | 工程经验规则 |

### 逐篇结论

这篇论文不直接解决曝气或加药控制，但它非常适合你的项目里“机理模型怎么校准”这一层。它的最大价值是告诉你：先找敏感参数，再做动态校准，能显著降低复杂模型落地难度。

## 14. Aeration Control with Gain Scheduling in a Full-scale Wastewater Treatment Plant

| 项目 | 内容 |
| --- | --- |
| 年份 | 2014 |
| 期刊/来源 | IFAC World Congress Proceedings |
| 论文定位 | 曝气控制 |
| 对应问题 | 药耗能耗高—用量问题；响应滞后---需要预判 |
| PDF路径 | [Aeration Control with Gain Scheduling in a Full-scale.pdf](/D:/part3data/文献/Aeration%20Control%20with%20Gain%20Scheduling%20in%20a%20Full-scale.pdf) |
| 最适合借鉴模块 | 传统曝气控制基线与节能对照 |
| 提取难点 | 论文更偏控制策略与实验结果，显式编号公式较少，重点是控制规则和节能效果。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 控制策略 | gain scheduling PI control | controller = PI + gain scheduling | 调度变量可为 NH4+ 或 DO | 根据工况动态调整曝气控制器参数/限制 | 响应滞后；药耗能耗高 | 摘要/引言 | 高 | 核心控制思想 |
| 规则 | 调度变量 | scheduling on NH4+ concentration and/or DO concentration | scheduling_var ∈ {NH4+, DO} | NH4+ 为目标变量；DO 为过程状态变量 | 实现按工况变化调整曝气强度 | 响应滞后 | 摘要/实验图8-11 | 高 |  |
| 规则 | 最优策略 | best results when scheduling the controller output limit | best_strategy = schedule output limit | 调度输出上限而非只调 Kp/Ti | 获得更高节能效果 | 药耗能耗高 | 摘要/结论 | 高 | 论文核心结论 |
| 规则 | 工程结果 | peak NH4 events: S3 saved 6.7% vs constant DO; yearly average saved 11.4% vs constant DO | energy_saving ≈ 6.7% peak, 11.4% annual | 与恒定DO控制比较 | 作为曝气节能基线和对照结果 | 药耗能耗高 | Table 3 / conclusions | 高 | 工程结果 |

### 逐篇结论

这篇论文最适合你项目里的“传统曝气控制基线”。它说明即便不使用深度学习，只要把 `NH4+` 负荷和 `DO` 状态用于 gain scheduling，也能带来明显节能效果。

## 15. Statistical monitoring and dynamic simulation of a wastewater treatment plant: A combined approach to achieve model predictive control

| 项目 | 内容 |
| --- | --- |
| 年份 | 2017 |
| 期刊/来源 | Journal of Environmental Management |
| 论文定位 | 预测+动态仿真 |
| 对应问题 | 响应滞后---需要预判；精准度低—用量问题 |
| PDF路径 | [Statistical monitoring and dynamic simulation of a wastewater treatment plant.pdf](/D:/part3data/文献/Statistical%20monitoring%20and%20dynamic%20simulation%20of%20a%20wastewater%20treatment%20plant.pdf) |
| 最适合借鉴模块 | 统计预测 + 动态仿真 + MPC桥接模块 |
| 提取难点 | 重点在 PCA、多元回归和动态仿真的组合思路，显式控制公式较少。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 特征筛选 | PCA used to find out significant variables for COD and TP prediction | PCA -> significant variables | 关键变量包括 inflow, ammonia, pH 等 | 解决在线COD/TP监测不足的问题 | 响应滞后 | 摘要/方法 | 高 | 统计监测步骤 |
| 规则 | 预测模型 | multiple regression applied the variables suggested by PCA to predict influent COD and TP | PCA-selected vars -> multiple regression -> COD/TP | 用统计模型预测进水COD/TP | 给动态仿真与MPC提供输入 | 响应滞后；精准度低 | 摘要/方法 | 高 | 桥接控制的关键步骤 |
| 规则 | 评价指标 | R2 of predicted COD and TP versus measured data are 81.6% and 77.2% | R2_COD=0.816; R2_TP=0.772 | 预测与实测对比 | 验证统计预测可用性 | 精准度低 | 摘要/结论 | 高 | 关键结果 |
| 规则 | 控制用途 | predicted COD and TP data by multiple regression served as model input for dynamic simulation, enabling MPC of aeration and coagulant dosing | predicted influent -> dynamic simulation -> aeration/coagulant dosing support | MBBR + 混凝分离工艺 | 把统计预测接到控制层 | 响应滞后；精准度低 | 摘要/结论 | 高 | 对项目很有启发 |

### 逐篇结论

这篇论文最像“从预测走向控制”的桥梁。它告诉你：即使没有高质量的在线 COD/TP 仪表，也可以先用 `PCA + 多元回归` 预测，再送入动态仿真模型，为曝气和加药控制提供可用输入。

## 16. A Dynamic Physicochemical Model for Chemical Phosphorus Removal

| 项目 | 内容 |
| --- | --- |
| 年份 | 2015 |
| 期刊/来源 | Water Research |
| 论文定位 | 化学除磷机理 |
| 对应问题 | 精准度低—用量问题；药耗能耗高—用量问题 |
| PDF路径 | [A dynamic physicochemical model for chemical phosphorus removal.pdf](/D:/part3data/文献/A%20dynamic%20physicochemical%20model%20for%20chemical%20phosphorus%20removal.pdf) |
| 最适合借鉴模块 | 化学除磷机理约束与精准投药模块 |
| 提取难点 | 这篇包含大量理化反应矩阵和附录表，当前先提炼最关键的机理路径和建模目标。 |

### 公式清单

| 公式编号 | 公式类别 | 原始公式/规则 | 标准化公式 | 变量释义 | 公式用途 | 对应解决问题 | 页码/段落位置 | 提取置信度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 规则 | 建模目标 | dynamic physico-chemical model for chemical phosphorus removal | dynamic physicochemical phosphorus-removal model | 目标是在达标前提下优化药剂投加 | 用于加药优化与出水TP约束 | 精准度低；药耗能耗高 | 摘要 | 高 | 论文主目标 |
| 规则 | 机理路径 | precipitation of hydrous ferric oxides (HFO), phosphates adsorption and co-precipitation | HFO precipitation + phosphate adsorption + co-precipitation | HFO 为水合氧化铁 | 解释铁盐除磷的真实机理路径 | 精准度低 | 摘要/引言 | 高 | 关键机理 |
| 规则 | pH耦合 | combined with chemical equilibrium and physical precipitation reactions in order to model observed bulk dynamics in terms of pH | dose + equilibrium + precipitation + pH dynamics | pH 是影响加药效果的重要状态量 | 说明加药量不能只看TP，还要看pH与沉淀平衡 | 精准度低；药耗能耗高 | 摘要/引言 | 高 | 适合项目解释层 |
| 规则 | 模型结论 | model describes adequately the mechanisms of adsorption and co-precipitation of phosphate species onto HFO and is robust under various experimental conditions | robust HFO-phosphate dynamic model | 基于前人研究与实验数据校准验证 | 适合作为化学除磷精准控制的机理约束 | 精准度低；药耗能耗高 | 摘要/结论 | 高 | 适合项目机理支撑 |

### 逐篇结论

这篇是你做“化学除磷精准加药”的机理核心文献。它最重要的价值不是某一个简短公式，而是把 `HFO沉淀 + 吸附 + 共沉淀 + pH耦合` 这些过程统一进了动态模型，非常适合给你的 AI 加药决策加机理约束。
