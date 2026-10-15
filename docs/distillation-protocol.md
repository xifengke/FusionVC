# 固定融合 Teacher → 原架构 MLP 的均值响应蒸馏

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15
- Verification Status: UNVERIFIED（协议固定时尚未运行；最终核验状态见结果报告）
- Version Label: distillation-v1

用户已授权完成知识蒸馏与整个项目。以下规则在本轮融合 MMD 的完整成绩汇总前固定，
实现不修改旧训练器、模型或历史产物。实际 Teacher 只能在七架构三损失 63 次正式验证
和 fusion-mmd 的产物/六组复现/测试记录全部通过后选择。

## 预先固定的 Teacher 规则

候选为 Concat、Gated、Cross-Attention × mean、energy、hybrid 九组。
按每组种子 42/43/44 的完整官方验证 MAE 算术均值最小选择；精确并列按 experiment
字典序。胜出组使用固定 seed42 的单 checkpoint，全部 Student 共用，不筛选幸运种子、
不使用 ensemble，不根据测试集选 Teacher。

`teacher-selection.json` 保存九组均值/标准差及选定 checkpoint SHA256。
报告必须区分胜出组均值与实际固定 Teacher 成绩。Teacher 即使不优于简单 MLP/无扰动
仍照协议完成，并如实报告负结果。

## Student 与损失

Student 直接使用原 `ResidualPredictor(model=mlp)`，宽度64、原训练对照拟合的基因
描述、256候选输入及18,080全基因输出，与原MLP相同的1,266,144个可训练参数。
纯监督对照复用原 MLP mean 三种子，已知它与最早01 MLP逐tensor一致。

对同一批32个训练控制细胞 C，定义

```
mean_S = apply_residual(C, delta_S, training_target_sum).mean(cell_axis)
mean_T = apply_residual(C, delta_T, training_target_sum).mean(cell_axis)
task = MSE(mean_S, real_training_pseudobulk)
KD = MSE(mean_S, mean_T)
loss = task + 1.0 * KD
```

KD系数固定1.0，不搜索；每个基因/靶点等权，无细胞逐一配对。
Teacher的delta只用于计算同bank的表达均值。直接raw-delta MSE不作为训练目标，因为
截断、expm1、library归一化和log1p使raw-delta距离不等于最终表达距离。
均值KD不宣称蒸馏完整条件分布或细胞状态机制，本轮不交叉扩展Student的三种task loss。

Teacher以float32 eval/no_grad仅对固定150个训练靶点生成 `[150,18080]` delta缓存
（约10.8MB）。不生成验证/测试训练软标签，不读测试表达。缓存按靶点名称/基因顺序
绑定到Teacher、来源和代码SHA256；准备与Student训练分进程，并在初始化Student前
重设原训练种子，避免Teacher构造改变Student随机初始化。

Student推理沿用MLP输入路径，只需训练控制拟合的描述表、控制矩阵和MLP。
Teacher缓存与基础编码器不在Student推理调用路径中；训练来源元数据不等于运行依赖。

## 不变的训练选择与独立公平性检查

沿用原AdamW lr=0.001、weight_decay=0.0001、最多120轮、patience20、batch16、
32控制细胞、bf16头、float32残差/损失、梯度裁剪1与CosineAnnealingLR。
仍按固定256个训练对照上的验证pseudobulk MAE选择checkpoint（改进阈值1e-6）。
验证标签只用于早停/比较，不进入KD或输入构建。

独立蒸馏trainer复用原load_inputs/label_tensors/validate，保留初始化和训练抽样RNG
顺序。首先实际运行KD=0、seed42的独立目录，逐tensor、最佳轮次和核心逐轮训练日志
严格复现原 `outputs/01_mlp/seed_42`；失败时停止并诊断，不扩大容差或改旧结果。
正式KD运行42/43/44，再独立重训KD seed42，比较张量、最佳轮次、缓存来源及日志。

## 评价、完整性与解释

完整官方验证沿用98,927细胞、18,080基因、50未见靶点、相同控制抽样、VCC profile
和全细胞DE/FDR规则；附加Energy MMD仍为64+64固定抽样，只作分布诊断。
三种子均值/样本标准差，与实际单Teacher、同架构纯监督Student及无扰动对照分开报告。

配对区间先平均Student/MLP种子，再对50相同靶点做5000次bootstrap，Teacher保持
固定；仅对当前拟合模型、Teacher/checkpoint选择和控制抽样成立。验证同时用于早停、
Teacher选择和比较，存在选择乐观偏差；无多重比较校正，种子不是生物重复。
单项变化不能扩大为整体能力提升或正式测试泛化优势。

准备前记录旧69 checkpoint及所有历史results文件的SHA256快照；本阶段只写新目录。
核验正式Student发布分数/源分数、逐靶点聚合、完整规模、抽样、checkpoint、缓存、
训练目标组合、单独推理与临时矩阵清理。CPU teacher缓存检查只抽三个训练靶点并
显式记录CPU/GPU float32容差；精确缓存由SHA256保护，GPU同硬件重训要求逐tensor相等。

## 成本测量范围

纯监督MLP、蒸馏MLP、固定Teacher均用seed42、相同150训练靶点、32训练对照、
18,080基因、目标batch4和float32 eval。一轮预热后测五轮，记录原始值/中位数/p95。
分别测缓存模型与输入加载、暖预测头、暖预测头+完整表达变换，以及包含加载/推理的
GPU峰值。加载包含现有来源校验和数据转移，操作系统文件缓存不清除；推理不含文件
输出或DE评分，不称为从原始数据开始的新冷启动测量。

原缓存已有编码器提取和共同预处理耗时作为历史实测的一次性共享成本单列。
同时报告checkpoint、静态特征表、共享控制矩阵及离线源模型权重大小。
现Teacher暖推理也使用缓存，不用总冻结参数比宣称端到端加速，checkpoint大小不等于
完整部署大小。成本结果及每项包含内容保存在 `cost-benchmark.json`。

## 完整运行 recipe

```powershell
./.venv/Scripts/python.exe scripts/run_distillation.py --phase prepare
./.venv/Scripts/python.exe scripts/run_distillation.py --phase smoke
./.venv/Scripts/python.exe scripts/run_distillation.py --phase all
```

各子阶段也支持train/evaluate/diagnostics/benchmark/verify/report。每个子进程硬超时
1800秒，stdout/stderr保存独立日志；失败停止，完整运行可核验后复用既有阶段。

```powershell
./.venv/Scripts/python.exe -m pytest -q tests/test_distillation.py --basetemp tmp/pytest-distillation
./.venv/Scripts/python.exe scripts/check_distillation_reproducibility.py --zero-only
./.venv/Scripts/python.exe scripts/check_distillation_reproducibility.py
./.venv/Scripts/python.exe scripts/benchmark_distillation.py
./.venv/Scripts/python.exe scripts/verify_distillation_artifacts.py
./.venv/Scripts/python.exe -m fusionvc.cli report-distillation
```

最终结果在 `results/distillation`。报告Material Passport仅在正式三运行、KD0公平性、
KD重训、测试、Ruff及成本记录全部通过且绑定当前产物时为VERIFIED；此标记不评价
生物学结论或最终测试集泛化。正式测试由全项目的最终冻结协议另行执行。
