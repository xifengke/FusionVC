# 实验 01/02：VCC 2025 基线协议

## Scope and selection

训练：官方 150 个靶点；验证：官方 50 个未见靶点；测试：暂不读取。
两组预先固定种子 42/43/44、最多 120 epochs、patience 20、AdamW 与学习率预算。
最佳 checkpoint 根据验证 pseudobulk MAE 选择；最终报告官方评价工具的验证 MAE/PDS/DES。
不以教师/测试成绩选择模型，不将本阶段成绩写成历史比赛总分。

## Input fitting and no leakage

数据分块读取。归一化训练 counts 到训练全集的非零 library size 中位数后 log1p。
输入基因按训练对照的 log-expression 方差选 256 个；输入描述仅在训练对照上拟合。
随机固定抽 2,048 个训练对照细胞，按每个基因的均值/标准差标准化，得到 32 维 SVD gene
loadings，再附加训练对照均值与标准差，共 34 维靶点描述。验证靶点无需未训练的 ID 参数。
这属于使用比赛对照表达的无监督特征，不能称为 RNA foundation model。

验证表达只参与验证损失和评价；预处理模块隔离 train/validation 统计文件，模型输入始终
来自训练控制。各分组是否共享官方对照 cell ID 记录于 cache manifest，不把共享控制当作
独立的生物学重复。训练和验证扰动靶点必须互斥。

## Architecture and loss

MLP：`[256 control means, 34 target features] -> 256 -> 64`，GELU/dropout。
Transformer：训练对照 gene descriptor 与均值投影到 64 维；target descriptor 作为专门
token；两层、4 heads、FFN 256、pre-layer-norm；读取 target token 后 LayerNorm。
不使用人为基因序列位置编码。decoder 相同：`64 -> 18,080`。
decoder 以零位移初始化，其余参数随机初始化；第一步等于无扰动预测。

训练每个 batch 抽一个 32 细胞的控制 bank，同 batch 的靶点共用这组控制；各靶点分别
用模型的条件位移作用于控制表达：

1. `s = clamp(log1p(x_ctrl) + delta(p), 0, 14)`。
2. `a = expm1(s)`，逐细胞重归一化到固定训练 library scale，返回 `log1p(a)`。
3. 比较预测集合均值与该靶点所有真实训练细胞的均值，MSE 按靶点、基因平均。

不存在随机对照与扰动细胞的一对一 MSE。靶点等权，训练细胞数只决定真实 pseudobulk
统计量的稳定性，不让细胞数最多的靶点主导训练。
验证选择使用固定 256 个训练对照细胞，以限制中间张量大小；最终评价使用官方 CSV
规定的完整预测细胞数，与早停代理不同。

该结构只预测 condition-specific shift，不学习额外状态依赖响应和完整分布。
clipping/归一化会改变方差，但异质性仍主要来自训练对照。后续若换成分布训练，应新建协议
版本并重跑基线，不能把不同目标训练的八组直接混在同一公平比较表。

## Evaluation

固定官方 cell-eval 0.6.6 源码修订 `27c18c4f215a0345c1428329a41fdf7b24c24b50`，
pdex 0.1.27，VCC profile：`mae`、`discrimination_score_l1`、`overlap_at_N`。
DE 使用 Wilcoxon、默认 FDR 0.05；PDS 按官方默认排除靶基因，保留官方 rank/n 规则。
为控制内存，DE 使用分块的等价渐近 Mann-Whitney 实现：复用排序后的控制值，保留
tie correction、continuity correction、两侧检验及跨全基因/全靶点的 BH。测试必须先与
SciPy 和 pdex 0.1.27 核对 p 值、U、fold change、FDR；不使用 pdex 的实验性整数近似。
MAE/PDS 只依赖 group means，传入紧凑均值 AnnData；DES 仍传入完整细胞计算的 DE。
紧凑路径也须与官方全细胞路径对照。原始指标本身来自固定的官方实现。

评价时真实验证 counts 按该验证全集的 library 中位数 log-normalize，这是旧版 evaluator 的
默认 counts 转换。预测表达从训练归一化空间转换到同一验证尺度，保持连续值和所有输出基因。
训练对照抽样规则、输出细胞数与 evaluator 配置在每次输出中记录。

只报告可复现的原始验证指标，不自行发明 VCC 历史归一化总分。参考无扰动和训练平均
扰动预测的廉价辅助对照，用于判断复杂模型是否具有实际价值。

## Storage and reproducibility

唯一完整控制 log-expression cache 约 2.76GB；统计矩阵只有 MB 级。
每个种子保留一个最佳 checkpoint 和小型日志。评价缓存可复用真实 DE；预测矩阵按模型
顺序处理，避免六份完整矩阵同时保留。所有运行记录含配置、软件版本、GPU、参数量、耗时、
峰值显存和 checkpoint SHA256。小型汇总与按靶点结果进入 `results/`，大数据不提交 Git。

## Actual completion

两组各三个种子完成，完整验证集预测数为 98,927（含 38,176 个对照），实际结果在
`results/baselines/`。Transformer 采用确定性 math attention；初次默认后端探索记录
归档于 `outputs/archive_initial_attention`，不进入正式表。
同一环境重复 seed 42 的 MLP 与 Transformer，所有权重 tensor 完全相同，记录在
`reproducibility-check.json`。这不代表跨设备/软件版本的 bitwise 一致性。
真实数据抽选 3 个扰动、16 个基因，使用全部官方控制，适配器与 native pdex 的最大
p 值误差 2.22e-16，FDR 误差 2.78e-16。其他基因仍使用同一经验证的算法，未声称逐项
与 native evaluator 重复计算整份矩阵。MAE/PDS 的全细胞与紧凑路径另有测试。

最初辅助无扰动和训练平均位移只评估 MAE/PDS。2026-10-09 已为无扰动对照补算全细胞 DES，
结果为 0.00018707589325086215，MAE/PDS 与原逐靶点成绩一致（最大差异小于 1e-16）。
使用与神经模型完全一致的控制抽样、98,927 个细胞、18,080 个基因、50 个靶点、
Mann–Whitney/tie/continuity correction、全基因×全靶点 BH 和 FDR < 0.05。
完整记录见 [四模型与无扰动比较](../results/controls/README.md)，复现入口
`scripts/evaluate_no_perturbation.py`。训练平均位移的 DES 仍未评估，不把未计算项目填成零。
两组均未在 MAE 上超过无扰动对照；Transformer 所有最佳 epoch 为 1，显示当前训练
目标/表示在未见靶点上泛化不足。以上负结果应保留，不能换称为已实现预测提升。
