# MMD 对照实验：mmd-v1

目标：检验改变训练损失能否改善 H1 未见基因扰动预测，区分编码器效果与损失效果。

## 固定的实验矩阵

MLP、随机初始化 Transformer，各比较均值 MSE、Energy MMD、均值 MSE + 0.1 × Energy MMD。
每组种子 42/43/44，共 18 次训练。新均值对照须与旧均值对照的最佳 checkpoint 权重逐 tensor 相同。
结果写入 results/mmd，保留 results/baselines 和原输出。

150 个训练靶点、50 个验证靶点，测试集不使用。模型、256 个输入基因、34 维靶点描述、
18,080 个输出基因、优化器、学习率、batch 16、最多 120 epochs、patience 20 和
验证 pseudobulk MAE 的 checkpoint 选择规则均固定。预算按相同优化步数上限定义；
MMD 的额外计算及读取耗时单独如实报告，并非声称墙钟算力成本完全相等。

## 细胞集合与损失

每批每个靶点预测 32 个控制细胞的扰动后表达；MMD 真实标签从同一训练靶点的全部原始细胞
中随机抽 32 个。细胞数不足时有放回抽样，并明确记录策略。真实细胞独立 RNG seed+100000，
不改变原先控制抽样或靶点顺序。全基因、训练固定 library scale、连续 log1p 空间。
同 batch 的靶点共用一个控制 bank，与旧基线代码一致；真实扰动标签则按靶点独立抽样。

同一靶点的预测集合与真实集合单独比较，再按靶点等权平均。
不存在细胞逐一配对，不将不同干预混成一组。均值 MSE 仍使用全部真实训练细胞的均值。
数据读取不删除或预先筛掉训练细胞。

为降低 18 次重复训练的随机 HDF5 读取成本，首次正式运行建立约 16.00GB 的 float32
训练细胞缓存 data/processed/mmd-v1/training_cells.npy。包含所有原始训练细胞，不减少基因、
不做半精度转换，数值与按需读取后归一化逐项相同。缓存源文件、大小、时间戳、表达尺度和
shape 均核对；完成标记只在整个缓存写入成功后生成。分块处理，磁盘保留 18GB 评价余量。
新增的是可重建缓存，不是新增训练数据。原始矩阵与旧缓存均保留。

论文：Adduri et al., Predicting cellular responses to perturbation across diverse contexts with State,
Cell 189, 5914–5931.e20 (2026), DOI 10.1016/j.cell.2026.07.052。
STAR Methods e5 / 提供 PDF 第 24 页给出 energy kernel 的经验 MMD。
此处采用包含对角项的 V-statistic：

energy = 2 × mean(||prediction-observation||₂)
         - mean(||prediction-prediction||₂)
         - mean(||observation-observation||₂)

为使系数容易解释，将距离除以 sqrt(18,080)，即仅以常数重缩放核；不等同于改变每个细胞
的归一化。原论文表达空间训练直接使用 MMD，本项目 hybrid 是额外的对照方案。
纯 MMD 不混入均值 MSE。混合权重预先固定 0.1，未针对官方验证标签调参。
以 PyTorch 实现论文公式，无需额外安装 geomloss；不复制 State 权重或训练代码。

## 评价及边界

继续使用固定的官方 cell-eval 0.6.6 全细胞 MAE/PDS/DES；输出与官方基因、细胞数一致。
附加 Energy MMD 只用于分布核对：每个验证靶点独立抽取 64 个预测细胞和 64 个真实细胞，
种子分别为 2026/2027；使用官方评价转换后的相同表达尺度。该附加指标不是历史比赛总分。
小样本经验分布有抽样误差，不作为全分布已匹配的证明。

仍按验证均值 MAE 选择最佳 checkpoint，以隔离损失改变的影响；这个选择规则可能限制
以分布恢复为目标的收益，须在结论中注明。未来改选择规则应另建协议。

本轮保持共同条件位移模型。细胞之间不同的变化主要继承控制异质性及既有非线性，
没有实现 State 的细胞集合 attention 或额外状态依赖响应。MMD 不能绕过这一表示限制，
也不能恢复同一真实细胞干预前后的轨迹。添加 MMD 属于已有方法的借鉴。

## 运行

```powershell
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe scripts/run_mmd.py --phase smoke
./.venv/Scripts/python.exe scripts/run_mmd.py --phase all
```

也可分别执行 train、evaluate、report；完成结果会校验配置与 checkpoint 后复用。
正式结果及验证发现更新在 results/mmd/README.md。

## 本轮结果

2026-10-08 完成全部 18 次正式训练及完整验证；15 项测试通过。
六次均值对照与旧模型的权重、选择 epoch、官方评分全部一致。
MLP 纯 MMD 的 MAE 约降低 0.35%，Transformer 混合组约降低 0.12%，但 DES 没有同步提高；
Transformer 纯 MMD 的四项平均指标均变差。所有新增组仍未超过无扰动 MAE 对照 0.027547。
本轮仅改变损失未带来明显的整体提升，原均值对照继续作为后续实验基线。
该结论限定于固定架构、抽样规模、混合权重及按均值 MAE 选择 checkpoint 的本轮设置。
