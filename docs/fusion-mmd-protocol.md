# 三种融合的 MMD 损失扩展（fusion-v1 / mmd-v1）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15
- Verification Status: UNVERIFIED（待正式运行和核验完成）
- Version Label: fusion-mmd-v1

## 运行前固定的扩展

用户要求为已完成的三种融合加入之前单模型使用的MMD训练损失。
保持三种融合的模型实现、冻结特征、输入、64维解码器、输出表达变换及评价规则不变，
新增纯Energy MMD和均值MSE + 0.1 × Energy MMD，各42/43/44三种子。
新增3架构 × 2损失 × 3种子 = 18次；原9次均值对照保留，融合比较总计27次。
合入原四架构3损失36次形成七架构 × 3损失 × 3种子的63次比较。

新增配置使用原`fusion-v1`，因为该协议的训练器已支持这两种损失；本文件单独记录
扩展时间和预先规定的比较，绝不把最初“仅均值损失”阶段改写成事先已计划全部损失。
输出独立运行名`05_concat_energy/hybrid`、`06_gated_energy/hybrid`、
`07_cross_attention_energy/hybrid`。小型产物在`results/fusion/<experiment>`，
综合报告在`results/fusion-mmd`，旧`results/fusion/README.md`仍为首轮均值比较报告。

## 与旧单模型完全相同的损失和选择规则

沿用[原MMD协议](mmd-protocol.md)的Energy kernel和包含对角项的V-statistic，
距离除以sqrt(18,080)。纯MMD只反向传播Energy项，不加入MSE；混合组为
`mean_mse + 0.1 * energy_mmd`。每个靶点的预测与真实细胞集合分别比较、靶点等权，
不存在细胞逐一配对，也不把不同扰动混为一组。

每批16个训练靶点、32个共享训练对照；真实细胞按训练靶点各独立抽32个，
采样RNG为seed+100000。复用已有约16GB精确float32全训练细胞缓存，
不增加新的表达数据，不读取真实验证表达用于损失或构造特征。
基因列表、150训练/50验证靶点、完整18,080输出、训练library scale完全不变。

沿用AdamW lr=0.001、weight_decay=0.0001、最多120轮、patience20、梯度裁剪1、
bf16预测头；仍按固定256个训练对照上的验证pseudobulk MAE选checkpoint。
不按MMD或DES选模型，不因某个种子的结果调整系数、架构或预算。
该选择规则可能限制分布损失表现，须如实报告。

完整官方评价仍覆盖98,927个验证细胞，MAE/PDS/DES分开报告；附加Energy MMD
仍用固定64+64细胞及2026/2027种子，只作为采样分布诊断。
测试集保持未使用，未扩展到知识蒸馏或额外细胞状态响应。

## 核验和报告

正式运行前记录原51个checkpoint（旧四架构42个和融合均值9个）及旧逐次小产物SHA256。
新增18次均使用独立目录。冻结模型资产/缓存复用原manifest并重新核对。
六个新“架构×损失”组的seed42在独立目录重训，逐tensor、最佳epoch、特征来源比较。
核验完整基因/细胞/靶点、控制抽样、聚合分数、训练损失组合、缓存来源、原51个
checkpoint与旧小产物未变、临时预测矩阵清理。

主对照为每个架构MMD/混合相对自身均值组，三个种子报告均值与标准差。
逐靶点区间先平均种子，再对相同50个靶点进行5000次配对bootstrap。
这些仅对当前模型/选择规则/固定控制/验证靶点有条件成立，未做多重比较校正，
不包含选择和重训不确定性，也不证明生物重复或正式测试泛化。

所有模型仍学习共同条件位移；MMD不会自动消除这一表示限制。
跨架构还存在参数量、预训练信息与token粒度差异，保持原融合协议的解释边界。
运行串行，每个子进程30分钟硬超时；过程及输出日志持续监控，错误先诊断并保留记录。

```powershell
./.venv/Scripts/python.exe scripts/run_fusion.py --phase smoke --variants energy hybrid
./.venv/Scripts/python.exe scripts/run_fusion.py --phase all --variants energy hybrid
./.venv/Scripts/python.exe scripts/check_fusion_mmd_reproducibility.py
./.venv/Scripts/python.exe scripts/verify_fusion_mmd_artifacts.py
./.venv/Scripts/python.exe -m fusionvc.cli report-fusion-mmd
```

也可用`--models concat`、`gated`、`cross_attention`逐组执行；均值已有结果会核验后复用。
