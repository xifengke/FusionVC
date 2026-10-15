# 最终保留测试集评价（final-test-v1）

执行完成：27个checkpoint及一次无扰动对照的完整测试与来源核验已通过。
本文件保留预先固定的规则；实际结果见[最终报告](../results/final-test/README.md)。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15
- Verification Status: UNVERIFIED（本协议预先固定时的状态；执行核验见最终报告）
- Version Label: final-test-v1

## 冻结后才访问测试表达

用户在2026-10-15授权：融合MMD完成且核验无误后继续完成知识蒸馏和整个项目。
当前阶段先固定本文件的选择规则。实际下载和读取test表达前必须生成
`docs/final-model-selection.json`，记录全部候选验证成绩、选择规则、选定run、
checkpoint及成绩文件SHA256，并标记`frozen=true`、`test_used=false`。
读取测试标签后不改模型、损失、超参数、Teacher、Student或种子选择。

## 八组模型及公平对照

前七种架构（MLP、随机Transformer、冻结scGPT、冻结Geneformer、Concat、Gated、
Cross-Attention）分别在mean/energy/hybrid三个已完成的组中，按三个种子
42/43/44的**完整官方验证MAE均值**选择损失。精确并列按experiment字符串字典序。
选中整组三种子全部进入test，不选“最好的种子”。

第八种为固定均值监督+KD的蒸馏MLP，三种子全部进入test。
为了单独检验蒸馏，原MLP的mean三种子必须也进入test；如果主MLP已选mean则不重复，
否则作为额外`kd_control`。因此共有24或27个checkpoint评价，加一次无扰动对照。
实际Teacher为按蒸馏协议选择的融合组固定seed42，已经包含在对应主模型test组中。

不同架构允许选择不同损失，这是相同候选预算下的验证选择结果，不是纯架构消融；
报告必须同时列出每架构选中的损失。原63次验证矩阵及均值比较报告保持保留。
蒸馏Student与未蒸馏MLPmean的比较才是同架构、同task损失的蒸馏对照。

## 数据和空间

使用公开固定generation的VCC2025真实test counts及官方扰动细胞数CSV。
元数据审计和小CSV访问不读取表达响应；下载全矩阵之前必须检查冻结selection。
匿名HTTP HEAD及小CSV已核验可读，不配置GCP账号或计费项目。
矩阵预期11,950,739,168 bytes，generation1765905295402700，CRC32C`JMP5RQ==`。
CSV包含100个扰动、132,670个扰动细胞，但不含non-targeting行；完整细胞数
必须从下载后h5ad读取，不能把该总数或旧验证对照细胞数当成已知test总数。

不改变训练预处理、基因顺序、18,080个输出或任何输入特征。只在最终评价中
按test全集实际library-size中位数做官方counts→log1p转换，预测从原训练尺度
转换到同一test评价尺度。统计量只用于评价，不回流训练/模型选择。

使用独立test原始文件、审计、float32真实表达cache和真实DE缓存；不改原
`baselines-v1/manifest.json`、验证矩阵cache、成绩或checkpoint。
根据实际shape动态检查真实cache+一个临时预测矩阵所需空间，串行评分并清理预测。

## 官方评分与复现

固定原cell-eval0.6.6 / pdex0.1.27、官方VCC profile、全部输出基因、全部实际细胞、
100个测试靶点。DE沿用经过native数值对照的分块Mann–Whitney/Wilcoxon流程、
tie/continuity correction和全基因×全靶点BH。保留训练对照细胞异质性，
按sorted test条件名、seed2025固定各组共同控制抽样。

只报告原始test MAE/PDS/DES，附加固定64+64细胞Energy MMD单列，
不创造历史排行榜总分。无扰动对照按test完整流程实际计算，不预设DES为零。
所有test产物明确`test_used=true`，独立保存到`results/final-test`及
`outputs/final-test*`，绑定冻结selection SHA和checkpoint来源。

三种子报告均值/样本标准差；逐靶点比较可先平均种子再做100靶点配对bootstrap，
仅是给定拟合模型与固定控制抽样的条件性区间，不代表独立生物重复或因果检验。
H1 hESC单基因CRISPRi的保留测试结果不外推成通用虚拟细胞能力。

最终报告包含八组选择、蒸馏对照、实际Teacher、资源成本、负结果、可复现命令与
尚未解决的限制。远端GitHub发布/CI只有实际发生才可声明完成。
