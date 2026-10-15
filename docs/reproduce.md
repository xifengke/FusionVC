# 从数据到最终报告

## 先区分四种使用方式

| 方式 | 资源与含义 |
| --- | --- |
| 源码检查 | 源码与Python依赖，无需数据、权重或缓存；不能据此宣称完整实验已复现 |
| Student演示 | 需要另行取得或训练后导出的完整模型包；演示抽样不同于正式评分 |
| 重建实验 | 数据、来源模型、GPU、磁盘和本次来源记录；原本机哈希不能证明新训练结果 |
| 原工作区验收 | 需要原历史checkpoint、Student、源资产、缓存及日志；发布副本不含全部文件 |

## 源码检查

在干净源码目录中先创建独立检查环境。以下为Windows PowerShell示例，使用自己的Python3.12。
CPU PyTorch用于源码检查，不替代完整GPU环境。

```powershell
python -m venv .venv-check
./.venv-check/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
./.venv-check/Scripts/python.exe -m pip install -e '.[evaluation,dev,geneformer,final_test]' -c requirements/windows-constraints.txt
./.venv-check/Scripts/python.exe -m ruff check src scripts tests
./.venv-check/Scripts/python.exe -m pytest -q --basetemp tmp/pytest-source
```

安装依赖需要网络。记录的69项通过来自完整本机环境；纯CPU会跳过GPU专用测试，通过数量可能不同。
新机器从零安装和Linux/macOS全量实验尚未经本机验证，远端Actions状态以实际运行记录为准。

## 完整实验的前提

在项目根目录运行。实际环境为 Windows、Python 3.12、RTX 4060 Laptop 8GB；
完整版本锁见 `requirements/windows-training.lock.txt`。原始数据、预训练模型、
处理缓存及训练输出不包含在小型源码产物中。

下面列出阶段入口，不是新克隆即可一键重建全部历史收据的承诺。
新实验需在扩展阶段前建立本次来源记录，不能通过删掉校验或改旧收据来绕过。
完整核验包含实际测试、复现与来源检查，不会因只运行report命令自动建立。
更多资源与支持边界见[资源说明](resources-and-data.md)。

## 环境与前期资源

```powershell
./scripts/setup_download.ps1 -Python 'E:/Python/python.exe'
./.venv-download/Scripts/python.exe scripts/download_assets.py
./.venv-download/Scripts/python.exe scripts/prepare_sources.py
./.venv-download/Scripts/python.exe scripts/validate_assets.py
./scripts/setup_training.ps1 -Python 'E:/Python/python.exe'
./.venv/Scripts/python.exe -m pytest -q --basetemp tmp/pytest-reproduce
```

下载记录与源码版本在 `docs/download-status.json`、`docs/source-provenance.json`。
大型数据需要充足磁盘；训练细胞缓存单独约16GB，最终评价还会保存测试原文件、
真实表达缓存以及一个串行复用的临时预测矩阵。最终下载/评价脚本会检查空间，
至少保留18GB余量，不删除旧结果来腾空间。

## 七架构、三损失、三种子验证矩阵

```powershell
./scripts/run_baselines.ps1
./.venv/Scripts/python.exe scripts/run_mmd.py --phase all
./.venv/Scripts/python.exe scripts/run_scgpt.py --phase all
./.venv/Scripts/python.exe scripts/run_geneformer.py --phase all
./.venv/Scripts/python.exe scripts/evaluate_no_perturbation.py
./.venv/Scripts/python.exe scripts/run_fusion.py --phase all --variants mean
./.venv/Scripts/python.exe scripts/run_fusion.py --phase all --variants energy hybrid
./.venv/Scripts/python.exe scripts/check_fusion_mmd_reproducibility.py
./.venv/Scripts/python.exe scripts/verify_fusion_mmd_artifacts.py
./.venv/Scripts/python.exe -m fusionvc.cli report-fusion-mmd
```

最终比较矩阵有63个独立模型方案/种子结果。最早六个基线 checkpoint 另行保留，
不当作独立重复；各阶段还有 smoke、零KD和复现目录。
以上是训练和评分入口。阶段正式核验还需要对应 `validation-checks.json`、
各自复现/实现一致性收据及新增实验前的来源快照，不能只运行 report 就获得 VERIFIED。
首次重建必须在扩展实验之前按各阶段协议生成本次来源记录；仓库中的快照绑定本机
已完成实验的实际文件，不能拿来证明新训练结果。完整规则及命令见各阶段协议。

## 蒸馏及独立推理

```powershell
./.venv/Scripts/python.exe scripts/run_distillation.py --phase smoke
./.venv/Scripts/python.exe scripts/run_distillation.py --phase all
./.venv/Scripts/python.exe scripts/predict_student.py --export --target ACLY --cells 32
```

Teacher 按九个融合组的三种子官方验证 MAE 均值选择，使用固定 seed42；
Student 沿用原 MLP 和三种子，仅对150个训练靶点蒸馏，KD系数固定1。
详见 [蒸馏协议](distillation-protocol.md) 和 [独立推理说明](student-inference.md)。

## 冻结选择与最终保留测试

```powershell
./.venv/Scripts/python.exe -m fusionvc.cli freeze-final-selection
./.venv/Scripts/python.exe scripts/download_test_assets.py
./.venv/Scripts/python.exe scripts/run_final_evaluation.py --phase all
./.venv/Scripts/python.exe -m fusionvc.cli report-final-test
./.venv/Scripts/python.exe scripts/verify_project_completion.py
```

冻结前必须完成 MMD 与蒸馏的全部验证核验。冻结清单记录63个候选、八组最终
方案、三个种子、原MLP均值蒸馏对照、实际Teacher及各文件SHA256。
测试集下载固定官方对象generation并校验CRC32C与SHA256；不配置云账号或计费项目。
测试表达访问后不再调整模型、Teacher、损失、超参数或选择规则。

所有阶段可从已有完整产物继续；部分完成的下载支持分块恢复，最终评分可用
`--phase evaluate --label <冻结清单中的label>` 继续指定运行。
每次评分后清理其临时预测，保留checkpoint、控制抽样、真实DE和逐靶点分数。

历史结果与其校验快照必须保持不变。蒸馏准备后不要重写前面阶段的报告或收据；
冻结最终选择后不要重写任何选择来源。若改变科学协议，应使用新的版本和目录，
保留已完成实验的来源记录。

最终结果入口为 `results/final-test/README.md`；验证矩阵为
`results/fusion-mmd/README.md`；蒸馏公平性、成本和完整性为
`results/distillation/README.md`。验证成绩、保留测试成绩及历史阶段记录各自独立。
本地通过检查不等于已在远端运行 GitHub Actions 或发布仓库。
