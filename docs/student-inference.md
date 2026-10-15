# 蒸馏 Student 独立推理

## 新克隆仓库时

源码发布包默认不包含 `outputs/08_distillation/seed_42/best.pt`、完整预处理缓存或
`outputs/deployment/student-v1`。本机演示通过不代表克隆后这些文件自动存在。
需要另行取得完整推理包，或先按完整实验流程训练并导出；下面命令不是无资源的开箱演示。

## 已有正式checkpoint或推理包时

正式 Student 是与原 MLP 同架构的 1,266,144 参数模型。导出固定 seed42，
不按测试成绩挑选种子。推理包只包含权重、基因描述、基因名和训练对照的固定子集；
推理时不需要 Teacher、scGPT、Geneformer、原始 h5ad 或蒸馏软标签缓存。

在完整训练和验证核验通过后，从项目根目录执行：

```powershell
./.venv/Scripts/python.exe scripts/predict_student.py --export --target ACLY --cells 32
./.venv/Scripts/python.exe scripts/predict_student.py --target GATA1 --cells 64 --output outputs/demo/GATA1.npz
```

默认推理包在 `outputs/deployment/student-v1`，默认预测在
`outputs/demo/student_prediction.npz`。运行前会验证包内五个文件的 SHA256；
已有包与当前 checkpoint 不同则报错，不会静默覆盖。
可用 `--bundle` 指定已有包，`--device cuda` 切换到 GPU。

预测文件可直接读取，无需允许 pickle：

```python
import json
import numpy as np

with np.load("outputs/demo/student_prediction.npz", allow_pickle=False) as result:
    expression = result["expression"]  # [预测细胞数, 18080]
    genes = result["genes"]
    metadata = json.loads(str(result["metadata"]))
```

输出是训练 library-size 尺度上的连续 log1p 表达，不是整数 counts。
靶点必须在项目基因表内；表内未见靶点可以计算，但不代表已验证其预测可靠性。
单细胞差异主要来自抽样的训练对照，模型学习的是共同条件位移。

默认包使用 256 个固定训练对照，便于搬运。正式评分使用完整训练对照库，
因此此演示的抽样与官方完整评价不同。调整 `--control-cells` 时应使用新的
`--bundle` 目录。模型和依赖代码仍需安装 FusionVC、PyTorch、NumPy 及基础依赖；
本项目没有将第三方数据和权重重新许可或上传到远端。

实际导出、有限值、表达尺度和 checkpoint 绑定的验收记录见
`student-demo-verification.json`；这些检查不代替完整官方测试评分。
