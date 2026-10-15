# 前期资源与环境状态

**2026-10-15 更新。** 八组模型、融合MMD与知识蒸馏均已完成本机训练和完整验证。
本地69项测试与Ruff通过，最终保留测试已按冻结清单完成。独立Student包已导出，
CPU推理验收通过。测试下载增加固定requests2.34.2 / google-crc32c1.9.0，
仍使用项目内`.venv`，未替换系统Python或修改云账号/计费设置。
当前状态以[项目记录](project-status.md)和[全项目完成状态](project-completion-state.json)为准。
下面内容为早期环境搭建的历史记录。

**2026-10-09 更新。** 冻结 scGPT 已完成正式验证，Geneformer V1-10M 已在同一 GPU 环境
完成严格权重加载、实际训练对照前向和官方实现对照。Geneformer 增加固定版本
Transformers 4.57.1 / safetensors 0.6.2，系统 Python 不变；详细边界见
[scGPT 协议](scgpt-protocol.md)、[Geneformer 协议](geneformer-protocol.md)。
下面“尚未前向/适配”的文字为下载及最初基线阶段的历史记录。

**更新：2026-10-03 基线阶段已完成。** GPU 训练环境位于 `.venv`：Python 3.12、
PyTorch 2.8.0+cu128，RTX 4060 CUDA 实际运算已验证。MLP 与 Transformer 各完成三个种子。
依赖锁位于 `requirements/windows-training.lock.txt` 与 `requirements/windows-constraints.txt`。
以下原下载阶段说明保留为历史记录，其中“GPU 环境尚未配置”仅描述当时状态。
本阶段尚未进行 scGPT/Geneformer 端到端前向；不能把基线环境可运行等同于所有基础模型已适配。

本次任务是下载前期所需资源。已建立项目内独立的 `.venv-download`，只用于 HTTP 下载、CRC/SHA 校验、HDF5 与权重结构检查，不是训练环境。

已检查本机 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 显存。系统现有 `E:/Python/python.exe` 为 Python 3.12，PyTorch 2.12.0+cpu，CUDA 不可用。本次未替换系统 PyTorch，也未安装大型 CUDA Toolkit、Conda 或 WSL。

后续训练需要单独配置 GPU 版 PyTorch 和模型适配依赖，按文件锁定版本。下载环境的 lock 文件仅记录下载工具，不能作为训练 requirements。

保存了以下官方源码快照：

- `vendor/scGPT`：当前官方版本，含不依赖 torchtext 的词表兼容实现。
- `vendor/scGPT-paper2024`：2024 年最后一个官方提交，原 pyproject 指定 Python <3.11，且依赖 torchtext；不能直接认为其完整依赖可在当前 Python 3.12 中安装。
- `vendor/Geneformer`：官方 tokenizer、embedding 等源码及字典，权重只取得 V1-10M。
- `vendor/cell-eval-vcc2025`：2025 比赛结束前的官方提交；后续优先据此核对 2025 评价设置。
- `vendor/cell-eval`：当前官方实现，仅作参考。两份实现不代表评分配置已经完全锁定。

独立 GPU 环境配置、预训练编码器端到端前向、批量与显存测量尚未进行。模型文件可读和数据 checksum 正确，不等同于八组模型已实现或可直接训练。

下载脚本可用 `.venv-download/Scripts/python.exe scripts/download_assets.py` 恢复。多流下载以 `.resume.json` 记录完成块；未完成的 `.part` 文件不能用于训练。校验和来源记录位于 `docs/download-status.json` 与 `docs/source-provenance.json`。
