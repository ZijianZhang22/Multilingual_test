# 路线 B：英文为主模型的中文持续预训练与旧语言退化

默认模型是 HuggingFaceTB/SmolLM2-360M。它以英文为主，不是严格仅见过英文的模型。全参数训练，无 SFT/chat 模板、无 LoRA。

## 一键运行

使用 PyTorch >=2.6 的 RunPod GPU 镜像（建议至少 16GB 显存，24GB 更宽裕）。请预留约 15GB 以上磁盘，checkpoint 是 FP32 全量权重。

```bash
cd /workspace/Multilingual_test
git pull --ff-only origin feature/literature-measurement-suite
bash run_route_b_forgetting.sh
```

新 Pod 先 clone：

```bash
cd /workspace
git clone --single-branch --branch feature/literature-measurement-suite https://github.com/ZijianZhang22/Multilingual_test.git
cd Multilingual_test
bash run_route_b_forgetting.sh
```

默认三组依次运行，一次只加载一个模型。下载/准备数据、训练、评测、绘图、激活对比和 ZIP 打包都会自动完成。每一步有日志，失败立即停止。不要复用已有结果目录来覆盖实验。

```bash
# 后台运行，终端断开后仍继续（日志在当前仓库）
nohup bash run_route_b_forgetting.sh > route_b_run.log 2>&1 &
tail -f route_b_run.log
```

## 实验设置

| 设置 | 默认值 |
|---|---|
| 初始模型 | 同一个 SmolLM2-360M |
| 中文组 | 100% 中文 Wikipedia |
| 英文组 | 100% 英文 Wikipedia |
| 混合组 | 英文/中文 microbatch 交替，准确 50/50 |
| 每组总预算 | 256 optimizer steps × 2 microbatch × 8 accumulation × 256 tokens = 1,048,576 input tokens |
| 学习率 | 3e-5，16 step 线性 warmup 后恒定 |
| 评测步数 | 0 / 32 / 64 / 128 / 256 |
| 保存完整模型 | 64 / 256 |
| 热图 | 两个保存点与基础模型分别比较，三种池化，每个模型 100 句 |
| held-out | 每种语言分别 128 个验证块和 128 个测试块 |

三组匹配的是**总 token 数**；混合组只接触中文组一半的中文 tokens。因此它是相同总预算下回放的对照，不是相同中文曝光量的对照。

数据：`wikimedia/wikipedia` 的 `20231101.en` 和 `20231101.zh`，固定仓库 revision。先对整篇文章的文本哈希划分 train/val/test（80/10/10），再在文章内部切块；不把同一文章拆到不同集合。重复文本去重。每个评测文章最多贡献四块，减少长文章主导。小于 block-size 的文章和尾部不足一个块的文本被丢弃。精确去重不能排除近重复，也不能保证模型初始预训练未见过这些文章；这里测试的是持续预训练造成的前后差异。

各组从相同基础权重、相同随机种子重新加载。训练数据不循环复用，足额准备后按固定排列取块。模型参数/optimizer 保持 FP32，支持的 GPU 使用 bf16 autocast；开启 gradient checkpointing。每层热图使用原来的 activation_heatmaps 工具。

## 输出和判读

默认 `route_b_results/smol360_seed0/`：

- `forgetting_curves.png`：英文损失变化、中文损失变化、保留/新语言增益散点。
- `evaluation_summary.csv`：逐组逐 checkpoint 的 held-out NLL 与变化置信区间。
- `conclusions.json`：是否有英文退化且中文改善；中文组相对两个对照的英文损失差。
- `{zh_only,en_only,mixed}/metrics.json`：每个验证/测试块的 NLL，可重新分析。
- `{arm}/step_000064/`、`step_000256/`：完整训练 checkpoint。
- `heatmaps/{arm}/step_*/index.html`：各阶段三套激活图。
- `route_b_results.zip`：图片、CSV、日志、设置、诊断、图册。

英文 delta >0 表示原有英文建模能力退化；中文 delta <0 表示中文改善。主要证据组合：中文组英文变差、中文改善，而且英文退化超过英文对照/混合对照。两个语言都变差时优先检查训练稳定性和领域影响。

CI 使用配对的**文章聚类 bootstrap**，不是把每个 token 当独立样本。它只反映评测文章变化，没有涵盖训练 seed 的变化。默认 `--large-delta .5` 是探索性幅度标记（约 1.65 倍 PPL），不是公认的灾难性遗忘门槛。脚本不会仅凭热图或 NLL 自动宣称广泛能力的灾难性遗忘。仍需独立任务、更广英文语料与多个 seed。

每个阶段都评测测试集是为了观察诊断轨迹；如果根据这些结果调整学习率/训练预算，后续确认必须使用新的独立测试集，避免把测试集变成调参依据。

**ZIP 不包括 checkpoint、原始 NPZ 和训练数据块**。关闭/删除 Pod 前如需继续做 patching，请另外下载这些文件或保留持久卷。

## 调整预算

```bash
# 先用 135M 验证流程；正式结论仍须足够数据
bash run_route_b_forgetting.sh --model HuggingFaceTB/SmolLM2-135M \
  --steps 64 --eval-marks 16 32 64 --save-marks 32 \
  --out route_b_results/smol135_pilot

# 训练预算扩大，新的独立实验目录
bash run_route_b_forgetting.sh --steps 512 --eval-marks 32 64 128 256 512 \
  --save-marks 128 --out route_b_results/smol360_2m_seed0

# 只训练并生成曲线，跳过热图
bash run_route_b_forgetting.sh --skip-heatmaps --out route_b_results/loss_only

# 显存不足时，保持总 batch/token 预算相同
bash run_route_b_forgetting.sh --micro-batch 1 --grad-accum 16 \
  --out route_b_results/smol360_micro1
```

旧输出禁止直接复用；没有 optimizer 断点续训支持。中断时保留日志和已有 checkpoint，正式完整重跑请指定新的 `--out`。无需先训练英文 anchor：这里明确使用基础模型已有的英文能力作为 step 0。

## 验证

```bash
python -m unittest discover -s experiments/route_b_forgetting/tests
```

覆盖文章划分/精确去重、等预算语言调度、配对文章 bootstrap 和默认 token 预算。真实模型短训练验证见同目录的 `VALIDATION.md`（若存在）。
