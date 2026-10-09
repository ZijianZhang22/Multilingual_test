# 同一输入的激活热图：RunPod 一键运行（v2）

下载公开模型配对，对同一批英文输入比较激活。一次只加载一个模型；默认在同一次前向计算里提取三种池化表示，不重复加载三遍。

## 新 Pod

使用 PyTorch >=2.6 的 RunPod 镜像：

```bash
cd /workspace
git clone --single-branch --branch feature/literature-measurement-suite https://github.com/ZijianZhang22/Multilingual_test.git
cd Multilingual_test
bash run_activation_heatmaps.sh --pair all --pool all
```

已有 clone 时在该仓库执行 `git pull --ff-only origin feature/literature-measurement-suite`。先跑小模型可用 `--pair smol360_ru` 或 `--pair smol135_ru`。

默认输出到 `activation_heatmap_results_v2/`，与旧版目录分开。默认 100 条输入，自动选择 GPU bf16/fp16 或 CPU fp32。`--limit 10` 可检查流程；`--resume` 可复用设置匹配且完整的缓存。`SKIP_INSTALL=1` 跳过依赖安装。

## v2 改了什么

- 新默认输入 `inputs_100_en_diverse.jsonl`：100 条人工编写的不同句子，分为对话、叙事、描述、指令、科学、技术、推理、数量、问题、论证十组。它是探索性诊断集，不是正式 benchmark，也不是随机抽取的真实语料。保留旧的五模板输入文件供对照。
- `--pool all`（默认）：同时得到 `mean`、`last_nonpunct`、`last` 三套图。
- `mean`：平均所有非 special token 的隐藏向量，包含标点。
- `last_nonpunct`：从实际输入 token 序列向前寻找最后一个包含非标点、非空白字符的 token，排除 special token。只含标点的 token 会跳过；包含词和标点的混合 token 保留。没有改写或重新编码文本。
- `last`：原版最后一个输入 token，保留作为标点/EOS 对照。
- 元数据记录实际池化位置、对应解码文本以及是否截断。新旧缓存因 pooling_version 不同不能混用。
- 新增幅度比热图、每个句子组的逐层 CKA/漂移/幅度比/漂移与 NLL 相关性。每组只有十句，分组结果仍然很不稳定。
- NPZ 先写临时文件再原子替换，避免中断后复用半个文件。

三种池化使用同一前向计算和同一输入，因此它们的 NLL **应该完全相同**；改变的是用于比较的激活表示。

## 输出

```text
activation_heatmap_results_v2/
  index.html
  activation_heatmaps_results.zip
  invocation.json
  tiny_zh/                         另外两组为 smol135_ru、smol360_ru
    before.log / after.log
    compare_mean.log / compare_last_nonpunct.log / compare_last.log
    run_manifest.json
    mean/                          另有 last_nonpunct/、last/
      before.npz / after.npz        原始激活，不放入 ZIP
      comparison/
        00_overview.png
        01_before.png / 02_after.png
        03_rms_difference.png
        04_relative_drift.png
        05_directional_drift.png
        06_rms_ratio.png            after RMS / before RMS，1=幅度相同
        module_*_before/after/difference.png
        per_sample_layer.csv
        layer_summary.csv
        group_layer_summary.csv
        diagnostics.json           token、截断、损失退化数、整体相关性
        summary.json
```

从 RunPod 文件浏览器下载 `activation_heatmap_results_v2/activation_heatmaps_results.zip`，解压打开 `index.html`。完整原始 NPZ 留在服务器，如想以后重新计算指标请另行备份。

优先比较同一模型的 `mean` 与 `last_nonpunct` 总览：若只在 `last` 出现强烈条纹，很可能与句末位置有关；若三种都出现，则变化更广泛，但仍然不是机制的因果证据。

单独一种池化可用 `--pool mean`。此时输出为 `配对/comparison/`，不增加池化子目录。

旧输入对照：

```bash
bash run_activation_heatmaps.sh --pair all --pool all \
  --inputs experiments/activation_heatmaps/inputs_100_en.jsonl \
  --out /workspace/heatmaps_old_inputs_controls
```

自己的完整 checkpoint：

```bash
bash run_activation_heatmaps.sh \
  --before /workspace/anchor --after /workspace/adapted \
  --tokenizer /workspace/anchor --inputs /workspace/old_language_100.jsonl \
  --pool all --out /workspace/own_forgetting_heatmaps
```

输入是 JSONL，每行 `{"id":"unique", "group":"optional", "text":"nonempty text"}`。before/after 须同架构、同词表语义、共同初始权重；未合并 LoRA adapter 不支持。支持 layers.N 风格解码器。

## 解释限制

热图横轴为层，纵轴为输入句子；每格是整条池化向量的统计量，并非单个参数。模块图横轴才是通道编号。RMS 大不表示能力好；相对漂移大不等于遗忘百分比；CKA 衡量样本间结构，不保证功能保留。

模块图选当前样本集上差异最大的通道，是探索性选择，不是功能重要性结论。NLL 是原始输入整段的 next-token loss，不是问答正确率。公开模型配对的语言/SFT/分支混杂见 MODEL_PAIRS.md；要确证遗忘需要独立旧任务评测，并检查新语言是否真正改善。

不要修改 checkpoint 权重而保留原路径后使用 `--resume`。不同设置请用不同输出目录。

验证：`python -m unittest discover -s experiments/activation_heatmaps/tests`。覆盖数值对照、Unicode 标点与 special token 池化选择、缓存、命令、CSV/图册/ZIP；本地没有 GPU 模型运行验证。
