# 同一输入的激活热图：RunPod一键运行

不需要先训练新模型。脚本下载公开模型配对，向两个模型分别输入同一批100句英文，直接生成PNG、CSV和HTML图册。模型依次加载，一次一个。

## 运行

使用带PyTorch>=2.6的RunPod镜像，进入仓库根目录：

```bash
# 推荐：1.1B英文为主分支与中英持续预训练分支
bash run_activation_heatmaps.sh --pair tiny_zh

# 更小的135M英俄SFT，先检查流程
bash run_activation_heatmaps.sh --pair smol135_ru

# 顺序运行所有三组：1.1B中英、135M英俄、360M英俄
bash run_activation_heatmaps.sh --pair all

# 已有依赖时跳过安装，失败重跑时复用匹配的已完成提取
SKIP_INSTALL=1 bash run_activation_heatmaps.sh --pair all --resume
```

默认自动选择GPU bf16/fp16或CPU fp32。`--limit 10`用于快速检查；默认100句。若内存/显存紧张可改`--max-length 64`。一次处理一条输入，不使用生成或KV cache，也不安装FlashAttention。

```bash
# 自己的遗忘前后完整checkpoint；输入需要来自固定的旧语言测试集
bash run_activation_heatmaps.sh \
  --before /workspace/path/to/anchor \
  --after /workspace/path/to/adapted \
  --tokenizer /workspace/path/to/anchor \
  --inputs /workspace/old_language_100.jsonl \
  --out /workspace/own_forgetting_heatmaps
```

`before`和`after`必须同架构、同tokenizer/词表语义、共同初始权重。需要完整模型目录，不支持直接输入未合并LoRA adapter。支持Qwen/Llama/Gemma风格decoder.layers。

## 输出

默认位于仓库根目录`activation_heatmap_results/`：

```text
activation_heatmap_results/
  index.html                       所选配对全部PNG的图册
  activation_heatmaps_results.zip   可下载的图片、CSV、日志、设置包
  tiny_zh/                         每组单独目录
    before.npz / after.npz          原始池化激活，留在服务器，不包含在ZIP
    before.log / after.log / compare.log
    comparison/
      00_overview.png               六格总览，建议先看
      01_before.png / 02_after.png  句子×层，隐状态RMS，相同色标
      03_rms_difference.png         后减前的幅度变化
      04_relative_drift.png         向量差范数 / 训练前范数
      05_directional_drift.png      1−cosine
      module_*_before/after/difference.png
      per_sample_layer.csv
      layer_summary.csv            每层CKA与平均漂移
      summary.json
```

运行完成后，从RunPod文件浏览器下载`activation_heatmaps_results.zip`，解压即可打开`index.html`查看全部图片。日志逐条打印进度；任何阶段失败会停止，不生成成功提示。

## 你在比较什么

默认`inputs_100_en.jsonl`是人工组合的试跑句子，不是正式benchmark。两个模型使用同一个tokenizer，对相同文本得到相同token IDs。先比较旧语言英文的响应，也可以准备100句中文/俄语另跑，使用不同`--out`。

JSONL每行格式：

```json
{"id":"en_001","group":"greeting","text":"Hello, how are you today?"}
```

- 默认取每层block输出的最后输入token向量；也可以`--pool mean`，两次都使用相同设置。
- module图默认记录约1/3、2/3、最后层的Q/K/V/O、gate/up/down投影输出。`--layers 7 14 21`可指定0开始的层号。
- module图选本批输入上平均绝对激活差最大的64通道；前/后图同通道顺序、同色标。它是探索性选取，不代表已经定位到功能重要单元。
- down_proj输出通道不是SwiGLU中间神经元；这里的模块通道与MLP原始中间神经元需要区分。
- RMS变化小不代表方向没变，需结合relative drift与cosine。CKA比较固定样本间的表示结构，不保证功能保留。
- NLL是原始输入整段文本的下一个token预测损失，不是问答正确率或answer-only loss。
- 各配对的训练与解释限制见`MODEL_PAIRS.md`。TinyLlama是共同起点的两个分支；Eagle有SFT和领域/格式因素，俄语能力有限。
- 这些公开配对没有预先确认发生灾难性遗忘。真正的“遗忘前后”分析要先在独立旧任务上确认退化，再解释激活变化。

## 断点与重复运行

`--resume`只复用输入、模型名、tokenizer、池化、精度、长度与层设置匹配的完整NPZ，并重新生成图。不同设置请使用不同`--out`，防止混淆。不要在保留路径的同时修改checkpoint权重后用`--resume`；模型名/本地路径不变并不能自动识别权重内容变化。完整提取文件写完后才可复用，中断的提取会重跑。

## 验证

覆盖身份对照、向量符号翻转、CKA、输入不一致拒绝、断点缓存核对、命令生成、HTML/ZIP生成与数值绘图。测试通过不代表真实模型的结果。执行`python -m unittest discover -s experiments/activation_heatmaps/tests`。
