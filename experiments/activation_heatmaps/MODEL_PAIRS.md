# 已核对的三组小模型（2026-10-09）

注意：这些是英文为主的基础模型与语言适应分支，并非严格控制所有变量的“纯单语 vs 完美双语”实验。这里的 before/after 文件名只是两个比较对象的标签，TinyLlama两者是共同基础checkpoint的两个分支，并不是前者直接训练得到后者。没有先证明遗忘，不应把差值热图叫作遗忘热图。

## 首选：TinyLlama，1.1B，中英

- https://huggingface.co/TinyLlama/TinyLlama_v1.1
- https://huggingface.co/TinyLlama/TinyLlama_v1.1_chinese

官方模型卡说明：先共同使用SlimPajama训练1.5T tokens，再分支持续预训练与cooldown到2T。标准分支继续使用100% SlimPajama；中文分支使用50% SlimPajama + 50%中文SkyPile。两个分支同架构、同tokenizer。SlimPajama包括不同来源文本及代码，英文为主不等于绝对只含英语。相较SFT模型，这组更适合探索加入中文持续预训练后的变化。不要换成v1.0 Chat或旧3T checkpoint：那是不同训练版本。

已检查配置：22层、hidden_size=2048、intermediate_size=5632、vocab_size=32000，LlamaForCausalLM；两者一致，无quantization_config。

## 最轻：SmolLM2-135M，英俄SFT

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://huggingface.co/nyuuzyou/SmolLM2-135M-Eagle

Eagle模型卡标明从SmolLM2-135M训练，使用英俄问答EagleSFT，2 epochs。它不是英俄从头预训练模型；作者明确说明俄语能力极弱，仅有SFT带来的有限改善。因此适合快速观察同骨干微调后的激活变化，不能将差异全部归因于双语知识形成。还混入聊天/指令格式与领域变化。

已检查配置：30层、hidden_size=576、intermediate_size=1536、vocab_size=49152，LlamaForCausalLM；两者一致，无quantization_config。

## 尺度复查：SmolLM2-360M，英俄SFT

- https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- https://huggingface.co/nyuuzyou/SmolLM2-360M-Eagle

同样由对应360M英文基础模型在英俄EagleSFT上微调，可与135M结果做尺度比较，但不能按同编号神经元跨135M和360M模型相减。具有与135M版本相同的SFT和能力限制。

已检查配置：32层、hidden_size=960、intermediate_size=2560、vocab_size=49152，LlamaForCausalLM；两者一致，无quantization_config。

## 一键试跑

```bash
# 解压后进入activation_heatmap_trial；安装依赖见README。
# 最合适的中英持续预训练对照
python run_pair.py --pair tiny_zh

# 先用最小模型检查流程
python run_pair.py --pair smol135_ru

# 再用360M复查
python run_pair.py --pair smol360_ru

# CPU可用，但速度依赖机器。不要在CPU上沿用默认bf16。
python run_pair.py --pair smol135_ru --device cpu --dtype float32

# 只打印命令，不加载模型
python run_pair.py --pair tiny_zh --dry-run
```

默认每组用同一批100句英文原始文本，不生成答案，不套各自chat template。首次可 `--limit 10` 检查下载、加载、hook是否成功，再跑100句。若测试中文或俄语，使用自己的JSONL并传 `--inputs`，分别指定 `--out` 避免覆盖英文结果。比较两模型时保持完全相同的输入token IDs；脚本使用基础模型tokenizer并检查两次实际token序列。

模型权重加载前仍应确认adapter/词表语义；本次核对了公开模型卡、完整权重文件列表和配置，没有下载权重运行真实模型。两组Eagle均公开safetensors；TinyLlama两组公开完整PyTorch权重。实际推理图尚未生成。

优先看：100句×层的相对漂移、1−cosine，再看同一层选定通道的前/后/差值。差异大的区域是后续验证候选，不是自动定位到“语言知识”或“遗忘神经元”。
