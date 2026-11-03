# WeMM-JEV 适配说明

本文简要说明 vllm-jev 如何适配 [Rover00/WeMM-JEV](https://huggingface.co/Rover00/WeMM-JEV)：它的结构有什么特殊之处、怎么在 vLLM 上实现、做了哪些验证，以及还有哪些限制。

## 1. 模型特点

WeMM-JEV 是基于 Qwen3.8-27B 的多模态决策模型，官方格式为 `d3-code-readout-v1`：

| 项 | 内容 |
|---|---|
| 主干 | `Qwen3_5Model`（不含 lm_head），64 层：48 层 Gated DeltaNet 线性注意力 + 16 层全注意力，带视觉塔，BF16 约 52 GB |
| 读出头 | `readout.safetensors`：`[255, 5120]` 的 FP32 线性层，在最后一个位置给 255 个答案码（`A`…`Z`、`AA`…）打分 |
| Prompt | 固定 system prompt；user 内容为 `State / Question / Options`，每个选项一行 `码: 描述`；图片放在文本之前 |
| 题型 | Choice 1–255 个选项；Noul 视为 2 个选项；Score 把 2–10 个等级当作选项，返回期望分数 |
| 注意力 | `attention_mode: noncausal_full_attention`：全注意力层**双向**，GDN 层保持因果 |

原有代码没有这种格式的探测和导出逻辑，所以 `vllm-jev serve Rover00/WeMM-JEV` 会报 "no supported native vLLM Jev checkpoint layout"。

### 与 Qwen3.5 底座的差异

WeMM-JEV 声明的底座是 `Qwen/Qwen3.8-27B`（revision `1d4bf0f2`），使用 Qwen3.5 架构。与底座相比：

| 项 | Qwen3.8-27B 底座 | WeMM-JEV |
|---|---|---|
| 层结构 | 64 层，每 4 层为 3 层 GDN + 1 层全注意力，带视觉塔 | 相同，层数、维度、视觉塔一致 |
| 全注意力层（16 层） | 因果 | **双向**（`noncausal_full_attention`） |
| GDN 层（48 层） | 因果 | 仍为因果：GDN 是从左到右的递推，本身就是因果的 |
| 输出头 | `lm_head` `[248320, 5120]`，预测下一个 token | 去掉 `lm_head`，改为 `[255, 5120]` 的 FP32 readout，只给答案码打分 |
| MTP 模块 | 有（`mtp.*`） | 无 |
| 权重保存格式 | `Qwen3_5ForConditionalGeneration`，权重名为 `model.language_model.*`、`model.visual.*` | 裸 `Qwen3_5Model`，权重名为 `language_model.*`、`visual.*` |
| 权重大小 | 约 55.6 GB | 约 52.2 GB（少了 `lm_head` 和 MTP） |
| 用法 | 逐 token 自回归生成 | 每道题只做一次 prefill，读出最后位置，不生成文本 |
| tokenizer、chat template、图片/视频预处理配置 | — | 与底座完全一致 |

体现到 vLLM 的运行方式上：

| | 底座 | WeMM-JEV |
|---|---|---|
| 全注意力层的 `AttentionType` | `DECODER`：因果，有 KV cache | `ENCODER_ONLY`：双向，无 KV cache |
| 整个模型的 `attn_type` | `hybrid`（生成模型） | `encoder_only`（pooling 模型） |
| 分块 prefill | 支持 | 不支持，每个 prompt 一次算完 |
| 前缀缓存 | 支持 | 不支持，前面 token 的表示受后面内容影响 |

## 2. 关键问题：双向注意力

双向注意力是唯一不能直接套用现有适配模式的地方。vLLM 0.29.0 已经提供了所需能力（来自 ColQwen3.5 的提交 `3c8e49596c`）：

- 配置中 `is_causal=False` 时，`Qwen3NextAttention` 会把全注意力层建成 `AttentionType.ENCODER_ONLY`（双向），GDN 层不受影响；
- `ModelConfig.attn_type` 随之变为 `encoder_only`，vLLM 自动关闭 chunked prefill 和 prefix cache，保证每个 prompt 一次性完整 prefill。

因此适配时不需要改 vLLM 本身。导出时在 `config.json` 的顶层和 `text_config` 中都写入 `is_causal: false` 即可：顶层字段供 vLLM 判断调度方式，`text_config` 中的字段决定各层的构建方式。

但只能使用 **V1 model runner**。vLLM 0.29 在 CUDA 上默认用 V2 runner，而 V2 选择 model state 时先看有没有 `ENCODER_ONLY` 层、再看是不是 hybrid 模型，会给 WeMM-JEV 选中面向 BERT 的 `EncoderOnlyModelState`，缺少 GDN 状态管理。ColQwen3.5 的双向支持是在 V1 runner 上验证的，V1 会把 encoder-only 层和 GDN 层分组并存。因此 CLI 在 serve d3 时默认设置 `VLLM_USE_V2_MODEL_RUNNER=0`；若显式开启 V2，模型会拒绝启动。

官方实现会把多道题左 padding 后组成一个批次一起前向；vLLM 则把每道题作为一条独立序列，不做 padding。我们用实验确认了两者等价（见第 5 节）。

## 3. 实现方案

整体沿用 Clef / JevAny 的模式：**先导出成 checkpoint，再由 vLLM pooling 模型输出隐状态，由服务层完成读出**。

```
请求 ──> D3Service._compile ──> 每题一条 prompt（token ids + 图片）
                                     │
                     vLLM encode(task=embed, LAST pooling，不做归一化)
                                     │
               最后位置的隐状态（FP32） ──> readout + 按选项数 mask + softmax
                                     │
                              按官方格式组装 Choice / Noul / Score 答案
```

| 文件 | 职责 |
|---|---|
| [vllm_jev/d3.py](../vllm_jev/d3.py) | Prompt 与答案码（逐字节复刻官方 `d3_format.py`）、题目校验、图片处理（与官方 1.6 MP 缩放策略一致）、FP32 readout、答案格式、vLLM 服务层 |
| [vllm_jev/d3_export.py](../vllm_jev/d3_export.py) | 校验 release（Hub 版本固定在 commit `a463f840`，并比对 sha256），硬链接权重，改写 `config.json`（架构名 + `is_causal`），生成 `d3_manifest.json` |
| [vllm_jev/model.py](../vllm_jev/model.py) | `VllmD3Qwen35ForEmbedding`：权重名映射（`language_model.*` → `language_model.model.*`）；在双向模式下检测到 chunked prefill 或 prefix cache 被开启时拒绝启动 |
| `native_models.py` / `cli.py` / `endpoint.py` / `checkpoint.py` | Hub 和本地目录的自动探测、serve 默认参数、`/v1/systemone` 路由（接收顶层 `images`）、checkpoint 校验 |

几项设计取舍：

- **不执行官方代码**：只下载权重、readout、tokenizer 和 processor 文件；`d3_runtime.py`、`inference/` 等代码既不下载也不运行。
- **超长不截断**：单题 token 数（含图片 token）超出上限时直接返回 422。官方没有设上限，默认取 16,384，可用 `--max-model-len` 调整。
- **视频暂不支持**：官方推理包默认也是关闭的。
- `context_order`、`instructions-first` 等模板可以使用：图片始终紧贴在 State 之前，结尾的答题提示始终在最后。

## 4. 使用方式

```bash
# 从 Hub 下载（固定版本，约 52 GB）
vllm-jev serve Rover00/WeMM-JEV

# 本地已有 release（普通目录或 HF cache 的 snapshot 目录都可以），全程离线
vllm-jev serve /path/to/WeMM-JEV --workspace /同一文件系统/.vllm-jev --served-model-name WeMM-JEV
```

请求示例：

```json
{
  "state": "Look at the image.",
  "images": ["data:image/png;base64,..."],
  "questions": {
    "object": {"type": "choice", "instructions": "What is shown?",
               "criteria": {"A": "A cat", "B": "A dog"}}
  }
}
```

本地部署时需要注意：

- workspace 和模型在同一文件系统时，权重通过硬链接复用；跨文件系统时会完整复制，需要额外约 52 GB 空间。
- 用原始目录启动时，每次都会对全部文件重新做 sha256 校验；直接 serve 导出后的 `<workspace>/checkpoint/local/<指纹>` 目录可以跳过这一步。

## 5. 验证情况

目前只在 CPU 上做了验证，**还没有在 GPU 上加载过真实权重**。

| 验证项 | 结果 |
|---|---|
| 255 个答案码 | 与 `decision_config.json` 一致 |
| 文本 prompt（6 种题型 × 7 种 state） | 与官方 `d3_format.render` 逐字节一致 |
| 图片 prompt、图片 token 数 | 与官方 processor 一致 |
| 图片 1.6 MP 预缩放 | 与官方 `inference/_images.py` 的输出像素逐字节一致 |
| vLLM 0.29 的图片占位展开 | 与服务层预期的 token 序列完全一致 |
| vLLM 0.29 对导出 config 的判定 | `attn_type=encoder_only`，chunked prefill 和 prefix cache 均关闭 |
| 双向注意力的必要性（随机小模型） | 因果与非因果最后状态相差 0.14–0.29，`is_causal` 必须设置 |
| 逐条无 padding vs 官方左 padding 批处理 | 最大差 2.4e-7（float32 舍入误差），二者等价 |
| 单元测试 | 新增 46 个，全套 255 个通过 |

## 6. 限制与待办

1. **GPU 实测**：需要确认 27B + 图片能正常加载和推理，并对比与官方 `d3_runtime` 的概率差、答案是否翻转。仓库外有现成的对比脚本 `compare_gpu.py`。
2. **性能**：双向注意力下不能复用前缀缓存，同一 state 下的多道题会各自完整 prefill，只能靠 vLLM 的连续批处理提高吞吐。
3. **数值差异**：vLLM 的注意力 kernel 与官方的 SDPA 不同，概率会有细微差异，接近平局时答案可能翻转。
4. **本地 release 的哈希校验**：目前只检查结构完整，不与 Hub 固定版本的 sha256 比对；需要的话可以加一个识别逻辑。
