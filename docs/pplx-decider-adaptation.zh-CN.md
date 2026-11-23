# pplx-decider 适配说明

本文简要说明 vllm-jev 如何适配 Perplexity 的 [pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3)（27B）和 [pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3)（4B）：它们和 WeMM-JEV 有哪些相同和不同、怎么在 vLLM 上实现、做了哪些验证，以及还有哪些限制。

## 1. 模型特点

两个尺寸使用同一套推理包 `pplx_decider`（协议 v4，`joint-chunked-readout-v1`），格式一致：

| 项 | pplx-decider-v1.3 | pplx-decider-flash-v1.3 |
|---|---|---|
| 底座 | Qwen3.8-27B（64 层，16 层全注意力） | Qwen3.5-4B（32 层，8 层全注意力） |
| 主干 | 裸 `Qwen3_5Model`，带视觉塔，BF16 约 52 GB | 同左，约 9 GB |
| 读出头 | `readout.safetensors`：`[255, 5120]` BF16 | `[255, 2560]` BF16 |
| 温度 | 1.2264 | 1.3303 |
| 注意力 | `noncausal_full_attention`：全注意力层双向，GDN 层因果 | 同左 |

仓库里还有 `locate-span/`（Locate 片段抽取：LoRA + span head）和 `source/`（官方推理代码）。

### 与 WeMM-JEV（d3）的异同

主干结构、双向注意力、255 个答案码的 readout 都与 WeMM-JEV 相同，因此 vLLM 侧直接复用 `VllmD3Qwen35ForEmbedding`。不同之处都在 prompt 和答案格式：

| 项 | WeMM-JEV（d3） | pplx-decider |
|---|---|---|
| Prompt 结构 | 完整 ChatML，以 assistant 开头结束 | 分块：`system + 图片 + "State:\n" + state`（user 轮不闭合）+ `<\|fim_middle\|>` 分隔符 + `"Question:\n…Options:\n…"` + `<\|fim_suffix\|>` 读出标记 |
| 读出位置 | 最后一个 token | `<\|fim_suffix\|>`；每题单独一条序列时它就是最后一个 token |
| 分块分词 | 整段一起分词 | context 和每个问题块各自分词再拼接，块边界不会合并 token |
| 特殊 token 转义 | 拒绝图片占位符 | 用户文本里的特殊 token 改成全角括号（如 `＜\|fim_suffix\|＞`） |
| 空 state | `(empty)` | 空字符串原样保留，`None` 序列化为 `null` |
| Score | 2–10 级，级别不能重复 | 2–255 级，可以重复 |
| Choice 置信度 | 1 − 归一化熵 | 最高概率相对均匀分布的提升，缩放到 [0, 1] |
| Score 置信度 | 1 − 归一化熵 | 1 − 期望距离 / 均匀基线 |
| 图片 | 先缩放到 1.6 MP 并重新编码，再交给 processor | 只用 processor 缩放（`max_pixels` 取自 `decision_config.json`） |
| 长度上限 | 默认 16,384 | 默认 32,768（与官方一致） |

## 2. 实现方案

沿用 d3 的整体方案：**导出 checkpoint → vLLM pooling 输出最后位置的隐状态 → 服务层做 readout**。导出产物仍是 `d3_manifest.json`，用 `prompt_protocol: pplx_joint_readout_v4` 区分。

| 文件 | 职责 |
|---|---|
| [vllm_jev/pplx.py](../vllm_jev/pplx.py) | 复刻官方 `protocol.py`：分块 prompt、特殊 token 转义、题目校验、答案格式；`PplxService` 继承 `D3Service` |
| [vllm_jev/pplx_export.py](../vllm_jev/pplx_export.py) | 校验 `joint_config.json` 和 `decision_config.json`，按官方 `release-manifest.json` 的 sha256 固定两个仓库的版本，硬链接权重，改写 `config.json`（架构名 + `is_causal: false`），生成 manifest（含 marker、分隔符和转义 token 列表） |
| [vllm_jev/d3.py](../vllm_jev/d3.py) | 抽出 `_inputs` / `_prompts` / `_answer`，并支持按 manifest 配置图片上限，供子类复用；WeMM-JEV 行为不变 |
| `native_models.py` / `cli.py` / `endpoint.py` / `d3_export.py` | 自动识别（有 `joint_config.json` 的 d3 布局即为 pplx）、serve 默认参数、按协议选择 service、manifest 校验 |

几项设计取舍：

- **只支持 `separate` 模式**：这是官方默认、评测分数所用的模式，每题一条序列，与 vLLM 的“每条序列只取最后位置”天然对应。`isolated` 模式需要逐层修改注意力 mask 和重启 GDN 状态，vLLM 原生不支持。
- **不支持 Locate**：它需要动态开关的 LoRA、替换 marker 的 embedding，以及读取全部 state token 的 span head。请求中有 `locate` 题时返回 422。
- **只支持 native 顺序**：官方协议固定为 state 在前、问题在后，`instructions-first` 和其他 `context_order` 在启动时就拒绝；`state_template` 和 `instruction_template` 仍然可用。
- **readout 按 BF16 计算**：与官方在 CUDA 上的 `nn.Linear(dtype=bf16)` 一致，之后转成 FP32 做温度缩放和 softmax。
- **不执行官方代码**：只下载权重、readout、协议、tokenizer 和 processor 文件，`source/` 和 `locate-span/` 都不下载。

## 3. 使用方式

```bash
vllm-jev serve perplexity-ai/pplx-decider-flash-v1.3   # 4B，约 9 GB
vllm-jev serve perplexity-ai/pplx-decider-v1.3         # 27B，约 52 GB

# 本地已有 release（离线）
vllm-jev serve /path/to/pplx-decider-v1.3 --workspace /同一文件系统/.vllm-jev
```

请求与官方 schema 相同，图片放在顶层 `images`：

```json
{
  "state": "My integration keeps failing with a 401 since we rotated our API key.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"},
    "frustration": {"type": "score", "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]}
  }
}
```

## 4. 验证情况

所有验证都在 CPU 上完成，**还没有在 GPU 上跑过推理**。

| 验证项 | 结果 |
|---|---|
| 文件 sha256 | 4B 的全部下载文件和 27B 的非权重文件在本地计算，27B 的权重 shard 与 HF LFS 记录比对；均与官方 `release-manifest.json` 一致，已固定到代码中 |
| token 序列（同一 transformers 版本） | 7 种 state、3 组图片，每组 5 道题（含 255 个选项、特殊 token 转义、前后空白、多图），与官方 `Encoder.plan` 逐 token 一致，读出标记都在最后一位；`pixel_values`、`image_grid_thw` 一致 |
| token 序列（跨 transformers 版本：官方要求 5.17，vLLM 0.29 要求 <5.11） | 27B 和 4B 各 55 条序列，token、`pixel_values`、`image_grid_thw` 全部一致 |
| 答案格式（Choice / Noul / Score，随机概率） | 与官方 `protocol.answer` 一致（误差 < 1e-12） |
| vLLM 0.29 的图片占位展开 | 15 条图片 prompt 展开后与服务层预期的序列完全一致 |
| vLLM 对导出 config 的判定 | `attn_type=encoder_only`，chunked prefill 和 prefix cache 均关闭 |
| 权重加载（4B 真实权重；27B 用 shard header 中的形状在 meta 设备上加载） | 4B：627 个参数全部加载；27B：991 个参数全部加载，总参数量 26,085,330,160，与 release 一致；全注意力层均为 `ENCODER_ONLY` |
| 单元测试 | 新增 52 个（含 2 个需本地 release 的对照测试），原有测试全部通过 |

对照测试的运行方式：`PPLX_DECIDER_RELEASE=/path/to/release pytest tests/test_pplx.py -m reference`（只导入官方的 `protocol.py` 和 `types.py`）。

## 5. 限制与待办

1. **GPU 实测**：需要确认能正常加载和推理，并与官方 `Decider` 对比概率差和答案翻转率。仓库外有对比脚本 `compare_gpu.py`：先在官方环境里运行 `reference` 子命令生成参考答案，再对 vllm-jev 服务运行 `compare`。
2. **性能**：与 WeMM-JEV 一样，双向注意力下不能复用前缀缓存，同一 state 的多道题会各自完整 prefill。官方的 `isolated` 模式可以只读一次 state，但 vLLM 原生不支持。
3. **数值差异**：vLLM 的注意力和 GDN kernel 与官方的 SDPA / transformers 实现不同，概率会有细微差异，接近平局时答案可能翻转。官方 fast path 与参考实现的 top-1 一致率为 99.6%–99.7%，可作参考。
4. **Locate 和 isolated 模式**：如有需要，需在 vLLM 侧单独实现。
5. **本地 release 的哈希校验**：与 WeMM-JEV 相同，用本地目录启动时只检查结构，不与固定版本的 sha256 比对。
6. **图片输入**：只接受 PNG / JPEG / WebP 的 base64 data URL（单张不超过 8 MB、1600 万像素）；官方还接受文件路径和 PIL 图片。
