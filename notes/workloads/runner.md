# `kvpim/runner.py` —— 喂给 vLLM 什么、从 vLLM 取什么（359 行）

> 术语见 [`README.md`](README.md)｜数据流全景见 [`CODE.md`](CODE.md)

## 这个文件是什么

**唯一和 vLLM 打交道的地方。** 一组配置（一个拓扑 × 一个 block 档 × 一个容量档）
从头到尾由它跑完，产出 `manifest.json` / `calls.jsonl`，并驱动
[`events.py`](events.md) 和 [`dump.py`](dump.md)。

---

## 1. `RunConfig` —— 矩阵里的一个格子

```python
RunConfig(topology="T3", block_tier=16, capacity_tier="ample")
```

### block 档 → 引擎参数

```python
_TIER_TO_PHYSICAL = {4: 16, 8: 16, 16: 16, 32: 32, 64: 64}

block_size        = _TIER_TO_PHYSICAL[block_tier]
prefix_match_unit = block_tier if block_tier in (4, 8) else None
```

**依据**：计划 §6.1。本仓库所有可用 CUDA 后端都要求 block 是 16 的倍数
（`flash_attn.py:141` 的 `if block_size % 16 != 0: raise`），**物理 4/8 跑不了**。
所以 4/8 档用 `block_size=16` + `prefix_match_unit=4/8`——匹配粒度可以细于物理块
（`vllm/config/cache.py:56-67` 的文档串明说"can be set finer than the physical KV
cache block sizes … enabling cache hits at boundaries inside a physical block"）。

**如实标注的限制**（计划 §6.1）：4/8 档的**驱逐仍按 16-token 物理块发生**，
只有前缀匹配粒度是 4/8。现有内核做不到驱逐粒度也到 4/8。

### 哪些档 dump

```python
@property
def dump_enabled(self):
    return self.capacity_tier == "ample" and self.block_tier not in (4, 8)
```

- **受限档不 dump**：块随时被踢，dump 时机不可控（计划 §6.3）
- **4/8 档不 dump**：它们的物理块就是 16 token，张量与 16 档是同一批数据，
  按计划 §6.1 在分析侧切分复用。存储从 740 GB 降到 444 GB

---

## 2. 输入侧：喂给 vLLM 什么

### 2.1 分词（`_tokenize`，runner.py:109）

```python
encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
return list(ids)
```

**为什么 driver 自己分词、传 token id 而不是字符串** —— 计划 §3 的「铁律」：

> *"driver 自己 tokenize，传 `prompt_token_ids` 而非字符串。否则你不知道 vLLM 内部
> 切出的 token 序列，位置对不上，de-RoPE 全错。"*

我们要给每个 dump 出来的块记 `positions`，位置必须是我们自己确知的。

**为什么有 `hasattr(encoded, "keys")` 这个分支**：实测 transformers 5.14.1 的
`apply_chat_template(tokenize=True)` 返回的是 **`BatchEncoding`**（含
`input_ids`/`attention_mask`），不是 list。直接当 list 用，`max()` 会遍历出字符串键，
最终报 `TypeError: '>' not supported between 'str' and 'int'`。
**计划成文时的写法在本版本上是错的**，这条如果没抓到，token 全错。

验证：`tokenize=False` 拿文本再编码，与 `enc["input_ids"]` **完全一致**。

### 2.2 提交请求（runner.py:286）

```python
output = llm.generate(TokensPrompt(prompt_token_ids=prompt_token_ids), sampling, use_tqdm=False)[0]
```

**依据**：本 commit 的签名是 `generate(prompts: PromptType, ...)`
（`vllm/entrypoints/llm.py:414`），**旧的 `prompt_token_ids=[ids]` 关键字已删除**。
`TokensPrompt` 定义在 `vllm/inputs/llm.py:106`。计划 §3 的示例代码是旧 API。

### 2.3 采样参数（runner.py:229）

```python
SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=cfg.max_tokens)
```

计划要求 `temperature=0`。但**光设 temperature 不够**：模型自带的
`generation_config.json` 里有

```json
{"temperature": 0.7, "top_k": 20, "top_p": 0.8, "do_sample": true}
```

vLLM 默认会把它合并进来（`vllm/config/model.py:1594-1625` 的 `get_diff_sampling_param`）。
所以额外在引擎侧设了 `generation_config="vllm"` —— 依据是同文件 :312-317 的文档串：
*"If set to `"vllm"`, no generation config is loaded, vLLM defaults will be used."*

不设的话，实测会走进 flashinfer 的 top-k/top-p 采样路径（因为 `top_k=20` 非空），
既污染 `temperature=0` 口径，又触发 JIT 编译崩溃。

### 2.4 引擎参数逐条依据（`build_llm`，runner.py:127）

| 参数 | 值 | 为什么 |
|---|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | 只有同进程才能读 `kv_caches` 做 dump（计划 §7.2）|
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 装的 flashinfer 是 cu13 版，首次采样 JIT 编译，集群无 ninja 直接崩；贪心解码用不上 |
| `enforce_eager` | `True` | 关 CUDA graph，计划要求 |
| `kv_cache_dtype` | `"auto"` | 计划：**绝不 fp8**，量化会掩盖我们要测的数值差异 |
| `enable_prefix_caching` | `True` | 前缀共享是被测对象本身 |
| `gpu_memory_utilization` | `0.95` | 比常用的 0.90 多约 9% 池子（93,424 → 102,048 token），让两个容量档能跑在同一种卡上 |
| `max_model_len` | `65536` | 模型 262144 上下文，vLLM 要求池子装得下一条满长请求，不设起不来 |
| `seed` | `0` | 可复现 |

⚠️ **`max_model_len` 与受限档的耦合**：设了 `num_gpu_blocks_override` 之后，
准入检查是按**压低后的**池子算的（`vllm/v1/core/kv_cache_utils.py:2125-2145`
把 `available_memory` 改写成 `override * bytes_per_block`）。
所以受限档的 `max_model_len` 必须 ≤ 池子 token 数，否则引擎起不来。

---

## 3. 输出侧：从 vLLM 取什么

### 3.1 `calls.jsonl`（runner.py:300-315）

每次调用一行：

```json
{"topology", "workflow_id", "call_idx", "agent_id", "parent_idx",
 "prompt_token_ids", "output_token_ids", "num_cached_tokens",
 "finish_reason", "t_start", "t_end", "meta"}
```

- **`num_cached_tokens`** 是引擎自己报的前缀命中数
  （`vllm/outputs.py:105`：*"The number of tokens with prefix cache hit"*），
  是 sanity #4 的对账基准——**它是引擎对自身状态的权威说法**。
- **`prompt_token_ids` 存完整的**，不是长度。这样离线可以重建任何东西，
  也是 Tier 2 结构不变量校验的输入。
- **`t_start` / `t_end`** 是我加的（计划没要求），用于把事件流对齐到具体调用。

### 3.2 另外两条

- KV 事件流 → `blocks.jsonl`，见 [`events.md`](events.md)
- 显存张量 → `dumps/`，见 [`dump.md`](dump.md)

---

## 4. 清场时机（runner.py:251, 267）

```python
llm.reset_prefix_cache()              # 一组配置开始前
for task in tasks:
    llm.reset_prefix_cache()          # 每个 workflow 开始前
    if dumper: dumper.start_workflow()
```

**为什么每个 workflow 都清**：实测单 workflow 工作集 8–13.5 万 token，
10 道题累计约 100 万，L40S 的 22 万、A5000 的 9.3 万**都装不下**。
所以充足档的判据从"整组 10 题不驱逐"改成"**单 workflow 内不驱逐**"，
计划 §6.2 已按此修订。

**这不损失跨 workflow 的重复信号**：我们的判定是对 **dump 出来的块做离线比对**，
不依赖 vLLM 运行时是否合并。清场后同内容在两个 workflow 里是两块独立的物理块，
**正是要数的重复对**。真正损失的只有跨 workflow 的**命中率**这一个量。

---

## 5. `working_set_tokens`（runner.py:173）

按 `AllBlocksCleared` 把事件流分段（一段 = 一个 workflow），段内**按 block hash 去重**
求 token 数，返回 `per_workflow` / `max` / `mean` / `num_blocks_removed`。

**为什么按 hash 去重**：同一 hash 出现多次说明是同一个逻辑块（驱逐后重算存回），
不去重会高估工作集。

`manifest.ample_criterion_met` = 充足档且 `num_blocks_removed == 0`。
**每组跑完必须查这一项**——为 false 说明有 workflow 撑破了池子。

---

## 6. 审查点

1. **`_tokenize` 的兼容分支**是运行时判断（`hasattr`），不是版本判断。
   换 transformers 版本时行为会跟着变，但不会静默出错（两条路都返回 list[int]）。
2. **`enforce_eager=True` 让推理慢约一倍**。这是计划要求的，但它也意味着
   我们测的数值路径与开了 CUDA graph 的生产环境可能不同。
3. **`head_size`、采样层**等常量散落在 `dump.py` / `runner.py`，没有统一从模型配置读。
4. **`build_llm` 用 `os.environ` 设两个 vLLM 开关**。这是进程级副作用，
   在一个进程里跑多组配置时不会互相干扰（值一样），但不够干净。
   目前每组配置都是独立子进程，所以没问题。
