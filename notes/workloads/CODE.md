# 代码逐文件说明（供审阅）

> 目的：让你能逐行审我的代码。
> 每一个"为什么这么写"都给出**依据**——要么是 vLLM 源码的具体位置，要么是实测。
> 术语见 [`README.md`](README.md)。

---

## 0. 数据流全景

```
①  抽题              workloads/t*_.py::load_tasks     10 道题，seed=0 固定
        ↓ dict
②  编排              workloads/t*_.py::build          生成器，yield 一个 Call，接住回复
        ↓ Call(messages=[{role, content}, ...])
③  分词              runner.py::_tokenize             聊天模板 → list[int]
        ↓ prompt_token_ids
④  推理              runner.py::run → llm.generate    vLLM 单实例，同进程
        ↓ 三条输出
    ├─ RequestOutput          → calls.jsonl    （回复、命中数、时刻）
    ├─ zmq KV 事件流           → blocks.jsonl   （events.py 后台线程接收）
    └─ gpu_model_runner.kv_caches → dumps/     （dump.py 直接读显存张量）
        ↓
⑤  离线分析          analyze.py                       校验 + 计数 + README
        ↓
    counts.parquet / sanity.log / 每组 README.md
```

**关键**：④ 里三条输出是**同一次运行**的三个侧面。calls 是我们自己记的，
events 是引擎主动播报的，dumps 是我们从显存里拷的。三者互相印证——
`sanity #4` 就是拿 events 重建的树去对 calls 里引擎报的命中数。

---

## 1. `kvpim/call.py` —— 统一调用单元

只有一个 dataclass，17 行。

```python
@dataclass
class Call:
    agent_id: str          # 角色名 = 系统提示词的身份
    messages: list[dict]   # 聊天消息，由 runner 分词
    parent_idx: int | None # DAG 边：本次调用消费了哪次的输出
    topology: str = ""     # 以下三项由 runner 填
    workflow_id: str = ""
    call_idx: int = -1
    meta: dict = ...       # 拓扑自定义的标注（层号、轮次、重试次数……）
```

**依据**：计划 §3 定义的统一接口。九个拓扑必须产出同一种流，否则横向比较无从谈起。

**审查点**：`parent_idx` 只能表达一条边。T3/T4 这类 fan-in（一个 agent 消费上一层
全部输出）表达不了，所以我把完整的父列表放在 `meta["parents"]` 里。
这是我加的，不是计划规定的。

---

## 2. `kvpim/workloads/t*_.py` —— 九个拓扑的编排

每个文件两个函数：

```python
def build(task: dict) -> Iterator[Call]:   # 生成器：yield Call，用 send() 接住回复
def load_tasks(num_tasks=10, seed=0) -> list[dict]
```

**生成器协议**（`runner.py:270-282`）：

```python
gen = workload(task)
reply = None
while True:
    call = gen.send(reply) if call_idx else next(gen)   # 拿下一个 Call
    ...跑 LLM...
    reply = completion.text                             # 回复喂回生成器
```

**为什么用生成器而不是回调**：拓扑逻辑（"下一步问谁"）与执行（"怎么调 vLLM"）解耦，
且拓扑代码读起来就是计划 §3 的伪代码原样。

**提示词的来源与校验**：

| 拓扑 | 提示词 | 校验函数 |
|---|---|---|
| T1 | `ref/MetaGPT/metagpt/roles/role.py:51-52` 的模板 + 各角色 goal/constraints | `analyze.check_metagpt_reference` |
| T3 | `ref/MoA/utils.py::inject_references_to_messages` | `analyze.check_moa_reference` |
| T4 | `ref/llm_multiagent_debate/gsm/gen_gsm.py` | 未写 |
| T5 | AutoGen **v0.2.34** `groupchat.py` 的两个模板 | 未写 |
| T6 | `ref/swarm` + `ref/tau-bench`，**运行时读取** | 结构上等价 |
| T8 | `ref/camel/camel/prompts/ai_society.py` | `analyze.check_camel_reference` |
| T9 | `ref/tree-of-thought-llm/src/tot/prompts/game24.py` | `analyze.check_tot_reference` |
| T2 / T7 | 无原文可对照（自拟，计划授权） | — |

**审查点**：T2/T7 的提示词是我写的。T1 的角色提示词虽然逐字节对齐，
但**产出格式**与真实 MetaGPT 不同（它用 Action 类产结构化文档，我这里是自由文本）。

---

## 3. `kvpim/runner.py` —— 喂给 vLLM 什么、从 vLLM 取什么

### 3.1 输入侧

**`_tokenize`（runner.py:109）** —— 唯一的分词入口：

```python
encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
return list(ids)
```

**为什么 driver 自己分词、传 token id 而不是字符串**（计划 §3「铁律」）：
只有自己分词，才知道每个 token 的**绝对位置**。dump 出来的 K 要做 de-RoPE，
必须知道位置；位置错一位，逆旋转全错，而且**不会报错**。

**为什么要 `hasattr(encoded, "keys")` 这个分支**：实测 transformers 5.14.1 的
`apply_chat_template(tokenize=True)` 返回 `BatchEncoding` 而不是 list。
直接当 list 用会把字符串当 token 传进去。计划成文时的写法在本版本上是错的。

**`llm.generate`（runner.py:286）**：

```python
output = llm.generate(TokensPrompt(prompt_token_ids=prompt_token_ids), sampling, use_tqdm=False)[0]
```

**依据**：本 commit 的 `LLM.generate` 签名是 `generate(prompts: PromptType, ...)`，
旧的 `prompt_token_ids=[ids]` 关键字**已删除**。`TokensPrompt` 定义在
`vllm/inputs/llm.py:106`。

**采样参数（runner.py:229）**：

```python
SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=cfg.max_tokens)
```

**为什么显式写 `top_p`/`top_k`**：模型自带的 `generation_config.json` 里有
`temperature=0.7 / top_k=20 / top_p=0.8`，vLLM 默认会把它合并进来。
所以还在 `build_llm` 里设了 `generation_config="vllm"`（不加载模型默认值）。
依据：`vllm/config/model.py:312-317` 的文档串明说 `"vllm"` = 不加载、用 vLLM 中性默认。

### 3.2 引擎构建（`build_llm`，runner.py:127）

```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # driver 与 engine 同进程
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
LLM(model=..., enforce_eager=True, dtype="auto", kv_cache_dtype="auto",
    seed=..., max_model_len=..., generation_config="vllm",
    block_size=..., gpu_memory_utilization=0.95,
    enable_prefix_caching=True, kv_events_config=KVEventsConfig(...),
    [prefix_match_unit=...], [num_gpu_blocks_override=...])
```

逐条依据：

| 参数 | 为什么 |
|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 只有同进程才能直接读 `gpu_model_runner.kv_caches` 做 dump。计划 §7.2 |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | 装的 flashinfer 是 cu13 版，首次采样会 JIT 编译，集群 PATH 无 ninja 直接崩。贪心解码用不上 |
| `enforce_eager=True` | 关 CUDA graph。计划要求；也让 dump 时机可控 |
| `kv_cache_dtype="auto"` | 计划：**绝不 fp8**，否则量化会掩盖我们要测的数值差异 |
| `max_model_len` | 模型 262144 上下文，vLLM 要求池子装得下一条满长请求，不设就起不来 |
| `block_size` / `prefix_match_unit` | 4/8 档物理块跑不了（FA 要求 16 的倍数），改用 `block_size=16` + 细粒度匹配。计划 §6.1 |
| `num_gpu_blocks_override` | 受限档压容量。**注意**：设了它之后准入检查按压低后的池子算（`kv_cache_utils.py:2125-2145`）|

### 3.3 输出侧：三条

**① `RequestOutput` → `calls.jsonl`（runner.py:300-315）**

每次调用记录：`agent_id`、`parent_idx`、**完整** `prompt_token_ids`、
`output_token_ids`、`num_cached_tokens`、`finish_reason`、`t_start`/`t_end`、`meta`。

- `num_cached_tokens` 是**引擎自己报的前缀命中 token 数**（`vllm/outputs.py:105`），
  是 sanity #4 的对账基准。
- `t_start`/`t_end` 是我加的，用于离线把事件流按时间对齐到具体调用。

**② KV 事件流 → `blocks.jsonl`**（见 §4）

**③ 显存张量 → `dumps/`**（见 §5）

### 3.4 清场时机（runner.py:251, 267）

```python
llm.reset_prefix_cache()          # 一组配置开始前
for task in tasks:
    llm.reset_prefix_cache()      # 每个 workflow 开始前
    dumper.start_workflow()
```

**为什么每个 workflow 都清**：实测单 workflow 工作集 8–13.5 万 token，10 道题累计
约 100 万，任何卡都装不下。清场后充足档的判据变成"单 workflow 内不驱逐"。
计划 §6.2 已按此修订。

**审查点**：这**不损失**跨 workflow 的重复信号，因为我们的判定是对 **dump 出来的
block 做离线比对**，不依赖 vLLM 运行时是否合并。清场后同内容在两个 workflow 里
是两块独立的物理 block，正是要数的重复对。真正损失的只有跨 workflow 的**命中率**。

### 3.5 `working_set_tokens`（runner.py:173）

按 `AllBlocksCleared` 把事件流分段 → 每段就是一个 workflow → 段内按 **block hash 去重**
求 token 数。输出 `per_workflow` / `max` / `mean` / `num_blocks_removed`。

**为什么按 hash 去重**：同一个 hash 出现多次说明是同一个逻辑块（被驱逐后重算存回），
不是新块。不去重会高估工作集。

---

## 4. `kvpim/events.py` —— 引擎播报的 KV 事件

后台线程跑一个 zmq SUB socket，收 `KVEventBatch`（msgspec 编码），逐条写 JSONL。

```python
sub.connect("tcp://localhost:<port>"); sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")
_, seq_bytes, payload = sub.recv_multipart()
batch = Decoder(type=KVEventBatch).decode(payload)
```

**依据**：`vllm/distributed/kv_events.py:464` 的
`self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))` —— 三帧格式。
订阅样例见 `examples/features/kv_events/kv_events_subscriber.py`。

**补洞**：PUB/SUB 会丢消息。用序列号检测缺口，从 replay socket 补回来
（`_replay_gap`）。`manifest.num_missed_events` 记录最终缺口数，**目前所有组都是 0**。

**三种事件的语义**：

| 事件 | 含义 |
|---|---|
| `BlockStored` | **新增**缓存的块（带 `block_hashes` / `parent_block_hash` / `token_ids` / `block_size`）|
| `BlockRemoved` | 驱逐 |
| `AllBlocksCleared` | `reset_prefix_cache()` |

⚠️ **关键依据**：`BlockStored` 默认**只代表新增**。复用也发事件的路径在
`vllm/v1/core/kv_cache_manager.py:266-272`，条件是
`request.kv_cache_report_mode == "full"`，而默认是 `"incremental"`。
**改了这个默认值会让 N_total 静默高估一倍以上。**

---

## 5. `kvpim/dump.py` —— 从显存里把 K/V 拷出来

### 5.1 怎么摸到张量

```python
worker = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker
runner = getattr(worker, "worker", worker).model_runner      # GPUModelRunner
runner.kv_caches[layer]                                       # 该层的 KV 张量
```

**依据**：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 时 `EngineCoreClient.make_client`
返回 `InprocClient`（`vllm/v1/engine/core_client.py:306-317`），engine 就在本进程内。
`kv_caches` 定义在 `gpu_model_runner.py:562`。这条链路我用探针脚本实测验证过。

### 5.2 K 和 V 怎么切开（最容易搞错的一处）

```python
def split_kv(layer_tensor, head_size):
    return layer_tensor.transpose(1, 2).split(head_size, dim=-1)
```

**依据**：`vllm/v1/attention/backends/flash_attn.py:1114`，kernel 自己就是这么取的：

```python
key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
```

即逻辑形状 `(块, head, token, 2×128)` → transpose → `(块, token, head, 2×128)`
→ 末维前 128 是 **K**、后 128 是 **V**。

**验证**：sanity 脚本在 rotary 模块挂钩子抓到真实的 post-RoPE K，与 dump 出来的
缓存内容逐 token 比对，**74 个 token 全部 max abs diff = 0.000e+00**。
布局、K/V 顺序、position 映射三件事一次性验证通过。

### 5.3 position 怎么来（第二个容易错的地方）

```python
size = key_cache.shape[1]              # 物理块大小
end   = block.block_hash_num_tokens    # 该 hash 覆盖到的前缀长度（不含）
start = ((end - 1) // size) * size
block["positions"] = list(range(start, end))
```

**依据**：`KVCacheBlock._block_hash_num_tokens`（`kv_cache_utils.py:130`）的注释是
"Number of prefix tokens covered by `_block_hash`. For full blocks this is the full
block boundary; **partial entries can end inside a cache block**."

**为什么不能用 `end - size` 反推**：`prefix_match_unit=4/8` 时 hash 可以在物理块
内部结束，用 `end - 16` 会让整块 position 前移，**de-RoPE 全错且不报错**。
这是我写这个文件时真犯过的 bug，改成按块边界向下取整后修复，并在 4 档上验证过。

### 5.4 去重键

```python
fresh = [b for b in cached_blocks(pool) if b["block_hash"] not in self.seen]
```

**按 hash 而不是 block_id**：块被驱逐后重算存回会拿到新的 `block_id`，
按 id 去重会把同一个逻辑块 dump 两次、`N_total` 重复计数。
`seen` 在每个 workflow 开始时清空（`start_workflow`）——跨 workflow 的同内容块
是**真正独立的物理块**，正是要数的重复对。

### 5.5 为什么只 dump 5 层

运行时缓存里是**全部 36 层**（引擎一行没改）。dump 是只读旁路，只拷 0/8/17/26/35。
全存要多 7.2 倍硬盘（444 GB → 3.2 TB）。

⚠️ **这带来一个上界偏差**：真要去重两个块，36 层都得相同；只查 5 层可能放过
在未采样层其实不同的块对。所以"5 层全判同"是"36 层全判同"的**上界**。
宸逸决定不做 36 层对照，该偏差如实记录。

---

## 6. `kvpim/derope.py` —— 位置编码的逆变换

```python
# 正变换（vLLM 做的）        逆变换（我们做的）
o1 = x1*cos - x2*sin         x1 = o1*cos + o2*sin
o2 = x2*cos + x1*sin         x2 = o2*cos - o1*sin
```

**依据**：`vllm/model_executor/layers/rotary_embedding/common.py:145-185`
的 `ApplyRotaryEmb.forward_static`，neox 分支就是上面左边两行，
向量按 `torch.chunk(x, 2, dim=-1)` 前后半分。

`cos/sin` 的构造照抄 `base.py:80-102` 的 `_compute_cos_sin_cache`：
`inv_freq = 1/base^(2i/d)`，`freqs = outer(位置, inv_freq)`。
Qwen3-4B 的 `rope_theta = 5,000,000`（读自 config.json），`rope_scaling = null`。

**`cache_dtype=torch.bfloat16` 这个参数**：vLLM 把 cos/sin 表算完之后
**存成模型 dtype**（`base.py:61` 的 `cache = cache.to(dtype)`）。
我们也走一遍 bf16 舍入，逆变换才是它实际所做的那个变换的逆。

**验证（sanity #1）**：钩子抓真实的 pre-RoPE K 与 post-RoPE K，
`cos(derope(k_post), k_pre) = 0.9999999692`，逐 token 最小 0.9999999254。

**审查点**：原判据"max abs err < 1e-2"实测不可达——K 有 attention-sink 式离群值，
`|k_post|` 最大到 314 而均值 2.35，bf16 存储下 1e-2 的绝对阈值低于 vLLM 自身噪声
（用我的正变换对 vLLM 的 k_post，误差同为 6.2e-2 量级）。已改成 cosine 判据。

---

## 7. `kvpim/analyze.py` —— 离线校验与计数

### 7.1 `RadixTree` + `reconcile_events`（sanity #4）

从 `blocks.jsonl` 重建缓存前缀树，预测每次调用应该命中多少 token，
与引擎报的 `num_cached_tokens` 对账。

**这个函数我修过四次，每次都是工具错不是数据错**，都记在这里供你判断：

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | T3 有 7 处不符 | 树没处理 `AllBlocksCleared` / `BlockRemoved` | 补上 |
| 2 | T7 有 9 处，全是各 workflow 第 0 次调用 | 清场事件由**发布线程异步打时间戳**，可能晚于 `t_start`，按时间过滤清不干净 | 改成按 `AllBlocksCleared` **分段**，每个 workflow 只看自己那段 |
| 3 | T1 b16 受限档 2 处，仍是第 0 次调用 | 上个 workflow 的事件因发布延迟落进下一段开头 | 段内再丢掉时间戳早于该 workflow 首次调用的事件 |
| 4 | 7 组 T4，gap 恰好 16 | 引擎永远留**最后一个物理块**重算，我按 `prefix_match_unit`（4/8 档是 4）写容差太小 | 容差改成物理 `block_size` |

**防止"调到变绿"的反向测试**：故意删掉 5% 的 `BlockStored` 事件，
判据立即失败（7 处失配）。**判据仍然能失败。**

### 7.2 Tier 1 / Tier 2（sanity #5 的替代）

**Tier 1**：用 `ast` 从 `ref/` 里把论文自己的函数/常量**单独编译出来**
（那些文件顶层 import 了未安装的 openai/requests，不能整体导入），与我们的实现逐字节比。

**Tier 2**：把 `calls.jsonl` 里的 `prompt_token_ids` 解码回聊天轮次，
检查拓扑必须满足的结构关系（同层 user turn 相同、注入段相同、角色顺序、
context 单调增长、Ω=1 记忆上限、transcript 镜像、模板只填 input……）。

**这一条为什么重要**：vLLM 看不见拓扑，让 T1 成为 T1 的**只有编排代码**。
编排出错不会报错，会变成一个看起来可发表的结论。

**同样做了反向测试**：植入六个 bug（user turn 混入 agent 编号、references 顺序
per-agent 不同、persona 错配、aggregator 带 persona、layer 0 误注入、
Tier 1 编号改 0-based），**六个全被对应不变量抓到**。

### 7.3 `count_duplicates` —— 实际的测量

```python
# ① 按 token 内容分组（元数据，很轻）
groups[tuple(token_ids)].append(block_index)
# ② 只有 ≥2 成员的组才加载张量
for members in [g for g in groups.values() if len(g) > 1]:
    for layer in sample_layers:
        # ③ 三个视角各算一次 cosine 矩阵
        similarity = _cosine_matrix(vectors[view]); similarity.fill_diagonal_(-1)
        hit = similarity.max(dim=1).values >= tau[layer]
```

**为什么先按内容分组**：计划 §4 规定只有 **token 内容完全相同**的块才是去重候选
（不同内容的块不视为可去重对象，这样也避开了 ContextPilot 对"近似复用"的批评）。
副作用是极省算力——T7 那组 1,843 块里只有 696 块进入比对，3,020 对，**5.6 秒**跑完。

**τ 是逐层的**，来自 sanity #6 的重算噪声 P1 分位。

**三种口径都输出**：逐层、strict（5 个采样层全判同）、any（至少一层）。

---

## 8. `scratch/` 下的编排脚本（不入库）

| 文件 | 作用 |
|---|---|
| `run_config.py` | 跑一组配置：抽题 → `runner.run` → sanity #4/#5 → 写 `sanity.log` |
| `run_config.sbatch` | 单组配置的 Slurm 作业，带 1 小时看门狗 |
| `run_group.sbatch` | **一次分配跑多组**，中途不放卡；每组独立子进程 + 看门狗 |
| `humaneval_baseline.py` | T7 抽题用的 baseline pass（164 题真跑真判） |
| `probe_layout.py` / `sanity_12.py` / `sanity_6.py` | 探针与 sanity 实验 |

**受限档容量的自算逻辑**（`run_config.py`）：

```python
w_mean = ample["working_set"]["mean"]
longest_call = max(len(prompt) + len(output) for 充足档的每次调用)
floor_tokens = ceil(longest_call * 1.5 / block_size) * block_size
pool_tokens  = max(int(0.5 * w_mean), floor_tokens)
```

- **用 mean 不用 max**：T1 实测 W 从 3,872 到 23,520 波动，用 max 会让池子比
  9/10 个 workflow 的整个工作集还大，它们全程不受限。
- **为什么要下限**：T2 的 `0.5×mean = 1,024` 小于它最长的单次调用 2,922，
  请求装不进池子，vLLM **不报错、直接挂死**。
- **为什么下限是 1.5 倍不是 1.1 倍**：实测 T1 b32 充足档最长 4,681，
  但受限档实际跑出 5,008——驱逐后的重算噪声翻转了 greedy 选词，对话分叉，
  prompt 长过了充足档的最大值，撞上 5,152 的池子后挂死。

⚠️ 下限生效时 KVFlow 的 0.5 比例达不到，`manifest.extra` 里如实记录
`capacity_floor_applied` 与 `achieved_ratio_vs_mean`。
