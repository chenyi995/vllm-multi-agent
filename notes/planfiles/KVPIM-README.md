# Multi-Agent KV Cache Trace Study

测量不同 multi-agent 拓扑下的 KV cache 行为，重点是 **de-RoPE 前后的跨 agent token 相似度** 与 **驱逐 (eviction) 的代价**。

---

## 0. 三层架构（先明确边界）

```
Layer 3  编排 orchestration     纯 Python 控制流，决定拓扑。不碰 GPU。
             ↓ prompt_token_ids
Layer 2  推理 serving           vLLM / SGLang 单实例。看到的是无状态请求流。
             ↓
Layer 1  缓存 KV cache          RadixTree + BlockPool。共享与驱逐发生在这里。
```

**关键**：agent 不是进程，是「一段 system prompt + 循环里的一个变量」。vLLM 不知道拓扑的存在。

---

## 1. 九类拓扑

| # | 拓扑 | 代表论文 | 参考实现 |
|---|---|---|---|
| T1 | **Sequential / Pipeline** | MetaGPT (Hong et al., ICLR'24)；ChatDev (Qian et al., ACL'24) | MetaGPT、CrewAI、LangGraph |
| T2 | **Supervisor / Orchestrator-Worker** | AutoGen (Wu et al., 2023)；Anthropic *Multi-Agent Research System* (eng blog, 2025) | LangGraph `supervisor`、AG2 |
| T3 | **Fan-out / MoA** | *Mixture-of-Agents* (Wang et al., 2024)；*More Agents Is All You Need* (Li et al., 2024) | MoA repo、LangGraph `Send` |
| T4 | **Debate** | Du et al., ICML'24；Liang et al., EMNLP'24 (*Encouraging Divergent Thinking*) | 原 repo、AgentVerse |
| T5 | **Group Chat** | AutoGen GroupChat (Wu et al., 2023) | AutoGen/AG2、AgentScope |
| T6 | **Handoff / Swarm** | OpenAI Swarm 设计文档 | OpenAI Agents SDK、LangGraph network |
| T7 | **Reflection / Generator-Critic** | Reflexion (Shinn et al., NeurIPS'23)；Self-Refine (Madaan et al., NeurIPS'23)；CRITIC (Gou et al., ICLR'24) | 各框架内建 |
| T8 | **Role-play / Simulation** | CAMEL (Li et al., NeurIPS'23)；Generative Agents (Park et al., UIST'23)；AgentVerse (Chen et al., ICLR'24) | CAMEL/OWL、Generative Agents |
| T9 | **Tree Search** | Tree of Thoughts (Yao et al., NeurIPS'23)；LATS (Zhou et al., ICML'24) | 多为自研 |

### 系统侧背景阅读

- **缓存机制**：PagedAttention/vLLM (SOSP'23)、RadixAttention/SGLang
- **PIC 位置无关缓存**：PromptCache (MLSys'24)、CacheBlend (EuroSys'25)、EPIC、KVLink
- **agentic 调度与驱逐**：KVFlow (NeurIPS'25)、Parrot (OSDI'24)、Preble、TokenCake、TokenDance
- **跨模型复用**：DroidSpeak
- **真实 trace**：TraceLab / SyFI

---

## 2. 编排（统一接口）

所有拓扑产出同一种 `Call` 流，便于横向比较：

```python
@dataclass
class Call:
    topology: str          # T1..T9
    workflow_id: str
    call_idx: int
    agent_id: str          # 角色名，等价于 system prompt 身份
    parent_idx: int | None # DAG 边，用于重建拓扑
    messages: list[dict]
```

**铁律**：driver 自己 tokenize，传 `prompt_token_ids` 而非字符串。否则你不知道 vLLM 内部切出的 token 序列，位置对不上，de-RoPE 全错。

```python
ids = tok.apply_chat_template(call.messages, add_generation_prompt=True, tokenize=True)
out = llm.generate(prompt_token_ids=[ids], sampling_params=sp)
```

### T1 Sequential
```python
ctx = TASK
for role in ["架构师", "开发", "测试", "评审"]:
    yield Call(agent_id=role, parent_idx=prev,
               messages=[SYS[role], {"role":"user","content":ctx}])
    ctx += "\n" + reply          # 单调增长，但 system prompt 每步都换
```

### T2 Supervisor
```python
plan = ask("supervisor", TASK)                    # orchestrator context 单调增长
subtasks = parse(plan)
for st in subtasks:                               # workers 共享任务描述前缀
    yield Call(agent_id=f"worker_{st.id}", parent_idx=sup_idx,
               messages=[SYS["worker"], {"role":"user","content":CTX + st.text}])
yield Call(agent_id="supervisor", messages=sup_history + worker_results)
```

### T3 Fan-out / MoA
```python
shared = [SYS["proposer"], {"role":"user","content":TASK}]   # 完全相同的长前缀
for i in range(N):
    yield Call(agent_id=f"proposer_{i}", messages=shared, parent_idx=None)
yield Call(agent_id="aggregator", parent_idx=None,          # N 份输出拼接，无前缀可用
           messages=[SYS["agg"], {"role":"user","content":TASK + join(all_replies)}])
```

### T4 Debate  ← 位置错配主战场
```python
transcript = {a: [] for a in AGENTS}
for r in range(ROUNDS):
    speeches = {}
    for a in AGENTS:
        others = [s for b, s in last_round.items() if b != a]   # 顺序因 a 而异
        yield Call(agent_id=a, parent_idx=prev_round_idx,
                   messages=[SYS[a], {"role":"user","content":Q + join(others)}])
        speeches[a] = reply
    last_round = speeches
```

### T5 Group Chat
```python
transcript = []
for turn in range(T):
    speaker = select_speaker(transcript)         # 可以是 LLM 调用，也计入 trace
    yield Call(agent_id=speaker, parent_idx=turn-1,
               messages=[SYS[speaker], *transcript])   # 共享 transcript，但 SYS 在最前
```
> 优化提示：把公共 transcript 前置、SYS 后置，能把不可复用的分叉点往后推。**主实验保持"SYS 在前"的默认写法**，这个变体单独做一组对照。

### T6 Handoff / Swarm
```python
cur, ctx = "triage", [{"role":"user","content":TASK}]
while cur != "done":
    yield Call(agent_id=cur, messages=[SYS[cur], *ctx])
    nxt, ctx = handoff(reply, ctx)      # ctx 可能被裁剪/重写 → 前缀被截断
    cur = nxt
```

### T7 Reflection
```python
draft = None
for it in range(K):
    yield Call(agent_id="generator", parent_idx=crit_idx,
               messages=[SYS["gen"], *history])        # 单调 append
    yield Call(agent_id="critic", parent_idx=gen_idx,
               messages=[SYS["critic"], *history])     # radix tree 上的兄弟分支
```

### T8 Simulation
```python
for step in range(STEPS):
    for a in POP:                                      # POP 可达数十
        mem = retrieve(a, k=8)                         # RAG 式 chunk 拼接
        yield Call(agent_id=a, messages=[SYS[a], {"role":"user","content":join(mem)+OBS}])
```

### T9 Tree Search
```python
frontier = [root]
while frontier and budget:
    node = select(frontier)
    for _ in range(BRANCH):                            # 同一路径前缀 fan-out
        yield Call(agent_id="expander", parent_idx=node.idx,
                   messages=[SYS["exp"], *node.path])
    yield Call(agent_id="evaluator", parent_idx=node.idx, messages=[SYS["eval"], ...])
```

---

## 3. 谁自带驱逐，谁没有

### 3.1 两种「驱逐」不要混淆

| | 发生在哪 | 单位 | 后果 |
|---|---|---|---|
| **KV block eviction** | 推理层 (vLLM/SGLang) | KV block | prompt 不变，KV 要重算。**语义无损** |
| **Context trimming / summarization** | 编排层 (LangGraph/AutoGen/CrewAI) | message | prompt **内容变了**，前缀断裂，下游全部 miss。**语义有损** |

⚠️ 编排层的 trimming 会**从断点开始废掉整条前缀**，破坏力远大于 block eviction。做实验时**必须关掉**所有框架自带的 memory/summarization，否则你测的是 trimming 不是 eviction。

### 3.2 推理层：有内置驱逐

| 系统 | 前缀复用 | 驱逐策略 | 关键开关 |
|---|---|---|---|
| **vLLM V1** | RadixTree + BlockPool，默认开 | free-block queue 上的 **LRU** | `--enable-prefix-caching`（V1 默认 on）、`--num-gpu-blocks-override`、`/reset_prefix_cache` |
| **SGLang** | RadixAttention | radix tree 上的 **LRU**（叶节点优先） | `--enable-cache-report`、`--max-total-tokens` |
| **LMCache** | 多级 GPU/CPU/Disk | 各层独立驱逐 | `chunk_size`、`max_local_cpu_size` |
| **TensorRT-LLM** | block reuse | LRU | `enable_block_reuse`、`tokens_per_block` |

### 3.3 无驱逐（不适合本实验）

| 系统 | 为什么不行 |
|---|---|
| **HF Transformers `DynamicCache`** | 无跨请求共享、无驱逐，cache 随对象生死 |
| **Ollama / llama.cpp** | 只有单 slot 的前缀复用，无全局 radix tree |
| **多实例 vLLM 负载均衡** | radix tree 是 per-engine 的，**跨实例零共享**，测出的命中率是假的 |
| **编排框架本身** | LangGraph / AutoGen / CrewAI 完全不管 KV，它们只管 message |

### 3.4 各拓扑的驱逐压力

驱逐痛不痛，取决于 **重激活间隔 (reuse distance)**：

| 拓扑 | reuse distance | 驱逐压力 | 主要损失形态 |
|---|---|---|---|
| T3 Fan-out | 极短（并发共用同一前缀） | **低** | 几乎不驱逐 |
| T7 Reflection | 短（两分支交替） | 低 | — |
| T2 Supervisor | orchestrator 短、worker 长 | 中 | orchestrator 应 pin 住 |
| T1 Sequential | **长**（转一圈才回来） | **高** | KVFlow 型过早驱逐 |
| T5 Group Chat | 长且不可预测（speaker 动态） | **高** | 同上 + SYS 前置分叉 |
| T4 Debate | 长 | **高** | **主要损失不是驱逐，是位置错配** |
| T6 Handoff | 不定 | 中高 | 裁剪导致前缀断裂 |
| T8 Simulation | 极长（agent 数多） | **最高** | chunk 级复用需求 |
| T9 Tree Search | 分支间短、回溯时长 | 中 | 回溯路径被踢 |

> T4/T5 的损失要**拆开归因**：多少来自驱逐（可用更好的策略救），多少来自位置错配（只能靠 PIC 救）。这是核心分析。

---

## 4. Block / Page 大小

### 4.1 各系统默认值

| 系统 | 参数 | 默认 | 含义 |
|---|---|---|---|
| **vLLM** | `--block-size` | **16 tokens** | KV block = hash 粒度 = 复用最小单位；可选 1/8/16/32/64/128 |
| **SGLang** | `--page-size` | **1 token** | radix tree 到 token 级，匹配最精细 |
| **LMCache** | `chunk_size` | **256 tokens** | 存储/传输 chunk（demo 配置常写 8，生产用 256） |
| **TensorRT-LLM** | `tokens_per_block` | **32 tokens** | |
| **CacheBlend / PromptCache** | — | message / 文档边界 | 语义 chunk，非固定长度 |

### 4.2 block size 的影响

- **越大**：元数据少、kernel 效率高，但**尾部不满一个 block 的 token 无法缓存**（平均浪费 `block_size/2` 个 token 的复用机会），且前缀匹配变粗。
- **越小**：匹配精细、命中率高，但 block table 变长、寻址开销上升。
- 对 **T4 Debate**：block size 基本不影响结论——位置一错，多大的 block 都 miss。
- 对 **T1/T5**：block size 会影响边界对齐，值得单独扫。

**实验建议**：主实验固定 `block_size=16`；另做一组 `{16, 32, 64}` 的敏感性分析。

### 4.3 KV 显存换算

```
KV bytes/token = 2(K,V) × n_layers × n_kv_heads × head_dim × dtype_bytes
```

| 模型 (BF16) | layers | kv_heads | head_dim | KV/token | 权重 | 32GB 卡上可缓存 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 8 | 128 | 112 KB | 1.2 GB | ~24 万 token |
| Qwen3-1.7B | 28 | 8 | 128 | 112 KB | 3.4 GB | ~22 万 token |
| **Qwen3-4B** | 36 | 8 | 128 | **144 KB** | 8 GB | **~14 万 token** |
| Qwen3-8B | 36 | 8 | 128 | 144 KB | 16.4 GB | ~8.5 万 token |

一个 block (16 tokens) 在 Qwen3-4B 上 = **2.25 MB**。

> 注意 GQA 让 KV 大小主要由 `n_kv_heads`（不是总参数量）决定，所以 1.7B 和 4B 的 KV/token 差别很小。**换小模型省的是权重，不是 KV。**

### 4.4 Dump 存储预算

全 36 层 × 10 万 token = **14.4 GB**，不可交互扫描。

**采样 5 层**（0 / 8 / 17 / 26 / 35，浅中深覆盖）× 10 万 token = **2 GB** ✅

层深本身就是核心自变量（上下文污染逐层累积），所以采样不是妥协，是设计。

---

## 5. 实验矩阵

```
A. 相似度实验（dump KV 张量）
   9 拓扑 × 10 workflow × 5 采样层        → ~2 GB/拓扑
   固定容量（充足，不触发驱逐，避免污染变量）

B. 驱逐实验（只采事件 + timing，不 dump 张量）
   9 拓扑 × 5 容量档 × 30 workflow
   --num-gpu-blocks-override: 2k / 4k / 8k / 16k / 32k

C. 重算保真度（受控小实验）
   单请求 → dump → 人为挤出 → 重发 → dump → 比对相对误差
```

**固定量**：单 vLLM 实例、单模型、`temperature=0`、`--kv-cache-dtype auto`（**绝不开 fp8**）、锁定 vLLM 版本。

**每换一个拓扑必须 `/reset_prefix_cache`**，否则上一轮残留 KV 污染命中率。

---

## 6. Sanity Check 清单

按顺序做，不通过就停：

1. `derope(k_post, pos) ≈ k_pre`，max abs err < 1e-2 ← **最关键**，验证 RoPE 逆变换（注意 HF/vLLM 用 NeoX 风格 `rotate_half`，配对是 `(j, j+d/2)`）
2. G1 组（同前缀同位置）`cos ≈ 1.0` ← 验证 token 对齐无 bug
3. `blocks.jsonl` 中 `BlockRemoved` 非空 ← 否则 cache 太大，容量再降一个数量级
4. `cached_tokens` 与 radix tree 重建结果一致 ← 验证事件流没丢
5. T3 Fan-out 的 `prefix_hit_ratio` 应接近上界 ← 拓扑实现正确性的反向验证
