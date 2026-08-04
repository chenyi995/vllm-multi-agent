# Block-wise KV 重复度实验计划（v2，全覆盖版，待审阅）

> 目标：在 9 类 multi-agent workload 中，统计**存储在 vLLM cache 里的全部 KV 中，有几个 block 是一样的**。
> 覆盖三个维度的全叉积：① RoPE 编码前后、② block size {4, 8, 16, 32, 64}、③ 容量两档（受限 / 充足）。
> 本文自包含（不依赖 KVPIM-README），全部决策已定。

---

## 1. 研究问题

- **Q1（RoPE）**：post-RoPE 的 K 里有多少 block 一样？de-RoPE 之后又有多少 block 变成一样？差值 = RoPE 位置错配吞掉的可去重 block 数。（V 不经过 RoPE，作为天然对照）
- **Q2（block size）**：粒度从 4 到 64 token，"一样的 block"的数量和占比怎么变？
- **Q3（容量）**：容量受限、驱逐发生后，被驱逐的 block 里有多少是"本来和别人一样"的？

---

## 2. 三层架构（边界）

```
Layer 3  编排 orchestration     纯 Python 控制流，决定拓扑。不碰 GPU。
             ↓ prompt_token_ids
Layer 2  推理 serving           vLLM 单实例。看到的是无状态请求流。
             ↓
Layer 1  缓存 KV cache          RadixTree + BlockPool。共享与驱逐发生在这里。
```

**关键**：agent 不是进程，是「一段 system prompt + 循环里的一个变量」。vLLM 不知道拓扑的存在。

---

## 3. 九类 workload：拓扑 × 代表论文 × 公开 benchmark

**benchmark 已定**（宸逸确认按提议列执行）。§12 阅读完成后，把每行的任务抽样方式与 agent 配置（轮数/人数）补进表格。

| # | 拓扑 | 代表论文 | 公开 benchmark（提议） |
|---|---|---|---|
| T1 | Sequential / Pipeline | MetaGPT (ICLR'24)；ChatDev (ACL'24) | **HumanEval**（MetaGPT 评测集）；备选 ChatDev 的 SRDD |
| T2 | Supervisor / Orchestrator-Worker | AutoGen (2023)；Anthropic Multi-Agent Research System | **GAIA**（supervisor 框架通用评测）；备选 MATH |
| T3 | Fan-out / MoA | Mixture-of-Agents (2024)；More Agents Is All You Need (2024) | **AlpacaEval 2.0**（MoA 原文主评测） |
| T4 | Debate | Du et al. (ICML'24)；Liang et al. (EMNLP'24) | **GSM8K + MMLU 子集**（Du et al. 原文任务） |
| T5 | Group Chat | AutoGen GroupChat (2023) | **MATH 子集**（AutoGen 论文场景 A1 数学求解） |
| T6 | Handoff / Swarm | OpenAI Swarm 设计文档 | **τ-bench**（airline/retail 客服，标准 handoff 评测） |
| T7 | Reflection / Generator-Critic | Reflexion (NeurIPS'23)；Self-Refine (NeurIPS'23) | **HumanEval**（Reflexion 设置） |
| T8 | Role-play / Simulation | CAMEL (NeurIPS'23)；Generative Agents (UIST'23) | **CAMEL AI Society**（公开数据集）；备选 Smallville 复现片段 |
| T9 | Tree Search | Tree of Thoughts (NeurIPS'23)；LATS (ICML'24) | **Game of 24**（ToT 原文任务） |

### benchmark 实施配置（§12 读文后逐行填）

| # | benchmark 与题数 | agent 配置（论文口径） | 抽 10 题方式 | 获取方式 |
|---|---|---|---|---|
| T1 | HumanEval 全量 164 题（MetaGPT 报 Pass@1） | 5 角色瀑布流水线：PM→Architect→Project Manager→Engineer→QA；executable feedback ≤3 次重试；每题调用数论文未写（按角色估 5–8 次） | 固定种子从 164 题均匀抽 10，或按 canonical solution 行数分层（短/中/长） | HF `openai/openai_humaneval`；`ref/MetaGPT` |
| T2 | GAIA 共 466 题（L1:146/L2:245/L3:75），可用的是带答案的 validation 165 题 | benchmark 不规定 agent 结构，用本计划 §3 的 supervisor 编排；**注意**：无工具环境下 worker 靠参数化知识作答，答案准确率不影响 KV 统计口径 | L1:4 / L2:4 / L3:2，优先选无附件（`file_name` 为空）的题 | HF `gaia-benchmark/GAIA`（**gated，需先同意条款**） |
| T3 | AlpacaEval 2.0 共 805 条指令（MoA 原文主评测） | 原文 3 层 × 每层 6 proposer + aggregator（6 个不同开源模型）；**单模型适配**：Qwen3-4B 以 6 个不同 system prompt 扮演 proposer（与"agent = system prompt"一致）；aggregator 用原文 Aggregate-and-Synthesize prompt，按编号列表拼接各回答；可用 MoA-Lite（2 层）压工作集 | 固定种子抽 10，可按指令长度分层 | HF `tatsu-lab/alpaca_eval`；`ref/MoA`（含 prompt 模板） |
| T4 | GSM8K（Du et al. 用 100 题）+ MMLU 随机 100 题 | **3 agent × 2 轮**（第 1 轮独立作答，第 2 轮把其余所有 agent 的完整回答拼进 prompt，模板论文有逐字原文，在 `ref/llm_multiagent_debate`）；context 随轮数线性膨胀 | GSM8K test（1319 题）固定种子抽 10；要拉长 trace 可扩到 3 轮 | HF `openai/gsm8k`、`cais/mmlu` |
| T5 | MATH level-5，AutoGen A1 口径抽题（六类不含 geometry） | **修正**：AutoGen 的 A1 是 2-agent 不是 GroupChat；改用其 **A5 Dynamic GroupChat 结构**：3 agent（User_proxy/Solver/Checker）+ `max_round=12`，每轮先由 manager 用一次 LLM 调用（role-play prompt）选 speaker，消息 broadcast 全员共享增长历史 | level-5 六类各抽 1–2 题凑 10，固定种子 | HF `hendrycks/competition_math`；`ref/autogen`（0.2 分支 `GroupChat`） |
| T6 | τ-bench airline 域 50 个任务 | Swarm airline 5 agent（Triage/Modification/Cancel/Change/Lost Baggage）层级 handoff；**handoff 不裁剪历史，只换 system prompt**（README 原文），另有 context_variables 传递；τ-bench 配 **LLM user simulator**（其调用同样进 KV cache，计入 trace），"###STOP###" 终止 | repo 自带 `--task-ids` 直接指定 10 个，或先看 `historical_trajectories/` 的轮数分布选中等长度任务 | `ref/tau-bench`（任务在 `tau_bench/envs/airline/tasks_test.py`）、`ref/swarm`（`examples/airline/`） |
| T7 | HumanEval 164 题（Reflexion 报 pass@1） | 三组件循环：Actor（prompt = 签名 + 上版实现 + 单测结果 + 反思）→ Evaluator（LLM 生成单元测试并**实际执行**）→ Self-Reflection；memory 只留最近 Ω=1 条反思；官方 repo 默认 `max_iters=2`（1 次生成 + 1 轮反思重试），要拉长 trace 可调大 | **必须从 baseline 第一轮失败的题里抽**（否则只有 1 轮、无迭代 trace）：先跑一遍 pass@1，从失败集抽 10（或 8 失败 + 2 通过混合） | HF `openai/openai_humaneval`；`ref/reflexion`（自带 `humaneval-py.jsonl` 与 prompt） |
| T8 | CAMEL AI Society：50×50 role pair × 10 task = 25,000 段对话 | **Inception Prompting 双 agent**（AI User 只发 "Instruction:/Input:" 指令、AI Assistant 以 "Solution:" 作答），非 §3 伪代码的多 agent 模拟式（那是 Generative Agents 备选）；5 个终止条件，硬上限 **40 条 message**；工作集因此可控，之前对 T8 撑爆显存的担心解除 | 抽 10 个 (assistant_role, user_role, specified_task) 三元组、按 role pair 分层，用 `ref/camel` 重放生成 trace | HF `camel-ai/ai_society`（含 termination_reason/num_messages 字段可过滤）；`ref/camel` |
| T9 | Game of 24：4nums.com 共 1362 题，ToT 原文取 **901–1000 共 100 题** | BFS 深度 3、**breadth b=5**；propose prompt 单次列全部候选算式，value prompt 对每候选打 sure/likely/impossible；原文 value 采样 3 次 @ temperature 0.7，**我们 temperature=0 的适配：value 采 1 次**（3 次会完全相同） | 在 901–1000 内等距抽 10（905, 915, …, 995），同时覆盖早停与满树两种 KV 行为，不要只抽最难尾部 | `ref/tree-of-thought-llm`（`data/24/24.csv` 按行号切 901–1000） |

### 编排统一接口

所有拓扑产出同一种 `Call` 流：

```python
@dataclass
class Call:
    topology: str          # T1..T9
    workflow_id: str
    call_idx: int
    agent_id: str          # 角色名 = system prompt 身份
    parent_idx: int | None # DAG 边，用于重建拓扑
    messages: list[dict]
```

**铁律**：driver 自己 tokenize，传 `prompt_token_ids` 而非字符串，否则位置对不上、de-RoPE 全错：

```python
ids = tok.apply_chat_template(call.messages, add_generation_prompt=True, tokenize=True)
out = llm.generate(prompt_token_ids=[ids], sampling_params=sp)
```

### 各拓扑编排要点

```python
# T1 Sequential：ctx 单调增长，但 system prompt 每步都换
ctx = TASK
for role in ["架构师", "开发", "测试", "评审"]:
    yield Call(agent_id=role, messages=[SYS[role], user(ctx)]); ctx += reply

# T2 Supervisor：orchestrator context 单调增长；workers 共享任务描述前缀
plan = ask("supervisor", TASK)
for st in parse(plan):
    yield Call(agent_id=f"worker_{st.id}", messages=[SYS["worker"], user(CTX + st.text)])
yield Call(agent_id="supervisor", messages=sup_history + worker_results)

# T3 Fan-out/MoA：N 个 proposer 完全相同的长前缀；aggregator 无前缀可用
shared = [SYS["proposer"], user(TASK)]
for i in range(N): yield Call(agent_id=f"proposer_{i}", messages=shared)
yield Call(agent_id="aggregator", messages=[SYS["agg"], user(TASK + join(all_replies))])

# T4 Debate ← 位置错配主战场：每个 agent 看到的他人发言顺序不同
for r in range(ROUNDS):
    for a in AGENTS:
        others = [s for b, s in last_round.items() if b != a]
        yield Call(agent_id=a, messages=[SYS[a], user(Q + join(others))])

# T5 Group Chat：共享 transcript，但 SYS 在最前 → 每个 speaker 前缀在 SYS 处分叉
for turn in range(T):
    speaker = select_speaker(transcript)          # speaker 选择调用也计入 trace
    yield Call(agent_id=speaker, messages=[SYS[speaker], *transcript])

# T6 Handoff/Swarm：Swarm 实际语义 = 历史不裁剪、只换 SYS
#   → 前缀在最前面的 SYS 处断裂，且 SYS 长度不同使整条历史 token 位置平移（de-RoPE 相关）
cur, ctx = "triage", [user(TASK)]
while cur != "done":
    yield Call(agent_id=cur, messages=[SYS[cur], *ctx])
    cur, ctx = handoff(reply, ctx)

# T7 Reflection：generator/critic 是 radix tree 上的兄弟分支，单调 append
for it in range(K):
    yield Call(agent_id="generator", messages=[SYS["gen"], *history])
    yield Call(agent_id="critic",    messages=[SYS["critic"], *history])

# T8 Simulation：agent 数可达数十，RAG 式 chunk 拼接
for step in range(STEPS):
    for a in POP:
        yield Call(agent_id=a, messages=[SYS[a], user(join(retrieve(a, k=8)) + OBS)])

# T9 Tree Search：同一路径前缀 fan-out，回溯时重激活
while frontier and budget:
    node = select(frontier)
    for _ in range(BRANCH):
        yield Call(agent_id="expander", messages=[SYS["exp"], *node.path])
    yield Call(agent_id="evaluator", messages=[SYS["eval"], *node.path])
```

**必须关掉所有编排层 memory/summarization**（拓扑循环全部手写纯 Python，不用 LangGraph/AutoGen 等框架跑），否则测到的是编排层 trimming 不是缓存层行为——trimming 会从断点废掉整条前缀，破坏力远大于 block 驱逐。

---

## 4. 指标定义：一样的 block 计数

**不算 block 之间的相似度分布。指标是一个计数**：一次运行中存储过的全部物理 block 里，有几个 block 与至少一个其他 block **一样**（即理论上可去重），以及去重后能省几个 block。

### 判定"一样"

- **候选对生成**：`BlockStored` 事件带每个 block 的 `token_ids`（`vllm/distributed/kv_events.py:50`）。只有 token 内容完全相同的 block 才进入数值比对（不同内容的 block 不视为可去重对象）。
- **数值判定**：候选对逐 token 比 K（或 V）向量（cosine = 归一化内积，BF16 转 FP32 计算），block 内所有 token 的 cosine ≥ **τ** 判为一样。**τ 由校准得出，且逐层标定（已定，2026-08-03 实测后修订）**：取 sanity #6 重算保真度实验中"同内容同位置、重算后"block 对的 cosine 分布 **P1 分位数**，判定口径为 "identical up to recomputation noise"。

**τ 是每个采样层一个值，不是全局一个值。** 实测噪声底线逐层累积（第 N 层的输入是第 N−1 层带误差的输出）：

| 采样层 | 0 | 8 | 17 | 26 | 35 |
|---|---|---|---|---|---|
| τ（K_post，P1） | 0.99999 | 0.99998 | 0.97809 | 0.96400 | 0.99648 |

跨度近三个数量级。若用全局单值（该实验的全局 P1 = 0.98054）：在 layer 26 上**过严**——那层真正相同的 block 有 >1% 会掉到阈值下被误判为"不一样"，N_dup 系统性低估；在 layer 0 上**过松**——那层噪声只到 1e-5，0.98 等于不设防。计划本就要求按采样层分别计数，逐层 τ 与之一致。

**τ 必须在"部分前缀命中"条件下标定**（2026-08-03 实测的关键发现）：

| 重算条件 | 结果 |
|---|---|
| A 完全驱逐 + 整条重新 prefill | 1050 对中 930 对**逐位相同**，cosine 精确 1.0 |
| B 部分前缀命中、只重算后缀 | 1050 对中仅 144 对逐位相同，K_post P1 = 0.98054 |

条件 A 之所以逐位复现：重算时 batch 形状与首次完全相同 → GPU 归约顺序相同 → 浮点结果逐位一致。噪声不是硬件的固有属性，**是 batch 形状的属性**。前缀缓存的常态是部分命中（后缀以更短的 batch 重新 prefill，归约顺序改变），故 τ 只能取自条件 B；按条件 A 标定会得出"噪声为零、τ = 1.0"的假结论。

V 的重算噪声比 K 更大（条件 B 下 P1 = 0.96098，min 0.88306）。"V 是干净对照"仅对**位置无关性**成立，对**数值稳定性**不成立。

当前 τ 取自单条 prompt × 7 个命中边界（1050 对），为工作值；正式跑批后须在真实 trace 上重标。

**敏感性扫描点（2026-08-03 依实测修订）**：附录给 τ 与 **{0.99, 0.995, 0.999, 0.9999}** 的阈值敏感性，主结果只用 τ。
原写的 {0.90, 0.95} 已删除——实测**不同内容**的 block 对 cosine 就已经是 K_post 0.90–0.96 / K_derope 0.90–0.96
（V 仅 0.002–0.17），0.90/0.95 落在假阳性基线**之内**，用作阈值会把毫不相干的 block 判成"一样"。
配套要求：每组结果必须同时报告"不同内容 block 对"的 cosine 分布作为**假阳性基线**，
任何低于该基线 P99 的数值不得作为判定点。（该基线目前只有 5 block / 10 对的小样本，正式跑批后按真实数据复核。）
- **三个视角分别计数**：
  - `K_post`：cache 里原样的 post-RoPE K → 只有同内容**同位置**的 block 能一样
  - `K_derope`：逆旋转还原后的 K → 同内容**不同位置**的 block 也可能一样
  - `V`：不经过 RoPE，天然的位置无关对照
- **层的处理**：vLLM 中一个 block id 贯穿所有层。按采样层分别计数（层深是核心自变量，上下文污染逐层累积），同时给"全部采样层都一样"的严格口径。

### 每次运行的输出

```
N_total     存储过的物理 block 总数
N_dup[v]    v ∈ {K_post, K_derope, V}：与他人一样的 block 数（分层 + 严格口径）
去重节省率   1 - N_distinct / N_total
Q1 核心数   N_dup[K_derope] - N_dup[K_post]
```

**RoPE 前后不用跑两次**：cache 里存的就是 post-RoPE K，de-RoPE 是离线逆变换（NeoX 风格 `rotate_half`，配对 `(j, j+d/2)`），dump 时记下每 token 的 position 即可。Q1 是分析维度，运行成本 ×1。

### 判定口径的 sanity 锚点

- 同内容同位置的 block 对（vLLM 因 radix 分叉未合并、但前缀相同的场景）必须判为一样（接近逐位相等）→ 验证阈值不太严
- 随机抽取的不同内容 block 对必须判为不一样 → 验证阈值不太松

### 文献依据（2026-08-03 两轮核实，均有原文引句）

- **度量先例**：KV 相似用 cosine/角距离有先例——MiniCache（NeurIPS'24，arXiv:2405.14366）以角距离 + SLERP 做跨层 KV 合并；KVMerger（arXiv:2407.08454）用 **key 状态 cosine ≥ 0.75** 识别序列内相邻可合并 token（其发现 value **无**局部相似性；与我们无冲突——我们比的是跨 block 同内容对，不是序列内相邻 token）。
- **阈值先例**：没有任何论文把某个数值定义为"两段 KV 相同"。近似复用文献的阈值都远松于我们：KVMerger 0.75（合并）、SemShareKV（arXiv:2509.24832）LSH 0.8（跨 prompt 复用）、KVSharer（arXiv:2410.18517）输出表征 cosine 0.5。跨请求相似 block 的合并编码 Joint Encoding（arXiv:2601.03067，自适应 cosine 阈值）是与我们最接近的系统先例。
- **存储去重先例**：现有系统全是 token 精确匹配、恒等保证（vLLM APC / SGLang radix / KVCache in the Wild 的 SipHash 块哈希，arXiv:2506.02634），无数值判定。我们"内容相同 + 数值容差"的口径介于精确匹配与近似复用之间，是文献空档。
- **反面证据**：ContextPilot（arXiv:2511.03475）指出 KV 值相似度不是跨上下文可复用的可靠指标（近似匹配可损 9–11% 质量）。我们只在 token 内容**完全相同**的候选对内做数值判定，避开该批评。
- **de-RoPE 先例**：KVLink（arXiv:2502.16002）存储时剥离 RoPE、推理拼接时按全局位置重加旋转（该步无需训练）——`K_derope` 的直接方法学先例。
- **更正记录**：DroidSpeak（arXiv:2411.02820）选层依据是离线端到端质量画像（F1 的 Pareto 前沿），**不是**逐层 KV cosine，不作为 cosine 度量先例引用。
- **阈值口径（已定）**：主阈值 τ 不取约定值，由 sanity #6 重算噪声底线校准（P1 分位数），比文献所有阈值都严格；
  {0.99, 0.995, 0.999, 0.9999} 仅作敏感性扫描点（2026-08-03 依实测上调，见 §4）。
  文献里 KVMerger 0.75 / SemShareKV 0.8 / KVSharer 0.5 这些阈值，按本实验实测的假阳性基线（不同内容 K 对已达 0.90+）
  在我们的口径下全部不可用——这不是否定它们，而是因为它们做的是"近似复用"，我们做的是"是否同一"。

---

## 5. 驱逐背景（Q3 需要）

### 5.1 两种「驱逐」不要混淆

| | 发生在哪 | 单位 | 后果 |
|---|---|---|---|
| **KV block eviction** | 推理层 (vLLM) | KV block | prompt 不变，KV 要重算。语义无损 |
| **Context trimming** | 编排层框架 | message | prompt 内容变了，前缀断裂，下游全 miss。语义有损 |

本实验只测前者，后者已通过"手写编排、不用框架"排除。

### 5.2 vLLM 的驱逐机制

- RadixTree + BlockPool，前缀复用默认开（`--enable-prefix-caching`）
- 驱逐 = free-block queue 上的 **LRU**
- 容量控制：`--num-gpu-blocks-override`（`vllm/config/cache.py:87`）
- 换 workload 清场：`llm.reset_prefix_cache()`（`vllm/entrypoints/llm.py:792`）

### 5.3 各拓扑的驱逐压力（先验，供解读 Q3 结果）

| 拓扑 | reuse distance | 驱逐压力 |
|---|---|---|
| T3 Fan-out | 极短（并发共用前缀） | 低 |
| T7 Reflection | 短（两分支交替） | 低 |
| T2 Supervisor | orchestrator 短、worker 长 | 中 |
| T6 Handoff | 不定（SYS 更换致前缀断裂 + 历史位置平移） | 中高 |
| T9 Tree Search | 分支间短、回溯时长 | 中 |
| T1 Sequential | 长（转一圈才回来） | 高 |
| T5 Group Chat | 长且不可预测 | 高 |
| T4 Debate | 长（主要损失是位置错配） | 高 |
| T8 Simulation | 极长（agent 数多） | 最高 |

---

## 6. 实验矩阵：全叉积

```
9 workload × 5 block size 档 × 2 容量档 = 90 组配置
每组 10 个 workflow 实例（benchmark 里抽 10 题）
执行顺序：T1/T2/T3（30 组）→ T7/T8/T9（30 组）→ T4/T5/T6（30 组）
```

固定量：Qwen3-4B BF16、`temperature=0`、`--kv-cache-dtype auto`（绝不 fp8）、单 vLLM 实例、锁定本仓库 commit。每个 workflow 开始前 `reset_prefix_cache()`（见 §6.2）。

**`max_tokens=1024`、`gpu_memory_utilization=0.95`（2026-08-03 定）**：实测池子 **102,048 token**。
两个参数一起，是为了让**充足档与受限档跑在同一种卡上**（A5000）：

- MoA 原文的 `max_tokens=2048` 几乎不生效（实测输出 291–1365），但会让单 workflow 涨到 13.5 万 token，
  24GB 卡装不下。1024 只截尾部，W 落在 9.2 万左右。
- `0.95` 比常用的 `0.90` 多约 9% 池子（93,424 → 102,048），给上面这个 9.2 万留出约 10% 余量。

**为什么值得为此改参数**：原分工（充足档 L40S / 受限档 A5000）把**一条 GPU 架构边界埋进了 Q3 的推理链**——
Q3 拿受限档的驱逐名单去 join 充足档算出的重复判定，而 sanity #6 的保真度只在**同一张 A5000 内部**测过，
跨卡（Ampere sm_86 ↔ Ada sm_89，FlashAttention 的 tile 划分与 kernel 分发可能不同 → 归约顺序不同）**从未测量**。
两档同卡则该变量直接消失。次要理由：L40S 仅 3 张且长期零可用，A5000 有 5 张空闲。

⚠️ **须如实记录的混淆**：压短输出会让 reference 变短，而 persona + Aggregate-and-Synthesize 提示词是**固定长度**的，
于是固定部分在 prompt 中的占比上升 → **重复率可能被人为抬高**。方向可判断，幅度未知。
若日后拿到 L40S，应用 `max_tokens=2048` 复跑一组同配置作对照。

### 6.1 block size 五档的实现（实际硬件约束）

本仓库全部可用 CUDA 后端的内核约束：FlashAttention/Triton 要求 16 的倍数（`flash_attn.py:84`、`triton_attn.py:291`），FlashInfer 支持 {16,32,64,…}（`flashinfer.py:374`）。**物理 block 4/8 跑不了**。实现方式：

| 档 | 物理 block_size | prefix_match_unit | 说明 |
|---|---|---|---|
| 4 | 16 | 4 | 匹配/hash 粒度 4 token（`vllm/config/cache.py:56`，匹配粒度可细于物理块） |
| 8 | 16 | 8 | 同上，粒度 8 |
| 16 | 16 | 默认 | 基准 |
| 32 | 32 | 默认 | |
| 64 | 64 | 默认 | |

- 4/8 档的"一样 block 计数"在分析侧按 4/8-token 单元切分 dump（边界与 16 对齐，4|16、8|16，无错位）。
- **限制（如实说明）**：4/8 档的驱逐仍按 16-token 物理块发生，只有前缀匹配粒度是 4/8。若要求驱逐粒度也到 4/8，现有内核做不到。
- 尾部效应可测：block 越大，尾部不满一块的 token 无法缓存（平均浪费 block_size/2），5 档正好扫出这条曲线。

### 6.2 容量两档

**读文结果（2026-08-03）**：四篇论文都**没有**显式的 "KV 池 = X GB / X token" 设置——

| 论文 | 容量设置的事实 |
|---|---|
| KVFlow (arXiv:2507.07400) | 无显式池大小；用大 fixed prefix（10 agent × 8192 token ≈ 8.2 万 token 工作集）撑爆 A10G-24GB 的自然容量，推导出**容量 ≈ 工作集的 40–60%**。四篇中唯一的 agentic 驱逐实验 |
| TokenCake (arXiv:2510.18586) | GPU 池未写（vLLM 默认），只固定 CPU offload 池 100GB；agent 上下文 1K–5K token × 20 并发应用 |
| Parrot (OSDI'24) | 非容量受限实验，无池设置 |
| Preble (arXiv:2407.00023) | 无显式池；Toolbench 潜在前缀工作集 ~2900 万 token，远超单卡容量（比例 ~0.01 量级），属前缀调度场景 |

**受限档定义（按文献对齐，宸逸已确认 2026-08-03）**：采用 KVFlow 比例——**受限档容量 = 该 workload 工作集的 50%**：

| 档 | 定义 | 数值 |
|---|---|---|
| 受限 | KVFlow 式：容量 = 工作集 × 0.5 | 每个 workload 先在充足档跑完，从事件流测出**单 workflow** 工作集 token 数 W，受限档 `num_gpu_blocks_override = 0.5 × mean(W) / block_size`。各 workload 数值不同、自动适配 |

**用 mean 不用 max（2026-08-03 实测后定）**：W 在 workflow 之间波动很大，T1 实测
`[8176, 3872, 7232, 7760, 7696, 23520, 8416, 6624, 5776, 6352]`——第 6 个是离群值（该题重试多轮），
是均值 8,542 的 2.75 倍。若按 `0.5 × max` 得池子 11,760 token，**比其余 9 个 workflow 各自的整个工作集还大**，
它们全程不受限，`BlockRemoved` 虽非空（sanity #3 技术上通过）但 Q3 实际只有 1/10 个 workflow 的数据支撑：

| 规则 | 池子 | 真正受限的 workflow |
|---|---|---|
| `0.5 × max(W)` | 11,760 | 1 / 10 |
| **`0.5 × mean(W)`（采用）** | **4,271** | **9 / 10** |
| `0.5 × median(W)` | 3,464 | 10 / 10 |

mean 对应"该 workload 典型的工作集"，与 §6.2 原文"该 workload 工作集的 50%"的单数措辞一致。
| 充足 | **单个 workflow 内**不发生驱逐 | 每个 workflow 开始前 `reset_prefix_cache()`，以整轮 `BlockRemoved` 为空验证 |

**W 是每个 workflow 一份，不是每组配置一份（2026-08-03 实测后修订）**：实测 T3 单 workflow 工作集在 **90,192 / 135,088 / 80,480 token** 之间（19 次调用；prompt 从 layer 0 的 50 token 涨到 aggregator 的 6,801 token）。一组配置 10 题、题目互不相同、跨 workflow 几乎无共享，累计约 **100 万 token**——L40S 只有 22 万、A5000 只有 9.3 万，原判据"整组不驱逐"在任何卡上都不成立。

改为**每个 workflow 之间清场**。注意 **W 随题目波动很大（9 万–13.5 万），不是常数**：

| 卡 | KV 池 | 能否承担充足档 |
|---|---|---|
| A5000 24G | 9.3 万 token | **不能**——实测 135,088 token 的 workflow 在 node4 上驱逐了 2,606 个 block，`ample_criterion_met=false` |
| L40S 48G | 22 万 token | 能。19 次调用全部打满 `max_tokens=2048` 的**最坏情况**约 20.1 万 token，余量约 9% |

即充足档仍然**只能跑 L40S**（与 athena 方案 §2 的分工一致）；改清场解决的是"整组 100 万"这个不可能，不是把充足档下放到 A5000。每组配置结束后必须查 `manifest.ample_criterion_met`，为 false 说明该组有 workflow 撑破了池子。

**这不损失跨 workflow 的重复度信号**：本实验的判定是对 **dump 出来的 block 做离线比对**，不依赖 vLLM 是否在运行时把它们合并成同一物理块。清场后同内容在两个 workflow 里是两块独立的物理 block，正是要计数的重复对（dump 侧因此必须按 workflow 边界重置去重集合，否则会漏 dump）。真正损失的只有**跨 workflow 的前缀命中率**这一个量——T3 里它本来就接近 0。

这也决定了运行次序依赖：**同一 workload 必须先跑充足档、后跑受限档**（受限档数值来自充足档实测）。

容量以 **token 数**定义、跨 block size 可比：`num_gpu_blocks_override = 容量token数 / block_size`。

### 6.3 两档容量各采什么

- **充足档（45 组）**：事件流全采；**KV 张量只在 16/32/64 三档 dump（27 组）** → 一样 block 计数的主结果（Q1/Q2 全部出自这里）
  - **4/8 档不 dump（2026-08-03 定）**：这两档的物理 block 就是 16 token（见 §6.1），KV 张量与 16 档是同一批数据，只有匹配粒度不同；按 §6.1 本来就要"在分析侧按 4/8-token 单元切分 dump"，所以直接复用 16 档张量离线切分。4/8 档照常跑、照常采事件流（W、命中率、驱逐全都还在），只是不落张量。存储从 740 GB 降到 444 GB，不损失任何分析能力。
- **受限档（45 组）**：只采事件流（`BlockStored`/`BlockRemoved`）+ `num_cached_tokens`，不 dump——block 随时被踢、dump 时机不可控。与同配置充足档的计数结果按 block 内容 join：**被驱逐的 block 中有多少属于"一样"集合**（Q3）。join 的前提是 temperature=0 下同内容同前缀的 KV 可复现，由 sanity #6 保真度实验支撑。

---

## 7. 实施方案（已对本仓库核实）

### 7.1 driver

离线 `LLM()`，不起 server。所有拓扑产出统一 `Call` 流，自己 tokenize 传 `prompt_token_ids`。

### 7.2 KV dump（充足档）

- `VLLM_ENABLE_V1_MULTIPROCESSING=0`（`vllm/envs.py:148`）+ `enforce_eager=True`，driver 与 engine 同进程，直接读 `gpu_model_runner.kv_caches`（`vllm/v1/worker/gpu_model_runner.py:562`，逐层张量）。
- 每个 Call 完成后：从 kv_cache_manager 取该请求 block_ids → 采样层的对应 block 拷 CPU 存盘（safetensors），记录 `(workflow_id, call_idx, agent_id, layer, block_id, block_hash, token_ids, positions)`。
- position 是 token 在本请求内的绝对位置，直接供 de-RoPE 用。

### 7.3 事件采集（两档都开）

- `KVEventsConfig`（zmq publisher）→ 订阅进程写 `blocks.jsonl`。`BlockStored` 带 `block_hashes/parent_block_hash/token_ids/block_size`，可离线重建 radix tree；`BlockRemoved` 给驱逐时刻。
- 每个 `RequestOutput.num_cached_tokens` 落盘，与事件流交叉验证。

### 7.4 de-RoPE

独立离线函数（NeoX `rotate_half` 逆变换），不改 vLLM 代码。上线前过 sanity #1。

### 7.5 硬件分工

- **5090 (32GB)**：全部推理（radix tree 是 per-engine 的，跨实例零共享，绝不做多实例负载均衡）
- **5060 (8GB)**：离线分析、de-RoPE、计数

---

## 8. 模型与显存换算（背景）

```
KV bytes/token = 2(K,V) × n_layers × n_kv_heads × head_dim × dtype_bytes
```

| 模型 (BF16) | layers | kv_heads | head_dim | KV/token | 权重 | 32GB 卡可缓存 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 8 | 128 | 112 KB | 1.2 GB | ~24 万 token |
| Qwen3-1.7B | 28 | 8 | 128 | 112 KB | 3.4 GB | ~22 万 token |
| **Qwen3-4B** | 36 | 8 | 128 | **144 KB** | 8 GB | **~13–14 万 token** |
| Qwen3-8B | 36 | 8 | 128 | 144 KB | 16.4 GB | ~8.5 万 token |

**选用模型：Qwen3-4B-Instruct-2507**（与 Qwen3-4B 同架构：36 层 / 8 kv-heads / head_dim 128）。理由：
- 权重 8GB，32GB 卡上留下 ~13 万 token 的 KV 池，充足档才成立（8B 只剩 ~8.5 万，余量不够）
- 0.6B/1.7B 太弱，跑不动 9 类 agent 编排（对话会退化，trace 不具代表性），且 GQA 下 KV/token 和 4B 差别很小，省不了 KV
- 非 thinking 版本，输出短且确定，不会被 thinking token 撑爆 decode 长度
- 标准 NeoX 风格 RoPE，de-RoPE 逆变换直接适用

- GQA 下 KV 大小由 `n_kv_heads` 决定，换小模型省的是权重不是 KV。
- 一个 16-token block 在 Qwen3-4B 上 = 2.25 MB（全层）。
- **dump 采样 5 层**：0 / 8 / 17 / 26 / 35（浅中深覆盖），20KB/token。层深本身是核心自变量（上下文污染逐层累积），采样是设计不是妥协。

### 存储预算

| 项 | 估算 | 小计 |
|---|---|---|
| ~~充足档 dump（原估）~~ | ~~9 × 5 档 × ~10 万 token × 20KB ≈ 2GB × 45~~ | ~~90 GB~~ |
| 充足档 dump（2026-08-03 实测修订） | 实测 1.6 GB/workflow（80,480 token × 20KB）→ 16.5 GB/组 × **27 组**（只 16/32/64 档） | **~444 GB** |
| 事件 jsonl（90 组） | | <2 GB |
| 需要空闲盘 | | **≥500 GB** |

原估算低了 8 倍：把"每组配置 ~10 万 token"当成了工作集，实测是**每个 workflow** 8 万、每组 10 题合计 80 万。

**落盘路径（已核实本服务器）**：`/mnt/hdd_8t`（7.3T 盘，3.9T 空闲，用户可写）→ 建议 `/mnt/hdd_8t/kvpim-traces/`。dump 是顺序写，HDD 带宽足够。备选 `/home`（414G 空闲）。`/mnt/ssd` 无写权限，`/eda` 只剩 23G，均不可用。

### 单卡 5090 够不够

**够，瓶颈是墙钟时间不是显存。**

- 显存：权重 8GB + KV 池 ~13 万 token，充足档成立的前提是单 workflow 工作集 ≤ ~10 万 token。唯一有风险的是 T8 Simulation（agent 数多、工作集最大），通过控制 POP/STEPS 规模压在 10 万 token 内，以充足档 `BlockRemoved` 为空来验证。
- 时间粗估：每 workflow 约 20–40 个 Call，decode 合计 ~1 万 token，5090 上 Qwen3-4B eager 单流 ≥150 tok/s → 每 workflow 1–2 分钟 → 每组配置（10 workflow）15–20 分钟 → 90 组 ≈ **25–30 小时纯推理**，加 dump/清场开销按 **2–3 天墙钟**规划。
- 5060 (8GB) 不参与推理（放不下且 radix tree 跨实例零共享），只跑离线分析。

---

## 9. Sanity Check 清单（按序执行，不通过就停）

1. `derope(k_post, pos) ≈ k_pre`，判据 **`cos(k_rec, k_pre) > 1 − 1e-6`（逐 token 取最小值）** ←— 最关键
   （在 `rotary_emb` 上挂 **pre-hook** 抓真 pre-RoPE K：vLLM 的旋转是原地操作，post-hook 拿到的已经被转过了）
   - **判据为何不是绝对误差**（2026-08-03 实测，原写 `max abs err < 1e-2` 已作废）：K 有 attention-sink 式离群值，
     实测 `|k_post|` 最大 314 而均值仅 2.35，BF16 存储下 1e-2 的绝对阈值物理上到不了。
     佐证：用本仓库的 `rope()` 正变换去对 vLLM 自己的 `k_post`，误差同为 6.2e-2 量级——1e-2 比 vLLM 自身数值噪声还低。
     且本实验的判定指标本来就是 cosine，绝对误差不是相关量。
   - 实测值：cosine 0.9999999692（cos/sin 走 BF16，与 vLLM 一致）/ 0.9999999805（cos/sin 走 FP32）。
2. 同内容同位置 block 对判定为"一样"（cos ≈ 1.0，验证 token 对齐与阈值）
3. 受限档 `BlockRemoved` 非空 ←— 否则容量档还要降
4. `num_cached_tokens` 与事件流重建的 radix tree 命中一致
5. **编排正确性验证（2026-08-03 重写）**。原判据"T3 Fan-out 的 prefix hit ratio 接近理论上界"**已作废**：
   宸逸选定 6 个不同 system prompt 扮演 proposer 后，6 个 proposer 从第 0 个 token 就分叉，
   实测命中率仅 6.6%——低命中率成了"正确"的表现，与"编排 bug 导致前缀断裂"读数相同，探针失灵。

   这一条守的是整条推理链里**唯一一个外部效度环节**：其余 sanity 都在验"把跑出来的东西测准了"，
   只有它验"跑出来的东西是对的"。而 vLLM 看不见拓扑（§2），让 T1 成为 T1、T9 成为 T9 的**只有编排代码**，
   所以 fig1 的横轴物理上就是那九个 driver；编排出错不会报错，会变成一个看起来可发表的**结论**。

   替代判据分两层（实现在 `kvpim/analyze.py`）：

   - **Tier 1 —— 与论文自己的实现逐字节对齐**（真正非循环，参照系是原作者的代码）。
     用 AST 从 `ref/` 里只取出目标函数编译执行（这些文件顶层 import 了未安装的 openai/requests，不能整体导入）。
     T3 实测：`_with_references` 与 `ref/MoA/utils.py::inject_references_to_messages` 在 6 个 proposer +
     aggregator 共 7 个用例上**逐字节相等**。可同样处理的还有 Reflexion、Du et al. debate、CAMEL、ToT
     （其 prompt 模板在 repo 中有逐字原文）；T2/T5/T6 的编排耦合在框架里，只能退到 Tier 2。
   - **Tier 2 —— token 级结构不变量**（覆盖全部 9 个拓扑）。断言从拓扑定义写出，不从 driver 抄。
     T3 的六条：persona 前缀正确、layer 0 无注入、同层 user turn 逐 token 相同、同层注入段相同、
     AGG 提示词逐字出现、aggregator 不含任何 persona。

   **判据本身必须能失败**：植入五个 bug（user turn 混入 agent 编号、references 顺序 per-agent 不同、
   persona 错配、aggregator 带 persona、layer 0 误注入）后逐一验证，**五个全部被对应不变量抓到**；
   Tier 1 把 references 编号从 1-based 改成 0-based 也立即失败。

   **覆盖边界（如实说明）**：这套判据防的是**意外与回归**，防不住**理解偏差**——若对拓扑的理解本身就错，
   Tier 2 的断言会跟着错。防理解偏差只有 Tier 1 的逐字节对齐和人工 review 两条路。原判据同样防不住。
6. 重算保真度：单请求 dump → 人为挤出 → 重发 → 再 dump → 比对相对误差（支撑 §6.3 的 join 前提）；其 cosine 分布的 P1 分位数即 §4 的判定阈值 τ——**本条必须在计数分析开始前完成**

---

## 10. 产出物

```
data/
  calls.jsonl        # Call 流：拓扑、workflow、agent、parent、token_ids
  blocks.jsonl       # BlockStored / BlockRemoved 事件（90 组全有）
  dumps/<topo>/<block档>/<workflow>/<call_idx>_<layer>.safetensors   # 充足档
  counts.parquet     # 每组配置：N_total、N_dup[K_post/K_derope/V]、分层计数
analysis/
  fig1  9 workload × {K_post, K_derope, V} 的一样 block 占比（Q1 主图）
  fig2  一样 block 占比随 block size 4→64 的变化，9 workload 各一条线（Q2）
  fig3  受限档被驱逐 block 中"一样"block 的占比，按 workload（Q3）
  fig4  阈值敏感性附录（τ 与 {0.90, 0.95, 0.99, 0.999}）
```

## 11. 执行顺序

1. 环境搭建：`uv venv` + `VLLM_USE_PRECOMPILED=1 uv pip install -e .`（锁定 commit）
2. driver + T3（最简单、hit ratio 可反向验证）打通事件采集
3. dump 通路 + de-RoPE，过 sanity #1/#2
4. **T1/T2/T3 × 5 档 × 2 容量 = 30 组**全跑，出 fig1–fig3 的前三列
5. T7/T8/T9 的 30 组
6. T4/T5/T6 的 30 组
7. 汇总分析

---

## 12. 阅读清单（定 benchmark 与容量值）

读每篇的**评测章节**，确认任务集、题目数、agent 配置（轮数/人数），抄进 §3 表格。
**状态（2026-08-03）：全部读完**，T1–T9 结果已填入 §3 配置表，容量四篇的结论在 §6.2。

| 拓扑 | 要读的文章 | 要从中拿到的东西 |
|---|---|---|
| T1 | MetaGPT (Hong et al., ICLR'24)；ChatDev (Qian et al., ACL'24) | HumanEval/MBPP 评测设置；SRDD 数据集 |
| T2 | AutoGen (Wu et al., 2023)；Anthropic *Multi-Agent Research System* (eng blog, 2025)；GAIA (Mialon et al., ICLR'24) | supervisor 场景任务集；GAIA 题目分级与抽题方式 |
| T3 | Mixture-of-Agents (Wang et al., 2024)；More Agents Is All You Need (Li et al., 2024) | AlpacaEval 2.0 设置；proposer 层数/个数 |
| T4 | Du et al. (ICML'24)；Liang et al. MAD (EMNLP'24) | GSM8K/MMLU 抽题、debate 轮数与 agent 数 |
| T5 | AutoGen GroupChat 章节（同 T2 论文） | 场景 A1 数学求解的 GroupChat 配置、MATH 抽题 |
| T6 | OpenAI Swarm 设计文档；τ-bench (Yao et al., 2024) | airline/retail 域任务、handoff 规则 |
| T7 | Reflexion (Shinn et al., NeurIPS'23)；Self-Refine (Madaan et al., NeurIPS'23) | HumanEval 迭代轮数 K、critic 提示词 |
| T8 | CAMEL (Li et al., NeurIPS'23)；Generative Agents (Park et al., UIST'23) | AI Society 数据集用法；Smallville 人数/步数（用于压工作集规模） |
| T9 | Tree of Thoughts (Yao et al., NeurIPS'23)；LATS (Zhou et al., ICML'24) | Game of 24 题目集、branch/depth 参数 |
| 容量档 C | KVFlow (arXiv:2507.07400)；TokenCake (arXiv:2510.18586)；Parrot (OSDI'24)；Preble (arXiv:2407.00023) | **已读完**，结论见 §6.2。KVFlow/TokenCake 无开源 repo（已确认搜不到）；Parrot/Preble 已克隆在 `ref/` |

---

**决策状态（全部已定）**：
- ~~A~~ 已定：benchmark 按 §3 表格提议列执行
- ~~B~~ 已定（2026-08-03 更新）：主阈值 τ 由 sanity #6 重算噪声底线校准（P1 分位），0.99/0.999/0.95/0.90 为敏感性点
- ~~C~~ 已定：受限容量档按 case 对齐相应论文数值，读 §12 后填 §6.2 映射表
- ~~D~~ 已定：落盘路径 `/mnt/hdd_8t/kvpim-traces/`（3.9T 空闲、可写）
