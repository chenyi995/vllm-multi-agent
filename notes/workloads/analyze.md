# `kvpim/analyze.py` —— 离线校验与计数（1240 行）

> 术语见 [`README.md`](README.md)｜数据流全景见 [`CODE.md`](CODE.md)

## 这个文件是什么

**跑完之后的一切都在这里。** 四块职责：

| 块 | 函数 | 干什么 |
|---|---|---|
| 1 | `RadixTree` / `reconcile_events` | sanity #4：证明事件流完整 |
| 2 | `check_*_reference` / `check_structure` | sanity #5：证明编排代码是对的 |
| 3 | `count_duplicates` / `build_counts_table` | **实际的测量** |
| 4 | `write_config_readme` | 给每组配置生成 README |

---

## 1. sanity #4：事件流完整性（`reconcile_events`）

### 想法

从 `blocks.jsonl` 重建缓存的前缀树，预测每次调用**应该**命中多少 token，
与引擎自己报的 `num_cached_tokens` 对账。对不上说明事件丢了。

### `RadixTree.from_events`

按事件流顺序重放：

```python
AllBlocksCleared → 清空整棵树
BlockRemoved     → 摘掉这些块（以及以它们为父的子树，因为链断了就不可达）
BlockStored      → 沿 parent_block_hash 挂上，token_ids 按 block_size 切成 chunk
```

`longest_prefix` 从根开始贪心走：找一个 chunk 与 prompt 下一段完全相同的孩子，走过去。

### ⚠️ 这个函数我修过四次，每次都是工具错不是数据错

**必须让你知道全过程，好判断我有没有把判据调到变绿。**

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | T3 有 7 处不符 | 树完全没处理 `AllBlocksCleared` / `BlockRemoved` | 补上 |
| 2 | T7 有 9 处，**全是各 workflow 第 0 次调用** | 清场事件由**发布线程异步打时间戳**，可能晚于 driver 的 `t_start`，按时间过滤清不干净 | 改成按 `AllBlocksCleared` **分段**，每个 workflow 只看自己那段 |
| 3 | T1 b16 受限档 2 处，仍是第 0 次调用 | 上个 workflow 的事件因发布延迟落进下一段开头 | 段内再丢掉时间戳早于该 workflow 首次调用的事件 |
| 4 | 7 组 T4，gap 恰好 **16** | 引擎永远留**最后一个物理块**重算，我按 `prefix_match_unit`（4/8 档是 4）写容差，太小 | 容差改成物理 `block_size` |

**第五次不是修，是缩范围。** 五组 `T4_*_limited` 各有 2 处失配，且 gap 是**负的**
（引擎报 3,328、我预测 96）。查明：

```
T4_w03 这一段：601 个块存过，325 个被驱逐
```

在一个 workflow 内驱逐 325 次、事件又异步打时间戳的情况下，
**"重建某一瞬间的精确缓存状态"这件事做不到**。

决定性检验——忽略驱逐、只问「引擎报的命中能不能被本 workflow 存过的块解释」：
**五个档、每一次调用全部通过，一处不差。事件流是完整的。**

所以把判据按容量档拆成两种模式，各自只声称数据能支撑的东西：

| 容量档 | 模式 | 判据 | 为什么 |
|---|---|---|---|
| ample | `exact-replay` | `predicted == reported`（容忍一个物理块）| 无驱逐，瞬时状态可精确重放 |
| limited | `completeness-only` | `predicted >= reported`（忽略驱逐）| 瞬时状态不可重建，但**完整性**仍可验，而这正是 sanity #4 存在的目的 |

**当前 76 组全部通过。**

### 防"调到变绿"的反向测试

故意删掉 5% 的 `BlockStored` 事件，两种模式都必须失败：

```
充足/exact-replay     丢 529 个事件 → passed=False，3 处失配
受限/completeness     丢  69 个事件 → passed=False，10 处失配
```

---

## 2. sanity #5：编排正确性（Tier 1 + Tier 2）

### 为什么需要这一条

计划 §2 的三层架构里，**vLLM 看不见拓扑**。让 T1 成为 T1、T9 成为 T9 的
**只有编排代码**。所以 fig1 的横轴物理上就是那九个 driver。

其余 sanity 都在验"把跑出来的东西测准了"，**只有这一条验"跑出来的东西是对的"**。
编排出错不会报错，会变成一个**看起来可发表的结论**。

原判据（"T3 命中率接近理论上界"）在 6-SYS 方案下失效——实测命中率仅 6.6%，
低命中率成了"正确"的表现，与"编排 bug 导致前缀断裂"读数相同，探针失灵。

### Tier 1：与论文自己的代码逐字节比

```python
load_reference_function("MoA/utils.py", "inject_references_to_messages", {"copy": copy})
load_reference_strings("camel/camel/prompts/ai_society.py", {"ASSISTANT_PROMPT", ...})
```

用 `ast` 把目标函数/常量**单独取出来编译**，不整体导入——因为那些文件顶层
`import openai / requests / loguru`，集群没装。这样比对的就是**论文作者写的那几行**。

| 拓扑 | 参照 | 结果 |
|---|---|---|
| T1 | `ref/MetaGPT/metagpt/roles/role.py:51-52` 模板 + 五个角色的 goal/constraints | 五个角色全部逐字节相等 |
| T3 | `ref/MoA/utils.py::inject_references_to_messages` | 7 个用例全部相等 |
| T4 | `ref/llm_multiagent_debate/gsm/gen_gsm.py` 的字符串字面量 + `construct_message` | 2 个用例相等 |
| T5 | `ref/autogen/.../groupchat.py@v0.2.34` 的两个选人模板 | 2 段相等 |
| T6 | `ref/swarm/.../configs/agents.py` 里 `flight_modification` 的 instructions | 1 段相等 |
| T8 | `ref/camel/.../ai_society.py` 的 ASSISTANT_PROMPT / USER_PROMPT | 2 段相等 |
| T9 | `ref/tree-of-thought-llm/.../game24.py` 的 propose/value prompt | 2 段相等 |

三种取参照物的方式，按参照物在原仓库里的存在形态选：

- **常量** → `load_reference_strings`（T5/T8/T9）。T5 的模板是 0.2.34 的
  `GroupChat` 类字段，而 `ref/autogen` 检出的 0.4 分支已经删掉了 GroupChat，
  所以用 `reference_source(..., rev="v0.2.34")` 走 `git show` 从同一个克隆的 tag 里读。
- **函数** → `load_reference_function`（T3/T4）。喂合成输入，比对输出。
- **埋在调用参数里** → 只能走 AST。T6 的 `flight_modification` 提示词写在
  `swarm.Agent(instructions=...)` 里，driver 执行不了这个文件（要 import swarm），
  所以从语法树上把那个 keyword 取出来比。

**T4 的题面提示词没有常量可比**——原文把它内联在 `__main__` 的一行 f-string 里。
所以判据是"我们这份出现在该文件的字符串字面量集合中"。这不如取常量精确
（同文件另一处出现同样的串也会通过），但比不查强。

未覆盖：**T2/T7**——两者的提示词都是自拟的（T2 是 GAIA 只给题不给编排，
T7 是 Reflexion 的提示词和框架耦合、取不出独立常量）。
这两类在 `TIER1_UNAVAILABLE` 里登记了原因，
每组配置的 `sanity.log` 会写下"不适用 + 理由"，
**而不是留空**——空值读起来像"忘了做"，和"做不了"是两回事。

### Tier 2：token 级结构不变量

把 `calls.jsonl` 里的 `prompt_token_ids` **解码回聊天轮次**，检查拓扑必须满足的关系。
断言从**拓扑定义**写出，不从 driver 抄。

| 拓扑 | 不变量 |
|---|---|
| T1 | 角色系统提示词与 MetaGPT 重建的一致、流水线顺序、context 单调增长 |
| T2 | worker 共享同一 SYS、共享任务描述前缀、汇总调用含 plan |
| T3 | persona 前缀、layer 0 无注入、同层 user turn 相同、同层注入段相同、AGG 提示词逐字、aggregator 不含 persona |
| T7 | 阶段循环合法（actor→evaluator→reflection→actor）、Ω=1 记忆上限 |
| T8 | 各自的规则块、共享任务串、40 条消息上限、transcript 镜像 |
| T9 | 模板只填 input、无 system 轮 |

### 同样做了反向测试

植入六个 bug，**六个全被对应不变量抓到**：

```
user turn 混入 [Proposer 3]        → user_turn_identical
references 顺序 per-agent 不同      → injection_identical
persona 错配                        → persona_prefix
aggregator 带 persona               → aggregator_no_persona
layer 0 误注入 AGG 提示词           → layer0_no_injection
Tier 1 references 编号改 0-based    → Tier 1 整体
```

### 覆盖边界（必须说清）

**防意外与回归，防不住理解偏差。** 若我对拓扑的理解本身就错了，
Tier 2 的断言会跟着错。防这个只有 Tier 1 的逐字节对齐和你的 review。
原判据同样防不住。

---

## 3. 实际的测量（`count_duplicates`）

### 算法

```python
# ① 按 token 内容分组（只读元数据，很轻）
groups[tuple(token_ids)].append(block_index)

# ② 只有 ≥2 成员的组才加载张量
for members in [g for g in groups.values() if len(g) > 1]:
    for layer in sample_layers:
        # ③ 三个视角各算一次 cosine 矩阵
        similarity = _cosine_matrix(vectors[view])
        similarity.fill_diagonal_(-1.0)          # 不和自己比
        hit = similarity.max(dim=1).values >= tau[layer]
```

**为什么先按内容分组** —— 计划 §4 规定：

> *"只有 token 内容完全相同的 block 才进入数值比对（不同内容的 block 不视为
> 可去重对象）"*

这既是口径要求（避开 ContextPilot 对"近似复用"的批评：近似匹配可损 9–11% 质量），
也顺带极省算力——T7 那组 1,843 块里只有 696 块进入比对，3,020 对，**5.6 秒**跑完。

### τ 是逐层的

```python
TAU_BY_LAYER = {0: 0.99999, 8: 0.99998, 17: 0.97809, 26: 0.96400, 35: 0.99648}
```

来自 sanity #6：让同一批内容在**部分前缀命中**的条件下重算，量出浮点噪声分布，
取 P1 分位。逐层是因为噪声**逐层累积**。

⚠️ **当前是工作值**，取自单条 prompt × 7 个命中边界。正式跑批后要用真实 trace 重标。

### 三种口径都输出

| 口径 | 含义 |
|---|---|
| `n_dup_layer_<L>` | 第 L 层判同的块数（对该层精确）|
| `n_dup_strict_all_layers` | **5 个采样层全都**判同（最严）|
| `n_dup_any_layer` | 至少一层判同（最松）|

⚠️ 真要在硬件上去重，**36 层都得相同**。只查 5 层 → "5 层全判同"是
"36 层全判同"的**上界**。

### `build_counts_table`

遍历所有带 `dumps/` 的配置，每组 × 三视角一行，写 `traces/counts.parquet`。

---

## 4. `write_config_readme`

从 `manifest.json` / `sanity.log` / 计数结果生成每组配置的 `README.md`：
跑了什么、`sbatch` 复现命令、五个产出文件各是什么、实测工作集与容量、
sanity 结果、重复度计数表、口径提醒。

受限档若触发了容量下限，会在 README 里**显式标注 KVFlow 的 0.5 比例未达成**
及实际比例。

---

## 5. 审查点

1. **`reconcile_events` 改过五次**（四次修 bug + 一次缩范围）。每一次的症状、根因、
   修法都在上面第 1 节。两种模式都做了反向测试，但**判据是我定的**，
   你如果觉得 `completeness-only` 太弱，可以要求受限档也做精确重放——
   那需要改 vLLM 让事件带调用标记。
2. **`_cosine_matrix` 在 double 精度下算**，向量先归一化再内积。
   块内所有 token 拉平成一个向量比较（不是逐 token 比再取最小）。
   计划 §4 写的是"block 内所有 token 的 cosine ≥ τ"，**逐 token 取最小会更严**，
   我用的是整块拉平——这是一处口径差异，需要你确认。
3. `TAU_BY_LAYER` 硬编码在文件里，没从 `tau.json` 读。重标后要手动同步。
4. Tier 2 的断言是我从拓扑定义写的，**没有第三方参照**。
