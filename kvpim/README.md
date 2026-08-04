# kvpim — KV-PIM block-wise 实验代码

测量 multi-agent workload 在 vLLM 缓存里留下的 KV block 有多少是重复的。
实验设计见 `../notes/planfiles/blockwise-similarity-plan.md`。

## 三层架构

```
Layer 3  编排        kvpim/workloads/*.py   纯 Python 控制流，决定拓扑，不碰 GPU
             ↓ prompt_token_ids
Layer 2  推理        kvpim/runner.py        vLLM 单实例，看到的是无状态请求流
             ↓
Layer 1  缓存        kvpim/events.py        RadixTree + BlockPool，共享与驱逐在这里
         + dump.py
```

**vLLM 看不见拓扑**。让 T1 成为 T1、T9 成为 T9 的只有 `workloads/` 里的代码——
所以那一层出错不会报错，只会变成一个看起来可发表的结论。这就是 `analyze.py` 里
Tier 1 / Tier 2 校验存在的原因。

## 模块

| 文件 | 职责 |
|---|---|
| `call.py` | `Call` dataclass —— 所有拓扑产出的统一调用单元 |
| `runner.py` | 引擎构建、tokenize、跑一组配置、写 `manifest.json` / `calls.jsonl` |
| `events.py` | zmq SUB 线程订阅 KV 事件 → `blocks.jsonl`，带 replay 补洞 |
| `dump.py` | 采样层 K/V 拷盘 + 每块的 `(block_id, block_hash, positions)` |
| `derope.py` | NeoX `rotate_half` 的逆变换（离线，不改 vLLM） |
| `analyze.py` | 事件流重建、sanity 校验、重复度计数、README 生成 |
| `workloads/t{1..9}_*.py` | 九类拓扑的 driver |

## 九个拓扑

| | 拓扑 | benchmark | 原文实现来源 |
|---|---|---|---|
| T1 | Sequential（MetaGPT 五级瀑布） | HumanEval | `ref/MetaGPT` 角色 `goal`/`constraints` **逐字节** |
| T2 | Supervisor / Worker | GAIA | 无（benchmark 不规定 agent 结构，用计划 §3 编排）|
| T3 | Fan-out / MoA | AlpacaEval 2.0 | `ref/MoA` 的 Aggregate-and-Synthesize **逐字节** |
| T4 | Debate | GSM8K | `ref/llm_multiagent_debate` **逐字节** |
| T5 | Group Chat | MATH level-5 | AutoGen **v0.2.34** 的选 speaker 模板 **逐字节** |
| T6 | Handoff / Swarm | τ-bench airline | `ref/swarm` + `ref/tau-bench`，**运行时读取** |
| T7 | Reflection（Reflexion） | HumanEval | 真执行单元测试判定 |
| T8 | Role-play（CAMEL） | AI Society | `ref/camel` Inception Prompting **逐字节** |
| T9 | Tree Search（ToT） | Game of 24 | `ref/tree-of-thought-llm` **逐字节** |

## 跑一组配置

```python
from kvpim.runner import RunConfig, run
from kvpim.workloads import t3_fanout

cfg = RunConfig(topology="T3", block_tier=16, capacity_tier="ample")
out = run(cfg, t3_fanout.build, t3_fanout.load_tasks(10, seed=0))
```

实际提交见 `scratch/run_config.sbatch` 与 `scratch/run_group.sbatch`。

**必须的环境变量**（`build_llm` 会自己设前两个）：

```
VLLM_ENABLE_V1_MULTIPROCESSING=0   driver 与 engine 同进程，才能读到 kv_caches
VLLM_USE_FLASHINFER_SAMPLER=0      装的 flashinfer 是 cu13 版且首次采样会 JIT
HF_HOME=/home/cw636/chenyi/KVPIM/hf-cache
```

## 分析

```python
from kvpim.analyze import (
    reconcile_events,      # sanity #4：事件流 vs 引擎报的命中数
    TIER1_CHECKS,          # sanity #5 Tier 1：提示词 vs 论文原文，逐字节
    check_structure,       # sanity #5 Tier 2：拓扑结构不变量
    count_duplicates,      # 重复度计数（一组配置）
    build_counts_table,    # 汇总成 counts.parquet
    write_config_readme,   # 给一组配置写 README
)
```

## 本仓库 commit 上容易踩的坑

1. `apply_chat_template` 在 transformers 5.x 返回 `BatchEncoding` 不是 list ——
   直接用会把字符串当 token 传进去，**de-RoPE 全错且不报错**。用 `runner._tokenize`。
2. `llm.generate(prompt_token_ids=...)` 旧 API 已删除，改用 `TokensPrompt`。
3. 模型 `generation_config.json` 带 `temperature=0.7/top_k=20/top_p=0.8`，vLLM 默认会合并，
   污染 `temperature=0` 口径。已设 `generation_config="vllm"`。
4. `max_model_len` 必须显式设：模型是 262144 上下文，vLLM 要求池子装得下一条满长请求。
   **受限档尤其注意**：设了 `num_gpu_blocks_override` 后准入检查按压低后的池子算。
5. `BlockStored` 事件默认只代表**新增**缓存的块；只有 `kv_cache_report_mode="full"`
   才会为复用也发事件。改了这个默认值会让 `N_total` 静默高估。
6. FlashAttention 的 KV 张量把 K/V 打包在最后一维：
   `kv_cache.transpose(1, 2).split(head_size, dim=-1)`，**K 在前 128 维、V 在后 128 维**。
7. `prefix_match_unit` 比物理 block 细时，hash 覆盖的结束位置会落在块内部，
   起点必须按块边界取，不能用 `end - block_size` 反推。
