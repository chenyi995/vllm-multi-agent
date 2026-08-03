# KV-PIM Block-wise 实验交接文档

> 给在另一台服务器（A5000 + L40S）上继续本实验的人/agent。
> 详细实验设计**必读** `notes/planfiles/blockwise-similarity-plan.md`（下称"计划"），本文只讲怎么落地。

## 0. 这是什么

vLLM fork（upstream `vllm-project/vllm`，base commit `e279f7158`）+ multi-agent KV cache 研究文件。目标：9 类 multi-agent workload × 5 个 block size 档 × 2 个容量档，统计存储的 KV 中**有几个 block 是一样的**（K post-RoPE / K de-RoPE / V 三视角）。

```bash
git clone https://github.com/chenyi995/vllm-multi-agent.git
```

文件地图：

| 路径 | 内容 |
|---|---|
| `notes/planfiles/blockwise-similarity-plan.md` | 实验计划（指标定义、矩阵、benchmark 配置表、sanity 清单）|
| `notes/planfiles/KVPIM-README.md` | 早期背景笔记 |
| `CLAUDE-Chenyi.md` | 工作守则（AI 助手必须遵守）|
| `workloads/` | 9 类拓扑的 driver（**尚未实现**，见 §5 进度）|
| `ref/` | 论文开源仓库克隆（4.4GB，**不入库**，按 §2 重新拉取）|

## 1. 环境搭建

```bash
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto   # 纯 Python 模式
uv pip install safetensors datasets pyzmq msgspec pandas pyarrow
```

- 模型：`Qwen/Qwen3-4B-Instruct-2507`（BF16，权重 8GB，KV=144KB/token，36 层，采样层 0/8/17/26/35）
- HF token：需要登录；**GAIA 是 gated**（T2 用），先在网页 `huggingface.co/datasets/gaia-benchmark/GAIA` 同意条款
- 锁定本仓库 commit，不要 rebase 上游

## 2. 拉取论文开源仓库（ref/ 不入库，链接如下）

```bash
mkdir -p ref && cd ref
for url in \
  https://github.com/FoundationAgents/MetaGPT https://github.com/OpenBMB/ChatDev \
  https://github.com/microsoft/autogen https://github.com/togethercomputer/MoA \
  https://github.com/composable-models/llm_multiagent_debate https://github.com/Skytliang/Multi-Agents-Debate \
  https://github.com/openai/swarm https://github.com/sierra-research/tau-bench \
  https://github.com/noahshinn/reflexion https://github.com/madaan/self-refine \
  https://github.com/camel-ai/camel https://github.com/joonspk-research/generative_agents \
  https://github.com/princeton-nlp/tree-of-thought-llm https://github.com/lapisrocks/LanguageAgentTreeSearch \
  https://github.com/microsoft/ParrotServe https://github.com/WukLab/preble \
  https://github.com/openai/human-eval https://github.com/openai/grade-school-math \
  https://github.com/hendrycks/math https://github.com/hendrycks/test \
  https://github.com/tatsu-lab/alpaca_eval \
; do git clone --depth 1 -q "$url" && echo "OK $(basename $url)"; done
```

KVFlow (arXiv:2507.07400) 与 TokenCake (arXiv:2510.18586) 无开源代码（已确认），只读论文。

## 3. 实验怎么跑

矩阵与顺序（计划 §6）：**9 workload × 5 block 档 × 2 容量档 × 每组 10 题**；
拓扑顺序 **T1/T2/T3 → T7/T8/T9 → T4/T5/T6**；同一 workload **先充足档、后受限档**（受限档容量 = 充足档实测工作集 W × 0.5）。

每组配置的 vLLM 关键参数：

| 项 | 值 |
|---|---|
| 引擎 | 离线 `LLM()`，`enforce_eager=True`，环境变量 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（同进程才能读 `gpu_model_runner.kv_caches` 做 dump）|
| block 档 | 16/32/64 → `block_size=N`；4/8 → `block_size=16` + `prefix_match_unit=4或8`（内核不支持物理 4/8，见计划 §6.1）|
| 容量 | 充足档：默认 profiled；受限档：`num_gpu_blocks_override = 0.5 × W / block_size` |
| 事件 | `KVEventsConfig`（zmq），订阅进程落 `blocks.jsonl`（BlockStored 带 token_ids，BlockRemoved 是驱逐时刻）|
| 采样 | `temperature=0`，`kv-cache-dtype auto`（绝不 fp8）|
| 清场 | 每组配置开始前 `llm.reset_prefix_cache()` |

铁律：driver 自己 tokenize、传 `prompt_token_ids`（否则位置对不上，de-RoPE 全错）；
编排全部手写纯 Python，**不用** LangGraph/AutoGen 框架跑（框架的 memory/trimming 会污染测量）。
充足档判据：`BlockRemoved` 为空；若仅出现"已完成 workflow 的块被清理"，记录后可接受。
9 类 workload 的 benchmark、agent 数/轮数、抽题方式：**逐行照计划 §3「benchmark 实施配置」表执行**。

## 4. GPU 分工（A5000 24G / L40S 48G / 5090 32G）

KV 池 = 显存×0.92 − 权重 8GB − 激活 ~1.5GB，÷ 144KB/token：

| 卡 | KV 池 | 能跑什么 |
|---|---|---|
| **A5000 24G**（768 GB/s） | ~12.6GB ≈ **8.7 万 token** | **全部 45 组受限档**（只采事件不 dump，需求 ~15GB）；sanity 实验；离线分析 |
| **L40S 48G**（864 GB/s） | ~34.7GB ≈ **24 万 token** | **全部 45 组充足档 dump**（池比 5090 还大，10 题累计不驱逐最有保障）；需挂 ≥120GB 数据盘 |
| **5090 32G**（1792 GB/s） | ~19.9GB ≈ **13.8 万 token** | 任意配置（速度约为 L40S 2 倍、A5000 2.3 倍），纯加速用 |

**没有哪部分非 5090 不可，A5000 + L40S 就能跑完全部实验。**
流水线：L40S 跑 workload i 的充足档 → 事件流算出 W → A5000 跑 i 的受限档，两卡错位并行。

## 5. 当前进度（2026-08-03）

- ✅ 计划定稿（唯一待宸逸确认项：受限档 = 工作集 50%，计划 §6.2）
- ✅ §12 论文全部读完，benchmark 配置表（计划 §3）与容量结论（§6.2）已填
- ✅ ref/ 21 个仓库已验证可克隆
- ❌ `workloads/` driver 未实现；dump/事件订阅/分析脚本未写；**实验一组都没跑**
- 实现顺序建议照计划 §11：先 T3 打通事件采集 → dump + de-RoPE 过 sanity #1/#2 → 再铺开

## 6. Return 清单：跑完必须带回的东西（分析输入）

每组配置一个目录 `<topo>_<block档>_<容量档>/`，缺一不可：

1. **`manifest.json`** — 拓扑、block_size、prefix_match_unit、容量档与 override 实值、实测 W、vLLM commit、模型 revision、tokenizer/chat-template hash、随机种子、采样层、时间
2. **`calls.jsonl`** — 每次 Call：workflow_id、call_idx、agent_id、parent_idx、完整 `prompt_token_ids`、输出 token_ids、`num_cached_tokens`
3. **`blocks.jsonl`** — 全部 BlockStored（含 block_hashes/parent_hash/token_ids）与 BlockRemoved 事件，带时间戳
4. **`dumps/`**（仅充足档）— `<workflow>/<call_idx>_<layer>.safetensors`，附每 block 的 `(block_id, block_hash, token_ids, positions)` 元数据；**positions 丢了 de-RoPE 就废了**
5. **`sanity.log`** — 6 条 sanity（计划 §9）的执行结果，尤其 derope 误差数值与受限档 BlockRemoved 非空确认

分析流程（任意 ≥8GB GPU 或纯 CPU 均可，不占推理卡）：
de-RoPE 逆变换 → 按 token_ids 生成候选对 → cos ≥ 0.99 判定（K_post/K_derope/V 三视角，分层 + 严格口径）→ `counts.parquet` → 计划 §10 的 fig1–fig4。
硬盘：充足档 dump 全量 ~90GB，预留 ≥120GB。
