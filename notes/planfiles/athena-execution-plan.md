# Athena 集群执行方案（KV-PIM block-wise 实验）

> 本文是 `blockwise-similarity-plan.md`（下称"计划"）在 **Duke Athena Slurm 集群**上的落地方案。
> 只讲"在这台机器上怎么跑"，实验设计、指标、抽题一律以计划为准。
> 勘察时间 2026-08-03，仓库 HEAD `e087b82a4`。
> 资源配额：**最多 3 张 L40S + 5 张 A5000**（宸逸指定）。

---

## 1. 机器勘察结论

### 1.1 集群拓扑

- **登录节点**（本机，ZFS 服务端 node0）：48 核 / 251GB 内存 / **无 GPU、无 nvidia 驱动**
- **计算节点 node1–node6**：通过 NFSv3 挂载 node0 的存储
- Slurm 21.08.5，无抢占，`OverSubscribe=NO`

### 1.2 GPU 清单（实测 `srun ... nvidia-smi`）

| 节点 | 分区 | GPU | 显存 | 驱动 | 勘察时空闲 |
|---|---|---|---|---|---|
| node1 | `athena-mini` | 8× RTX A5000 | 24564 MiB | 535.154.05 | 6 |
| node2 | `athena`（默认） | 8× RTX A5000 | 24564 MiB | 580.105.08 | 6 |
| node3 | `athena` | 8× RTX A5000 | 24564 MiB | 550.67 | 2 |
| node4 | `athena-small` | 8× RTX A5000 | 24564 MiB | 535.154.05 | **8（最闲）** |
| node5 | `athena-genai` | **8× L40S** | 46068 MiB | 550.54.15 | **3** |
| node6 | `athena-genai` | 8× L40S | — | — | **DRAIN，不可用** |

- `athena-genai` 限 `genai-ux` 组，账号 `cw636` 在组内 ✅
- **全集群 L40S 仅 node5 可用**（node6 掉线）；**没有 5090**
- MaxTime：`athena` / `athena-mini` / `athena-small` = 14 天；`athena-genai` = 无限
- ⚠️ `gres.conf` 中 GPU **无 type 标注**（只有 `gpu:8`），`--gres=gpu:1` 无法按卡型筛选 → **必须用 `-w <node>` 指定节点**

### 1.3 存储

| 路径 | 登录节点 | 计算节点 | 可用 | 判定 |
|---|---|---|---|---|
| `/home/cw636`（= `/zpool-00/home/cw636`） | ✅ 可写 | ✅ 挂为 `/home` | **6.36T**（ZFS 池共享，无 quota，lz4） | ✅ **主存储** |
| `/zpool-00/data` | ✅ 777 | ❌ 该路径不存在（挂为 `/data`） | 同池 | 路径不通用，不用 |
| `/zpool-00/software`、`/archive` | 只读 | — | — | 不可用 |
| 计算节点本地 | — | node4 `/` 867G 空；node5 `/` 180G 空；`/dev/shm` 504G | 作业内临时 | staging 备用 |

- `/zpool-00/home` 在计算节点是 `/home` 的符号链接 → **`/home/cw636/...` 是唯一两端通用的路径，脚本一律用它**
- 实测 NFS 写带宽（node4 → `/home`，`dd oflag=direct`）：**284 MB/s**，dump 顺序写足够
- ⚠️ ZFS 池已用 93%（6.36T 空闲为全所共享、无 reservation）。实验只需 120GB，余量充裕，但需每日巡检

**替代计划 §8 的落盘路径**：计划里定的 `/mnt/hdd_8t/kvpim-traces/` 本机不存在，改为：

```
/home/cw636/chenyi/KVPIM/
  vllm-multi-agent/          仓库（.venv、ref/ 在内）
  hf-cache/                  HF_HOME，~10GB
  traces/<topo>_<block档>_<容量档>/    HANDOVER §6 的 5 类产出，~120GB
```

`traces/` 放仓库外，避免 90GB dump 被误 `git add`。

---

## 2. 卡怎么分

KV = 147456 B/token（2 × 36 层 × 8 kv-head × 128 dim × 2B），权重 7.49 GiB，
`gpu_memory_utilization=0.90` + `enforce_eager=True`：

| 卡 | KV 池 | token 容量 | 承担 |
|---|---|---|---|
| **L40S ×3**（node5） | ~31 GiB | **~22 万** | **45 组充足档 + KV dump** |
| **A5000 ×5**（node4） | ~12.3 GiB | **~9 万** | **45 组受限档（只采事件）** |

**硬约束**：A5000 原生池 ~9 万 token < 计划 §8 假设的单 workflow 工作集 ~10 万 →
**A5000 绝不能跑充足档**（必然驱逐，充足档 `BlockRemoved` 为空的判据直接失效）。
L40S 的 22 万是原计划 5090 口径（13.8 万）的 1.6 倍，计划 §8 对 T8 Simulation 撑爆显存的担心在本机解除。

受限档跑在 A5000 上无风险：容量本来就由 `num_gpu_blocks_override = 0.5 × W / block_size` 强制压低，
远小于 9 万的原生池。

---

## 3. 环境搭建

前提：`uv` 未安装，系统 python 3.10.12，无 torch，`.venv` / `ref/` 均不存在。

1. 装 `uv` 到 `~/.local/bin`
2. 仓库内 `uv venv --python 3.12`
3. ⚠️ **不能用 `VLLM_USE_PRECOMPILED=1`，必须源码编译**（2026-08-03 实测结论，宸逸拍板）。
   上游对本 base commit `e279f7158`（以及往前 400 个 commit 全部抽查）**只发 cu130 轮子**，
   `libcudart.so.13` 要求驱动 ≥580；node4 是 535、node5 L40S 是 550，两台都跑不了。
   `--torch-backend=auto` 选的是 torch cu129，与 cu130 轮子也不配套。

   做法：免 root 装 CUDA 12.9.1 toolkit 到 `/home/cw636/chenyi/KVPIM/cuda-12.9`，
   在 node4（96 核）编 sm_86（A5000）+ sm_89（L40S）：

   ```bash
   uv pip install "cmake>=3.26.1" ninja
   srun -p athena-small -w node4 --gres=gpu:1 -c 64 -t 180 bash -c '
     export CUDA_HOME=/home/cw636/chenyi/KVPIM/cuda-12.9
     export PATH=$CUDA_HOME/bin:$PATH
     export TORCH_CUDA_ARCH_LIST="8.6;8.9"
     export VLLM_USE_PRECOMPILED_RUST=1   # rust 产物无 CUDA 依赖，可复用
     uv pip install -e . --torch-backend=auto --no-build-isolation'
   ```

   `VLLM_USE_PRECOMPILED_RUST=1` 是因为集群无 cargo/rustc，而 `vllm-rs` 与
   `_rust_tool_parser.abi3.so` 经 `ldd` 确认不依赖 CUDA，直接复用轮子里的即可。
   装完分别在 node4（A5000）与 node5（L40S）各跑一次 `import vllm` + 单请求 generate 验证。
4. `uv pip install safetensors datasets pyzmq msgspec pandas pyarrow`
5. `HF_HOME=/home/cw636/chenyi/KVPIM/hf-cache`，在**登录节点**预下 `Qwen/Qwen3-4B-Instruct-2507`
   （登录节点有外网，实测 `huggingface.co` 200 OK）——避免 8 个并行作业抢下载
6. `hf auth login` + 网页同意 GAIA 条款（T2 需要，gated）
7. 登录节点执行 HANDOVER §2 的 21 个 `ref/` 克隆（4.4GB，不入库）

---

## 4. 冒烟与 sanity（全部在 A5000 上做）

L40S 只有 3 张，不拿来试错。按计划 §11 第 2–3 步：

- T3 打通事件采集（`KVEventsConfig` + zmq 订阅 → `blocks.jsonl`）
- dump 通路（`VLLM_ENABLE_V1_MULTIPROCESSING=0` + `enforce_eager` 同进程读 `gpu_model_runner.kv_caches`）+ de-RoPE
- sanity #1（de-RoPE 误差）、#2（同内容同位置判"一样"）、#5（T3 hit ratio）

**sanity #6 是硬闸门**（计划 2026-08-03 更新）：重算保真度实验的 cosine 分布 **P1 分位数就是主判定阈值 τ**，
`#6 未完成则计数分析不能开始`。它需要人为制造驱逐 → 用 A5000 + `num_gpu_blocks_override` 压容量即可，
不占 L40S。sanity #3（受限档 `BlockRemoved` 非空）本来就在 A5000 相里验证。

---

## 5. 矩阵调度

拓扑顺序按计划 §6：**T1/T2/T3 → T7/T8/T9 → T4/T5/T6**。
每个 workload 内部 5 个 block 档（4/8/16/32/64），充足档必须先于受限档（受限容量来自充足档实测 W）。

### 5.1 作业形态

- **L40S 相**（充足档 + dump）
  `sbatch -p athena-genai -w node5 --gres=gpu:1 -c 16 --array=<...>%3`
- **A5000 相**（受限档，只采事件）
  `sbatch -p athena-small -w node4 --gres=gpu:1 -c 16 --array=0-4%5`
  用 `--dependency=afterok:<对应 L40S array 元素>` 挂在同 workload 的充足档之后

**配额由 array 并发上限 `%3` / `%5` 结构性强制**，不可能越界。
5 个 block 档 ↔ 5 张 A5000 正好 1:1，受限档一轮打完。

`-c 16` 必须给：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 下 driver 与 engine 同进程，
tokenize + dump 拷盘都在这几个核上，默认 1 核会成为瓶颈。

### 5.2 流水

全部 array 一次性提交，Slurm 按依赖排程：
L40S 跑 workload *i+1* 的充足档时，A5000 在跑 workload *i* 的受限档，两相错位并行。

### 5.3 墙钟估算

带宽比：L40S 864 GB/s ≈ 5090 的 0.48×，A5000 768 GB/s ≈ 0.43×。
计划 §8 的 5090 口径为 15–20 min/组：

| 相 | 组数 ÷ 卡数 | 每组耗时 | 小计 |
|---|---|---|---|
| 充足档（L40S ×3） | 45 ÷ 3 = 15 组/卡 | ~40 min（含 dump I/O） | **11–13 h** |
| 受限档（A5000 ×5） | 45 ÷ 5 = 9 组/卡 | ~40 min | **~6 h** |

流水后关键路径 ≈ **13–15 h 纯跑**，含失败重跑按 **1.5–2 天**规划。
（比计划 §8 的单卡 2–3 天更快，靠的是 8 路并行。）

---

## 6. 本机特有风险

1. **L40S 供给恰好等于配额（3 张），且 node6 的 8 张 L40S 处于 DRAIN**。
   node5 另外 5 张被他人 allocate 但 `nvidia-smi` 显示 0 MiB 占用（空占）。
   → 建议向管理员确认 node6 能否恢复；恢复后充足档相可翻倍提速。
   → 3 张一旦拿到，用长时作业持有，不要频繁释放。
2. **登录节点无 nvidia 驱动** → 装包必须进 `srun`（见 §3.3），否则拿到 CPU 轮子。
3. **A5000 四节点驱动版本不一致**（535 / 550 / 580）→ 受限档全部钉死 **node4**，保证可复现；
   node4 同时是最闲的节点、本地盘 867G。
4. **`/zpool-00/data` 在计算节点是 `/data`**，路径不通用 → 脚本只用 `/home/cw636/...`。
5. **ZFS 池 93% 满且全所共享**，写入性能可能因碎片下降。
   真掉速时：dump 先写 node5 的 `/dev/shm`（504G）或本地盘，作业结束 rsync 回 `/home`。
6. **GPU 无 type** → 任何作业都要 `-w node5`（L40S）/ `-w node4`（A5000），
   只写 `--gres=gpu:1` 会拿到卡型不确定的资源。

---

## 7. 开跑前的待决项

无。计划 §6.2 的 **受限档容量 = 工作集 × 0.5**（KVFlow 口径）已由宸逸确认（2026-08-03），
即 §5.1 中 A5000 相 `num_gpu_blocks_override = 0.5 × W / block_size`。
