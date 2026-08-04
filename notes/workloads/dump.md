# `kvpim/dump.py` —— 从显存里把 K/V 拷出来（157 行）

> 术语见 [`README.md`](README.md)｜数据流全景见 [`CODE.md`](CODE.md)

## 这个文件是什么

在**不改 vLLM 一行代码**的前提下，把缓存里的 K/V 张量拷到硬盘，供离线比对。

运行时缓存里是**全部 36 层**（和真实 serving 一模一样）；dump 是**只读旁路**，
每次调用结束后拷 5 层（0/8/17/26/35）。详见 [`README.md`](README.md) §10。

## 三个关键问题

### 一、怎么摸到张量

```python
worker = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker
runner = getattr(worker, "worker", worker).model_runner   # GPUModelRunner
tensor = runner.kv_caches[layer]
```

**依据**：`VLLM_ENABLE_V1_MULTIPROCESSING=0` 时，`EngineCoreClient.make_client`
返回 `InprocClient`（`vllm/v1/engine/core_client.py:306-317`），
`self.engine_core = EngineCore(...)` —— 引擎就在本进程内，属性直接可达。
`kv_caches` 声明在 `vllm/v1/worker/gpu_model_runner.py:562`。

计划 §7.2 就是这么设计的：*"driver 与 engine 同进程，直接读
`gpu_model_runner.kv_caches`"*。这条链路我用探针脚本逐级打印验证过。

### 二、K 和 V 怎么切开（最容易搞错的一处）

```python
def split_kv(layer_tensor, head_size):
    return layer_tensor.transpose(1, 2).split(head_size, dim=-1)
```

**依据**：`vllm/v1/attention/backends/flash_attn.py:1114`，**kernel 自己就是这么取的**：

```python
# (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
```

即：逻辑形状 `(块, head, token, 2×128)` → transpose(1,2) → `(块, token, head, 2×128)`
→ 末维**前 128 是 K、后 128 是 V**。

实测张量：`(5839, 8, 16, 256)` bf16，stride `(32768, 256, 2048, 1)`，非 contiguous。

**验证**：sanity 脚本在 rotary 模块挂钩子抓到真实的 post-RoPE K，
与 dump 出来的缓存内容逐 token 比对：

```
74 个 token 全比，max abs diff = 0.000e+00
```

**布局、K/V 顺序、position 映射三件事一次性验证通过**，block_size=16 与
prefix_match_unit=4 两档都过。

### 三、position 怎么算（第二个容易错的地方）

```python
size  = key_cache.shape[1]              # 物理块大小
end   = block.block_hash_num_tokens     # 该 hash 覆盖到的前缀长度（不含）
start = ((end - 1) // size) * size
block["positions"] = list(range(start, end))
```

**依据**：`KVCacheBlock._block_hash_num_tokens` 的注释
（`vllm/v1/core/kv_cache_utils.py:128-130`）：

> "Number of prefix tokens covered by `_block_hash`. For full blocks this is the
> full block boundary; **partial entries can end inside a cache block**."

**我在这里犯过一个真 bug**：原先写的是 `range(end - size, end)`，即用"结束位置减块大小"
反推起点。对满块没问题，但 `prefix_match_unit=4/8` 时 hash 可以在物理块**内部**结束
（比如块覆盖 16..31，hash 只到 20），`20 - 16 = 4` 会让整块 position 前移 12 位。

**后果：de-RoPE 全错，而且不会报任何错。** 改成按块边界向下取整后修复，
并在 4 档上重新验证（`max abs diff = 0.000e+00`）。

## 去重键：为什么按 hash 不按 block_id

```python
fresh = [b for b in cached_blocks(pool) if b["block_hash"] not in self.seen]
```

块被驱逐后重算存回会拿到**新的 `block_id`**，按 id 去重会把同一个逻辑块 dump 两次、
`N_total` 重复计数。按 hash 去重则正确——同一 workflow 内同 hash 就是同一个逻辑块。

`seen` 在每个 workflow 开始时清空（`start_workflow`）：

> 跨 workflow 的同内容块是**真正独立的物理块**（中间清过场），
> 正是我们要数的重复对，不能去掉。

## 增量 dump，不是全量

`dump_call` 每次只写**本次调用新增**的块。理由：

- 全量写会让共享前缀被反复写盘，T3 一组配置会膨胀十几倍
- 一个块被创建后到被驱逐之间，**至少隔着一次调用边界**（单次调用最长约 8k token，
  远小于 10 万的池子），所以增量不会漏

## 写盘格式

```
dumps/<workflow_id>/<call_idx:04d>_<layer:02d>.safetensors    张量 k / v
dumps/<workflow_id>/<call_idx:04d>_meta.json                  该次新增块的元数据
```

`_meta.json` 里每块记 `block_id` / `block_hash` / `num_tokens` / `positions`，
外加 `block_size` / `num_kv_heads` / `head_size`。
**`positions` 丢了 de-RoPE 就废了**（HANDOVER §6 明确列为必带项）。

## 审查点

1. `cached_blocks` 扫的是 `pool.blocks` 全表（几千条），每次调用扫一遍。
   之所以不用 `cached_block_hash_to_block`，是因为实测那个结构**不可遍历**
   （只有 `contain` / `get_one_block` / `insert` / `pop`）。
2. `head_size` 默认写死 128。当前模型是 128，换模型要改。
   更稳的做法是从 `kv_cache_spec` 读，我没做。
3. 只 dump 5 层带来的**上界偏差**：真要去重两个块，36 层都得相同；
   只查 5 层可能放过在未采样层其实不同的块对。所以"5 层全判同"是"36 层全判同"的
   **上界**。宸逸决定不做 36 层对照实验，该偏差如实记录。
