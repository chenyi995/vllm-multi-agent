# `kvpim/events.py` —— 订阅引擎播报的 KV 事件（157 行）

> 术语见 [`README.md`](README.md)｜数据流全景见 [`CODE.md`](CODE.md)

## 这个文件是什么

vLLM 可以把「哪个 block 被缓存了 / 被驱逐了」这些事件通过 zmq **主动播报**出来。
这个文件起一个后台线程订阅它，逐条写成 `blocks.jsonl`。

**这是我们唯一能看到缓存内部动态的通路**——`N_total`（存过多少块）、工作集 W、
驱逐了多少、sanity #4 的对账，全都建立在这条流上。

## 怎么打开这条流

在 `runner.py:139` 构建引擎时传：

```python
KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",
    endpoint=f"tcp://*:{port}",
    replay_endpoint=f"tcp://*:{port + 1}",
    topic="kv-events",
    buffer_steps=1_000_000, hwm=1_000_000, max_queue_size=1_000_000,
)
```

**依据**：`vllm/config/kv_events.py:11-48` 定义了这些字段。
后三个数我都调到 100 万——默认 `hwm=100_000` 是"排队超过这么多条就**开始丢**"，
本实验一组配置能产出几万到几十万条事件，用默认值有丢事件的风险。

## 线路格式

```python
sub.connect(f"tcp://localhost:{port}")
sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")
topic, seq_bytes, payload = sub.recv_multipart()
batch = Decoder(type=KVEventBatch).decode(payload)   # msgspec msgpack
```

**依据**：`vllm/distributed/kv_events.py:464` 发送端就是三帧：

```python
self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))
```

订阅样例见仓库自带的 `examples/features/kv_events/kv_events_subscriber.py`。
我直接 `from vllm.distributed.kv_events import KVEventBatch` 复用类型定义
（样例里是把类型抄一份，因为它假设订阅方是独立进程；我们同进程，直接导入更不容易错版）。

## 三种事件的语义（这一节最需要审）

| 事件 | 字段 | 含义 |
|---|---|---|
| `BlockStored` | `block_hashes` / `parent_block_hash` / `token_ids` / `block_size` | **新增**缓存的块 |
| `BlockRemoved` | `block_hashes` | 驱逐 |
| `AllBlocksCleared` | — | `reset_prefix_cache()` |

⚠️ **最关键的一条依据**：`BlockStored` **默认只代表新增**，不代表复用。

vLLM 里确实有一条"复用也发事件"的路径（`vllm/v1/core/kv_cache_manager.py:266-272`）：

```python
if (num_new_computed_tokens > 0
        and self.enable_kv_cache_events
        and getattr(request, "kv_cache_report_mode", "incremental") == "full"):
    ... emit_cached_block_events(...)
```

但它要求 `kv_cache_report_mode == "full"`，而**默认是 `"incremental"`**。
我们没有改这个默认值，所以流里的每个 `BlockStored` 都是一次真正的新增。

**这件事的后果**：`N_total` 直接数 `BlockStored` 里的块就是对的。
如果哪天有人把这个模式改成 `"full"`，`N_total` 会**静默高估**（一个块被命中十次
就会出现十次），而且不会有任何报错。我在冒烟数据上验证过：880 个块引用、
880 个唯一 hash，**零重复**，与"只发新增"一致。

## 丢消息怎么办

zmq 的 PUB/SUB 是**不保证送达**的（慢订阅者会被丢）。发送端给每条消息编了序号，
并提供一个 replay socket。`_replay_gap` 检测到序号缺口就去补：

```python
if self.last_seq >= 0 and seq > self.last_seq + 1 and replay:
    self._replay_gap(replay, decoder, seq)
```

`manifest.num_missed_events` 记录最终没补回来的缺口数。
**目前所有组都是 0。**

## 时间戳的坑（影响 sanity #4）

每条记录我写了三个时间：

- `ts` —— **引擎侧**的批次时间戳（`EventBatch.ts`）
- `recv_ts` —— 我们收到的时刻
- `seq` —— 发送端序号

⚠️ `ts` 是**发布线程**打的，不是事件发生的瞬间。实测它可能**晚于** driver 记录的
`t_start`，也可能因为发布延迟而让上一个 workflow 的事件落到下一段开头。
这个性质导致 `analyze.reconcile_events` 改了四次，最后不得不按容量档分成两种模式
（详见 [`analyze.md`](analyze.md)）。

**审查点**：如果你希望时间对齐更可靠，可行的方向是让 driver 在每次调用前后各发一条
自定义标记事件。我没做，因为那需要改 vLLM。

## 关停

```python
def stop(self, drain_s: float = 3.0):
    time.sleep(drain_s)      # 等在途事件
    self._stop.set()
    self._thread.join(timeout=10)
```

3 秒的 drain 是拍的，没有依据。如果某组配置的 `num_missed_events` 不为 0，
这里是第一个要怀疑的地方。
