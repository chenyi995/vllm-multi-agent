# `kvpim/call.py` —— 统一调用单元（26 行）

> 术语见 [`README.md`](README.md)｜数据流全景见 [`CODE.md`](CODE.md)

## 这个文件是什么

整个仓库里最小的文件，只有一个 dataclass。**九类拓扑无论内部逻辑差多远，产出的都是
同一种 `Call` 流**，`runner.py` 只认这一种东西。

```python
@dataclass
class Call:
    agent_id: str            # 角色名 = 系统提示词的身份
    messages: list[dict]     # 聊天消息，由 runner 负责分词
    parent_idx: int | None = None   # DAG 边：本次调用消费了哪次的输出
    topology: str = ""       # 以下三项由 runner 填，拓扑不用管
    workflow_id: str = ""
    call_idx: int = -1
    meta: dict = field(default_factory=dict)   # 拓扑自定义标注
```

## 依据

**计划 §3 逐字规定了这个接口**：

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

存在的理由（计划 §3 原话）：**"所有拓扑产出同一种 `Call` 流，便于横向比较"**。
如果每个拓扑各写各的输出格式，`runner`、`analyze`、9 × 5 × 2 的矩阵就没法统一处理。

## 与计划的两点差异（我改的，需要你确认）

**① 字段顺序和默认值。** 计划里六个字段都是必填。我把 `topology` / `workflow_id` /
`call_idx` 给了默认值，因为**拓扑代码不应该知道自己是第几个 workflow**——那是 runner
的职责。拓扑只写：

```python
yield Call(agent_id="architect", parent_idx=0, messages=[...])
```

runner 在 `runner.py:275-277` 填上其余三项。

**② 多了一个 `meta` 字段。** 计划里没有。加它是因为 `parent_idx` **只能表达一条边**，
而 T3/T4 这类 fan-in 拓扑里一次调用会消费上一层的**全部**输出：

```python
# t3_fanout.py：layer 2 的 proposer_0 消费了 layer 1 的六个 proposer
Call(parent_idx=6,                          # 只能记第一个
     meta={"parents": [6, 7, 8, 9, 10, 11]})  # 完整的父列表在这里
```

`meta` 同时承载各拓扑自己的标注：T3 的 `layer`/`proposer`、T4 的 `round`/`agent`、
T7 的 `stage`/`iteration`、T1 的 `attempt`。**Tier 2 的结构不变量校验依赖这些标注**
（见 [`analyze.md`](analyze.md)）。

## 审查点

- `meta` 是自由 dict，没有 schema 约束。写错 key 不会报错，只会让对应的 Tier 2
  检查取不到值。目前靠各拓扑的 driver 自己保证一致。
- `parent_idx` 在 fan-in 场景下取的是"第一个父"，这个选择是任意的。真正用于重建
  拓扑的是 `meta["parents"]`。如果你希望 `parent_idx` 语义更严格（比如干脆置 None），
  说一声。
