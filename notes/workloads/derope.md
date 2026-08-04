# `kvpim/derope.py` —— 位置编码的逆变换（129 行）

> 术语见 [`README.md`](README.md)（尤其 §6 RoPE）｜数据流全景见 [`CODE.md`](CODE.md)

## 这个文件是什么

**整个实验的核心问题 Q1 就靠它。**

缓存里存的 K 已经被按位置旋转过（RoPE）。同一段文字出现在不同位置，K 的比特完全不同。
这个文件把旋转**反过来转回去**，还原成"不带位置"的 K，好回答：
**去掉位置之后，有多少块会变成真的相同？**

纯离线数学逆运算，**不需要重新跑模型，不改 vLLM 一行代码**。
计划 §4 原话：*"RoPE 前后不用跑两次……de-RoPE 是离线逆变换，运行成本 ×1"*。

## 正变换是什么（必须逐字对上 vLLM）

**依据**：`vllm/model_executor/layers/rotary_embedding/common.py:145-185`，
`ApplyRotaryEmb.forward_static` 的 neox 分支：

```python
x1, x2 = torch.chunk(x, 2, dim=-1)     # 128 维前后各半
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
output = torch.cat((o1, o2), dim=-1)
```

注意配对方式是 **NeoX 风格 `(j, j+d/2)`**（前半 vs 后半），
不是 GPT-J 风格的 `(2j, 2j+1)`（相邻两维）。**配错就全错，而且不会报错**——
计划 §9 sanity #1 专门提示了这一点。

## 逆变换（本文件的 `_rotary(..., inverse=True)`）

```
x1 = o1 * cos + o2 * sin
x2 = o2 * cos - o1 * sin
```

推导：把正变换写成旋转矩阵 `[[cos, -sin], [sin, cos]]`，逆就是转置（旋转矩阵正交）。
代数验证：
```
o1*cos + o2*sin = (x1 cos − x2 sin)cos + (x2 cos + x1 sin)sin
                 = x1 cos² − x2 sin cos + x2 sin cos + x1 sin² = x1  ✓
```

## cos / sin 怎么算（三个必须对齐的细节）

**① 频率表**，依据 `rotary_embedding/base.py:80-102` 的 `_compute_cos_sin_cache`：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
cos, sin = freqs.cos(), freqs.sin()
```

**② 模型参数**从 `config.json` 读，不是硬编码猜的：

```
Qwen3-4B-Instruct-2507:  head_dim = 128,  rope_theta = 5,000,000,  rope_scaling = null
```

`RopeParams.from_hf_config` 在遇到非空 `rope_scaling` 时**直接抛异常**，
不猜测该怎么处理——换模型时会立刻暴露，而不是静默算错。

**③ `cache_dtype=torch.bfloat16` 这个参数（容易被忽略的一处）**

依据 `rotary_embedding/base.py:58-63`：

```python
cache = self._compute_cos_sin_cache()   # fp32 算
if not self.use_flashinfer:
    cache = cache.to(dtype)             # ← 存成模型 dtype，即 bf16
self.register_buffer("cos_sin_cache", cache, persistent=False)
```

vLLM 实际用的 cos/sin 表是 **bf16 精度**的。所以我们也走一遍 bf16 舍入，
逆变换才是"它实际所做的那个变换"的逆，而不是"理想 RoPE"的逆。
实测确认：`rotary.cos_sin_cache.dtype = torch.bfloat16`。

## 验证（sanity #1）

在 `layers.0.self_attn.rotary_emb` 上挂 **pre-hook**（不是 post-hook——vLLM 的旋转是
**原地操作**，post-hook 拿到的 k 已经被转过了），同时抓真实的 pre-RoPE K 与 post-RoPE K：

```
cos(derope(k_post, pos), k_pre) = 0.9999999692     逐 token 最小 0.9999999254
```

**原判据"max abs err < 1e-2"实测不可达，已作废。** 理由：

```
|k_post| 最大 314，均值 2.35        ← attention-sink 式离群值
用本文件的 rope() 正变换去对 vLLM 自己的 k_post，误差同为 6.2e-2 量级
```

也就是 1e-2 这个绝对阈值**比 vLLM 自身的数值噪声还低**，物理上到不了。
而本实验的判定指标本来就是 cosine，绝对误差不是相关量。宸逸拍板改成 cosine 判据，
已写进计划 §9。

## 审查点

1. 逆变换在 **fp32** 里做（`work = k_post.float()`），结果再转回原 dtype。
   如果你希望全程 bf16 以完全复现引擎的数值路径，说一声。
2. `positions` 由 `dump.py` 记录，来自 `KVCacheBlock._block_hash_num_tokens`。
   **position 错一位，de-RoPE 全错且不报错** —— 这条链路的正确性见 [`dump.md`](dump.md)。
3. 本文件不处理 `rope_scaling`（YaRN / NTK 之类）。当前模型没有，换模型要先补。
