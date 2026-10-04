# Metal prefill 和 decode 耗时与空间换时间研究

2026 年 10 月 4 日，Apple M2 Max，38 核 GPU，96 GiB 统一内存，macOS 26.5，接电、Automatic 模式。模型为当前 IQ2_XS，48 层，每层 512 个专家中激活 10 个，FP16 KV，prefill chunk 1024，容量 32768。实验保留当前模型、量化、专家数量和采样设置。

本轮确认了耗时大头，并在 decode 的专家 down 投影中验证到小幅整段收益。新内核将当前模型的 2560 × 640 尺寸交给编译器，保留原始量化权重和逐输出运算次序，不增加权重或 scratch 内存。其余密集投影、hyper-connection 和 prefill 候选没有确立可采用的收益。三种权重缓存不启用；混合专家 tile 保留实验补丁。详见下文的分项与整段对照。

## 实际整段耗时

日常 `engine/strata` 处理冻结的 10000 token 输入，生成 128 token，连续请求三次。关闭逐内核 profiler，保留现有 decode 计时和 command buffer 时间诊断。计时不含模型加载、tokenizer、HTTP 和图片编码。

| 请求 | 输入处理 | 生成 128 token | 合计 | 输入复用 |
| --- | ---: | ---: | ---: | ---: |
| 第一次 | 37.194 秒，268.9 tok/s | 4.390 秒，29.16 tok/s | 41.584 秒 | 0 |
| 第二次 | 0.256 秒 | 4.369 秒，29.30 tok/s | 4.625 秒 | 9993 token |
| 第三次 | 0.246 秒 | 4.365 秒，29.33 tok/s | 4.610 秒 | 9993 token |

第一次请求的 prefill 占 89.4%。这种负载中，prefill 节省 1% 约是 372 毫秒，decode 节省 1% 约是 44 毫秒。复用请求则是 decode 占约 94.6%。需要按实际使用场景排序，不能把短续写的输入处理速度作为完整 prefill 的优化目标。

decode 现有阶段计时约为每 token 34.1–34.3 毫秒：verify 34.0–34.2 毫秒，commit/emit 0.06 毫秒，draft 0。commit/emit 全部消除也只节省约 0.18%。它不是本轮优先项。三个请求的输出 ID 一致，专家命中率均为 100%，没有 CPU 专家计算或运行期专家文件读取。当前设备已经将全部 24576 个原始量化专家放在 GPU 可访问缓存中。

command buffer 诊断的稳态 prefill GPU busy 为 98.2%，decode 为 97.0%–98.3%。首组 prefill 缓冲包含初始化和捕获间隔，不能用其 79.7% 推断整段有 20% CPU 空闲。GPU busy 也不等于带宽或算力利用率。

## GPU 内核时间分布

另外运行逐内核 profiler 定位大头。prefill CSV 已落盘部分包含 22454 次直接 launch，合计 34.219 秒；decode 合并 verify 与 commit 两条图的每项平均时间，合计 36.465 毫秒，1841 项。以下占比是内核采样时间的占比。

| Prefill 类别 | 已落盘采样累计 | 占比 |
| --- | ---: | ---: |
| 专家 GEMM | 13.553 秒 | 39.6% |
| 密集 GEMM | 11.324 秒 | 33.1% |
| Prompt attention | 4.749 秒 | 13.9% |
| 其他 | 2.060 秒 | 6.0% |
| 独立反量化 | 1.523 秒 | 4.5% |
| GDN recurrence | 1.010 秒 | 3.0% |

密集 GEMM 中 FP16 是 7.454 秒，BF16 是 3.870 秒。专家 gate/up 和 down 内核已经将量化权重按 tile 解码后直接送进矩阵乘。其内部解码包含在“专家 GEMM”中；表里的“独立反量化”不能代表全模型所有解码成本。

| Decode 类别 | 每图迭代采样累计 | 占比 |
| --- | ---: | ---: |
| 密集量化投影 | 10.453 毫秒 | 28.7% |
| 专家 gate/up 与 down | 9.111 毫秒 | 25.0% |
| Hyper-connection | 6.937 毫秒 | 19.0% |
| QSA attention 与 indexer | 2.355 毫秒 | 6.5% |
| GDN，包括 conv 与 commit | 1.742 毫秒 | 4.8% |
| BF16 其他投影 | 1.362 毫秒 | 3.7% |
| 激活量化 | 1.344 毫秒 | 3.7% |
| 其他 | 3.161 毫秒 | 8.7% |

profiler 给每项增加 pass 边界和采样开销。该次生成 64 token 耗时 3.317–3.397 秒，明显慢于正常路径，不能用于计算 tok/s 收益。prefill 的 direct CSV 每 32 个已完成缓冲落盘，最后不足 32 个缓冲未必被保存；还混有初始化直接 launch，且不采样 blit copy。以上是定位大头的采样分布，并非完整请求的无开销时间账。应结合上一节整段时间看，不能把某个采样占比直接当作端到端可节省比例。

## Decode：投影、专家与 hyper-connection

三项合计约占上述采样时间的 72.7%，是本轮 decode 的主要研究对象。以下微基准使用生产内核生成变体，保持同一 lane 的浮点累加次序、量化 scale 步骤和 XOR 归约。数据来自轮换的权重或专家，避免只反复测一小块热权重；仍不能替代真实模型的整段对照。

| 对象 | 原路径与候选 | 微基准结果 | 决定 |
| --- | --- | --- | --- |
| GPU 密集 IQ4 投影 | 固定 K=2560/6144，保持四组虚拟 warp 的原累加链 | 五种实际矩阵形状中约快 0.5%–1.7%；词表投影 1149.06 → 1140.78 微秒 | 不启用，没有完整 decode 收益证据 |
| 专家 gate/up | 固定 2560 × 640 | IQ2_S 117.65 → 116.42、IQ2_XXS 110.87 → 109.49 微秒 | 不启用，收益很小 |
| 专家 Q2_0 down | 固定 2560 × 640，仍每 warp 处理四行 | 两次 65.89/66.06 → 50.02/50.33 微秒，耗时约降 24% | 进入完整模型验证；代码已实现 |
| 专家 down 行复用 | 固定尺寸后改为每 warp 2/8/16 行 | 57.84/51.20/52.22 微秒，均慢于四行 | 保留四行 |
| Hyper-connection down/up | 调整 4/8/16 个 SIMD-group，up tile 8/16/32 列 | down 基本不变；up 最好 25.02 → 24.29 微秒。原布局重命名的变体也有约 1% 差异 | 不启用，没有完整 decode 收益证据 |
| Hyper-connection norm＋down 融合 | 每个 down 工作组重复原 norm，分两段暂存 20 KiB，保持 dot8 顺序 | norm＋down 33.50 → 62.18 微秒；包含 up 的完整链 58.85 → 85.69 微秒 | 放弃本融合方案 |

专家 down 的输入宽度 640 对应 20 次 32 元素调用，少于 SIMD-group 的 32 个 lane。固定尺寸允许编译器简化动态循环和边界/地址计算；这只是从代码与尺寸提出的解释，本轮没有 ISA 或指令计数证明具体节省了多少条指令。内存格式、激活 Q8_1、整数 scale、逐 lane float 累加和最终归约全部沿用原 helper。

只在 `d_type=42`、`n_embd=2560`、`n_ff=640` 且原 direct-dot 路径启用时选择新内核。其他尺寸、量化类型和关闭 direct-dot 的情形继续调用原内核。`STRATA_METAL_DECODE_DOWN_DIMS=0` 可回到动态尺寸内核，默认开启。该变化也适用于 resident 接口的多 token 路由容量；batched prefill GEMM 路径不变。

### 专家 down 的完整 decode 对照

隔离实验二进制 SHA-256 为 `a1ff389079df6b7a39012af2985a33d412e3d0db1bf9a92bf54e0e1f7666c0ce`。每个进程处理同一冻结的 10000 token 输入，生成 256 token，先一次完整 prefill，再三次相同输入复用。没有逐内核 profiler，所有专家命中、没有 drafts。两组顺序分别为关闭→开启、开启→关闭。

| 配对 | 关闭：三次复用请求 decode | 开启：三次复用请求 decode | 中位耗时减少 / tok/s 增加 |
| --- | --- | --- | --- |
| 第一组 | 9.0615、9.0609、9.1057 秒 | 8.8079、8.8703、8.8805 秒 | 2.11% / 2.16% |
| 反向第二组 | 9.0038、8.9100、8.9098 秒 | 8.8573、8.8731、8.8699 秒 | 0.45% / 0.45% |

两组都观察到正收益，但幅度随同期环境变化，不能只选第一组宣称稳定提高 2.2%。分别为约 0.747 和 0.157 毫秒/token。两个开关的进程 footprint 均约 40.2G、峰值 40.3G，swap 均保持 2464.88 MiB；没有新增持久权重或 scratch 分配。

随后用完整生产构建重测，仍是同一二进制只切换开关。`build-metal/strata` 的 SHA-256 为 `1c137ffcb7d0ebe324735e6c11409acedec94d5b9a8af68fba7e17a3dec94900`，每侧一次完整 prefill 后五次输入复用：

| 开关 | 五次复用请求 decode | 中位耗时 | 中位 tok/s |
| --- | --- | ---: | ---: |
| 关闭 | 9.0772、9.0251、9.1452、9.0732、9.0631 秒 | 9.0732 秒 | 28.22 |
| 开启 | 8.8561、8.8323、8.8523、8.8566、9.0087 秒 | 8.8561 秒 | 28.91 |

该组中位 decode 耗时减少 2.39%，tok/s 提高 2.45%，约节省 0.848 毫秒/token；保留最后一次较慢的开启样本。复用请求的 prefill＋decode 合计中位耗时从 9.8822 降到 9.6731 秒，节省约 209 毫秒，仍不含 tokenizer、HTTP 等外部耗时。三组的 decode 速度收益范围是 0.45%–2.45%，只用于说明本次会话的测量波动，不是统计置信区间或其他设备的保证。

完整构建的首次请求，关闭为 prefill 35.9654 秒＋decode 9.1407 秒，开启为 36.2907＋9.1774 秒。首次整段分别为 45.1061/45.4681 秒，**未证明首次长输入请求加速**。全部六个进程、28 次请求的 256 个输出 ID 相同；内存 footprint 与 swap 没有可见增长。`decode/summary.json` 保留每组的冷/复用请求时间与输出哈希，`stage-runs.json` 另保留启动和配置。本轮已构建并测试新路径，日常 `engine/strata` 和配置未替换。

### 数值检查

专门的边界检查比较了 1424820 个 float 输出，零位差：包含统一/非统一 slot offset、重复/无效/不 resident 的路由、多 token、不同 cap、负零/极端/subnormal/Inf/NaN scale，以及非标准尺寸回退。它验证的是测试输入，不能称为全部输入的穷尽证明。

真实模型在同一实验二进制开关两侧运行 10K 输入、64 token 生成和 cached follow-up。99 个位置、24583680 个完整词表 logits 全部有限且逐位相同；输出 ID 和两个请求提交的 GDN、KV、PLE 等状态哈希一致，没有 drafts。follow-up 请求上限 24 token，实际在第 10 个 token 遇到 EOS，两侧相同。`decode/audit-comparison.json` 保留逐窗口结果。该测试覆盖当前 IQ2_XS 与 10K 输入，没有重新审计 128K 和视觉输入。

完整生产构建的 `metal_direct_dot_test`、`metal_mmvq_decode_test`、`iq_parity`、`iq_multi_parity` 四项通过；direct-dot 测试合计比较 51516951 个输出，零位差。GR 融合微基准的 93464 个中间值和输出也零位差，但因变慢而不进入生产路径。

## 第一性原理判断

每个依赖阶段的乐观下界是 `max(必须移动的字节 / 有效带宽, 必须执行的运算 / 有效算力)`。阶段之间有依赖，完整耗时还包含量化解码指令、归约、同步、padding 和调度。把整个模型的字节数除以设备峰值，只有一个宽松下界，不能作为可实现的速度承诺。

从实际 GGUF 和 native pack 的尺寸统计，单 token 激活专家的原始权重约 0.692 GB，密集量化矩阵约 1.884 GB，pack 中 BF16 权重约 1.528 GB。合计约 4.105 GB/token，不含 KV、状态、激活、重复事务和缓存命中差异。密集矩阵约 8.62 GFLOP/token，专家约 4.72 GFLOP/token，仍不含 attention 与递归操作。

M2 Max 的标称统一内存带宽是 400 GB/s。[Apple 硬件说明](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/) 本轮 256 MiB GPU copy 的读写合计约 361 GB/s。4.105 GB/400 GB/s 得到约 10.26 毫秒，仅是权重清单的理想下界，不能推出当前 34 毫秒能降到 10 毫秒。缺少实际 DRAM 事务、缓存和指令吞吐计数，不能判定剩余时间全是带宽浪费。

prefill 每个 1024-token chunk 产生约 10240 次专家路由，若覆盖 512 个专家，平均每个专家仅约 20 行。重复解码 tile 确实可能昂贵，但持久展开权重也会增加读流量和工作集；是否值得必须比较“解码加量化读取”与“缓存读取加构建”，不能只看少执行了多少次解码。

Prompt attention 使用按块 online softmax，已经避免完整 score 矩阵。QSA 使用 top-2048 稀疏选择，选中 KV 数量上限约 2051；indexer 扫描随上下文增长，因此长上下文应另测，不能将它视作始终完整的二次方 attention。本轮没有重新 profile 128K。

FlashAttention 的 IO 分块和 GatedDeltaNet 的 chunkwise 并行是可研究方向，但数学等价并不保证浮点逐位相同。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)、[GatedDeltaNet 论文](https://arxiv.org/abs/2412.06464) 对本项目没有本轮实测收益；GDN recurrence 当前只有约 3%，应排在专家与密集 GEMM 之后。

## 空间换时间实验

所有 GPU 实验串行执行。以下“输出一致”只表示相应测试通过，不是对全部输入的能力评测。

| 方案 | 内存代价 | 测量结果 | 决定 |
| --- | --- | --- | --- |
| 专家持久 FP16 视图 | 全部专家 225 GiB；单层 gate/up 3.125 GiB，down 1.5625 GiB | 512 专家合成路由微基准中，缓存已建好仍慢约 15%–58%，构建成本另计 | 放弃本布局 |
| IQ2 有符号 nibble 视图 | 140 字节/block，原格式 82/66；全模型额外约 17.00 GiB | IQ2_S 从 117.94 增至 155.73 微秒，IQ2_XXS 从 117.62 增至 161.75 微秒；包含 128 轮换专家、每次激活 10 个 | 放弃本布局 |
| 密集 prefill FP16 视图 | 实际 5580 MiB，304 矩阵 | 同一实验二进制第一组完整 prefill 从 38.50 增至 41.15 秒；第二组对照发生大波动，不能确认收益 | 不启用 |
| 每专家混合 16/32 行 tile | 少量 host descriptor，无额外权重缓存 | 专家 GEMM 微基准耗时下降 5.2%–9.4%；完整模型 A/B 不稳定 | 保留实验补丁，不启用 |

专家 FP16 微基准保持原 dequantizer、同一组半精度操作数和逐输出 MMA 次序，检查 324834560 个 float 输出，零位差。nibble 微基准检查 1305600 个输出，零位差，非法 code 数为零。这里的路由是合成输入，不能直接代表真实模型或全部浮点边界情况；负结果只否定本轮布局与路径。

密集缓存使用与原路径相同的 `dequant_f16`，缓存 immutable 权重，实际 2736 次命中、304 次构建。物理 footprint 峰值约从 40.3G 增至 45.7G，采样 swap 未增加。两组完整 prefill 的全部数据如下，保留慢样本，不用失真的中位数宣布加速：

| 密集缓存配对 | 关闭 | 开启 |
| --- | ---: | ---: |
| 第一组 | 38.495 秒 | 41.154 秒 |
| 第二组 | 61.874 秒 | 42.102 秒 |

混合 tile 的真实模型测量如下。同一二进制只切换实验环境变量；内存 footprint 均约 40.2G。运行期间系统 swap 增加约 53.5 MiB，GPU/CPU 的共享资源也未被隔离，原因不能由 swap 单独解释：

| 混合 tile 配对 | 关闭 prefill / decode | 开启 prefill / decode |
| --- | ---: | ---: |
| 第一组 | 38.276 / 4.504 秒 | 52.145 / 4.640 秒 |
| 第二组 | 87.441 / 5.715 秒 | 49.655 / 5.897 秒 |

实验参数不修改 decode，而 decode 也明显波动。不能用第二组表面的 43% prefill 缩短宣布成功。四个混合 tile 请求、十二个密集缓存请求的 128 个输出 ID 均与日常基线相同，但未做完整 logits 和提交状态对照，因为没有确立可采用的端到端收益。

## 内存预算如何随设备决定

本机物理内存为 96 GiB，Metal 返回的 `recommendedMaxWorkingSetSize` 为 77.760 GiB。runtime 的 `cudaMemGetInfo` 返回这个预算减 Metal 已分配量，**不代表 macOS 当前实际可用内存**。CPU、GPU、模型 mmap 和其他应用共享物理内存，预算必须同时考虑设备和系统。

未来缓存可用预算应取以下余量的最小值，并保证非负：

1. 设备工作集预算减已分配量，再减目标上下文的 KV 增量、prefill scratch 增量、视觉峰值和设备余量。
2. OS 安全可用内存估计，再减同样的未来增量和系统余量。不能把所有 inactive/file-backed 页都当成可用空间，尤其不能重复计算仍在使用的模型页。
3. 用户或自动设备策略的上限。

已分配的最大上下文 KV 和 scratch 不重复扣除。独立显卡则分别核算 VRAM 与系统 RAM，并实测 PCIe/file tier 的成本。初始化后要继续监测压力，支持低收益缓存驱逐、分配失败回退原路径，避免固定给所有设备增加几 GiB。

本轮密集缓存原型只做了 `min(请求上限, GPU 工作集预算/12, 当前 GPU 余量/4)` 的初始化约束。申请上限 6144 MiB，实际使用 5580 MiB；它没有 OS 压力驱逐，不能当作已完成的生产自动内存策略。225 GiB 的全专家 FP16 方案在预算阶段就不适合此设备；17 GiB 的 nibble 方案即使放得下，也因变慢而不值得占用。

缓存应按单位内存能节省的实测时间排序：`(命中次数 × 每次节省时间 - 构建和管理时间) / 新增字节`。如果每次节省时间为负，或者只是优化很短的小项，内存再空也不应自动扩大缓存。

## 后续优先级与验收条件

首先研究专家 GEMM 的实际路由分布、padding、weight tile 复用和解码指令隐藏。混合 tile 需要先捕获真实每专家行数，再确认分项下降；合成指数路由的 5%–9% 不能外推。按这次采样，专家 GEMM 耗时降低 10% 只对应约 4% 的 prefill 内核总时间，是 Amdahl 估算，不是预测。

第二项是密集 GEMM 的实际形状、重复读 tile、寄存器压力与流水。应保持 BF16 与 FP16 的原有操作数和累加次序；简单把 BF16 换 FP16 没有质量保证。第三项是 attention 的 K/V 载入复用和 softmax staging，在保持选中 token、归约次序与 precision 的前提下实验。GDN recurrence 和独立反量化优先级随后。

decode 的专家 down 尺寸特化已有小幅整段收益，随后应继续研究 IQ4/IQ3 密集投影、专家 gate/up 和 GR 投影的 load/decode 指令及访存组织。已经全部 resident 的专家缓存不能靠更多 residency 加速。图外 CPU/提交和短输入尾段应排在上述 GPU 大项之后。现有小幅布局变体和重复 norm 的融合均不进入默认路径；仍需要实际 DRAM 与指令计数判断更大的剩余空间。

任何进入默认路径的候选都需要在同一二进制、相同电源模式、相同输入和配置下，证明完整阶段时间收益超过同期波动；再用相同输入比较完整 logits、输出 ID、提交的 GDN/KV 状态和 cached follow-up。10K 通过不代表 128K 已验证。模型、量化精度、激活专家数、KV 精度、上下文、采样均保持当前设置。

## 证据和复现

仓库 HEAD 为 `e0307a0`。日常引擎和已有 build 对象的二进制不同，因此性能结论只使用同一实验二进制的开关对照。`stage-runs.json` 保留配置、引擎 SHA、整段计时、输出 ID 哈希、内存采样及 swap；`prefill-kernels.csv` 与 `decode-kernels.csv` 保留内核聚合数据；`summary.json` 保留权重清单和分项统计。原始日志位于被 git 忽略的 `run/`。

从仓库根目录运行，GPU 测试逐个执行：

```bash
.venv/bin/python bench/results/2026-10-04-metal-first-principles/analyze.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/analyze_decode.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_micro.py
/tmp/strata-first-principles/expert_cache /tmp/strata-first-principles/expert_cache.metallib 5 512
/tmp/strata-first-principles/expert_nibble /tmp/strata-first-principles/expert_nibble.metallib
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_expert_tiles.py
/tmp/strata-first-principles/expert_tiles /tmp/strata-first-principles/expert_cache.metallib 5 512
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_dense_cache.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_mixed_tiles.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_decode_micro.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_decode_engine.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_decode_check.py
.venv/bin/python bench/results/2026-10-04-metal-first-principles/build_gr_fusion.py
```

实验构建生成 `/tmp/strata-first-principles/` 内的对象和二进制。`dense-cache.patch` 与 `mixed-tiles.patch` 是未应用的实验补丁；`decode/down-dims.patch` 保存最初的 opt-in 原型，生产代码已经实现新内核并默认启用。`run_dense_cache.py` 可选择引擎、生成 token 数、重复次数、缓存上限和输出目录；混合 tile 使用 `STRATA_METAL_PREFILL_MIXED_TILES=0/1`。`run_audit.py` 复用上一轮 logits/state 检查驱动，使用当前模型路径；本轮已执行专家 down 的完整词表与状态对照。既有诊断存在尾部未落盘和初始化混入的问题，后续若要精确评价小幅分项收益，应先补齐最终 flush 并限定请求采样范围。

完整构建对照可从仓库根目录串行复现，使用新的结果目录名：

```bash
cmake --build build-metal -j 8
ctest --test-dir build-metal --output-on-failure -R '^(metal_direct_dot_test|metal_mmvq_decode_test|iq_parity|iq_multi_parity)$'
STRATA_METAL_DECODE_DOWN_DIMS=0 .venv/bin/python bench/results/2026-10-04-metal-first-principles/run_dense_cache.py --engine "$PWD/build-metal/strata" --tokens 256 --repeats 6 --output /tmp/strata-down-off-new
STRATA_METAL_DECODE_DOWN_DIMS=1 .venv/bin/python bench/results/2026-10-04-metal-first-principles/run_dense_cache.py --engine "$PWD/build-metal/strata" --tokens 256 --repeats 6 --output /tmp/strata-down-on-new
```
