# IQ3_S：2K chunk 的继续研究，2026-10-09

Apple M2 Max（38 核 GPU、96 GB）、macOS 26.5，接电源并保持 Low Power Mode。后台任务继续运行。
上下文容量为 **262,144**，实际输入 **30,000 token**，prefill chunk **2048**。IQ3_S 原权重、48 层、
每 token 10 个专家、FP16 KV、24,576 个专家全部驻留；没有 prefix reuse 或 drafts。

按用户选择，**细粒度专家尾块成为默认路径**：32 行 descriptor 根据有效行数使用 8 / 16 / 24 / 32 行 tile。
保留 `STRATA_METAL_PREFILL_MOE_TAIL=0` 回退开关。没有新增 GPU 全局缓冲区；每线程组仍为 256 个线程、
8 KiB 暂存。2 GiB 密集权重缓存没有取得稳定收益，已从生产代码移除。

**当前未确认稳定的整段 prefill 加速。** 微基准有收益，但完整请求的首次改善没有在重复测试中保持。
不要把本轮写成“再提高 14%”，也不要用上一轮较快的绝对值代表当前后台负载。

## 当前基线与候选的完整请求

每个配置使用一个独立引擎，依次读取相同的 A / B 两条新提示。A 是首次请求；B 在开头插入编号并等量移除 filler，
保持 30K 且 `reused=0`。固定语料、提示 SHA 与上一轮一致。启动时间不计入 prefill；请求中不运行 profiler、
数值审计或周期性内存采样，结束后只读取一次保存的峰值。

| 顺序 / 版本 / 方案 | A：prefill 秒 / tok/s | B：prefill 秒 / tok/s | 峰值 footprint |
| --- | ---: | ---: | ---: |
| 1 / V1 / 旧路径 | 170.124 / 176.34 | 163.013 / 184.03 | 61.8G |
| 2 / V1 / 粗粒度 16、32 行 | 185.013 / 162.15 | 160.213 / 187.25 | 61.6G |
| 3 / V1 / 粗粒度 + 2 GiB 密集缓存 | 179.728 / 166.92 | 162.206 / 184.95 | 63.5G |
| 4 / V2 / 细粒度 8、16、24、32 行 | 183.685 / 163.32 | 145.010 / 206.88 | 61.6G |
| 5 / V2 / 旧路径反向对照 | 188.748 / 158.94 | 165.549 / 181.22 | 61.6G |
| 6 / V2 / 细粒度重复 | 177.349 / 169.16 | 166.224 / 180.48 | 61.8G |

V2 采用同一二进制切换 tail 开关。第 4/5 次的 B 吞吐相差 **+14.16%**，第 6/5 次则为 **−0.41%**；
A 同时改善约 2.8% / 6.4%，但旧路径本身的 A 已从 170.1 变为 188.7 秒。后台 CPU/GPU 负载与 swap 在变化，
这组数据不足以把改善稳定归因于细粒度内核。两次细粒度 B 为 **180.48 / 206.88 tok/s**。
全部 12 个输出的 token ID 相同，校验字段均 3/3；这不能替代完整数值审计。

缓存增加约 1.9G 峰值 footprint，B 反而比相邻粗粒度对照慢约 1.2%。缓存开关仅在研究原型存在，
最终默认引擎不包含该缓存。各次配置、二进制 SHA、结果与 swap 状态在 [summary.json](summary.json)。

## 从耗时大头推导方案

另外对当前 2K 旧路径做了一次独立诊断，带 profiler 的采样分布如下：

| 类别 | 采样时间占比 |
| --- | ---: |
| 专家 GEMM（含 tile 内反量化） | 36.69% |
| 密集 GEMM：FP16 / BF16 | 23.14% / 9.92% |
| attention | 16.96% |
| 独立反量化 | 3.30% |
| GDN | 3.74% |
| 其它 | 6.25% |

原始聚合见 [profile-summary.json](profile-summary.json)。这是采样比例，不能直接相加为无开销请求的墙钟账；
`host grouping` 还包括同步等待，command-buffer 时间比例也不能当作 GPU 利用率。

主要专家矩阵的必要运算量没有减少。可以减少的是路由分组最后一块的 padding：一个完整 2K chunk 有
20,480 条路由记录，而诊断中约 900 个 32 行 descriptor 名义上要计算 28,800 行，比有效行多约 40.6%。
细粒度只减少尾块的补零行运算；满 32 行仍用旧几何。反量化调用、half 输入、FP32 累加及每个输出的
8-wide MMA 顺序保持一致。线程组统一选择分支，不增加 host dispatch 或全局缓冲区。

下列合成路由微基准每个配置跑 7 轮、每轮 5 次，交替顺序后取中位数；所有输出先与原 16 行路径逐位比较，
所列结果零差异。来源 [micro/rows-fine.txt](micro/rows-fine.txt)：

| 实际约 36–41 行/专家 | 固定 32 行 tile | 细粒度 tile |
| --- | ---: | ---: |
| type 18 gate/up | 55.64 ms | 46.92 ms |
| type 21 gate/up | 50.88 ms | 42.53 ms |
| type 22 gate/up | 68.61 ms | 62.87 ms |
| type 23 gate/up | 62.55 ms | 56.43 ms |
| type 20 down | 23.38 ms | 20.97 ms |
| type 42 down | 32.21 ms | 27.29 ms |

这是合成路由的局部收益，不能直接推广为模型吞吐。把尾行判断放进 MMA 内循环的另一方案反而更慢，
例如 type 21 约 37 行时细粒度 44.07 ms、循环判断 64.69 ms，已排除；见 [micro/rows-loop.txt](micro/rows-loop.txt)。

空间换时间也要计算省下的成本：当前独立反量化约占采样的 3.3%，全部密集 FP16 视图需要约 5.42 GiB，
本轮仅缓存 2 GiB，不能消除密集 GEMM 自身约三分之一的采样耗时。缓存额外内存没有换来可重复的请求收益。

## 权重读取候选：10K 筛选未确认收益

只在微基准中把专家权重指针的 Metal address space 从 `constant` 改为 `device`，保留解码字典与算术。
同样 32 行几何、7 轮中位数、零逐位差异：gate/up type 18 为 45.61 → 38.04 ms，21 为 54.18 → 44.44，
22 为 53.62 → 47.41，23 为 45.01 → 38.60；down type 20 却为 23.51 → 23.92 ms。
见 [micro/weights.txt](micro/weights.txt)。这只有局部收益，没有并入默认引擎。

与细粒度叠加后的另一轮微基准，fine → device fine：gate/up type 18 为 32.95 → 29.37 ms，21 为 44.23 → 37.60，
22 为 42.72 → 36.27，23 为 40.05 → 35.24；down type 20 为 20.99 → 22.53，type 42 为 27.35 → 26.64。
仍为 7 轮中位数、零逐位差异，见 [micro/weights-fine.txt](micro/weights-fine.txt)。研究模型候选仅切换四种 gate/up，
down 保持原路径。源代码保留为 [prototype-weight-device.patch](prototype-weight-device.patch) 和所引用的
[prototype-weight-device.metalh](prototype-weight-device.metalh)，未应用到默认生产代码。

GU 占上述诊断采样约 24.78%；即使全部 GU 缩短 15%，其它时间保持不变，Amdahl 估算整段只加速约 **3.86%**。
局部收益不能等比例外推。30K 达到 220 token/s 需要 prefill ≤ **136.364 秒**。若“220K/s”字面指 220,000 token/s，
仅专家的必要乘加工作量 `4,718,592,000 FLOP/token` 就对应约 **1.038 PFLOP/s**，还未包括密集投影、attention 等。
本轮没有达到这个字面目标的证据。用户随后要求先以 10K 实际输入筛选，确认提升后再复测 30K；两者容量都保持 256K。

同一候选二进制，细粒度一直开启，仅用 `STRATA_METAL_PREFILL_MOE_WEIGHT_DEVICE=0/1` 切换四种 gate/up。
各处理两条相同的 10K 新提示，计时方法仍不含周期性内存采样、profiler 或审计：

| 10K 筛选 | A：首次 prefill 秒 / tok/s | B：另一条完整新输入 秒 / tok/s | 峰值 footprint |
| --- | ---: | ---: | ---: |
| 当前细粒度基线 | 67.487 / 148.18 | 42.835 / 233.46 | 61.6G |
| 细粒度 + device gate/up | 61.236 / 163.30 | 42.819 / 233.54 | 61.7G |

B 只快 **0.016 秒 / 0.037%**，可视为持平。A 快约 10.2%，但此前已经观察到明显首次请求及后台负载波动，
且没有后续完整请求收益，不能确立稳定提升。输出 token ID 相同，均无 prefix reuse/drafts，字段均 3/3；
候选的独立 GEMM 边界对照通过（4.41 秒），但未做该候选的完整模型 logits/state 审计。
未启用此候选，也未进入 30K 复测。完整数据见 [screen-10k.json](screen-10k.json)。

10K 的 B 已超过 220 token/s；这不能推广为 30K，因为 attention 的历史长度与首次请求成本不同。
30K 稳定达到 220 token/s 尚未确认。此前已开始的 30K 候选对照按用户要求在模型加载阶段中止，未计入结果。

## 更大密集投影 tile：筛选后排除

保持原 half/float 操作数、FP32 累加与升序 8-wide MMA，只扩大输出 tile，以共享更多输入或权重。
FP16 测 64×128、128×64、128×128；BF16 测 64×128、128×64，GPU 线程组暂存从当前 FP16 的 9 KiB /
BF16 的 18 KiB 增至最高 28 KiB。测试时检查实际 pipeline 线程上限及设备线程组内存上限；所有输出逐位相同。
模型已退出后跑 7 轮、每轮 5 次，交替顺序，中位数见 [micro/dense.txt](micro/dense.txt)：

| dtype / M,N,K | 当前 64×64 | 最快较大 tile |
| --- | ---: | ---: |
| FP16 / 2048,10240,2560 | 17.002 ms | 17.546 ms |
| FP16 / 2048,2560,6144 | 8.115 ms | 8.352 ms |
| FP16 / 2048,6144,2560 | 11.110 ms | 10.737 ms |
| FP16 / 2048,12288,2560 | 21.807 ms | 21.805 ms |
| FP16 / 2048,640,2560 | 0.970 ms | 1.077 ms |
| BF16 / 2048,320,10240 | 2.657 ms | 3.454 ms |
| BF16 / 2048,10240,320 | 2.302 ms | 2.612 ms |

多数主要形状变慢。6144 输出列的 FP16 形状缩短约 3.36%，在实际旧路径诊断中该形状约占全采样的 4.2%，
即使假定这点局部改善可迁移，整段时间也只减少约 **0.14%**。这达不到本轮目标，没有进入模型长测试或默认配置。


## 数值核验与复现

最终细粒度引擎与原日常 2K 引擎对固定 A 输入分别审计，32 个位置的 **7,946,240 个完整词表 logits 逐位相同**，
输出 token、有效 KV/GDN/PLE/尾部/pooled/dead 状态指纹相同，全部 logits 有限，模型设置相同且 drafts 为 0。
见 [quality.json](quality.json)。这是一次明确的旧/新构建比较，不是同二进制比较，也不是新的任务准确率评测。
默认路径两项专项测试通过（共 9.10 秒）；回退路径 GEMM 测试通过（2.45 秒）。
通过后已替换本机日常 `engine/strata`，旧二进制留在本机 `run/`。完整 SHA 在汇总中。

生产 `metal_gemm_test` 扩展为旧 16×32、16×64、32×64 路径与独立反量化加 GEMM 的逐位对照，覆盖混合格式、
偏移、guard 与 8/9/16/17/24/25/31/32 行边界。默认细粒度与回退路径分别验证。

最终版本默认细粒度，可在仓库根目录依次复测，期间只加载一个模型：

```sh
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-followup/old --chunk 2048 --repeats 2 --env STRATA_METAL_PREFILL_MOE_TAIL=0
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-followup/fine --chunk 2048 --repeats 2 --env STRATA_METAL_PREFILL_MOE_TAIL=1
```

`--audit` 另跑一条输入；不得把读回全部 logits/state 的计时纳入速度表。`compare.py` 的
`--allow-different-binaries` 仅允许明确的旧/新构建比较，数值及模型设置检查仍必须全部通过。
微基准在模型退出后运行 `bash bench/results/2026-10-09-metal-iq3-s-30k-followup/micro/run.sh`。

复现 10K 权重读取筛选时，在单独的研究构建中对细粒度生产代码应用 `prototype-weight-device.patch`，重新编译，
然后用该同一二进制依次运行下列两条命令；它不是当前日常引擎的开关：

```sh
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --engine build-metal/strata --out /tmp/iq3-screen/old --tokens 10000 --chunk 2048 --repeats 2 --env STRATA_METAL_PREFILL_MOE_WEIGHT_DEVICE=0
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --engine build-metal/strata --out /tmp/iq3-screen/device --tokens 10000 --chunk 2048 --repeats 2 --env STRATA_METAL_PREFILL_MOE_WEIGHT_DEVICE=1
```

历史研究构建以 `13da722` 为基底：[prototype-v1.patch](prototype-v1.patch) 为粗粒度及缓存，
[prototype-v2.patch](prototype-v2.patch) 为细粒度及缓存；两者的 tail 开关当时默认关闭。最终生产代码只保留细粒度，
默认开启。本机 `run/` 保存原始日志、二进制和完整 trace，并被 git 忽略。
