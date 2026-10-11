# IQ3_S：低电量模式下的 30K prefill，2026-10-09

Apple M2 Max（38 核 GPU、96 GB）、macOS 26.5、Xcode 26.1，接电源，始终保持 Low Power Mode（`powermode 1`）。
实际输入为 30,000 token，上下文容量仍为 **262,144**。同一引擎二进制、IQ3_S 原权重与量化、全部 48 层、每 token
10 个专家、FP16 KV，24,576 个专家全部驻留 GPU。MTP 保持 `--mtp-max-t 1 --suffix-draft 0`，没有 drafts。
本机 `strata-iq3_s.json` 已将 `--prefill` 从 1024 调整为 **2048**。

## 完整请求的实测

每个 chunk 配置启动一个独立引擎，依次处理相同的两条固定提示。A 是进程的首次请求；B 在开头加编号并等量移除 filler
以保持 30,000 token，避免复用 A。各配置的 token ID SHA-256 相同，所有请求 `reused=0`。启动时间、tokenizer、请求后的
内存读取不计入 prefill。性能运行不启用 logits/state 审计、逐阶段或逐 kernel profiler，也不在运行中采样内存。

| prefill chunk | A：首次 30K 输入 | B：同进程另一条完整 30K 输入 | 相对 1K 的吞吐提升 A / B | 峰值 physical footprint | 额外 GPU 分配，相对 1K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 172.058 s，174.36 tok/s | 152.005 s，197.36 tok/s | — | 61.4G | — |
| **2048** | **149.633 s，200.49 tok/s** | **133.098 s，225.40 tok/s** | **14.99% / 14.20%** | **61.8G** | **465 MiB** |
| 4096 | 159.963 s，187.54 tok/s | 122.470 s，244.96 tok/s | 7.56% / 24.12% | 62.4G | 1393 MiB |

2K 分别节省 **22.425 / 18.906 秒**，两条输入都获得约 14–15% 的吞吐提升。4K 的第二条输入更快，但首次请求比 2K 慢；
两条合计为 282.432 秒，2K 为 282.732 秒，差约 0.1%，同时多分配 928 MiB。当前选择 2K，以较小内存代价获得两条输入
都有的收益。长期驻留服务、经常完整读入新长提示时，4K 值得按实际请求分布另测；这两条提示不足以证明所有负载的最优值。

峰值通过请求全部结束后的一次 `vmmap -summary` 读取，避免干扰计时。三组的 GPU cache 都保持 24,576 个专家，GPU 剩余
预算分别为 18,073 / 17,608 / 16,680 MiB；额外工作区没有挤掉专家。后台应用保留运行，系统已有 swap，绝对数字会随
负载变化；原始数据及每次 swap 状态见 [summary.json](summary.json)。本结果只测当前机器、低电量模式、文字路径与这两条
30K 提示，不能直接推广为 256K 输入、其他电源模式或其他模型的结果。

## 从第一性原理定位大头

每 token 的路由专家矩阵乘工作量为 `48 × 10 × 2 × (2 × 2560 × 640 + 2560 × 640)`；把乘加计为两个 FLOP，30K 输入
约为 **141.6 TFLOP**，还没算密集投影、attention 与其它算子。改 chunk 不减少这些数学运算的必要工作，而是减少重复
解码、加载与调度，以及 tile 填充造成的多余工作。

1K chunk 每个活跃专家平均约 20 行；2K 约 40 行，达到生产内核选择 32 行 tile 的阈值，同一份反量化权重可服务更多 token。
完整 30K 的 chunk 数从 30 变为 15，每层的密集权重反量化、host 分组与同步次数相应减半。4K 继续减少这些次数，但不再
扩大 32 行专家 tile；更大的全局工作区和不同请求阶段的成本，需要以完整请求计时判断。

另外做了一次诊断运行，逐 kernel 采样的时间分布约为：

| 类别 | 采样时间占比 |
| --- | ---: |
| 专家 GEMM（含 tile 内反量化） | 38.75% |
| 密集 GEMM：FP16 / BF16 | 20.52% / 10.59% |
| attention | 15.01% |
| 独立反量化 | 6.14% |
| GDN | 3.30% |
| 其它 | 5.70% |

这些是带 profiler 的相对分布，不是无开销请求的墙钟账。初始化直接 launch、未落盘的末尾采样、额外 pass 和事件都会影响
结果。特别不能把 phase 日志中的 `host grouping` 当成 CPU 分组本身的耗时：它还含同步、等待与事件计时的成本。
原始聚合在 [profile-summary.json](profile-summary.json)。IQ3_S 包混合使用 IQ2_S、IQ3_XXS、IQ3_S 等格式；只优化名称为
IQ3_S 的一个内核，无法覆盖全部专家的大头。

## 被实测排除的空间换时间方案

- **扩大专家 tile 的 K 暂存。** 16×64 tile 的线程组内存从 6 KiB 增到 11 / 21 KiB，尝试将 K 步长从 32 改为 64 / 128，
  保持每个输出的 8-wide MMA 累加顺序。IQ3_S gate/up 为 31.94 → 40.70 / 74.75 ms；其它主要格式也变慢。减少 barrier
  不自动产生收益；暂存和寄存器压力限制了能同时执行的工作。超过设备线程组内存上限的组合未纳入计时。
- **分批无损展开专家为 FP16。** 64 个专家的 gate/up 暂存为 400 MiB，down 可复用同一块，展开成本计入。IQ3_S gate/up
  参考 native 为 4.90 ms，展开加 32×64 GEMM 为 5.96 ms；其它主要格式也更慢。完整层（512 专家）的展开也更慢。
  这里的 native 参考是老的 16×32 tile，展开方案连这个较慢参考都未胜出。写入和再次读取展开权重是实在的额外成本。
  把整个模型的专家展开需要约 **225 GiB**，不适合这台 96 GB 设备。
- **按专家行数混用 16/32 行 tile。** 某些格式及密度的微基准更快，另一些变慢。例如约 20 行/专家，IQ3_S gate/up
  41.94 → 37.69 ms，IQ3_XXS 35.82 → 37.76 ms。未取得完整模型的收益证据，没有进入本机配置。

微基准每个输出先与原 kernel 按位比较，所列可运行方案均无差异；这只证明数值一致，不代表整段 prefill 获益。
源码和结果在 [micro/](micro/)。

## 数值验证

性能计时与数值审计分开。对固定 A 输入，分别运行 1K、2K、4K chunk，保留全部 28 个 verify trace 文件，包含
**32 个位置、7,946,240 个完整词表 logits**。2K 和 4K 都满足：

- 输入位置/token 相同，所有 logits 逐位相同且有限，输出 token 相同。
- 有效 KV、GDN、PLE、尾部与 pooled/dead key 状态指纹相同；不比较无效 padding 和 drafter 最后尚未计算的 cell。
- FP16 KV、context、全部专家驻留、spec/MTP/lookup/cvec 设置相同，drafts 为 0。
- 两条性能提示的三个字段均 3/3，输出 token 与 1K 对照相同。

详见 [quality.json](quality.json) 和 [quality-4096.json](quality-4096.json)。这是一条固定输入的数值审计与两条性能提示，
不是新的通用任务准确率或 SWE-bench 分数。独立 `metal_direct_dot_test` 与 `metal_gemm_test` 两项通过（共 6.39 秒），另覆盖多种专家格式、16/32 行 tile 和奇数行尾部。

## 计时方法修正

最初用了每 20 秒一次的 `vmmap -summary`，每次自身需要约 6–7 秒并产生明显 CPU 负载。那些运行的 1K / 2K prefill
为 242.328 / 221.527 秒，另一次 2K 为 230.516 秒；**不用于最终速度结论**。带逐阶段/逐 kernel profiler 的运行为
271.154 秒，也只用于定位算子。最终表使用不在请求中采样的独立运行，待请求结束才读取进程保存的峰值。
不能将去掉采样产生的差异计作模型内核加速；后台负载与首次初始化也会影响不同运行。

## 复现

在仓库根目录，保持 Low Power Mode，依次运行，一次只加载一个模型：

```sh
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-30k/1024 --chunk 1024 --repeats 2
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-30k/2048 --chunk 2048 --repeats 2
caffeinate -dims .venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-30k/4096 --chunk 4096 --repeats 2
```

以 `--audit` 另跑 A 输入用于数值审计；不能混入性能表。`--profile` 另跑诊断。`--memory-interval 20` 只保留为复现旧测法的
对照开关，默认是 0。固定 corpus 来自 `dc047b5`，A/B token SHA-256 在汇总中。GPU 微基准仅在模型退出后运行：

```sh
.venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-30k/audit-1024 --chunk 1024 --audit
.venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/measure.py --out /tmp/iq3-30k/audit-2048 --chunk 2048 --audit
.venv/bin/python bench/results/2026-10-09-metal-iq3-s-30k/compare.py /tmp/iq3-30k/audit-1024 /tmp/iq3-30k/audit-2048 --output /tmp/iq3-30k/quality.json
bash bench/results/2026-10-09-metal-iq3-s-30k/micro/run.sh
```

原始输出与完整 logits 保留在本机 `run/`，不提交到 git。引擎 SHA-256 是
`778df88d1f727922b06c3ffd77546305d4c07055b01079c67349d18d07329dba`；没有替换引擎或模型权重。
