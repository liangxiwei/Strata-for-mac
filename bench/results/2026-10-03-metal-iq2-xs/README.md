# IQ2_XS、关闭 MTP、10K 上下文实测，2026-10-03

Apple M2 Max（38 核 GPU、96 GB 内存），macOS 26.5。同一份冻结的 **10,000 个输入 token**
（含聊天模板），生成 **256 个 token**，temperature=0，thinking 关闭，无前缀缓存命中。
模型加载不计入请求时间；首个请求中的计算管线初始化计入时间。

IQ2_XS 来自 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 的固定版本
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`。两个分片均通过仓库 SHA-256 校验。
第二分片与旧 Q2_0 的文件逐字节相同，已复用；专家保持 GGUF 的原始量化格式。
使用项目 `.venv/bin/python`（Python 3.12）运行 `tools/iq_pack.py`，不生成额外的 `experts.bin`。

## IQ2_XS 结果

| 指标 | 实测 |
| --- | ---: |
| Prefill | **44.7 token/s** |
| Prefill 时间 | 223.787 秒 |
| HTTP 首字延迟 | **223.932 秒** |
| Decode | **7.6 token/s** |
| 256 token 生成时间 | 33.666 秒 |
| HTTP 总时间 | 257.508 秒 |
| 草稿生成 / 接受数 | **0 / 0** |
| 中段校验字段 | **3/3 找对**：银杏、10月18日、AX-7319 |
| 引擎 physical footprint 峰值 | **40.2G**（vmmap） |

设置：容量 32,768，FP16 KV，prefill 分块 1,024，`--spec 4 --mtp-max-t 1 --suffix-draft 0`。
全部 24,576 个专家驻留 GPU 缓存（33.02 GiB），`--mmap-experts --no-prefill-borrow` 避免
另一份固定在内存中的专家副本，decode 缓存命中率 100%。系统已有 swap，本轮占用采样
从 4,775.88 MiB 到 4,759.88 MiB，未观测到超过初始值的增长。RSS 未包含所有 Metal 内存，
因此没有把 RSS 用作总内存。

IQ 专家当前仍使用 Metal 的逐层主机调度路径。日志测得每 token 的验证为 131.34 ms，
其中等待 GPU 到达步骤为 67.05 ms，逐层主机阶段为 58.37 ms；后者包含调度过程中执行和
等待 GPU 工作的时间，不能全部当作 CPU 计算。CPU 专家计算为 0，MTP 草稿耗时为 0。
Q2 专用的整图和融合 prefill 优化尚未覆盖这条 IQ 路径，因此不能把速度差全部归因于
量化格式本身。

## 本轮修复及旧结果的限制

首次 IQ2_XS 请求在 prefill 入口报错：IQ3_S（类型 21）仍指向“未实现”的占位代码。
Metal IQ 计算核已存在，本轮将 `dequant_f16` / `dequant_f32` 的 IQ 分支接入对应计算核，
保留源行偏移。失败请求的日志保存在 `initial-unsupported-iq/`，没有计入性能结果。

新增的独立参考检查同时发现：Metal Q3_K 解量化组装 12 个缩放字节时漏了第 12 个字节，
导致错误的权重值。本轮补齐该字节。旧 Q2_0 模型使用 Q3_K 张量，所以此前 10K / 100K
回答质量失败受到实现错误影响，不能全部归因于 Q2_0 量化。100K 尚未用修复后版本复测。

`iq_parity` 已通过：10 种格式的 FP16/FP32 行切片都与独立 gguf-py / Q2_0 codec 参考一致，
涵盖非零行偏移和输出边界；原有 1–8 列矩阵向量检查也通过。修复前 Q3_K 行切片有
2,681 处差异，修复后为 0。详见 `iq-parity-before-q3-fix.log` 和 `iq-parity-detail.log`。

修复后又用同一引擎、同一 10K 输入和关闭 MTP 的设置跑了 Q2_0 对照：

| 模型 | Prefill | Decode | 首字延迟 | 校验字段 |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XS | 44.7 token/s | 7.6 token/s | 223.932 秒 | 3/3 |
| Q2_0，修复后对照 | 111.9 token/s | 19.2 token/s | 89.448 秒 | 3/3 |

Q2_0 的检索检查从修复前 0/3 变为修复后 3/3。这证明先前的质量判断受到实现错误影响，
不能用旧结果判断两种模型的普遍能力。IQ2_XS 在当前 Metal 路径下 decode 吞吐比修复后
Q2_0 低约 60.5%，prefill 约为其 40%；两条专家路径的优化程度不同，详见上文。
每种配置仅测一次，结果只覆盖本样本。机器可读数据为 `comparison.json`，对照原始数据在
`q2-fixed-control/`。

已按用户要求删除 `/Users/liangxw/src/xproject/petproject/Q2_0`。旧目录含约 105.85 GiB 文件，
其中约 27.59 GiB 的同内容 PLE 分片及 MTP 运行文件已迁移供 IQ2_XS 使用；删除了约
78.26 GiB 的单链接文件（实际磁盘可用空间变化还受 APFS 和其他进程影响）。IQ2_XS 运行
配置的所有依赖均位于保留的路径，测试服务已停止。删除清单和时间记录在 `migration.json`。
旧 Q2_0 对照脚本保留作历史证据，重新运行需要重新下载旧模型。

## 证据与复现

`result.json`、`output.txt`、`stream.jsonl`、`engine.log` 保存完整结果；`environment.json`
保存实测二进制哈希；`source-manifest.json` 保存模型来源；`memory-summary.json` 保存内存
摘要。`request.json` 与前轮 Q2_0 输入相同，新 tokenizer 的 token 序列也逐个核对一致。

从仓库根目录启动测试服务，再在另一终端发送请求：

```bash
STRATA_DECODE_TIMING=1 STRATA_TRACE=1 python3 -m serve.server --engine strata \
  --config bench/results/2026-10-03-metal-iq2-xs/server-config.json --port 18080
python3 bench/results/2026-10-03-metal-iq2-xs/run_benchmark.py
```

使用新的服务进程保证缓存命中为 0；脚本会覆盖本次请求日志。日用配置为
`/Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json`。
