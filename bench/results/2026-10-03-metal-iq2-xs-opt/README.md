# IQ2_XS Metal 优化，2026-10-03

实测机器：Apple M2 Max（38 核 GPU、96 GB 内存），macOS 26.5。继续使用已校验的 IQ2_XS，
MTP 关闭。相同的冻结输入 **10,000 token**（含模板），输出 **256 token**，temperature=0，
thinking 关闭，无前缀缓存命中。每个阶段使用新服务进程；模型加载不计入请求时间，首个请求
的计算管线初始化计入。每个配置测量一次，不能把中间几次 decode 的小幅变化当作独立结论。

## 性能

| 阶段 | Prefill token/s | Decode token/s | 首字延迟 | 总请求时间 |
| --- | ---: | ---: | ---: | ---: |
| 上轮基线 | 44.7 | 7.6 | 223.932 秒 | 257.508 秒 |
| GPU 整图执行原生 IQ 专家 | 43.4 | 18.8 | 230.485 秒 | 244.093 秒 |
| IQ 入口异步执行 | 71.3 | 18.1 | 140.420 秒 | 154.483 秒 |
| 融合 prefill，21 KiB 共享内存 | 75.5 | 18.1 | 132.559 秒 | 146.628 秒 |
| **最终：融合 prefill，5 KiB 共享内存** | **120.3** | **18.0** | **83.244 秒** | **97.416 秒** |

按原始计时计算，最终 prefill 为基线的 **2.69 倍**，decode 为 **2.37 倍**，总请求时间减少
**62.2%**。所有阶段都找对了“银杏、10月18日、AX-7319”。整图执行版与上轮基线的生成措辞
不同；从整图执行到最终版本，四次输出逐字一致。随后用保存的原始 round-19 二进制
（SHA-256 未变）复测，也与优化版的 256 个输出 token 完全一致。历史差异的原因尚未确定，
不能将它直接归因于此次优化。下述全模型数值审计通过；100K 本轮没有复测。

最终配置仍为容量 32,768、FP16 KV、prefill 1,024，
`--spec 4 --mtp-max-t 1 --suffix-draft 0 --mmap-experts --no-prefill-borrow`。
实际草稿生成数为 0；全部 24,576 个专家驻留缓存，decode 命中率 100%。
`vmmap` 的引擎 physical footprint 峰值为 **40.3G**；RSS 未用作总内存。

缓存续聊复用 **10,255 token**，新增 **35 token**，首字延迟 **2.335 秒**；56 个输出 token
的 decode 为 **18.2 token/s**，正确返回 AX-7319。短算术（17×23）、短字段检索、重复请求
另测 3 次，均得到预期答案。详见 `tiled/followup-result.json` 和 `tiled/smoke-requests.json`。

## 三个模块

1. **原生 IQ 专家调度：**将整个缓存作为 Metal buffer 绑定，在 GPU 上通过路由、驻留表和
   64 位槽偏移定位专家。每层的 gate/up 和 down 各一次批量启动，代替主机读取指针后逐个
   专家启动。所有专家驻留且没有 prefill 借用时，验证窗口可以整图回放；其他配置保留分段
   路径。最终 decode 日志中的逐层 GPU 到达等待和主机处理时间均为 0。
2. **异步量化、解量化：**去掉 IQ 数据生成入口的强制流同步。在 Metal runtime 中补充同步
   legacy copy 对 blocking stream 的等待，保留 nonblocking stream 的显式排序要求。
   数据依赖仍按流顺序执行，避免每个专家的两次解量化都让主机等待。
3. **原生 IQ 融合 prefill：**复用已有解量化公式，按 32 个值的切片直接写入转置的线程组
   矩阵块，再做 FP16 矩阵乘法、FP32 累加。原先整专家的 FP16 临时矩阵不再反复写入和读取。
   中间版本使用 21 KiB 共享内存；最终按切片写入，只需 5 KiB。640 列 down 的尾块只读取
   有效切片。矩阵乘加顺序与原 FP16 路径一致，保留原量化格式。

## 验证与复现

最终完整回归：**46 个 CTest 条目均无失败，126.93 秒**。其中 3 项 x86 AVX-512/VNNI
专用检查在 ARM 上按平台跳过；Metal 和本轮新增检查全部执行通过。详细输出保存在
`ctest-quality.log`，包含每个原生专家图回放与融合矩阵乘法的 0 差异记录。
原生驻留专家新增覆盖全部 11 种 gate/up、5 种 down 格式：67 个 fixture、201 次变化路由的
图回放，差异及非有限输出均为 0。

新增检查包括：捕获后改变 GPU 路由和驻留表、变长/定长缓存槽、缺失专家、重复专家、
1/2/4/8 token、图源对象销毁后回放；融合 prefill 覆盖 8 种 gate/up 与 3 种 down 格式、
2560×640 的真实专家形状、非零行偏移、部分分块和输出边界。结果与既有参考逐位一致。
原先的独立 gguf-py 解量化、FP16/FP32 行切片、矩阵向量检查继续通过。

### 模型计算等价性审计

用户明确要求不能牺牲模型能力。原始 GGUF、量化格式、层数、路由专家数量、FP16 KV 和
采样设置均保留，MTP 关闭。此前仅有数值单测和检索 3/3，证据不足以判断整个模型；
因此曾暂时关闭整图 decode 默认值，完成以下审计后恢复默认加速。

| 对照 | 同输入位置 | 完整词表 logits 数 | 逐位差异 | 提交后状态 |
| --- | ---: | ---: | ---: | --- |
| 4 个短固定输入，分段 vs 整图 | 290 | 72,012,800 | 0 | 4/4 相同 |
| 10K 预填充 + 续聊，原 FP16 专家路径 vs 融合/整图 | 99 | 24,583,680 | 0 | 2/2 相同 |

第二组读取完整 10K 提示，生成 64 token；随后复用 10,063 token、读取 20 个新增 token，
生成 10 token（含结束符），正确返回校验码。对照涵盖 prefill 留下的状态、逐 token decode、
多 token 验证窗口和 KV 续用。状态指纹包括循环状态、KV、PLE 和索引器；所有输入、输出
ID、模型设置也相同。短测试每题只生成 1 token，用于分布等价性检查，不是题目正确率测评。

另外在参考执行中，用真实激活和权重同时计算新旧专家路径：288 次逐层比较、
23,347,200 个输出值，逐位差异和非有限值均为 0。合成测试继续覆盖独立解量化参考、
图回放和尾部边界。这些结果支持**已测配置与输入下计算无损**，不代表已穷尽所有任务或
证明 100K 的模型质量。

审计脚本为 `audit_model.py`、`compare_audit.py`，结论在 `audit-short-comparison.json`、
`audit-10k-comparison.json` 和 `audit-experts.json`。原始 logits、输入 ID、输出 ID 和
引擎日志保存在 `audit-*` 子目录。`audit-round19-comparison.json` 另确认原始二进制与
当前参考路径的 4 次短请求输出 ID 和缓存状态指纹也一致。`quality-baseline-repeat/` 保存原二进制复测，
`quality-segmented/` 保存恢复分段 decode 的隔离测试。审计会读回完整 logits 并计算
缓存指纹，不能用它的时间作为性能测试。

示例（输出目录必须尚不存在）：

```bash
python3 bench/results/2026-10-03-metal-iq2-xs-opt/audit_model.py \
  --mode reference --case 10k --shadow-windows 3 --output /tmp/strata-audit-reference
python3 bench/results/2026-10-03-metal-iq2-xs-opt/audit_model.py \
  --mode resident --case 10k --output /tmp/strata-audit-candidate
python3 bench/results/2026-10-03-metal-iq2-xs-opt/compare_audit.py \
  /tmp/strata-audit-reference /tmp/strata-audit-candidate --output /tmp/strata-audit-comparison.json
```

`comparison.json` 保存汇总及二进制 SHA-256；各阶段子目录保存冻结请求、token 序列、
服务配置、原始输出及计时日志。`baseline/` 是上轮证据的副本。`resident/`、`async/`
使用的快照可执行文件名字不同于监控器原来的匹配规则，因此这两轮的 RSS 字段为 0，
不能作为内存测量；最终内存报告以 `tiled/vmmap-final.txt` 为准。

从仓库根目录执行最终版本：

```bash
cmake --build build-metal -j 8
ctest --test-dir build-metal --output-on-failure --timeout 300
STRATA_DECODE_TIMING=1 STRATA_TRACE=1 python3 -m serve.server --engine strata \
  --config bench/results/2026-10-03-metal-iq2-xs-opt/tiled/server-config.json --port 18080
```

服务 READY 后，在另一终端运行：

```bash
python3 bench/results/2026-10-03-metal-iq2-xs-opt/tiled/run_benchmark.py
python3 bench/results/2026-10-03-metal-iq2-xs-opt/tiled/run_benchmark.py --followup
```

脚本会覆盖对应日志。日用配置仍是
`/Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json`，已指向当前优化后的
`build-metal/strata`。`STRATA_METAL_IQ_RESIDENT=0` 或 `STRATA_METAL_LAYER_SYNC=1` 可恢复分段 decode；
`STRATA_METAL_PREFILL_F16_EXPERTS=1` 可恢复完整专家解量化后的 prefill，供对照。
