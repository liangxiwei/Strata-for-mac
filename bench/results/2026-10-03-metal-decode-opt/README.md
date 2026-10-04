# Metal decode：有收益的内存缓存，2026-10-03

> 仓库里保留本目录的 README、汇总、审计对照、驱动脚本和微基准；每次运行的原始输出子目录（`engine.log`、`results.json`、`config.json`、logits 跟踪）只留在测量的机器上，不进 git（提交 `f90370f` 里还有，可用 `git show f90370f:<路径>` 取回）。

Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，IQ2_XS，MTP 关闭。保留的主要改动是
**179 个密集投影矩阵的无损 IQ4 码字缓存，额外 2.24 GiB**。同一二进制开关对照的稳态
decode 为 **18.69 → 19.53 token/s，约 +4.5%**；每 256 token 节省约 **0.59 秒**。
这是 decode 吞吐收益，不能解释为整个 10K 请求同样提速 4.5%。

## 测量方法与结果

每个进程先处理冻结的 10,000-token 提示，再重复相同请求，每次都重新生成 256 token。
temperature=0、thinking 关闭；容量 32,768、FP16 KV、prefill 1,024，实际草稿数为 0。
第一次无前缀命中；后续均复用 9,993 个提示 token，但 256 个输出都重新计算。
下表使用三次热态 decode 的中位数，单独保存第一次结果。没有并行运行 GPU 测试。
模型加载、prefill、输出生成分别计时；数值审计读回完整 logits，其时间不用于性能结论。

| 对照 | 新增缓存 | 热态 decode（毫秒/256 token） | token/s |
| --- | ---: | ---: | ---: |
| `final-control`：缓存、compute copy 都关闭 | 0 | 13,700.4 | 18.69 |
| `final-repeat`：密集层缓存 + compute copy | 2.24 GiB | 13,110.9 | 19.53 |

这两项及 `final/` 使用相同二进制 SHA-256：
`eb3de0f71968def80e67e00c479b2ff15f285fb1743623b5d56d193b84cf65e0`。
另外一组同二进制隔离对照 `no-cache/` 和 `dense-only/` 保留相同的 compute copy：仅开启
密集层缓存使 13,591.5 → 13,018.8 毫秒，即 **18.84 → 19.66 token/s，+4.4%**。
因此缓存本身有可重复的整体 decode 收益，不是仅有单算子数字。

**首次请求没有稳定提速结论。** 相同功能的首次 decode 测到 17.69～19.52 token/s；同一
版本重启后测到 19.32。热态通常为 19.5～19.7。波动原因未隔离，不能归因于某一种
初始化或温度因素。10K prefill 约 127～130 token/s，本轮不声称稳定的 prefill 增益。
历史 `baseline/` 保存 round-20 原二进制；主结论采用更接近的同二进制开关对照。

所有性能运行的 256 个输出 token 均与原参考逐个相同，专家命中率均为 100%。
`summary.json` 由 `summarize.py` 校验输入、输出、参数和命中率后生成，包含所有中间试验。

## 保留的实现

1. **无损码字展开。** 原 IQ4_XS 的 136-byte block 保留供 prefill 和多 token 调用使用；
   额外视图为 264 bytes：原来的 8-byte scale/header 原样复制，256 个码字预查表为 signed
   int8。没有重新量化。decode 仍调用原格式的整数点积、缩放和浮点求和顺序。
   缓存对应 179 个每 token 都会执行的矩阵；`view-inventory.json` 保存完整名称和形状。
   不为使用独立内核的 PLE key 建无用视图，也不缓存输出词表层。
2. **短拷贝与二维拷贝。** 不超过 64 KiB 的注册 buffer 拷贝保留在同一个串行 compute pass，
   对齐时按 16-byte 搬运，否则按 byte 搬运。二维注册 buffer 拷贝合并成一个 dispatch。
   大的一维拷贝保留 blit，普通主机指针保留原 staging 语义；GraphLaunch 继续延迟提交，
   允许 CPU 在 launch 后、flush 前更新映射源。
3. **构建依赖。** `.metalh` 也触发 metallib 重编译，避免修改公共 shader 头后仍运行旧内核。

缓存实际分配 **2,409,031,680 bytes = 2,297.43 MiB = 2.244 GiB**，本机建立视图约
0.12 秒。该时间仅是视图构建，不是完整启动时间，也不代表冷请求全部额外开销。
缓存上限为 min(4 GiB, GPU 工作集预算/16)，单次申请还受空闲预算限制；不足时保留原路径。
全部 24,576 个专家仍驻留（33.02 GiB），该组开关对照前后的 swap 均为 2,754.75 MiB。
`metrics.child_peak_rss_bytes` 仅是子进程 RSS，不作为统一内存总占用。

通用构建的码字缓存默认关闭，仅在已验证的日用配置中设 `STRATA_METAL_IQ4_EXPAND=1`：
`/Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json`。
将该配置的值改成 `0` 可关闭缓存；`STRATA_METAL_COMPUTE_COPY=0` 恢复原 blit 拷贝。
配置修改前后副本分别保存在 `daily-config-before.json`、`daily-config-after.json`。

## 淘汰的方案

- 输出词表层额外 625.2 MiB 展开缓存：热态整体约 19.66，与不缓存它相当，已移除。
- FP16 整数码字视图：码字仍精确，但空间接近翻倍，大矩阵更慢，已移除。
- 共享专家 scalar gate 融合：完整模型只有约 0.1% 变化，属于波动，已移除。
- GR SIMD 改写：完整模型从约 19.66 降到 18.93 token/s，已撤回。
- 多种 warp 布局和整数点积改写：没有稳定胜出，均未保留。

`profile/` 是早期逐 command-buffer 的诊断，不是吞吐或正确性基准。该实验改变提交时机，
不适用于本项目的 mapped-source 延迟写入契约，已从最终源码删除；不能重新用于模型验证。
安全的 `STRATA_METAL_GRAPH_TIMING=1` 只记录 CPU 编码耗时，保留原提交时机。

## 回归与计算等价性

最终构建 SHA-256：`0c65709c5d8dd410578d2f2378a8258b7e2dbdfe2bbef43f719bf2b57ddbb86c`。
清理临时诊断后，直接使用日用配置复测的 `release/` 也使用这个 SHA：首次 decode **19.52**，
三次热态为 **19.670 / 19.687 / 19.645 token/s**，中位数 **19.67**；prefill **129.67**。
质量审计与该最终二进制一致。所有私有测试引擎已关闭。

**49 个 CTest 条目零失败，148.22 秒**；其中 3 个 x86 专用检查在 ARM 内部跳过。
新增 copy 两种路径各 471 个用例，包含非对齐、padding、图销毁、延迟主机写入和数据依赖。
新增 IQ4 测试比较原 shader、码字字节 oracle、捕获回放、单/多 token 和释放后原路径：
27,702 个输出逐位相同，非有限值和表示/生命周期错误均为 0。

| 原 round-20 对照 | 同输入位置 | 完整词表 logits | 逐位差异 | 提交后状态 |
| --- | ---: | ---: | ---: | --- |
| 4 个短固定输入 | 290 | 72,012,800 | 0 | 4/4 相同 |
| 10K + KV 续问 | 99 | 24,583,680 | 0 | 2/2 相同 |

共 **389 个位置、96,596,480 个 logits，6 个状态指纹相同**。输入、输出 ID、KV/循环/PLE/
索引器状态和模型设置均匹配。续问复用 10,063 token、读 20 个新增 token，正确返回
AX-7319。短测试是分布等价性 fixture，不是完整题目正确率测评。结论覆盖这些配置与输入；
100K、MTP 开启的 IQ2_XS 仍需单独验证。详见 `compare-short.json`、`compare-10k.json`、
`quality-summary.json`、`ctest-final.log` 和 `ctest-details.log`。

## 复现

在仓库根目录运行，输出目录必须不存在：

```bash
cmake --build build-metal -j 8
STRATA_METAL_IQ4_EXPAND=1 ctest --test-dir build-metal --output-on-failure
python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py \
  --config /Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json \
  --output /tmp/strata-decode-cache --tokens 256 --repeats 4
python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py \
  --output /tmp/strata-decode-reference --tokens 256 --repeats 4 \
  --env STRATA_METAL_IQ4_EXPAND=0 --env STRATA_METAL_COMPUTE_COPY=0
```

数值审计继续使用上一轮的 `audit_model.py` / `compare_audit.py`，对 candidate 设置
`STRATA_METAL_IQ4_EXPAND=1`；原始 trace 保存在本目录 `audit-short/` 和 `audit-10k/`。
