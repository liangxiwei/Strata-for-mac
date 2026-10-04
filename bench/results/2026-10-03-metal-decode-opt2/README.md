# Metal decode：直接码本与直接点积，2026-10-03（round 22）

> 仓库里保留本目录的 README、汇总、审计对照、驱动脚本和微基准；每次运行的原始输出子目录（`engine.log`、`results.json`、`config.json`、logits 跟踪）只留在测量的机器上，不进 git（提交 `f90370f` 里还有，可用 `git show f90370f:<路径>` 取回）。

Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，IQ2_XS，MTP 关闭，冻结的 10,000-token 提示，每次重新生成
256 token，temperature=0。同一二进制（SHA-256 `bc95a289…d2c7f0c`）用环境变量切换新旧 kernel：

| 运行 | 额外显存 | 热态 decode 中位数 | 三次热态 tok/s | 每 token |
| --- | ---: | ---: | --- | ---: |
| `control-a`：新 kernel 关闭，保留 round 21 的 2.24 GiB IQ4 视图 | 2,297.4 MiB | 18.90 | 18.90 / 18.98 / 18.84 | 53.07 ms |
| `control-b`：同上，第二遍 | 2,297.4 MiB | 18.83 | 16.83 / 18.83 / 19.08 | |
| `direct-a`：新 kernel，不建视图 | 0 | 21.78 | 21.78 / 21.63 / 21.82 | 45.84 ms |
| `direct-b`：同上，第二遍 | 0 | 21.99 | 21.99 / 22.01 / 21.92 | |
| `direct-view`：新 kernel 加视图 | 2,297.4 MiB | 19.76 | 19.76 / 19.63 / 19.80 | 50.50 ms |
| `release`：修改后的日用配置复测 | 0 | 21.75 | 21.62 / 21.75 / 21.77 | |

热态 decode 比对照快 **15.5–16.6%**，同时**少用 2.24 GiB**。视图在新 kernel 下反而更慢，所以日用配置
`STRATA_METAL_IQ4_EXPAND` 从 1 改为 0（`daily-config-before.json` / `daily-config-after.json`）。
prefill 115–118 tok/s，本轮没有 prefill 结论。运行串行进行，swap 前后都是 2,722.75 MiB。
所有运行的 256 个输出 token 都与 round 21 参考相同（`summarize.py` 检查），专家命中率 100%。

## 先测量：时间花在哪里

* `cb-timing/`：每个 command buffer 的 GPU 开始/结束时间（`STRATA_METAL_CB_TIMING=1`，不改变编码）。decode 时
  GPU 忙碌约 96%，CPU 编码 1–2.5 ms/token，因此主要瓶颈是 GPU kernel。
* `profile-before/`、`profile-before-kernels.csv`：逐 kernel 计时（`STRATA_METAL_KPROFILE=<csv>`），每个 tape
  entry 一个带 stage-boundary 时间戳的 pass（这块 GPU 不支持 dispatch-boundary 采样，见 `micro/counter-probe.log`），
  都在原本的 command buffer 里，不提前提交。每个 pass 会多 2–3 µs，所以只看相对比例。
* 权重读取只有 70–100 GB/s。IQ4 kernel 主要耗在模拟 `__byte_perm` 的码本查表上；emulated `dp4a` 本身不慢。

## 保留的改动（每项都可单独切回）

| 改动 | 切回 | 单独测量（`micro/`） |
| --- | --- | --- |
| IQ4_XS / IQ4_NL：线程组 float 码本，码值直接与激活字节 FMA，原 136/18-byte 权重 | `STRATA_METAL_IQ4_DIRECT=0` | 输出头 3.9 → 2.3 ms，密集层约 1.6–1.9 倍 |
| Q2_0 常驻专家 down：直接点积，每 warp 4 行共享激活 | `STRATA_METAL_EXPERT_DIRECT=0` | 110 → 73 µs |
| IQ3_S 单列：直接点积，每 warp 2 行 | 同上 | 约 1.1 倍 |
| hyper-connection norm 单遍 | `STRATA_METAL_GR_NORM1=0` | 18.5 → 9.5 µs/次，每 token 96 次 |
| 图回放的 `getenv` 只读一次 | — | CPU 侧 |

精确性依据：每次调用的整数和中，每一项与部分和都是绝对值小于 2^24 的整数，float 累加在任何顺序下都精确，
`(int)` 后就是原来的 int32；之后的整数缩放和 float 表达式、线程映射、warp 累加顺序和 butterfly 都不变。

## 淘汰的方案（日志在 `micro/`）

* `char4` / `float4` 写法的 `dp4a`：比原字节循环慢。常量表、寄存器 select、64-bit SWAR 查表：只快约 10–20%。
* IQ2_S / IQ2_XXS gate/up：直接点积、每 warp 多行、码本或激活放进线程组内存、更小的线程组都没有稳定收益，有的
  更慢。只加载字节的下限是 63–84 µs，完整 kernel 112–172 µs，需要换访存方式才可能再快。
* GR down/up：`device` 地址空间、每 warp 多行、预取都不比原 kernel 快（已约 250 GB/s）。

## 正确性

* `tests/core/metal_direct_dot_test.cpp`（CTest `metal_direct_dot_test`）：每个新 kernel 按名字与原 kernel 比较，
  覆盖随机、极值（整数和达到上界）、inf / NaN / -0 / subnormal 缩放，并检查公共入口的分派。把每个新 kernel
  各故意改坏一次都能检出。CTest 50 项全部通过。
* 真实模型（`audit.sh`，round 20 的脚本，未建视图）：短输入 290 个位置和 10K + KV 续问 99 个位置，共
  **96,596,480 个 logits 逐位相同，6 个提交后状态相同**，续问返回 AX-7319（`compare-short.json`、`compare-10k.json`）。
  这是数值等价性检查，不是题目正确率测评；100K 与 MTP 开启仍未验证。

## 复现

```bash
cmake --build build-metal -j 8 && ctest --test-dir build-metal --output-on-failure
bench/results/2026-10-03-metal-decode-opt2/ab.sh        # 一次只跑一个引擎，约 15 分钟
bench/results/2026-10-03-metal-decode-opt2/audit.sh
bench/results/2026-10-03-metal-decode-opt2/micro/run.sh # 无模型的 kernel 微基准
python3 bench/results/2026-10-03-metal-decode-opt2/summarize.py
```
