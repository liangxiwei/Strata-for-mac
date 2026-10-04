# Metal prefill：更大的 GEMM tile、寄存器 attention、更宽的专家 tile，2026-10-03（round 25）

> 仓库里保留本目录的 README、汇总、审计对照、驱动脚本和微基准；每次运行的原始输出子目录（`engine.log`、`results.json`、`config.json`、logits 跟踪）只留在测量的机器上，不进 git（提交 `f90370f` 里还有，可用 `git show f90370f:<路径>` 取回）。

Apple M2 Max（38 核 GPU、96 GB），IQ2_XS，日用配置（32K、FP16 KV、prefill chunk 1,024、无 IQ4 视图），冻结的
10,000-token 提示，新进程首次请求（无前缀复用）。同一二进制（SHA-256 `69aea4c6…e378a8`，保存为
`build-metal/strata-round25`），一次只运行一个引擎：

| 运行（`ab.sh`） | prefill 时间 | prefill tok/s | 输出 token |
| --- | ---: | ---: | --- |
| `old-a`：本轮三个开关关闭 | 80.58 s | 124.1 | 与 round 21 参考相同 |
| `new-a`：全部开启 | 51.89 s | 192.7 | 相同 |
| `old-b` | 81.60 s | 122.6 | 相同 |
| `new-b` | 51.11 s | 195.7 | 相同 |

prefill **快约 57%**（每 10K token 省约 29.6 s），不增加常驻内存。decode 不走这些 kernel，同一批运行中
decode 23.2–23.4 tok/s 没有变化。

## 先测量

* `phase-before/`：`STRATA_PREFILL_TIMING=1` 的分阶段时间。注意计时会在每个标记处提交，本身多花约 10 s，
  只看比例。
* `kprofile-before*.csv.direct.csv`：本轮给 `STRATA_METAL_KPROFILE` 加了未录制（直接 launch）的逐 kernel 采样，
  prefill 也能分 kernel 计时（每次 launch 一个带时间戳的 pass，只看比例）。改前：FP16 GEMM 31%，attention 21%，
  BF16 GEMM 14%，专家 tile GEMM 25%。FP16 GEMM 只有约 2.5 TFLOPS：每个线程组只算 16×32，加载远多于计算。

## 保留的改动（每项都可单独切回）

| 改动 | 切回 | 单独测量（`micro/`） |
| --- | --- | --- |
| 密集 GEMM 64×64 tile、向量加载、预取下一段 K（`pfl_gemm2_f16` 4 个 SIMD group，`pfl_gemm2_bf16` 8 个） | `STRATA_METAL_PREFILL_GEMM2=0` | FP16 2.4–2.7 → 6.0–7.7 TFLOPS，BF16 1.5–2.3 → 4.5–5.7 |
| prompt attention：累加值放寄存器（线程组内存 26.5 → 14.3 KB），一段内的 exp 分给所有线程 | `STRATA_METAL_PROMPT_ATTN_REG=0` | 每次 113–136 → 42–51 ms |
| 专家 tile GEMM：16 或 32 行 × 64 列（每专家平均行数 ≥ 40 时用 32 行） | `STRATA_METAL_PREFILL_MOE2=0` | 约 20 行/专家：快 12–18%；约 78 行：约 21% |

逐位一致的依据：每个输出元素的 SIMD-group MMA 序列不变（同类型、同值的操作数，8 宽 k 步从零累加器递增，
同样的 beta 处理）；attention 的 max 和 sum 仍按 32 个 cell 的顺序由一个线程求，只有 exp 并行、累加值换了存放
位置。K 不是 32 的倍数或操作数未 8 字节对齐时用原 kernel。

## 淘汰的方案

* prefill chunk 2,048 / 4,096：输出相同，但 prefill 分别只有 144.3 / 138.1 tok/s，比 1,024 慢。
* 先提交 router、让共享专家与 host 分组重叠：194.9 tok/s，对照 204.7，多出的提交开销超过重叠收益。
* attention 的 12 个 head 用 float4 打包 butterfly：寄存器太多（最多 448 线程），比 pa2 慢一倍多。
* 128×128 FP16 tile、64×64 BF16 4-SIMD-group：都比选定配置慢（`micro/prefill-gemm.log`）。

## 正确性

* `metal_direct_dot_test` 现在有 5,108 万个输出 0 位差异：
  * 新 GEMM 覆盖 FP16/BF16、不满 tile、ldy > N、beta 0/1，以及 inf/NaN/-0 输入。
  * attention 覆盖三种 KV 模式、masked page 和非有限 K。
  * 专家 tile 覆盖 16 种格式组合、奇数行数。
  * 每个新 kernel 都故意改坏过一次，均被检出。
* `metal_gemm_test`：原生专家 tile 与「全反量化 + 普通 GEMM」（现在走新 GEMM）逐位相同。CTest 50 项通过。
* 真实模型（`audit.sh`）：**389 个位置、96,596,480 个 logits 逐位相同，6 个提交后状态相同**，续问返回 AX-7319。
  其中 10K 用例覆盖 batched prefill。

## 现在的分布（`kprofile-r2.csv.direct.csv`，只看比例）

专家 tile GEMM 39%（受 IQ2 解码限制），FP16 GEMM 22%，attention 14%，BF16 GEMM 11%，反量化 4%，
GDN 递推 3%。
