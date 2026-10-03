# Metal decode：宽 IQ4_XS、GDN 状态写回、并行 top-k，2026-10-03（round 23）

同一台 M2 Max、同一冻结 10K 提示与日用配置（无 IQ4 视图）。同一二进制（SHA-256 `1a5300ad…d578d7b31`，
保存为 `build-metal/strata-round23`），`round1-*` 关闭本轮三个开关，`all-*` 全部开启，串行运行：

| 运行 | 热态 decode 中位数 | 三次热态 tok/s | 每 token |
| --- | ---: | --- | ---: |
| `round1-a`（round 22 的 kernel） | 21.75 | 21.68 / 21.75 / 21.85 | 45.76 ms |
| `round1-b` | 21.85 | 21.91 / 21.83 / 21.85 | |
| `all-a` | 22.46 | 22.46 / 22.46 / 22.40 | 44.65 ms |
| `all-b` | 22.23 | 22.23 / 21.72 / 22.55 | |

热态 decode **+1.9–3.0%**，约 1.1 ms/token；不增加内存。256 个输出 token 均与 round 21 参考相同。
round 21 的对照（18.83–18.90）在 round 22 的会话里测得；跨会话合计约 +18%，只作参考。

| 改动 | 切回 | 单独测量（`../2026-10-03-metal-decode-opt2/micro/`） |
| --- | --- | --- |
| IQ4_XS 每 simdgroup 一行、4 行（`native_iq4_xs_direct_sg4`，n_out >= 2560） | `STRATA_METAL_IQ4_SG=0` | 输出头 2.3 → 1.9 ms；n_out 640 时更慢，故不用 |
| GDN 递推：最后一个 token 的状态不写 shadow，commit 直接写 `state` | `STRATA_METAL_GDN_TAIL=0` | 省掉一半状态流量 |
| QSA 块 top-k：并行扫描（`block_topk_scan_kernel`） | `STRATA_METAL_TOPK_SCAN=0` | 10K：53–59 → 27–30 µs；32K：106–110 → 65–68 µs |

top-k 的输出与原 kernel 相同：阈值是同一个整数（256 个 digit 计数的后缀和），等值预算仍按块号从低到高分配，
输出仍按原来的连续线程区间写入。测试扫过 2,052–32,767 的上下文，让等值切点落在每个线程的区间里；
早期两个只在单一线程触发的注入错误没被测出，加了扫描后都能检出。

## 正确性

* `metal_direct_dot_test` 2,054 万个输出 0 位差异，包括 GDN（T = 1/2/4/8，verify、全部或部分 commit、
  t_out_begin）和 top-k（随机、大量并列、全相等、NaN 与 -0）。CTest 50 项通过。
* `audit.sh`：**389 个位置、96,596,480 个 logits 逐位相同，6 个提交后状态相同**，AX-7319 正确。

## 现在的时间分布（`profile/`，诊断用；这次会话整体慢约 15%，只看比例）

IQ4_XS sg4 18.7%，IQ2_S gate/up 12.4%，Q2_0 down 10.0%，GR up/down/norm 18.1%，IQ3_S 4.6%，
IQ2_XXS gate/up 3.8%，bf16 GEMV 3.4%，attention chunk 3.2%，GDN 递推 2.3%（commit 图另有 1.7 ms）。
