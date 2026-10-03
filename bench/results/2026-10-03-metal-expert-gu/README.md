# 专家 gate/up 访存（IQ2_S / IQ2_XXS 常驻 kernel），2026-10-03（round 24，未保留）

目标是 decode 时占约 16% 的 `native_resident_gu_22/16`。结论：在不改算术的前提下没有找到整模型可测的提速，
代码没有保留。

## 拆解（`gu-variants.log`，同一 dispatch 形状，只读真实字节）

| 只做 | 时间 |
| --- | ---: |
| ids / residency / offset 和写回 | 7.7 µs |
| 只读权重（原访存方式 / 按 word 合并） | 43.9 / 33.7 µs |
| 权重 + 激活 | 81.1 µs |
| 权重 + grid 查表 | 66.4 µs |
| 权重 + 激活 + grid | 107.2 µs |
| 完整 kernel | 164–175 µs |

## 试过的写法（全部逐位一致，`gu-variants.log`、`gu-candidates-*.log`）

* 展开调用、先发全部加载：更慢（寄存器多，`maxTotalThreadsPerThreadgroup` 从 1024 降到 512–832）。
* 激活放寄存器 / 线程组内存（转置、无 bank 冲突也试过）、权重先放进线程组内存、grid 压成 2 KB：都更慢。
* 更大或更小的线程组、按 word 存放的激活：更慢。
* 和共享专家大小的 MMVQ 并发：几乎不省（GPU 已经忙满）。
* 每 warp 2 行（`or2`）：单独测快 6–12%，但整模型 A/B（`ab.sh`，三轮交替）r1 22.70 / r2 22.69 tok/s，
  在噪声内，按「不保留噪声级收益」撤回。

这个 kernel 对占用率极其敏感：同样的算术只要多用一点寄存器或线程组内存就会慢 20–40%。
