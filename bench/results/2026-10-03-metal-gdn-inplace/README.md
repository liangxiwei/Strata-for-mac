# Metal decode：去掉 GDN commit 的重复递推，2026-10-03（round 26）

decode 每个窗口只验证 1 个 token，这个 token 一定会被提交（commit 要求 1 ≤ n_keep ≤ T）。原来的流程里，
verify 先用 GDN 递推算出输出 y 但不写回状态，commit 再从同一状态、同一输入把 36 层的递推重算一遍，只为写回
状态。本轮把这件事放到 verify 里一次做完：

* 只有在 Metal、全部专家常驻、单 stage 的条件下，1-token 窗口才录制成「写回」版本。写回用的是 round 23 的
  tail kernel，传入设备端常量 n_keep = 1，所以写回的状态和 commit 自己算出来的是同一组数。
* commit 图多录一份不含 GDN 递推的版本，conv 历史、索引器追加、PLE 历史这些步骤保持不变。窗口已经写回时
  只能走这一份；如果对不上，直接报错，不会把同一个 token 推进两次。
* T > 1 的窗口、分段路径、其他后端都不变。`STRATA_METAL_GDN_INPLACE=0` 切回原流程。
* 每个成功的 `run()` 后面都紧跟 `commit()`（`generate.cpp` 的三处调用），失败则结束请求或引擎。

## 测量（同一二进制 `build-metal/strata-round26`，SHA-256 `02bf0dfe…62336aa`，一次一个引擎）

| 测量 | 关闭 | 开启 |
| --- | ---: | ---: |
| commit 图（`kprofile-inplace*`，逐 kernel，只看比例） | 134 个 entry，GPU 2.43 ms/token | 98 个 entry，0.71 ms/token |
| verify 图里的 GDN 递推 | 1.31 ms | 1.28 ms（多写回一份状态，时间没变） |
| 热态 decode，全部 13 次的中位数（`off/on-*`、`cbt-inplace*`） | 21.78 tok/s | 22.55 tok/s |

GPU 时间每 token 减少约 1.7 ms（profile 计数，含每个 pass 的固定开销）。整模型热态中位数快 3.5%。
但这次会话的噪声较大（后台 `dasd` 占用约 88% CPU），单次结果在 20.2–22.8 之间浮动，单独一对 A/B 不足以
证明差异，结论以 GPU 计时和多次中位数为准。

## 正确性

* 所有 256-token 运行的输出都与 round 21 参考逐个相同。
* `audit.sh`：短输入 290 个位置和 10K 加续问 99 个位置，**96,596,480 个 logits 逐位相同，6 个提交后状态相同**
  （状态指纹包括 GDN），续问返回 AX-7319。10K 用例的 64 个 decode 窗口走的正是新路径。
* CTest 50 项通过。
