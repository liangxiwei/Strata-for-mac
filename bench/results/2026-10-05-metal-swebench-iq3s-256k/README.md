# Mac 上 IQ3_S 的 256K SWE-bench Verified 固定样本，2026-10-05

Apple M2 Max（38 核 GPU、96 GB 统一内存），macOS 26.5、Xcode 26.1，接电源、Energy Mode Automatic。
Strata 从提交 `ce33e47` 运行，Metal 后端；每次只运行 IQ3_S，`--max-context 262144`、文字路径（`--vision no`）、
全专家常驻 GPU。模型代理为 temperature 0.2、top-p 0.95、最多 2,048 token、`reasoning_effort: high`；
mini-swe-agent 2.4.6 对每题最多 120 步和 600 秒。

这不是完整的 500 题 SWE-bench Verified 分数。样本从 500 个 Verified 任务中确定性地选出：按
`SHA-256("strata-2026-10-04-iq3-swebench-8h-v1:" + instance_id)` 排序，取前 40 个，再按顺序分成两个
20 题轮次。完整题目列表和每轮结果在 [summary.json](summary.json)。因此此结果不能与采用不同 agent、采样次数、
运行限制或 500 题完整集的公开数字直接比较。

## 官方评分

每题由 agent 生成补丁；随后停止模型并用 SWE-bench 5.0.2 的 `swebench eval verified` 独立评分。官方 harness
在 Docker Desktop 的 `linux/amd64` 任务镜像中应用补丁，并运行该任务固定的 FAIL_TO_PASS 与 PASS_TO_PASS 测试。
它不调用模型。评分时使用 6 个 worker；只有非空补丁需要运行 harness。

| 固定轮次 | 题数 | 非空补丁 | resolved | unresolved | 空补丁 | 最终基础设施失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| round_1 | 20 | 6 | 5 | 1 | 14 | 0 |
| round_2 | 20 | 4 | 4 | 0 | 16 | 0 |
| **合计** | **40** | **10** | **9 (22.5%)** | **1** | **30** | **0** |

round_1 通过：`django__django-12741`、`django__django-14765`、`django__django-16612`、
`pytest-dev__pytest-7236`、`sympy__sympy-16886`。round_2 通过：`django__django-11239`、
`django__django-13363`、`django__django-13810`、`django__django-14559`。round_1 中
`psf__requests-5414` 的补丁未通过，harness 标为 `missing_module` 的歧义失败。

Docker 镜像启动而尚未开始模型轨迹的失败会重试；最终两个轮次的官方报告均为 0 个基础设施失败。不会重试已经
产生模型输出的任务。原始轨迹、补丁和官方容器日志保留在测量机的临时评测目录，不写入仓库。

## 复现边界

运行前确认 `GET /health` 的 `max_context` 是 `262144`，并一次只运行一个模型。需要 Docker、官方任务镜像、
SWE-bench 5.0.2 和 mini-swe-agent 2.4.6。该 40 题固定样本被设计为大约八小时的串行本地运行预算；它说明这个
具体配置在这些任务上的一次结果，不能替代全量评测或作为模型质量排序。
