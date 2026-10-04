# Mac 上的 256K 本地代码评测，2026-10-04

Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，Xcode 26.1，接电源、Energy Mode Automatic。Strata 从提交
`4d98709` 运行，engine `0.1.34`；每次只运行一个模型，`--max-context 262144`、文字路径（`--vision no`）、全专家常驻
Metal GPU。两种评测都只记录这个 256K 配置的结果。

这是一组可复现的本地冒烟测试，**不是** SWE-bench Verified 或 LiveCodeBench 的总分，更不能与完整排行榜比较。SWE-bench
只选一个固定任务；LiveCodeBench 只从 1,055 道 `release_v6` 题目中固定选 10 道。结果和范围均写在本目录的
`summary.json`。

## SWE-bench Verified

用 SWE-bench 5.0.2、mini-swe-agent 2.4.6 和官方任务镜像
`swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-20488:latest`。Docker Desktop 在 Apple Silicon 上以
`--platform=linux/amd64` 运行该镜像。代理的模型设置为 temperature 0、`reasoning_effort: medium`、每次最多
2,048 token、最多 40 步；它提交补丁后，独立的 `swebench eval verified` 再应用补丁并运行官方测试。

| 固定任务 | IQ2_XS | IQ3_S |
| --- | ---: | ---: |
| `matplotlib__matplotlib-20488` | **resolved** | **resolved** |
| agent API 回合 | 37 | 13 |
| 官方 harness 完成 / 基础设施失败 / 错误 | 1 / 0 / 0 | 1 / 0 / 0 |
| 官方 harness 运行时间 | 110.0 s | 106.3 s |

两个补丁都将 `LogNorm` 重采样范围的零值视为无效，并钳制为缩放 dtype 的最小正数；IQ2_XS 的补丁额外写了说明注释。这里的
`resolved` 仅表示这一条任务在官方 harness 中通过，不能外推到 500 条 Verified 任务。

## LiveCodeBench v6

LiveCodeBench 源码固定在 `28fef95`，读取 `release_v6`（1,055 题）。选择规则是对
`strata-2026-10-04-livecodebench-v6:<question_id>` 求 SHA-256 并取排名最小的 10 题，题号为：`3018`、`3748`、
`abc359_c`、`abc334_e`、`3709`、`3471`、`3765`、`3291`、`abc370_a`、`abc377_a`。每题用官方 OpenAI Chat
格式，temperature 0.2、top-p 0.95、最多 2,048 token、一个生成；再用 LiveCodeBench 的 `codegen_metrics` 和私有
测试计算 `pass@1`。

| | IQ2_XS | IQ3_S |
| --- | ---: | ---: |
| 单次 pass@1 | **5/10 (50%)** | **6/10 (60%)** |
| API 端到端时间（10 题） | 565.0 s | 625.6 s |
| 输入 / 输出 token | 6,215 / 13,349 | 6,215 / 12,331 |
| 引擎 prefill 合计 / 速率 | 46.1 s / 134.7 tok/s | 64.8 s / 95.9 tok/s |
| 引擎 decode 合计 / 速率 | 518.8 s / 25.7 tok/s | 560.7 s / 22.0 tok/s |
| 通过题号 | 3018、3471、3709、abc370_a、abc377_a | 3018、3471、3709、3748、abc370_a、abc377_a |

这一固定子集的题型、难度和时间分布不代表全量 release；每题只采样一次，且随机采样没有固定服务端 RNG seed。因此两个量化的
差别只能描述这一次测量，不能当作稳定的质量排序。原始回答、补丁和第三方镜像留在测量机的临时目录，不写入仓库。

## 复现边界

运行前确认 `GET /health` 的 `max_context` 是 `262144`，并一次只运行一个模型。评测需要 Docker、官方任务镜像、
SWE-bench 5.0.2、mini-swe-agent 2.4.6 和 LiveCodeBench 对应提交；模型的下载、启动和 256K 配置由 `./setup.sh`
完成。完整 500 / 1,055 题运行需要显著更多的模型调用、Docker CPU 时间和磁盘空间，不应把本目录的小样本替代为完整分数。
