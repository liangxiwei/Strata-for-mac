# 10K 上下文 MTP 对照，2026-10-03

> 仓库里保留本目录的 README、汇总、审计对照、驱动脚本和微基准；每次运行的原始输出子目录（`engine.log`、`results.json`、`config.json`、logits 跟踪）只留在测量的机器上，不进 git（提交 `f90370f` 里还有，可用 `git show f90370f:<路径>` 取回）。

后续更正：本报告的三组都使用了后来发现存在 Q3_K 解量化错误的 Metal 版本（遗漏一个
缩放字节）。开关 MTP 的输入、输出一致性及当时的耗时仍是实测记录，但检索失败不能
全部归因于 Q2_0 量化。修复后的对照与 IQ2_XS 结果另见 `../2026-10-03-metal-iq2-xs/`。

在 Apple M2 Max（38 核 GPU、96 GB 内存）、macOS 26.5 上，用相同 Q2_0 模型和
相同 10,000 个输入 token（含聊天模板），各生成 256 个 token。三组都从新服务进程
开始，缓存命中为 0，temperature=0、thinking 关闭，模型加载不计入请求耗时。

三组固定容量 32,768、FP16 KV、自动 1,024-token prefill、全量专家驻留 Metal、
`--spec 4 --spec-min-p 0.5 --suffix-draft 0`。只改变 `--mtp-max-t`。
窗口包含当前 token，所以 `max_t=2` 最多猜 1 个后续 token，`max_t=4` 最多猜 3 个。
置信度阈值会缩短实际验证窗口。

| 配置 | Prefill (token/s) | HTTP 首字延迟 | Decode (token/s) | 相对关闭 MTP | 草稿接受率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 关闭 MTP，max_t=1 | 107.6 | 93.039 秒 | 19.00 | 基准 | 无草稿 |
| 开启 MTP，max_t=2 | 107.1 | 93.466 秒 | 16.88 | -11.2% | 90/119，75.6% |
| 开启 MTP，max_t=4 | 105.5 | 94.891 秒 | 14.17 | -25.4% | 128/201，63.7% |

对应的 decode 耗时为 13.472、15.163、18.063 秒；HTTP 总耗时分别为 106.467、108.583、
112.908 秒。百分比按 token 数和 decode 毫秒数计算，没有用四舍五入的吞吐反推。

本样本开启 MTP 没有加速；较小窗口减少了损失，但仍比关闭慢。Prefill 三组为
105.5–107.6 token/s，未显示 MTP 有 prefill 加速效果。每种配置仅测一个请求，这不是
所有模型、文本和硬件上的普遍结论。

## 生成时间花在哪里

引擎的 `STRATA_DECODE_TIMING=1` 日志给出：

| 配置 | 轮数 | 平均验证窗口 | 每轮产出 token | 每轮验证 | 每轮草稿生成 |
| --- | ---: | ---: | ---: | ---: | ---: |
| max_t=1 | 256 | 1.00 | 1.00 | 52.17 ms | 0.00 ms |
| max_t=2 | 166 | 1.72 | 1.54 | 83.58 ms | 7.30 ms |
| max_t=4 | 130 | 2.55 | 1.97 | 122.80 ms | 15.81 ms |

MTP 减少了轮数，但每轮的批量验证和草稿开销增加，吞吐反而下降。当前应保留日用配置
中关闭 MTP 的设置。后续优化需要关注 Metal 批量验证的耗时和草稿开销。

## 校验与限制

`compare_results.py` 确认三组输入 token 文件完全相同、配置仅 MTP 窗口和日志路径不同、
输出都是 256 token、没有复用缓存、MTP 两组确实有草稿、引擎二进制哈希没有变化。
**三组生成文本完全相同，但都未检索出植入的三个校验字段。** 10K 输出没有 100K 样本
中的大量“RAM”重复，但回答质量仍未通过本次指令/检索校验。文本一致不能证明模型
输出正确，质量异常也不是只在开启 MTP 时出现。根因尚未定位。

`environment.json` 保存环境和二进制哈希；每组目录保存原始请求、token 文件、SSE
事件、完整回答、时延、配置、内存采样和引擎日志。`comparison.json` 为机器可读对照。
所有测试服务在完成后停止，日用配置没有修改。

## 复现

从仓库根目录运行，每组使用独立的新服务进程。以下以 `mtp-on` 为例：

```bash
STRATA_DECODE_TIMING=1 STRATA_TRACE=1 python3 -m serve.server --engine strata \
  --config bench/results/2026-10-03-metal-10k-mtp/mtp-on/server-config.json --port 18080
```

在另一终端发送冻结的输入（可改为 `mtp-off` 或 `mtp-t2`，同时更换服务配置）：

```bash
python3 bench/results/2026-10-03-metal-10k-mtp/run_benchmark.py --mode mtp-on \
  --request bench/results/2026-10-03-metal-10k-mtp/mtp-on/request.json
python3 bench/results/2026-10-03-metal-10k-mtp/compare_results.py
```

复测会覆盖对应目录中的日志和结果。不要在同一已处理过该输入的服务上直接重复，
否则会命中缓存；脚本会对此报错。

git 里只保留 `mtp-on/` 的 `server-config.json` 和 `request.json`，以及 `mtp-off/prompt-tokens.txt`：

- 三组的 `request.json` 完全相同。
- `mtp-off` 和 `mtp-t2` 的服务配置只把 `--mtp-max-t` 的 4 改成 1 和 2，日志路径也随之改变。
- `compare_results.py` 需要三组的原始输出，可以重跑生成，也可以从提交 `f90370f` 取回。
- 配置里的模型路径（`../Q2_0/`）是当时的位置，现在模型在 `Strata-data/` 里，复测前要改路径。
