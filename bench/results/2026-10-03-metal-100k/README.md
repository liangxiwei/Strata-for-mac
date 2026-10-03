# 100K 上下文实测，2026-10-03

后续更正：切换 IQ2_XS 时，独立数值检查发现并修复了 Metal Q3_K 解量化遗漏一个缩放
字节的问题。这里的 Q2_0 模型使用了 Q3_K 张量，因此本报告记录的是带该错误的历史
版本；回答退化不能全部归因于模型量化。100K 尚未用修复后的版本复测。

Apple M2 Max（38 核 GPU、96 GB 内存），macOS 26.5，Q2_0 模型。使用本目录
`server-config.json`，容量 131,072 token，FP16 KV，全量专家驻留 Metal，prefill 自动选择
1,024 token 分块。MTP 推测解码关闭：`--spec 2 --mtp-max-t 1 --suffix-draft 0`。
模型加载不计入请求时间，temperature=0，thinking 关闭。

| 指标 | 首次请求 | 同一会话追问 |
| --- | ---: | ---: |
| 完整输入 token（含模板） | 100,000 | 100,290 |
| 复用缓存 token | 0 | 100,255 |
| 新处理 token | 100,000 | 35 |
| Prefill 时间 | 960.978 秒 | 2.590 秒 |
| Prefill 吞吐 | 104.1 token/s | 13.5 token/s |
| HTTP 首字延迟 | 961.498 秒 | 3.081 秒 |
| 生成 token | 256 | 128 |
| Decode 时间 | 13.766 秒 | 7.674 秒 |
| Decode 吞吐 | 18.6 token/s | 16.7 token/s |
| HTTP 总时间 | 975.218 秒 | 10.701 秒 |

缓存追问只处理了 35 个新增 token。小批次的每 token 吞吐较低，但请求的等待时间大幅
缩短。新 token 仍需要访问历史状态，不能用 13.5 token/s 推断下一次完整 prefill 的速度。

**性能测试完成，回答质量未通过。** 首轮和追问都反复输出“RAM”，没有找出约第 50,000
token 处植入的三个校验字段（银杏、10月18日、AX-7319）。当前不能把这些吞吐数字视为
100K 上下文可用性验证。根因尚未定位，不能据此归因于量化、模型或 Metal 某一模块。

引擎 `vmmap -summary` 的 physical footprint 峰值为 **41.3G**。`ps` RSS 峰值约 9.98 GiB
没有计入大部分 Metal 内存，不能当作引擎总占用。系统已有约 5 GB swap，本轮采样中的
swap used 从 5,047.94 MiB 到 5,007.94 MiB，最大值等于初始值，没有观测到新增 swap 占用。

证据：`result.json`、`followup-result.json`、`summary.json`、`stream.jsonl`、
`followup-stream.jsonl`、`engine.log`、`monitor.jsonl`、`progress.jsonl`、
`memory-final.txt`。`request.json` 保存完整输入；`workload.json` 保存来源和输入哈希；
`environment.json` 保存二进制哈希和环境。原日用配置的 32K 容量没有修改。

复现（仓库根目录，先在另一终端启动服务）：

```bash
STRATA_DECODE_TIMING=1 STRATA_TRACE=1 python3 -m serve.server --engine strata \
  --config bench/results/2026-10-03-metal-100k/server-config.json --port 18080
python3 bench/results/2026-10-03-metal-100k/run_benchmark.py
python3 bench/results/2026-10-03-metal-100k/run_benchmark.py --followup
```

脚本会重新从仓库资料生成输入；修改源文件后输入哈希可能变化。精确复测应使用已保存的
`request.json`。首轮需启动新的服务进程，避免命中前缀缓存。
