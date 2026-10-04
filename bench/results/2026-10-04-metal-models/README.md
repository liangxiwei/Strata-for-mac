# Mac 上的 IQ2_XS 和 Q2_0，2026-10-04

> 仓库里保留本目录的 README、汇总（`iq2_xs.json`、`q2_0.json`）和驱动脚本；原始运行输出（`run/`）只留在测量的机器上，不进 git。

Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，Xcode 26.1。

- 两个模型都由 `./setup.sh --setup --model <size>` 装好，配置是 setup 写出的，图片开启。
- 引擎二进制相同，SHA-256 为 `71cb4981…`。
- `run.py` 和 `../2026-10-03-metal-readme/run.py` 是同一个脚本，只多了 `--config` 参数。测的是：
  - 4 段新的短对话，每段生成 256 token；
  - 一个 30,000-token 的文档，读入新上下文。
- 条件：一个引擎，没有其他引擎在跑，temperature 0，关闭思考，不复用前缀。

## Q2_0 的下载

- shard 1 共 37.6 GB，经 hf-mirror 用 4 条连接并行下载。单连接实测约 3.3 MB/s，4 条连接在 6.7-15.7 MB/s 之间波动，
  用时约 45 分钟。
- 下完后 SHA-256 为 `69820c02…`，与 Hugging Face 公布的一致。
- shard 2 与 IQ2_XS 的 shard 2 是同一个文件，SHA-256 都是 `316b46f3…`，所以直接硬链接，没有下载。
- `tools/iq_pack.py` 生成原生 pack 用时约 3 秒，setup 总共用了 3 s。

## 结果

| 运行 | IQ2_XS | Q2_0 |
| --- | ---: | ---: |
| 短对话 1-4 写回答 | 24.11 / 23.96 / 23.90 / 23.90 token/s | 22.81 / 22.80 / 22.55 / 22.49 token/s |
| 30,000-token 文档读取 | 148.7 s，**201.8 token/s** | 146.1 s，**205.3 token/s** |
| 读完后写回答 | 23.1 token/s | 21.9 token/s |
| 专家缓存（全部常驻） | 24,576 个，33.02 GiB | 24,576 个，31.64 GiB |
| 全部载入后剩余的 GPU 内存 | 39,453 MiB | 41,011 MiB |
| 启动 | 106.6 s | 29.1 s |

和 IQ2_XS 比，Q2_0 写回答慢约 5%，读提示快约 2%，专家少占 1.4 GiB。两个模型的引擎日志都显示 token graph 在 GPU
上决定专家，没有 CPU 专家。

- 启动时间取决于系统缓存里还剩多少专家页：Q2_0 的 shard 1 刚下完，还在缓存里。
- 两个模型的量化不同，输出文字本来就不一样。首页的 IQ2_XS 数字和 `../2026-10-03-metal-readme/` 的五组输出文字完全
  相同（那次没开图片）。

首页写的是中位数取整：IQ2_XS 24 / 196，Q2_0 23 / 205 token/s。IQ2_XS 的读取速度沿用首页那次实测的 196；这次是
201.8，在两次运行的波动范围内。

这些是速度测量，不是回答质量评测。

## 复现

```sh
.venv/bin/python bench/results/2026-10-04-metal-models/run.py --config strata-q2_0.json
.venv/bin/python bench/results/2026-10-04-metal-models/run.py --config strata-iq2_xs.json
```
