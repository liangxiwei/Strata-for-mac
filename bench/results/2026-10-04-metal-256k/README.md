# Mac 上的 256K 上下文，2026-10-04

Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，Xcode 26.1，接电源、Energy Mode Automatic（`powermode 0`）。引擎从提交
`94fbf69` 编译。每个模型由新进程启动，FP16 KV、1,024-token prefill chunk、全专家常驻 GPU；测试只走文字路径，去掉图片
参数。原始 `run/` 输出按仓库规则留在测量机器，目录内保留驱动和汇总 JSON。

## 测法

将 setup 写出的配置中的 `--max-context` 改为 262144，不加入 RoPE 参数。固定提示来自提交 `dc047b5` 的 `docs/` 和
`src/core/*.cpp`；套上对话模板后正好 256,000 token，token-id JSON 的 SHA-256 是
`44c53fdefffe8d78051852804152d1de508a5a541ea385778a3d460889dc4bbd`。提示中间有三个字段：项目代号“银杏”、验收日期
“10月18日”、校验码“AX-7319”。temperature 为 0。

首次完整读入后要求模型答出三项，再在同一会话追问校验码。`iq2_xs.json` 和 `q2_0.json` 保存了实际引擎参数、时间和回答。

## 结果

| | IQ2_XS | Q2_0 |
| --- | ---: | ---: |
| 读入 256,000 token | 1,602.5 s，**159.8 tok/s** | 1,556.5 s，**164.5 tok/s** |
| 首个 token | 1,602.6 s | 1,556.7 s |
| 读完后写回答 | 24.35 tok/s（26 token） | 21.48 tok/s（26 token） |
| 三个字段 | **3/3** | **3/3** |
| 追问：复用 / 新读 | 256,025 / 24 token，0.919 s | 256,025 / 24 token，0.819 s |
| 追问首个 token | 1.02 s | 0.92 s |
| physical footprint 峰值 | **47.0G** | **45.4G** |
| 启动 | 38.6 s | 93.1 s |

这只是每个模型一条固定长提示，证明该提示的三项能被读回，不能代表一般长上下文质量。首个回答只有 26 token，decode
也只是小样本。256K 在模型训练上下文内，两个运行都没有 RoPE 扩展、内存不足或专家回退。

## 复现

确认接电源、Energy Mode 不是 Low Power，并一次只启动一个引擎：

```sh
caffeinate -dimsu .venv/bin/python bench/results/2026-10-04-metal-256k/long_context.py --config strata-iq2_xs.json
caffeinate -dimsu .venv/bin/python bench/results/2026-10-04-metal-256k/long_context.py --config strata-q2_0.json
```

每个模型约 27 分钟。
