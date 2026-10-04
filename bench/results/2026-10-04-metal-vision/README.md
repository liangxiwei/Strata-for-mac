# Mac 上的图片：Metal 编码器和整条链路，2026-10-04

> 仓库里保留本目录的 README、汇总（`encoder.json`、`e2e.json`）、测试图片（`images/`）和驱动脚本；原始运行输出
> （`run/`：编码结果 `.sve`、引擎和服务端日志）只留在测量的机器上，不进 git。

Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，Xcode 26.1，IQ2_XS。

- 编码器：`strata-vision`，即 llama.cpp mtmd（pinned 3cf0325），用 `-DGGML_METAL=ON` 编译。
- 视觉权重：`mmproj-Qwen3.8-Flash-Next-BF16.gguf`，SHA-256 与 Hugging Face 公布的一致。
- 测试图片由 `make_images.py` 画出，内容已知：
  - 小票：店名加总额；
  - 三个形状：红圆、蓝方、绿三角；
  - 7 个橙色圆点；
  - 一行中文；
  - 仓库自带的 Strata 界面截图，缩到 1280 宽。

## 编码器：Metal 对 CPU（`encoder_parity.py`）

同一程序、同一批图片，一次用 `--gpu`（Metal），一次用 CPU，CPU 的 token 上限也设为 1024。两边是不同的 kernel，
所以只能期望数值接近，不能期望逐位相同。

| 图片 | image token | 网格 | 行余弦中位数 | 1% 分位 | <0.9 的行 | Metal | CPU |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| chinese.png | 225 | 25×9 | 0.99997 | 0.9954 | 0 | 0.46 s | 65.8 s |
| dots.png | 325 | 25×13 | 0.99998 | 0.9990 | 0 | 0.38 s | 95.5 s |
| receipt.png | 400 | 25×16 | 0.99999 | 0.9989 | 0 | 0.45 s | 117.5 s |
| screenshot.jpg | 920 | 40×23 | 0.99999 | 0.9979 | 2 | 1.38 s | 273.4 s |
| shapes.png | 308 | 28×11 | 0.99998 | 0.9990 | 0 | 0.44 s | 90.7 s |

- 五张图的网格两边完全相同，没有非有限值。
- 2,178 行里只有截图的 2 行余弦低于 0.9，最低的一行是 0.695。整体范数比在 0.9991 到 1.0013 之间。
- CPU 每张图要 1 到 4.5 分钟，在 Mac 上不能用，所以 Mac 上只用 Metal 编码。
- 编码器进程的 physical footprint 是 1.1G（在服务端里测，预热到 1024 token；`vmmap`）。

## 整条链路（`e2e.py`）

用 `./setup.sh --setup --model IQ2_XS` 写出的配置启动服务端（`strata-iq2_xs.json`，图片默认开启），再用
OpenAI API 发问。条件：temperature 0，关闭思考，每个问题从新会话开始。

| 问题 | 回答 | 检查 | 用时 |
| --- | --- | --- | ---: |
| 小票的店名和总金额 | 店名是 **STRATA COFFEE**，总金额是 **12.60** | 通过 | 8.7 s |
| 有哪几个形状、什么颜色 | 红色圆形、蓝色正方形、绿色三角形（从左到右） | 通过 | 5.9 s |
| 一共几个圆点 | 7 | 通过 | 3.0 s |
| 把文字原样写出来 | 苹果电脑运行大模型 | 通过 | 2.5 s |
| 描述截图 | 左边是 Strata 的监控面板（GPU 负载、显存、温度、请求记录），右边是写体素宝塔花园的代码编辑器 | 人工核对：对 | 9.8 s |
| 17×23（纯文字） | 391 | 通过 | 1.3 s |
| 追问：这些圆点什么颜色 | 橙色 | 通过 | 2.6 s |

用时是 HTTP 往返时间，包括编码图片、读入 248-941 个 token 和生成回答。追问那一次没有复用缓存（`cached_tokens`
为 0），371 个 token 全部重新读入，用了 2.6 s。

服务端从启动到 ready 用时 126.5 s，包括编码器预热。内存（`vmmap` physical footprint）：

- 引擎 40.2G，编码器 1.1G。
- 引擎的专家缓存仍然是 24,576 个、33.02 GiB，token graph 在 GPU 上决定专家，decode 命中率 100%。和不开图片时一样。

## 开图片后，纯文字是否不变

用 `../2026-10-04-metal-models/run.py`，配置里带 `--vision --vram-reserve-mib 700`，跑首页的五组输入（4 段短对话
和一个 30,000-token 文档）。和 `../2026-10-03-metal-readme/`（不开图片）对比，**五组输出文字完全相同**：

- 写回答：24.11 / 23.96 / 23.90 / 23.90 token/s，原来是 24.02 / 23.85 / 23.71 / 23.61。
- 读文档：201.8 token/s，原来是 196.2。

这些是速度和数值一致性检查。几个问题答对，不等于通用的看图能力评测。

## 复现

```sh
cd bench/results/2026-10-04-metal-vision
../../../.venv/bin/python make_images.py
../../../.venv/bin/python encoder_parity.py      # 需要 build-vision-metal/bin/strata-vision 和 mmproj
cd ../../.. && .venv/bin/python bench/results/2026-10-04-metal-vision/e2e.py   # 一个引擎，8080 端口要空着
```
