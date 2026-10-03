# Mac 一键安装（setup.sh）装出的版本，2026-10-03

> 仓库里保留本目录的 README、汇总和驱动脚本；原始运行输出（`run/engine.log`）只留在测量的机器上，不进 git。

验证内容：`./setup.sh` 在 Mac 上自己编出来的引擎，加上它写出的配置（`strata-iq2_xs.json`，路径相对于 Strata 目录），跑出的结果是否就是首页实测的那一版。

测试机器：Apple M2 Max（38 核 GPU、96 GB），macOS 26.5，Xcode 26.1。

## 安装

模型文件已经放在 `Strata-data/` 里，所以没有下载任何东西。运行命令是 `./setup.sh --yes --no-start`。PATH 里只放了 `.venv/bin` 和系统目录，所以用到的是 pip 装的 CMake 4.4.3 和 Ninja 1.13.2，没有用 Homebrew 的 CMake。

- 从空的 `build-metal-engine/` 开始编译，共 113 步，`setup.sh` 总耗时 **25 s**。
- 编译出的二进制 SHA-256 为 `71cb4981…`，和首页那次的 `02bf0dfe…` 不同：编译目录和 CMake 都换了。
- pack 由 `tools/iq_pack.py` 从挪过来的 GGUF 重新生成，用时 4.3 s，和原来的 pack 逐文件比较完全相同。

第一次运行时发现一个问题：`STRATA_BUILD_TESTS=OFF` 时 CMake 不定义 `strata` 目标，因为引擎被包在测试块里。已经改了 CMakeLists.txt；打开测试时注册的测试和改之前一样，共 50 个。

## 跑分

用 `run.py` 跑，内容同 `../2026-10-03-metal-readme/run.py`，只把默认配置换成了 setup 写出的配置。条件：一个引擎，没有其他引擎在跑，温度 0，关闭思考，不复用前缀。

| 运行 | 首页那次（`../2026-10-03-metal-readme/`） | 这次（setup 装出的） | 输出文字 |
| --- | ---: | ---: | --- |
| 短对话 1 decode | 24.02 token/s | 23.92 token/s | 相同 |
| 短对话 2 decode | 23.85 | 23.87 | 相同 |
| 短对话 3 decode | 23.71 | 23.76 | 相同 |
| 短对话 4 decode | 23.61 | 23.79 | 相同 |
| 30,000-token 文档读取 | 196.2 token/s | 198.3 token/s | 相同 |
| 读完后 decode | 22.7 | 23.6 | 相同 |
| 启动 | 95.9 s | 29.1 s | — |

启动快慢取决于系统缓存里还剩多少专家页。这次之前刚跑过一次服务端测试，缓存是热的。

参数去掉路径后两次完全相同，环境变量也相同。引擎日志显示：24,576 个专家全部常驻（33.02 GiB），token graph 在 GPU 上决定专家，decode 专家命中率 100%。这些都和首页那次一致。

## 服务端

在 `/tmp` 目录里启动 `serve/server.py --config <Strata>/strata-iq2_xs.json`，用来确认配置里的相对路径按配置所在目录解析，而不是按当前目录。结果：

- 94 s 后 `/health` 返回 `loaded: true`。
- 中文短问答回答正常，生成 36 个 token。
- `strata-iq2_xs.log` 写在 Strata 目录里。
- 发 SIGTERM 后服务端和引擎都正常退出。

这些是数值一致性和速度的检查，不是回答质量评测。
