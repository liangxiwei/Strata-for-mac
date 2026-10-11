#!/usr/bin/env python3
"""Summarize measured data; exclude the early build-overlap screening baseline."""
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def main():
    rows = json.loads((OUT / "summary.json").read_text())
    selected = json.loads((OUT / "selected-window.json").read_text())["window"]
    kinds = {
        "off": lambda r: r["label"].startswith("control-r"),
        "optimized_mtp": lambda r: r["label"] == f"w{selected}-r0" or r["label"].startswith("final-r"),
        "legacy_mtp": lambda r: r["label"].startswith("baseline-clean-r"),
        "snapshot_integration": lambda r: r["label"] == "gdn-integration",
    }
    groups = []
    for kind, match in kinds.items():
        for length in (1000, 4000):
            runs = [r for r in rows if match(r) and r["approx_prompt_tokens"] == length]
            if not runs:
                continue
            groups.append({"mode": kind, "prompt_tokens": runs[0]["prompt_tokens"], "runs": len(runs),
                           "window": runs[0]["window"],
                           "median_decode_tps": statistics.median(r["decode_tps"] for r in runs),
                           "decode_tps_range": [min(r["decode_tps"] for r in runs), max(r["decode_tps"] for r in runs)],
                           "individual_decode_tps": [r["decode_tps"] for r in runs],
                           "identical_outputs": sum(r["identical_to_strata_off"] for r in runs),
                           "accepted_drafts": sum(r["timings"]["draft_n_accepted"] for r in runs),
                           "offered_drafts": sum(r["timings"]["draft_n"] for r in runs)})
    for group in groups:
        off = next(g for g in groups if g["mode"] == "off" and g["prompt_tokens"] == group["prompt_tokens"])
        group["gain_over_off_percent"] = (group["median_decode_tps"] / off["median_decode_tps"] - 1) * 100
        legacy = next((g for g in groups if g["mode"] == "legacy_mtp" and
                       g["prompt_tokens"] == group["prompt_tokens"]), None)
        if legacy:
            group["gain_over_legacy_mtp_percent"] = (group["median_decode_tps"] / legacy["median_decode_tps"] - 1) * 100
    report = {"hardware": "Apple M2 Max, 38 GPU cores, 96 GiB unified RAM, Low Power Mode retained",
              "main_model": "existing IQ3_S GGUF shards; unchanged",
              "mtp_model": "original Strata Q2_0 experts, existing dense runtime; unchanged",
              "conditions": {"capacity": 131072, "generated_tokens": 512, "temperature": 0,
                             "thinking": False, "prompt_cache": False, "warmup_excluded": True},
              "selected_mtp_window": selected, "groups": groups,
              "hc_microbench": json.loads((OUT / "hc-microbench.json").read_text()),
              "gdn_microbench": json.loads((OUT / "gdn-microbench.json").read_text()),
              "correctness": {"hc_bitwise_cases": 120, "gdn_accepted_prefix_cases": 174,
                              "python_tests": 149, "measured_request_parity":
                                  [sum(r["identical_to_strata_off"] for r in rows), len(rows)]},
              "caveats": ["baseline-r0 overlapped CPU compilation; excluded from final comparisons",
                          "MTP window versus off was alternated in one loaded process",
                          "clean legacy MTP baseline uses a separate loaded process",
                          "GDN microbench combines verification and accepted-prefix commit; full-model snapshot case is diagnostic"],
              "cleanup": json.loads((OUT / "cleanup.json").read_text())}
    if (OUT / "installation.json").exists():
        report["installation"] = json.loads((OUT / "installation.json").read_text())
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    text = ["Strata Metal MTP 优化实测（2026-10-11）", "",
            report["hardware"], "主模型为已下载 IQ3_S；使用原 Strata Q2_0 MTP。没有修改或下载模型权重。",
            "输入 1139 / 4289 tokens，每次生成 512 tokens，temperature=0，关闭思考和提示缓存。",
            "排除 64-token 预热；任意时刻只加载一个推理模型。", "",
            "模式 | 输入tokens | 窗口 | 次数 | 解码中位数(tok/s) | 相对关闭MTP | 文本一致"]
    for g in groups:
        text.append(f"{g['mode']} | {g['prompt_tokens']} | {g['window']} | {g['runs']} | "
                    f"{g['median_decode_tps']:.3f} | {g['gain_over_off_percent']:+.2f}% | "
                    f"{g['identical_outputs']}/{g['runs']}")
    text += ["", "代码改动：",
             "1. HC down/up 将连续的 2–8-token 张量合并计算，每个权重片段只读取一次。保留每个 token 的 FMA 和归约顺序。",
             "   norm 仍逐 token；4-token HC 从 12 次 dispatch 减至 6 次。非连续布局和单 token 走原路径。",
             "   STRATA_METAL_HC_BATCH=0 可回退。默认启用。",
             "2. Metal 实际启用 MTP 草稿时，setup.start 自动安装已有的小词表；不覆盖自定义词表。",
             "   关闭草稿时不新增 draft-head 的内存占用。测试词表含 106299 个 token。",
             "3. 实现可选 GDN 前缀快照，提交只复制被接受的最后状态，不重跑递推。",
             "   STRATA_METAL_GDN_SNAPSHOT=1 启用，仅支持完整模型、专家全驻留的 Metal 验证。默认关闭。",
             "   当前4-token窗口额外432 MiB，微基准没有支持默认启用的收益。",
             "4. strata_tune.mtp_max_t 可按请求缩小 MTP 窗口，范围 1–8，且不超过启动配置的上限。",
             "   1 关闭 MTP 草稿；下一请求恢复启动默认值。用于校准和回退，不改变模型参数。", "",
             "HC 交替微基准（每个点持续至少10秒）："]
    for r in report["hc_microbench"]:
        text.append(f"T={r['window']}: {r['legacy_us']:.3f} -> {r['batched_us']:.3f} us，耗时减少{r['time_reduction_percent']:.2f}%")
    text += ["", "GDN 交替微基准（验证+提交）："]
    for r in report["gdn_microbench"]:
        text.append(f"T={r['window']}: replay {r['replay_us']:.3f}，snapshot {r['snapshot_us']:.3f} us，"
                    f"耗时减少{r['time_reduction_percent']:+.2f}%")
    text += ["", "验证：HC 120个逐位对照案例、GDN 174个接受前缀案例、149项Python测试全部通过。",
             f"完整模型请求文本一致：{report['correctness']['measured_request_parity'][0]}/"
             f"{report['correctness']['measured_request_parity'][1]}。", "",
             "范围与限制：只测本机 Metal/IQ3_S；不推及 CUDA/HIP、其他主模型或随机采样。",
             "早期 baseline-r0 与CPU编译重叠，已排除最终百分比。GDN局部微基准不代表整模型收益。",
             "窗口与关闭MTP在同一加载进程内交替；干净的旧MTP路径另起单个模型进程测量。",
             "原始数据、trace、配置、构建记录、复现脚本在本目录。", "",
             "最终安装和日常参数见 installation.json；进程清理见 cleanup.json。"]
    if "installation" in report:
        text += ["", "已安装验证过的生产构建；旧引擎及BUILD.json均有备份。",
                 "日常配置未改：mtp-max-t=1，保持关闭MTP草稿。HC批量默认启用，GDN快照默认关闭。"]
    (OUT / "REPORT.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text[:15]))


if __name__ == "__main__":
    main()
