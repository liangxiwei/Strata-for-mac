#!/usr/bin/env python3
"""Sequential local HTTP checks; start the server separately and keep the GPU idle."""
import json
import argparse
from pathlib import Path
import urllib.request

out = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="server-final")
args = parser.parse_args()

def ask(name, messages):
    name = name.replace("server-final", args.prefix, 1)
    body = dict(model="strata", messages=messages, max_tokens=128, temperature=0,
                chat_template_kwargs={"enable_thinking": False})
    (out / (name + "-request.json")).write_text(json.dumps(body, ensure_ascii=False, indent=2))
    request = urllib.request.Request("http://127.0.0.1:18080/v1/chat/completions",
                                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    (out / (name + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    text = result["choices"][0]["message"].get("content") or ""
    assert text, result
    print(name, result.get("timings"), text, flush=True)
    return text

ask("server-final-short", [{"role": "user", "content": "请用中文用两句话解释电脑中的内存和硬盘有什么区别。"}])
paragraphs = [
    "银杏项目的验收日期是10月18日，目标是降低本地推理的首字延迟。项目使用统一内存保存模型权重。",
    "工程师先记录原始速度，再分析每个算子的运行时间。提示词处理与逐字生成分别测量，因为两者的计算方式不同。",
    "矩阵运算在显卡上执行，主机负责读取请求、编码文本和返回结果。减少主机与显卡的反复同步可以缩短等待时间。",
    "混合专家模型每一层会根据输入选择少量专家。所有专家共享同一个权重区域，避免重复复制，也避免请求之间相互覆盖。",
    "测试数据包含中文问答、英文段落和代码片段。每一次修改都检查输出是否包含异常数值，并比较独立参考实现的误差。",
    "服务连续接收多个请求，也支持用户在同一段对话中追问。已经处理过的上下文可以缓存，但只能在状态一致时复用。",
    "测量时固定模型文件、机器和提示词，不同时运行其他推理任务。测试报告记录输入长度、输出长度和具体启动参数。",
    "研发小组每周整理一次测试结果。发现性能退化时，保留原始日志并恢复上一次经过验证的执行路径。",
]
document = "\n\n".join(paragraphs * 4)
messages = [{"role": "user", "content": document + "\n\n请只用三句话说明：项目代号、验收日期、项目目标。"}]
answer = ask("server-final-long", messages)
assert "银杏" in answer and "18" in answer, answer
messages += [{"role": "assistant", "content": answer},
             {"role": "user", "content": "根据上面的项目文档，主机具体负责哪三件事？用一句话回答。"}]
answer = ask("server-final-followup", messages)
assert "编码" in answer and "返回" in answer, answer
ask("server-final-repeat", [{"role": "user", "content": "请用中文用两句话解释电脑中的内存和硬盘有什么区别。"}])
