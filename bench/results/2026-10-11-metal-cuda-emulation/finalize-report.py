from pathlib import Path
import csv
import difflib
import hashlib
import json
import statistics
import subprocess

root = Path('/Users/liangxw/src/xproject/petproject/Strata')
p = root / 'bench/results/2026-10-11-metal-cuda-emulation'
previous = root / 'bench/results/2026-10-11-metal-mtp-kernel-optimization'

def rows(name):
    r = list(csv.DictReader((p / name).read_text().splitlines()))
    assert all(int(x['bitdiff']) == 0 for x in r)
    return r

def screens(name):
    r = rows(name)
    results = []
    for type_ in (18, 21, 22, 23, 42):
        for cap in (10, 20, 40):
            points = {int(x['mode']): float(x['gpu_us']) for x in r
                      if int(x['type']) == type_ and int(x['cap']) == cap and int(x['round']) == -2}
            for mode in (1, 2):
                results.append({'type':type_, 'cap':cap, 'mode':mode, 'baseline_us':points[0],
                                'candidate_us':points[mode], 'time_change_pct':100 * (points[mode]/points[0]-1)})
    return results

q6 = rows('q6-single-bench.csv')
q6_results = []
for K, N in ((2560, 248320), (2560, 106299), (6144, 2560)):
    r = [x for x in q6 if int(x['K']) == K and int(x['N']) == N and int(x['round']) >= 0]
    a = statistics.median(float(x['gpu_us']) for x in r if x['variant'] == 'old')
    b = statistics.median(float(x['gpu_us']) for x in r if x['variant'] != 'old')
    q6_results.append({'K':K, 'N':N, 'baseline_us':a, 'candidate_us':b,
                       'time_change_pct':100*(b/a-1), 'protocol':'ABBA, >=10 s per point'})

profile = json.loads((root / 'data/mac-metal.json').read_text())
installed = json.loads((previous / 'installed.json').read_text())
quality = json.loads((previous / 'quality/summary.json').read_text())
assert quality['bitwise_pass'] and quality['different'] == 0
assert hashlib.sha256((root / 'engine/strata').read_bytes()).hexdigest() == installed['sha256']
assert 'Ran 68 tests' in (p/'setup-tests.log').read_text() and '\nOK\n' in (p/'setup-tests.log').read_text()

inventory = [
    {'files':['src/kernels/metal/native_mmvq.metal','src/kernels/metal/native_mmvq.mm'],
     'hot_path':'native dense single-token and fallback quantized matrix-vector kernels',
     'decision':'keep validated multi-token fast paths; Q6 single-token candidate lacks a large whole-model gain'},
    {'files':['src/kernels/metal/iq_kernels.metal','src/kernels/metal/iq_kernels.mm'],
     'hot_path':'resident IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS gate/up, Q2_0 draft gate/up',
     'decision':'generic float-dot/vector-permutation and row-reuse candidates mostly regress; keep existing kernels'},
    {'files':['src/kernels/metal/s2_expert_grouped.metal'],
     'hot_path':'alternate S2 expert path, not current native IQ3_S runtime',
     'decision':'record packed-dot/byte-permutation/ballot emulation; no speed claim or unmeasured production edit'},
    {'files':['src/kernels/metal/sampler.metal'],
     'hot_path':'integer multiply-high for counter-based sampling RNG',
     'decision':'preserve exact random bits; this is intentional integer arithmetic'},
    {'files':['src/kernels/metal/strata_port.metalh','src/kernels/metal/kv_q4.metal','src/kernels/metal/kv_q8.metal'],
     'hot_path':'half/quantized storage conversion and bit packing',
     'decision':'rounding and special-value semantics are required; current daily KV is FP16'},
]

report = {
    'date':'2026-10-12', 'measurement':'Apple M2 Max, 38 GPU cores, 96 GiB, macOS 26.5, existing Low Power Mode',
    'baseline':installed, 'reference_policy':'latest validated MTP-on engine; no MTP-off inference in this scope',
    'inventory':inventory, 'generic_helper_screen':screens('iq-bench.csv'),
    'gu_row_reuse_screen':screens('gu-bench.csv'), 'q6_single_sustained':q6_results,
    'parity':{'primitives':{'cases_per_mode':262144,'values_per_mode':524288,'modes':3,'different':0},
              'generic_helpers':{'cases':180,'values':2210880,'different':0},
              'gu_row_reuse':{'cases':180,'values':2210880,'different':0},
              'q6_single':{'cases':168,'values':387828,'different':0}},
    'new_production_kernels':[], 'setup':{'revision':profile['revision'],'env':profile['env'],
        'mtp_max_t':4,'spec_min_p':0.5,'tests':68,'passed':True,
        'legacy_migration':'once, historical cap 1 with spec > 1; preserve existing overrides and subsequent edits',
        'no_start':'prepare selected draft subset before writing run script'},
    'installed_numerical_gate':{k:v for k,v in quality.items() if k != 'windows'},
    'limits':'Full-model speed/logits/state audit currently IQ3_S only; no new 100K-input MTP or vision audit.'
}
(p/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

lines = [
    'Strata Metal：剩余 CUDA 整数模拟评估与 MTP 默认配置（2026-10-12）', '',
    '结论：保留已安装且验证通过的 MTP 优化代码。本轮没有额外候选达到较大、可信的整模型收益，因此没有新增默认 GPU 算子。setup.sh 的 Mac 安装路径现在默认开启 MTP，并显式使用已验证开关。', '',
    '测量环境：M2 Max / 38 GPU 核 / 96 GiB / macOS 26.5，保留原有低电量模式。主模型为现有 IQ3_S，草稿为原有 Q2_0 MTP + CJK 词表。每次只运行一个 GPU 实验。此轮没有关闭 MTP 进行整模型对比。', '',
    '扫描范围与判断：',
    '1. native_mmvq：单 token 和回退路径仍有 dp4a/字节置换；多 token 的 Q6_K/Q4_K/Q5_K/IQ4 热路径已使用验证过的优化。',
    '2. iq_kernels：常驻 IQ gate/up 仍有 packed dot/SWAR；测试了通用替换和跨行复用，主要格式反而更慢。',
    '3. s2_expert_grouped：还有 dot、字节置换和 ballot 模拟，是另一类专家格式路径；当前 IQ3_S native 配置没有使用它，未声称这里的收益。',
    '4. sampler：multiply-high 用于 RNG，需要保持随机数位序列；量化、KV 打包和 half 转换中的整数位操作有精度/舍入用途，不能简单替换成浮点。', '',
    '实验结果（耗时越低越好）：',
    '通用 dp4a 浮点替换 / 加向量字节置换，type 18 cap 20：222.542 -> 278.883 / 299.937 us；cap 40：464.366 -> 640.453 / 829.070 us。其他主要格式也多数变慢，未加入默认实现。',
    'GU 两行/四行复用，type 21 cap 40：485.531 -> 650.020 / 875.527 us，未加入默认实现。IQ2_S cap 40 的小收益未达筛选阈值。Q2_0 gate/up cap 40 的短时收益没有对应当前草稿 cap 10 的较大收益。',
    '上述为 8 个命令缓冲采样的短时筛选，轮换专家权重，GPU 时间戳计时；失败候选未进入长测或整模型测试。', '',
    'Q6_K 单 token 候选持续测量（ABBA，每点至少 10 秒）：',
]
for x in q6_results:
    lines.append(f"K={x['K']}, N={x['N']}：{x['baseline_us']:.3f} -> {x['candidate_us']:.3f} us，耗时变化 {x['time_change_pct']:+.2f}%。")
lines += [
    '草稿头短时筛选曾显示接近减半，持续测量纠正为约 12% 的局部收益（每次约 0.234 ms）；主模型输出头没有收益。没有据此声称整模型提速，保留为实验原型。', '',
    '正确性：通用原语每模式 262144 组输入 / 524288 输出，对独立 CPU 参考，含 int32 溢出；三个模式零差异。通用 GU 与行复用各 180 例 / 2210880 值，Q6 单 token 168 例 / 387828 值，均逐位一致。Metal 编译使用 -fno-fast-math -ffp-contract=off。', '',
    '默认安装配置：--spec 4 --mtp-max-t 4 --spec-min-p 0.5，suffix draft 和 GDN snapshot 关闭；词表在 --no-start 时也准备好。旧 cap-1/spec>1 的 Mac 默认配置升级一次，记录 revision，保留自定义窗口/环境变量，以及升级后的手动编辑。源码指纹变化会触发 Metal 重编译。',
    f"配置版本：{profile['revision']}。68 项 Mac 安装与草稿词表测试通过（CPU，模拟下载/编译/启动）。", '',
    '已安装代码的证据：MTP 窗口 4 两种输入的三轮中位数为 21.50 / 21.32 tok/s（约 1K / 4K 提示，512 greedy 输出，128K 容量，prefill 2048）。MTP-on new/stage1/new 的单次 A/B/A 平均改善为约 4.9% / 19.4%；存在环境波动，不是置信区间。',
    '短/10K/缓存续聊，256K 容量，MTP 两边开启：117703680 个有限 logits / 474 位置零位差；输出 ID、草稿数和六组已提交状态指纹一致。详见上一轮报告与原始记录。', '',
    '适用范围：按张量格式、尺寸和 Metal 后端选核，不按 IQ3_S 文件名判断。其他受支持模型包含相应格式时也可使用；是否整模型更快还取决于格式占比、MTP 接受率和内存。全模型速度与 logits/状态审计目前只在 IQ3_S 验证，不能把同样百分比推广到其他模型或 CUDA/HIP 后端。', '',
    f"引擎 SHA-256：{installed['sha256']}", f"源码指纹：{installed['metadata']['src']}",
    '日常配置继续保留 262144 容量 / 2048 prefill。所有 owned 测试进程在结束后关闭；未修改其他应用或清理系统 swap。',
]
(p/'REPORT.txt').write_text('\n'.join(lines)+'\n')

patches = []
for rel in ('src/kernels/metal/iq_kernels.metal','src/kernels/metal/native_mmvq.metal',
            'src/kernels/metal/native_mmvq.mm','setup.py','setup.sh','data/mac-metal.json','tools/test_setup_mac.py'):
    before = p/'before'/rel
    if not before.exists():
        before = p/'before'/Path(rel).name
    if before.exists():
        patches.extend(difflib.unified_diff(before.read_text().splitlines(True),(root/rel).read_text().splitlines(True),
                                            fromfile='a/'+rel,tofile='b/'+rel))
(p/'setup-profile.patch').write_text(''.join(patches))
print(json.dumps({'report':str(p/'REPORT.txt'), 'q6_sustained':q6_results, 'new_default_kernels':0, 'setup_tests':68}, indent=2))
