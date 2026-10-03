#!/usr/bin/env python3
"""build-ple/gen/ple_fixture.py - deterministic fixtures for ple_parity (Metal port, K20).

Everything ple_parity's --selftest needs that the real artifact would provide, generated into THIS build
dir (the tools/iq_fixture.py precedent):

  bench/micro/ple_in.bin, ple_out.bin   the ggml-capture-shaped block fixture + its oracle, computed with
                                        the same arithmetic as ple_parity.cpp's own host f32 reference
                                        (f64 accumulators, f32-rounded products where that reference
                                        rounds them), so the in-test "host f32 reference" check sees ~1e-7
  pack/full/dense.bin                   the three PLE weight regions at the offsets pinned in
                                        ple_oracle_vectors.inc (sparse elsewhere): Q2_0 key codes/scales
                                        that decode EXACTLY to ple_in.bin's w_key, and the BF16-promoted
                                        ple_value
  table.gguf                            a sparse 28.8 GB IQ4_NL GGUF whose data starts at byte 192 with
                                        320001536 rows; rows 0 and 12345 are reconstructed so their first
                                        and last 8 values are BIT-EQUAL to kProbeHead/kProbeTail and their
                                        L1 sum matches kProbeSum to < 1e-9.  Every IQ4_NL value is
                                        cb[c] * d with d a positive-or-negative fp16, i.e. an integer
                                        multiple of 2^-24, so the row's L1 sum is an INTEGER times 2^-24
                                        and the reconstruction is an integer equation (solve_row below);
                                        the last row is a hole and decodes to zeros, which is exactly what
                                        its oracle vector says.

Deterministic: fixed seed, fixed geometry.  Needs numpy.
"""
import struct
from pathlib import Path

import numpy as np

BUILD = Path(__file__).resolve().parents[1]
SEED = 20261002
N_EMBD, HC, HC_DIM, KERN, DIL, NHIST = 2560, 4, 10240, 4, 3, 9
NT = 2
EPS = np.float32(1e-6)
CB = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int64)
ABS_CB = np.abs(CB)
# the pinned oracle values (ple_oracle_vectors.inc, verbatim)
K_TABLE_ROWS = 320001536
K_DATA_START = 192
K_FILE_SIZE = 28800138432
PROBE_ROWS = [0, 12345, 320001535]
PROBE_HEAD = [
    [-1.001119614e-02, 1.290917397e-02, -1.817822456e-02, 1.290917397e-02, -6.586313248e-03, 5.795955658e-03, -6.586313248e-03, 5.795955658e-03],
    [5.563795567e-03, -4.314780235e-03, 3.974139690e-03, -1.476109028e-03, -1.135468483e-04, 1.135468483e-03, -1.010566950e-02, -6.017982960e-03],
    [0.0] * 8,
]
PROBE_TAIL = [
    [2.571296692e-02, 1.607060432e-02, 3.139948845e-02, 5.439281464e-03, -1.705956459e-02, -3.214120865e-03, -2.200436592e-02, 2.472400665e-03],
    [-1.942408085e-02, 5.811929703e-03, -9.941458702e-03, 3.823637962e-03, -5.353093147e-03, -1.529455185e-03, -1.529455185e-03, -5.353093147e-03],
    [0.0] * 8,
]
PROBE_SUM = [2.011959553e00, 9.968549609e-01, 0.0]
K_KEY_CODES_OFF, K_KEY_CODES_BYTES = 1017723776, 6553600
K_KEY_SCALES_OFF, K_KEY_SCALES_BYTES = 1024277376, 819200
K_VALUE_OFF, K_VALUE_BYTES = 1025219456, 26214400


def f32(x):
    return np.float32(x)


def h2f(h):
    """fp16 bits -> f32 (every fp16 is an exact f32)"""
    return np.array([h], dtype=np.uint16).view(np.float16).astype(np.float32)[0]


def f2h_bits(x):
    return int(np.float16(np.float32(x)).view(np.uint16))


# ---------------------------------------------------------------- codebook magnitude-sum DP
INC = np.unique(ABS_CB - 1)
INC = INC[INC > 0]


def reach_inc(n):
    """bool array d: d[x] iff x is a sum of increments (|cb|-1) over AT MOST n lanes."""
    d = np.zeros(n * 126 + 1, dtype=bool)
    d[0] = True
    for _ in range(n):
        nd = d.copy()
        for i in INC:
            nd[i:] |= d[:-i]
        d = nd
    return d


def reachable(n_lanes):
    """bool array r: r[s] iff `n_lanes` codebook magnitudes sum to s (r has n_lanes*127+1 entries)."""
    r = np.zeros(n_lanes * 127 + 1, dtype=bool)
    r[n_lanes:] = reach_inc(n_lanes)
    return r


def codes_for_abs_sum(total, n_lanes, lut=None, cache=None):
    """n_lanes codebook POSITIVE-code magnitudes whose |cb| sum is exactly `total` (a witness for `lut`)."""
    if cache is None:
        cache = {0: np.ones(1, dtype=bool)}          # reach_inc(0) == [True]
    if 1 not in cache:
        cache[1] = reach_inc(1)
    mags = np.ones(n_lanes, dtype=np.int64)          # all |cb| = 1 (sum n_lanes, increment 0)
    rem = total - n_lanes                            # the increment budget left to place
    inc = sorted(int(x) for x in INC)
    for i in range(n_lanes):
        if rem == 0:
            break
        lanes_left = n_lanes - i - 1
        if lanes_left not in cache:
            cache[lanes_left] = reach_inc(lanes_left)
        sub = cache[lanes_left]
        for cand in sorted(inc, reverse=True):       # largest first, each step exactly feasible
            if cand > rem:
                continue
            rest = rem - cand
            if rest == 0 or (rest < len(sub) and sub[rest]):
                mags[i] = cand + 1
                rem -= cand
                break
        else:
            raise AssertionError(f"greedy stuck: rem {rem} at lane {i}/{n_lanes}")
    assert int(mags.sum()) == total, (int(mags.sum()), total)
    return mags


def mag_to_code_index(mag):
    return int(np.where(ABS_CB == mag)[0][0])


# ---------------------------------------------------------------- the table GGUF's probe rows
def recover_d(values):
    """The one fp16 scale whose f32 products with the codebook cover ALL pinned values exactly."""
    common = None
    for v in values:
        v = f32(v)
        cs = set()
        for c in CB:
            q = np.float64(v) / np.float64(c)      # the exact quotient: v = d*c exactly, d representable
            h = np.float16(f32(q))
            if not np.isfinite(h) or h == 0:
                continue
            d = f32(h)
            if any(f32(d * f32(cc)) == v for cc in CB):
                cs.add(float(d))
        common = cs if common is None else (common & cs)
        if not common:
            raise AssertionError(f"no scale covers value {v}")
    assert len(common) >= 1
    return f32(sorted(common)[0])


def solve_row(head, tail, want_sum):
    """Reconstruct one 160-element IQ4_NL row: 5 blocks x {fp16 d ; 16 qs bytes}.

    Head pins elements 0..7 (low nibbles of block 0's bytes 0..7), tail pins 152..159 (high nibbles of
    block 4's bytes 8..15).  Every value is cb[c]*d with d fp16, so |value| = |d|*|cb[c]| and the row's L1
    sum is (integer) * 2^-24.  With K(d) = |d|*2^24 the equation is
        K0*M0 + K4*M4 + 8192*(M1+M2) + 1*M3 = N
    over codebook-reachable M's: blocks 1,2 at d = 2^-11 (K = 8192) carry the bulk, block 3 at d = 2^-24
    (K = 1) is the fine residue, and the pinned blocks' 24 free lanes each absorb their K in steps of ~1.
    """
    d0, d4 = recover_d(head), recover_d(tail)

    def pinned_idx(d, values):
        out = []
        for v in values:
            v = f32(v)
            hits = [i for i in range(16) if f32(d * f32(CB[i])) == v]
            assert hits, (d, v)
            out.append(hits[0])
        return out

    head_codes = pinned_idx(d0, head)
    tail_codes = pinned_idx(d4, tail)
    P0 = sum(abs(int(CB[i])) for i in head_codes)
    P4 = sum(abs(int(CB[i])) for i in tail_codes)
    k0 = int(round(abs(np.float64(np.float32(d0))) * (1 << 24)))
    k4 = int(round(abs(np.float64(np.float32(d4))) * (1 << 24)))

    n_lo = int(np.ceil((want_sum - 1e-9) * (1 << 24)))
    n_hi = int(np.floor((want_sum + 1e-9) * (1 << 24)))
    assert k0 * (P0 + 24) + k4 * (P4 + 24) <= n_hi, "pinned minimum already overshoots the target sum"

    lut24 = reachable(24)
    lut32 = reachable(32)
    m0s = np.array([P0 + 24 + i for i in range(len(lut24)) if lut24[i]], dtype=np.int64)
    m4s = np.array([P4 + 24 + i for i in range(len(lut24)) if lut24[i]], dtype=np.int64)
    grid = k0 * m0s[:, None] + k4 * m4s[None, :]

    def split2(s, lo, hi):
        """s = m1 + m2 with both in the 32-lane reachable set (dense above ~150)."""
        for m1 in range(min(hi, s - lo), lo - 1, -1):
            m2 = s - m1
            if lo <= m2 <= hi and lut32[m1] and lut32[m2]:
                return m1, m2
        return None

    # the target N is pinned within ~1e-9/2^-24 = ~17 integers: try each, first hit wins
    for N in range(n_lo, n_hi + 1):
        R = N - grid                                 # must equal 8192*(M1+M2) + M3
        r = R % 8192                                 # the fine block's target (K = 1)
        ok_r = (r >= 150) & (r <= 4060)
        s12 = (R - r) // 8192                        # blocks 1+2's magnitude sum (K = 8192)
        ok_s = (s12 >= 300) & (s12 <= 8100)
        hit = np.argwhere(ok_r & ok_s)
        if len(hit) == 0:
            continue
        a, b = hit[0]
        m0, m4, m3, sm = int(m0s[a]), int(m4s[b]), int(r[a, b]), int(s12[a, b])
        pair = split2(sm, 150, 4060)
        if pair is None:
            continue
        m1, m2 = pair
        if not lut32[m3]:
            continue
        return dict(d0=d0, d4=d4, head_codes=head_codes, tail_codes=tail_codes,
                    m0=m0, m4=m4, m1=m1, m2=m2, m3=m3)
    raise AssertionError(f"no integer solution for sum {want_sum}")


def assemble_row(sol):
    """5 blocks -> 90 bytes (and the decoded 160 f32 values for the caller's self-check)."""
    d_bulk = h2f(0x1000)     # 2^-11 (exp field 4, mantissa 0): blocks 1,2, K = 8192, the bulk
    d_fine = h2f(0x0001)     # 2^-24 (the min fp16 subnormal, K = 1): block 3, the fine residue

    def block(d, elem):      # elem: 32 SIGNED codebook values, index = element
        qs = bytes((elem[j] & 0xF) | ((elem[j + 16] & 0xF) << 4) for j in range(16))
        return struct.pack("<H", f2h_bits(d)) + qs

    lut24 = reachable(24)
    e0 = [0] * 32
    e0[0:8] = sol["head_codes"]
    f0 = sol["m0"] - sum(abs(int(CB[c])) for c in sol["head_codes"])
    for i, m in enumerate(codes_for_abs_sum(f0, 24, lut24)):
        e0[8 + i] = mag_to_code_index(int(m))
    e4 = [0] * 32
    e4[24:32] = sol["tail_codes"]
    f4 = sol["m4"] - sum(abs(int(CB[c])) for c in sol["tail_codes"])
    for i, m in enumerate(codes_for_abs_sum(f4, 24, lut24)):
        e4[i] = mag_to_code_index(int(m))
    row = block(sol["d0"], e0)
    row += block(d_bulk, [mag_to_code_index(int(x)) for x in codes_for_abs_sum(sol["m1"], 32)])
    row += block(d_bulk, [mag_to_code_index(int(x)) for x in codes_for_abs_sum(sol["m2"], 32)])
    row += block(d_fine, [mag_to_code_index(int(x)) for x in codes_for_abs_sum(sol["m3"], 32)])
    row += block(sol["d4"], e4)
    assert len(row) == 90
    return row


def decode_row(row):
    out = np.zeros(160, dtype=np.float32)
    for b in range(5):
        blk = row[b * 18:(b + 1) * 18]
        d = h2f(struct.unpack("<H", blk[:2])[0])
        qs = blk[2:]
        for j in range(16):
            out[b * 32 + j] = f32(d * f32(CB[qs[j] & 0xF]))
            out[b * 32 + j + 16] = f32(d * f32(CB[qs[j] >> 4]))
    return out


def write_table_gguf(path):
    """GGUF v3, one IQ4_NL tensor [160, 320001536], data_start exactly 192 (kTableDataStart)."""
    def kv_str(key, val):
        return (struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8) +
                struct.pack("<Q", len(val)) + val.encode())

    def kv_u32(key, val):
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + struct.pack("<I", val)

    name = b"per_layer_token_embd.weight"
    tensor_info = struct.pack("<Q", len(name)) + name
    tensor_info += struct.pack("<I", 2)                       # n_dims
    tensor_info += struct.pack("<QQ", 160, K_TABLE_ROWS)      # ne0 = 160 is the FAST axis
    tensor_info += struct.pack("<I", 20)                      # GGML type IQ4_NL
    tensor_info += struct.pack("<Q", 0)                       # offset within the data section
    # data_start = align32(24 + kv + len(tensor_info)) must be 192 -> kv in [75, 106]
    kv = kv_u32("general.alignment", 32) + kv_str("strata.ple.padding", "x" * 16)
    assert 75 <= len(kv) <= 106, len(kv)
    head = struct.pack("<IIQQ", 0x46554747, 3, 1, 2) + kv + tensor_info
    data_start = (len(head) + 31) // 32 * 32
    assert data_start == K_DATA_START, (len(head), data_start)
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\0" * (data_start - len(head)))
        f.truncate(K_FILE_SIZE)
        for p, row_id in enumerate(PROBE_ROWS[:2]):
            sol = solve_row(PROBE_HEAD[p], PROBE_TAIL[p], PROBE_SUM[p])
            row = assemble_row(sol)
            got = decode_row(row)
            assert np.array_equal(got[:8], np.array(PROBE_HEAD[p], dtype=np.float32)), f"row {row_id} head"
            assert np.array_equal(got[152:], np.array(PROBE_TAIL[p], dtype=np.float32)), f"row {row_id} tail"
            err = abs(float(np.abs(got.astype(np.float64)).sum()) - PROBE_SUM[p])
            assert err <= 1e-9, (row_id, err)
            f.seek(K_DATA_START + row_id * 90)
            f.write(row)
            print(f"  row {row_id}: d0={sol['d0']!r} d4={sol['d4']!r} M={sol['m0']},{sol['m1']},"
                  f"{sol['m2']},{sol['m3']},{sol['m4']} sum err {err:.2e}")
    print(f"table.gguf: {K_FILE_SIZE} B (sparse)")


# ---------------------------------------------------------------- the capture + the pack
def gen_capture_and_pack():
    rs = np.random.RandomState(SEED)

    # ---- ple_key as Q2_0: 10240 rows x 40 groups of 64; codes 2-bit (one byte per quad), value (code-1)*s
    n_groups = N_EMBD // 64
    scales_f16 = rs.uniform(0.001, 0.05, size=(HC_DIM, n_groups)).astype(np.float16)   # fp16-exact storage
    scales_f32 = scales_f16.astype(np.float32)
    codes = rs.randint(0, 4, size=(HC_DIM, N_EMBD // 4)).astype(np.uint8)
    quads = np.stack([codes & 3, (codes >> 2) & 3, (codes >> 4) & 3, (codes >> 6) & 3], axis=2)
    vals = quads.astype(np.int32) - 1                                                   # {-1,0,1,2}
    w_key = (vals.reshape(HC_DIM, N_EMBD).astype(np.float32) *
             np.repeat(scales_f32, 64, axis=1)).astype(np.float32)                     # exact in f32

    # ---- ple_value as BF16-representable f32: bf16 bit patterns directly (sane exponents, mixed signs)
    # true bf16 bit patterns (1|8|7): exponent field 120..127 -> |w| in [2^-7, 2)
    m7 = rs.randint(0, 1 << 7, size=(N_EMBD, N_EMBD), dtype=np.uint32)
    efield = rs.randint(120, 128, size=(N_EMBD, N_EMBD), dtype=np.uint32)
    sign = (rs.randint(0, 2, size=(N_EMBD, N_EMBD), dtype=np.uint32) << np.uint32(15))
    bf = (sign | (efield << np.uint32(7)) | m7).astype(np.uint16)
    w_value = (bf.astype(np.uint32) << np.uint32(16)).view(np.float32)                 # exact promotion

    # ---- the norms (f32) and the conv kernel (f16-representable so the kernel's F16 weight IS the oracle's)
    w_nk = rs.uniform(0.5, 1.5, size=HC_DIM).astype(np.float32)
    w_nq = rs.uniform(0.5, 1.5, size=HC_DIM).astype(np.float32)
    w_nc = rs.uniform(0.5, 1.5, size=HC_DIM).astype(np.float32)
    w_conv = rs.normal(0.0, 0.05, size=HC_DIM * KERN).astype(np.float16).astype(np.float32)

    emb = rs.normal(0.0, 0.5, size=(NT, N_EMBD)).astype(np.float32)
    hidden = rs.normal(0.0, 1.0, size=(NT, HC_DIM)).astype(np.float32)
    hist = rs.normal(0.0, 0.7, size=(NHIST, HC_DIM)).astype(np.float32)

    # ================= the oracle: ple_parity.cpp's host f32 reference arithmetic =================
    def gnorm(x, w):
        y = np.zeros_like(x)
        for c in range(HC):
            xv = x[c * N_EMBD:(c + 1) * N_EMBD]
            sq = (xv * xv).astype(np.float32)               # the PRODUCT rounds in f32, then widens
            s = np.float64(sq.astype(np.float64).sum(dtype=np.float64))
            mean = f32(s / np.float64(N_EMBD))
            scale = f32(f32(1.0) / f32(np.sqrt(f32(mean + EPS), dtype=np.float32)))
            y[c * N_EMBD:(c + 1) * N_EMBD] = f32(f32(xv * scale) * w[c * N_EMBD:(c + 1) * N_EMBD])
        return y

    o_key = np.zeros((NT, HC_DIM), np.float32)
    o_value = np.zeros((NT, N_EMBD), np.float32)
    o_gate = np.zeros((NT, HC), np.float32)
    o_gated = np.zeros((NT, HC_DIM), np.float32)
    o_norm = np.zeros((NT, HC_DIM), np.float32)
    o_conv = np.zeros((NT, HC_DIM), np.float32)
    o_res = np.zeros((NT, HC_DIM), np.float32)
    hist_adv = hist.copy()
    idx = np.arange(HC_DIM)
    for t in range(NT):
        e64 = emb[t].astype(np.float64)
        k_ref = gnorm(f32(w_key.astype(np.float64) @ e64), w_nk)
        q_ref = gnorm(hidden[t].copy(), w_nq)
        v_ref = f32(w_value.astype(np.float64) @ e64)
        for c in range(HC):
            a = np.float64(np.sum(k_ref[c * N_EMBD:(c + 1) * N_EMBD].astype(np.float64) *
                                  q_ref[c * N_EMBD:(c + 1) * N_EMBD].astype(np.float64), dtype=np.float64))
            s = f32(f32(a / np.float64(1.0)) / f32(np.sqrt(f32(N_EMBD), dtype=np.float32)))
            mag = f32(np.sqrt(f32(max(abs(float(s)), 1e-6)), dtype=np.float32))
            sgn = f32(1.0) if s > 0 else (f32(-1.0) if s < 0 else f32(0.0))
            o_gate[t, c] = f32(f32(1.0) / f32(f32(1.0) + f32(np.exp(f32(-f32(sgn * mag)), dtype=np.float32))))
        gated = f32(v_ref[idx % N_EMBD] * o_gate[t][idx // N_EMBD])
        norm = gnorm(gated.copy(), w_nc)
        conv = np.zeros(HC_DIM, np.float32)
        for c in range(HC_DIM):
            acc = np.float32(0.0)
            for kk in range(KERN):
                row = NHIST - (KERN - 1 - kk) * DIL
                v = norm[c] if row == NHIST else hist_adv[row, c]
                acc = f32(acc + f32(w_conv[kk + KERN * c] * v))
            conv[c] = f32(acc / f32(f32(1.0) + f32(np.exp(f32(-acc), dtype=np.float32))))
        o_key[t], o_value[t], o_gated[t], o_norm[t] = k_ref, v_ref, gated, norm
        o_conv[t] = conv
        o_res[t] = f32(f32(hidden[t] + gated) + conv)
        hist_adv = np.concatenate([hist_adv[1:], norm[None, :]], axis=0)   # slide by one NORMALIZED row

    # ---- ple_in.bin / ple_out.bin (the byte sizes ple_parity's load_capture recomputes)
    micro = BUILD / "bench" / "micro"
    micro.mkdir(parents=True, exist_ok=True)
    with open(micro / "ple_in.bin", "wb") as f:
        f.write(struct.pack("<5i", N_EMBD, HC, NT, KERN, DIL))
        f.write(struct.pack("<f", EPS))
        for a in (emb, hidden, w_key, w_value, w_nk, w_nq, w_nc, w_conv):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        # `hist` is ROW-FASTEST in the capture (ggml ne=(hist, hc_dim)): flat = row + NHIST*channel
        f.write(np.ascontiguousarray(hist.T, dtype=np.float32).tobytes())
    # the oracle is TOKEN-MAJOR PER STAGE (load_capture reads each array over all tokens)
    with open(micro / "ple_out.bin", "wb") as f:
        f.write(struct.pack("<3i", N_EMBD, HC_DIM, NT))
        for a in (o_key, o_value, o_gate, o_gated, o_norm, o_conv, o_res):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

    # ---- pack/full/dense.bin (sparse): the three regions at the pinned offsets
    pack = BUILD / "pack" / "full"
    pack.mkdir(parents=True, exist_ok=True)
    need = max(K_KEY_CODES_OFF + K_KEY_CODES_BYTES, K_KEY_SCALES_OFF + K_KEY_SCALES_BYTES,
               K_VALUE_OFF + K_VALUE_BYTES)
    with open(pack / "dense.bin", "wb") as f:
        f.truncate(need)
        f.seek(K_KEY_CODES_OFF)
        f.write(np.ascontiguousarray(codes, dtype=np.uint8).tobytes())
        f.seek(K_KEY_SCALES_OFF)
        f.write(np.ascontiguousarray(scales_f16).view(np.uint16).tobytes())    # the fp16 BITS
        f.seek(K_VALUE_OFF)
        f.write(np.ascontiguousarray(w_value, dtype=np.float32).tobytes())

    # self-check: the pack's decode of row 0 == w_key row 0 exactly (what ple_parity asserts)
    d0 = np.arange(N_EMBD)
    byte = codes[0][d0 // 4]
    codev = (byte >> ((d0 % 4) * 2)) & 3
    dec = ((codev.astype(np.int32) - 1).astype(np.float32) * scales_f32[0][d0 // 64])
    assert np.array_equal(dec, w_key[0]), "pack/capture w_key mismatch"
    print(f"ple_in.bin / ple_out.bin in {micro}; dense.bin {need} B (sparse) in {pack}")


if __name__ == "__main__":
    write_table_gguf(BUILD / "table.gguf")
    gen_capture_and_pack()
    print("done")
