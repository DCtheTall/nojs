"""
NIST FIPS-197 AES-128 Reference Implementation & Combinational Circuit Builder.

Includes:
  - Algebraic FIPS-197 S-Box generation and pure-Python AES-128-ECB reference cipher
  - Boyar-Peralta (2010) 113-gate combinational AES S-Box circuit
  - Reusable AES-128 Round + KeyExpansion circuit builders
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from circuit import Circuit, LayerAssignment, format_mod2_sum, mobius_anf_4bit


# ---------------------------------------------------------------------------
# 1. Algebraic NIST FIPS-197 Reference Implementation
# ---------------------------------------------------------------------------

def gf_mul(a: int, b: int) -> int:
    """Multiply two bytes in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1 (0x11B)."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


def _build_fips_sbox() -> Tuple[int, ...]:
    table: List[int] = [0] * 256
    for x in range(256):
        # Multiplicative inverse in GF(2^8): x^254
        if x == 0:
            inv = 0
        else:
            # Find inverse via exponentiation / search
            inv = 1
            base = x
            exp = 254
            while exp > 0:
                if exp & 1:
                    inv = gf_mul(inv, base)
                base = gf_mul(base, base)
                exp >>= 1
        # Affine transform over GF(2) + 0x63
        s = inv
        res = 0x63
        for bit in range(8):
            b = (
                ((s >> bit) & 1)
                ^ ((s >> ((bit + 4) % 8)) & 1)
                ^ ((s >> ((bit + 5) % 8)) & 1)
                ^ ((s >> ((bit + 6) % 8)) & 1)
                ^ ((s >> ((bit + 7) % 8)) & 1)
            )
            res ^= b << bit
        table[x] = res
    return tuple(table)


SBOX: Tuple[int, ...] = _build_fips_sbox()
RCON: Tuple[int, ...] = (0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def key_expansion_ref(key16: bytes) -> List[bytes]:
    """Expand a 16-byte AES-128 key into 11 round keys (16 bytes each)."""
    assert len(key16) == 16
    w: List[List[int]] = [list(key16[4 * i : 4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        temp = list(w[i - 1])
        if i % 4 == 0:
            # RotWord + SubWord + Rcon
            temp = [SBOX[temp[1]] ^ RCON[i // 4], SBOX[temp[2]], SBOX[temp[3]], SBOX[temp[0]]]
        w.append([w[i - 4][j] ^ temp[j] for j in range(4)])
    return [bytes(sum(w[4 * r : 4 * r + 4], [])) for r in range(11)]


def aes128_encrypt_with_trace(plaintext16: bytes, key16: bytes) -> Tuple[bytes, List[bytes], List[bytes]]:
    """
    Encrypt a 16-byte block with AES-128, returning:
      (ciphertext, round_states[0..10], round_keys[0..10])
    """
    assert len(plaintext16) == 16 and len(key16) == 16
    rkeys = key_expansion_ref(key16)
    state = [plaintext16[i] ^ rkeys[0][i] for i in range(16)]
    states = [bytes(state)]

    for r in range(1, 11):
        # SubBytes
        state = [SBOX[b] for b in state]
        # ShiftRows (column-major: index = 4*col + row)
        sr = [0] * 16
        for row in range(4):
            for col in range(4):
                sr[4 * col + row] = state[4 * ((col + row) % 4) + row]
        state = sr
        # MixColumns (rounds 1..9)
        if r < 10:
            mc = [0] * 16
            for col in range(4):
                s0, s1, s2, s3 = state[4 * col : 4 * col + 4]
                mc[4 * col + 0] = gf_mul(2, s0) ^ gf_mul(3, s1) ^ s2 ^ s3
                mc[4 * col + 1] = s0 ^ gf_mul(2, s1) ^ gf_mul(3, s2) ^ s3
                mc[4 * col + 2] = s0 ^ s1 ^ gf_mul(2, s2) ^ gf_mul(3, s3)
                mc[4 * col + 3] = gf_mul(3, s0) ^ s1 ^ s2 ^ gf_mul(2, s3)
            state = mc
        # AddRoundKey
        rk = rkeys[r]
        state = [state[i] ^ rk[i] for i in range(16)]
        states.append(bytes(state))

    return bytes(state), states, rkeys


def aes128_encrypt_ref(plaintext16: bytes, key16: bytes) -> bytes:
    ct, _, _ = aes128_encrypt_with_trace(plaintext16, key16)
    return ct


# ---------------------------------------------------------------------------
# 2. Boyar-Peralta (2010) 113-Gate Combinational AES S-Box
# ---------------------------------------------------------------------------

def build_sbox_boyar_peralta(c: Circuit, u: Sequence[int]) -> List[int]:
    """
    Build the Boyar-Peralta (2010) 113-gate AES S-Box circuit.
    `u[0..7]` are 8 input bit node IDs ordered MSB..LSB (u[0] = weight 128, u[7] = weight 1).
    Returns `s[0..7]` ordered MSB..LSB (s[0] = weight 128, s[7] = weight 1).
    """
    assert len(u) == 8
    u0, u1, u2, u3, u4, u5, u6, u7 = u

    # Top linear transformation
    y14 = c.xor(u3, u5)
    y13 = c.xor(u0, u6)
    y9 = c.xor(u0, u3)
    y8 = c.xor(u0, u5)
    t0 = c.xor(u1, u2)
    y1 = c.xor(t0, u7)
    y4 = c.xor(y1, u3)
    y12 = c.xor(y13, y14)
    y2 = c.xor(y1, u0)
    y5 = c.xor(y1, u6)
    y3 = c.xor(y5, y8)
    t1 = c.xor(u4, y12)
    y15 = c.xor(t1, u5)
    y20 = c.xor(t1, u1)
    y6 = c.xor(y15, u7)
    y10 = c.xor(y15, t0)
    y11 = c.xor(y20, y9)
    y7 = c.xor(u7, y11)
    y17 = c.xor(y10, y11)
    y19 = c.xor(y10, y8)
    y16 = c.xor(t0, y11)
    y21 = c.xor(y13, y16)
    y18 = c.xor(u0, y16)

    # Middle non-linear section (GF(2^4) inversion & multiplication, 32 AND gates)
    t2 = c.and_(y12, y15)
    t3 = c.and_(y3, y6)
    t4 = c.xor(t3, t2)
    t5 = c.and_(y4, u7)
    t6 = c.xor(t5, t2)
    t7 = c.and_(y13, y16)
    t8 = c.and_(y5, y1)
    t9 = c.xor(t8, t7)
    t10 = c.and_(y2, y7)
    t11 = c.xor(t10, t7)
    t12 = c.and_(y9, y11)
    t13 = c.and_(y14, y17)
    t14 = c.xor(t13, t12)
    t15 = c.and_(y8, y10)
    t16 = c.xor(t15, t12)
    t17 = c.xor(t4, t14)
    t18 = c.xor(t6, t16)
    t19 = c.xor(t9, t14)
    t20 = c.xor(t11, t16)
    t21 = c.xor(t17, y20)
    t22 = c.xor(t18, y19)
    t23 = c.xor(t19, y21)
    t24 = c.xor(t20, y18)

    t25 = c.xor(t21, t22)
    t26 = c.and_(t21, t23)
    t27 = c.xor(t24, t26)
    t28 = c.and_(t25, t27)
    t29 = c.xor(t28, t22)
    t30 = c.xor(t23, t24)
    t31 = c.xor(t22, t26)
    t32 = c.and_(t31, t30)
    t33 = c.xor(t32, t24)
    t34 = c.xor(t23, t33)
    t35 = c.xor(t27, t33)
    t36 = c.and_(t24, t35)
    t37 = c.xor(t36, t34)
    t38 = c.xor(t27, t36)
    t39 = c.and_(t29, t38)
    t40 = c.xor(t25, t39)

    t41 = c.xor(t40, t37)
    t42 = c.xor(t29, t33)
    t43 = c.xor(t29, t40)
    t44 = c.xor(t33, t37)
    t45 = c.xor(t42, t41)

    z0 = c.and_(t44, y15)
    z1 = c.and_(t37, y6)
    z2 = c.and_(t33, u7)
    z3 = c.and_(t43, y16)
    z4 = c.and_(t40, y1)
    z5 = c.and_(t29, y7)
    z6 = c.and_(t42, y11)
    z7 = c.and_(t45, y17)
    z8 = c.and_(t41, y10)
    z9 = c.and_(t44, y12)
    z10 = c.and_(t37, y3)
    z11 = c.and_(t33, y4)
    z12 = c.and_(t43, y13)
    z13 = c.and_(t40, y5)
    z14 = c.and_(t29, y2)
    z15 = c.and_(t42, y9)
    z16 = c.and_(t45, y14)
    z17 = c.and_(t41, y8)

    # Bottom linear transformation
    t46 = c.xor(z15, z16)
    t47 = c.xor(z10, z11)
    t48 = c.xor(z5, z13)
    t49 = c.xor(z9, z10)
    t50 = c.xor(z2, z12)
    t51 = c.xor(z2, z5)
    t52 = c.xor(z7, z8)
    t53 = c.xor(z0, z3)
    t54 = c.xor(z6, z7)
    t55 = c.xor(z16, z17)
    t56 = c.xor(z12, t48)
    t57 = c.xor(t50, t53)
    t58 = c.xor(z4, t46)
    t59 = c.xor(z3, t54)
    t60 = c.xor(t46, t57)
    t61 = c.xor(z14, t57)
    t62 = c.xor(t52, t58)
    t63 = c.xor(t49, t58)
    t64 = c.xor(z4, t59)
    t65 = c.xor(t61, t62)
    t66 = c.xor(z1, t63)

    s0 = c.xor(t59, t63)
    s6 = c.xnor(t56, t62)
    s7 = c.xnor(t48, t60)
    t67 = c.xor(t64, t65)
    s3 = c.xor(t53, t66)
    s4 = c.xor(t51, t66)
    s5 = c.xor(t47, t65)
    s1 = c.xnor(t64, s3)
    s2 = c.xnor(t55, t67)

    return [s0, s1, s2, s3, s4, s5, s6, s7]


def build_xtime(c: Circuit, b: Sequence[int]) -> List[int]:
    """
    Multiply 8-bit byte `b[0..7]` (MSB..LSB) by 2 in GF(2^8) (modulo 0x11B).
    xtime([b0, b1, b2, b3, b4, b5, b6, b7]):
      out = [b1, b2, b3, b4 ^ b0, b5 ^ b0, b6, b7 ^ b0, b0]
    """
    b0, b1, b2, b3, b4, b5, b6, b7 = b
    return [
        b1,
        b2,
        b3,
        c.xor(b4, b0),
        c.xor(b5, b0),
        b6,
        c.xor(b7, b0),
        b0,
    ]


def build_aes_round_circuit(
    include_mix_columns: bool,
) -> Tuple[Circuit, Dict[str, int]]:
    """
    Build a single AES-128 round + key-schedule step circuit.
    Inputs:
      --s0..--s127   : 128-bit state entering the round (16 bytes * 8 bits MSB..LSB)
      --k0..--k127   : 128-bit previous round key
      --rc0..--rc7   : 8-bit Rcon[r] byte (MSB..LSB) for this round
    Outputs:
      --s0..--s127   : 128-bit state exiting the round
      --k0..--k127   : 128-bit expanded round key for this round
    """
    c = Circuit()
    s_in = [c.input(f"--s{i}") for i in range(128)]
    k_in = [c.input(f"--k{i}") for i in range(128)]
    rc_in = [c.input(f"--rc{i}") for i in range(8)]

    # 1. KeySchedule step: compute next 128-bit round key k_out[0..127]
    # Words W0..W3 are 32-bit slices of k_in.
    # RotWord(W3) = bytes [13, 14, 15, 12] of k_in
    rot_bytes = [
        k_in[8 * 13 : 8 * 14],
        k_in[8 * 14 : 8 * 15],
        k_in[8 * 15 : 8 * 16],
        k_in[8 * 12 : 8 * 13],
    ]
    sub_rot = [build_sbox_boyar_peralta(c, b_wires) for b_wires in rot_bytes]
    # XOR Rcon into first byte of SubWord(RotWord(W3))
    sub_rot[0] = [c.xor(sub_rot[0][bit], rc_in[bit]) for bit in range(8)]
    sub_word_flat = [wire for byte_w in sub_rot for wire in byte_w]  # 32 bits

    # Associative prefix XOR so all 4 key words complete 1 gate after SubWord
    k01 = [c.xor(k_in[i], k_in[32 + i]) for i in range(32)]
    k012 = [c.xor(k01[i], k_in[64 + i]) for i in range(32)]
    k0123 = [c.xor(k012[i], k_in[96 + i]) for i in range(32)]

    w0_new = [c.xor(k_in[i], sub_word_flat[i]) for i in range(32)]
    w1_new = [c.xor(k01[i], sub_word_flat[i]) for i in range(32)]
    w2_new = [c.xor(k012[i], sub_word_flat[i]) for i in range(32)]
    w3_new = [c.xor(k0123[i], sub_word_flat[i]) for i in range(32)]
    k_out = w0_new + w1_new + w2_new + w3_new

    # 2. SubBytes on all 16 state bytes
    sb_bytes = [build_sbox_boyar_peralta(c, s_in[8 * i : 8 * (i + 1)]) for i in range(16)]

    # 3. ShiftRows (column-major: byte index = 4 * col + row)
    sr_bytes: List[List[int]] = [[] for _ in range(16)]
    for row in range(4):
        for col in range(4):
            sr_bytes[4 * col + row] = sb_bytes[4 * ((col + row) % 4) + row]

    # 4. MixColumns (Rounds 1..9 only)
    if include_mix_columns:
        mc_bytes: List[List[int]] = [[] for _ in range(16)]
        for col in range(4):
            a0, a1, a2, a3 = sr_bytes[4 * col : 4 * col + 4]
            # Standard MixColumns identity:
            # t = a0 ^ a1 ^ a2 ^ a3
            # out0 = a0 ^ t ^ xtime(a0 ^ a1)
            # out1 = a1 ^ t ^ xtime(a1 ^ a2)
            # out2 = a2 ^ t ^ xtime(a2 ^ a3)
            # out3 = a3 ^ t ^ xtime(a3 ^ a0)
            t = [c.xor_many([a0[b], a1[b], a2[b], a3[b]]) for b in range(8)]
            xt01 = build_xtime(c, [c.xor(a0[b], a1[b]) for b in range(8)])
            xt12 = build_xtime(c, [c.xor(a1[b], a2[b]) for b in range(8)])
            xt23 = build_xtime(c, [c.xor(a2[b], a3[b]) for b in range(8)])
            xt30 = build_xtime(c, [c.xor(a3[b], a0[b]) for b in range(8)])

            mc_bytes[4 * col + 0] = [c.xor_many([a0[b], t[b], xt01[b]]) for b in range(8)]
            mc_bytes[4 * col + 1] = [c.xor_many([a1[b], t[b], xt12[b]]) for b in range(8)]
            mc_bytes[4 * col + 2] = [c.xor_many([a2[b], t[b], xt23[b]]) for b in range(8)]
            mc_bytes[4 * col + 3] = [c.xor_many([a3[b], t[b], xt30[b]]) for b in range(8)]
        post_linear = mc_bytes
    else:
        post_linear = sr_bytes

    # 5. AddRoundKey with k_out
    s_out: List[int] = []
    for byte_idx in range(16):
        for bit_idx in range(8):
            flat_idx = 8 * byte_idx + bit_idx
            s_out.append(c.xor(post_linear[byte_idx][bit_idx], k_out[flat_idx]))

    outputs: Dict[str, int] = {}
    for i in range(128):
        outputs[f"--s{i}"] = s_out[i]
    for i in range(128):
        outputs[f"--k{i}"] = k_out[i]

    return c, outputs


def build_aes_shared_subbytes_sr_keysched() -> Tuple[Circuit, Dict[str, int]]:
    """
    Shared across ALL 10 rounds (Rounds 1..10):
      - KeySchedule step (--k0..127 + --rc0..7 -> --k0..127)
      - SubBytes + ShiftRows (--s0..127 -> --s0..127)
    """
    c = Circuit()
    s_in = [c.input(f"--s{i}") for i in range(128)]
    k_in = [c.input(f"--k{i}") for i in range(128)]
    rc_in = [c.input(f"--rc{i}") for i in range(8)]

    rot_bytes = [
        k_in[8 * 13 : 8 * 14],
        k_in[8 * 14 : 8 * 15],
        k_in[8 * 15 : 8 * 16],
        k_in[8 * 12 : 8 * 13],
    ]
    sub_rot = [build_sbox_boyar_peralta(c, b_wires) for b_wires in rot_bytes]
    sub_rot[0] = [c.xor(sub_rot[0][bit], rc_in[bit]) for bit in range(8)]
    sub_word_flat = [wire for byte_w in sub_rot for wire in byte_w]

    k01 = [c.xor(k_in[i], k_in[32 + i]) for i in range(32)]
    k012 = [c.xor(k01[i], k_in[64 + i]) for i in range(32)]
    k0123 = [c.xor(k012[i], k_in[96 + i]) for i in range(32)]

    w0_new = [c.xor(k_in[i], sub_word_flat[i]) for i in range(32)]
    w1_new = [c.xor(k01[i], sub_word_flat[i]) for i in range(32)]
    w2_new = [c.xor(k012[i], sub_word_flat[i]) for i in range(32)]
    w3_new = [c.xor(k0123[i], sub_word_flat[i]) for i in range(32)]
    k_out = w0_new + w1_new + w2_new + w3_new

    sb_bytes = [build_sbox_boyar_peralta(c, s_in[8 * i : 8 * (i + 1)]) for i in range(16)]
    sr_bytes: List[List[int]] = [[] for _ in range(16)]
    for row in range(4):
        for col in range(4):
            sr_bytes[4 * col + row] = sb_bytes[4 * ((col + row) % 4) + row]

    outputs: Dict[str, int] = {}
    for byte_idx in range(16):
        for bit_idx in range(8):
            outputs[f"--s{8 * byte_idx + bit_idx}"] = sr_bytes[byte_idx][bit_idx]
    for i in range(128):
        outputs[f"--k{i}"] = k_out[i]
    return c, outputs


def build_aes_mixcolumns_ark() -> Tuple[Circuit, Dict[str, int]]:
    """Rounds 1..9 linear tail: MixColumns(--s0..127) XOR --k0..127 -> --s0..127."""
    c = Circuit()
    s_in = [c.input(f"--s{i}") for i in range(128)]
    k_in = [c.input(f"--k{i}") for i in range(128)]
    sr_bytes = [s_in[8 * i : 8 * (i + 1)] for i in range(16)]

    outputs: Dict[str, int] = {}
    for col in range(4):
        a0, a1, a2, a3 = sr_bytes[4 * col : 4 * col + 4]
        t = [c.xor_many([a0[b], a1[b], a2[b], a3[b]]) for b in range(8)]
        xt01 = build_xtime(c, [c.xor(a0[b], a1[b]) for b in range(8)])
        xt12 = build_xtime(c, [c.xor(a1[b], a2[b]) for b in range(8)])
        xt23 = build_xtime(c, [c.xor(a2[b], a3[b]) for b in range(8)])
        xt30 = build_xtime(c, [c.xor(a3[b], a0[b]) for b in range(8)])

        col_mc = [
            [c.xor_many([a0[b], t[b], xt01[b], k_in[8 * (4 * col + 0) + b]]) for b in range(8)],
            [c.xor_many([a1[b], t[b], xt12[b], k_in[8 * (4 * col + 1) + b]]) for b in range(8)],
            [c.xor_many([a2[b], t[b], xt23[b], k_in[8 * (4 * col + 2) + b]]) for b in range(8)],
            [c.xor_many([a3[b], t[b], xt30[b], k_in[8 * (4 * col + 3) + b]]) for b in range(8)],
        ]
        for row in range(4):
            for b in range(8):
                outputs[f"--s{8 * (4 * col + row) + b}"] = col_mc[row][b]
    return c, outputs


def build_aes_final_ark() -> Tuple[Circuit, Dict[str, int]]:
    """Round 10 linear tail: --s0..127 XOR --k0..127 -> --s0..127."""
    c = Circuit()
    s_in = [c.input(f"--s{i}") for i in range(128)]
    k_in = [c.input(f"--k{i}") for i in range(128)]
    outputs = {f"--s{i}": c.xor(s_in[i], k_in[i]) for i in range(128)}
    return c, outputs


def bytes_to_bits(data: bytes, prefix: str) -> Dict[str, int]:
    res: Dict[str, int] = {}
    for byte_idx, bval in enumerate(data):
        for bit_idx in range(8):
            res[f"{prefix}{8 * byte_idx + bit_idx}"] = (bval >> (7 - bit_idx)) & 1
    return res


def bits_to_bytes(env: Dict[str, int], prefix: str, num_bytes: int = 16) -> bytes:
    out = bytearray(num_bytes)
    for byte_idx in range(num_bytes):
        val = 0
        for bit_idx in range(8):
            val = (val << 1) | (env[f"{prefix}{8 * byte_idx + bit_idx}"] & 1)
        out[byte_idx] = val
    return bytes(out)


# ---------------------------------------------------------------------------
# 3. Compact 4-Byte Word-Serial `SubWord` + ANF `mod(..., 2)` CSS Spec
# ---------------------------------------------------------------------------

def build_compact_aes_css_spec() -> Dict[str, List[Tuple[str, str]]]:
    """
    Derives a highly compact, reusable CSS rule dictionary for AES-128 using:
      - A single 4-byte `SubWord` unit (`w0`, `w1`, `w2`, `w3`) shared 5x per round
        (once for KeyExpansion, 4x across the four 32-bit state columns)
      - Exact GF(2) Algebraic Normal Form + CSS `mod(..., 2)` flat sums so the
        entire S-box takes 4 stages and MixColumns+ShiftRows+AddRoundKey takes 1 stage.

    Returns `{class_name: [(lhs_var, rhs_expr), ...]}` for classes:
      `ki`, `w0`, `w1`, `w2`, `w3`, `p0`, `p1`, `p2`, `p3`, `mc`, `fn`.
    """
    # 1. Algebraically extract exact supports & 4-bit ANFs from Boyar-Peralta
    c_top = Circuit()
    u = [c_top.input(f"u{i}") for i in range(8)]
    u0, u1, u2, u3, u4, u5, u6, u7 = u
    y14 = c_top.xor(u3, u5)
    y13 = c_top.xor(u0, u6)
    y9 = c_top.xor(u0, u3)
    y8 = c_top.xor(u0, u5)
    t0 = c_top.xor(u1, u2)
    y1 = c_top.xor(t0, u7)
    y4 = c_top.xor(y1, u3)
    y12 = c_top.xor(y13, y14)
    y2 = c_top.xor(y1, u0)
    y5 = c_top.xor(y1, u6)
    y3 = c_top.xor(y5, y8)
    t1 = c_top.xor(u4, y12)
    y15 = c_top.xor(t1, u5)
    y20 = c_top.xor(t1, u1)
    y6 = c_top.xor(y15, u7)
    y10 = c_top.xor(y15, t0)
    y11 = c_top.xor(y20, y9)
    y7 = c_top.xor(u7, y11)
    y17 = c_top.xor(y10, y11)
    y19 = c_top.xor(y10, y8)
    y16 = c_top.xor(t0, y11)
    y21 = c_top.xor(y13, y16)
    y18 = c_top.xor(u0, y16)

    y_nodes = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17]
    extra_y = {"y18": y18, "y19": y19, "y20": y20, "y21": y21}
    top_outs = {f"y{idx + 1}": nid for idx, nid in enumerate(y_nodes)}
    top_outs.update(extra_y)

    # Probe unit vectors e_0..e_7 to get exact linear u-supports
    lin_support: Dict[str, List[int]] = {k: [] for k in top_outs}
    for bit in range(8):
        inp = {f"u{i}": (1 if i == bit else 0) for i in range(8)}
        ev = c_top.evaluate(inp, top_outs)
        for k, val in ev.items():
            if val:
                lin_support[k].append(bit)

    # 2. Extract exact 16-entry truth tables for GF(2^4) inverter (t21..t24 -> e0..e8)
    c_inv = Circuit()
    t21, t22, t23, t24 = [c_inv.input(f"d{i}") for i in range(4)]
    t25 = c_inv.xor(t21, t22)
    t26 = c_inv.and_(t21, t23)
    t27 = c_inv.xor(t24, t26)
    t28 = c_inv.and_(t25, t27)
    t29 = c_inv.xor(t28, t22)
    t30 = c_inv.xor(t23, t24)
    t31 = c_inv.xor(t22, t26)
    t32 = c_inv.and_(t31, t30)
    t33 = c_inv.xor(t32, t24)
    t34 = c_inv.xor(t23, t33)
    t35 = c_inv.xor(t27, t33)
    t36 = c_inv.and_(t24, t35)
    t37 = c_inv.xor(t36, t34)
    t38 = c_inv.xor(t27, t36)
    t39 = c_inv.and_(t29, t38)
    t40 = c_inv.xor(t25, t39)
    t41 = c_inv.xor(t40, t37)
    t42 = c_inv.xor(t29, t33)
    t43 = c_inv.xor(t29, t40)
    t44 = c_inv.xor(t33, t37)
    t45 = c_inv.xor(t42, t41)

    e_nodes = [t29, t33, t37, t40, t41, t42, t43, t44, t45]
    inv_outs = {f"e{i}": nid for i, nid in enumerate(e_nodes)}
    inv_tts: List[List[int]] = [[] for _ in range(9)]
    for mask in range(16):
        inp = {f"d{b}": (mask >> (3 - b)) & 1 for b in range(4)}
        ev = c_inv.evaluate(inp, inv_outs)
        for i in range(9):
            inv_tts[i].append(ev[f"e{i}"])

    # 3. Extract exact bottom affine matrix over z0..z17 -> s0..s7
    c_bot = Circuit()
    z = [c_bot.input(f"z{i}") for i in range(18)]
    t46 = c_bot.xor(z[15], z[16])
    t47 = c_bot.xor(z[10], z[11])
    t48 = c_bot.xor(z[5], z[13])
    t49 = c_bot.xor(z[9], z[10])
    t50 = c_bot.xor(z[2], z[12])
    t51 = c_bot.xor(z[2], z[5])
    t52 = c_bot.xor(z[7], z[8])
    t53 = c_bot.xor(z[0], z[3])
    t54 = c_bot.xor(z[6], z[7])
    t55 = c_bot.xor(z[16], z[17])
    t56 = c_bot.xor(z[12], t48)
    t57 = c_bot.xor(t50, t53)
    t58 = c_bot.xor(z[4], t46)
    t59 = c_bot.xor(z[3], t54)
    t60 = c_bot.xor(t46, t57)
    t61 = c_bot.xor(z[14], t57)
    t62 = c_bot.xor(t52, t58)
    t63 = c_bot.xor(t49, t58)
    t64 = c_bot.xor(z[4], t59)
    t65 = c_bot.xor(t61, t62)
    t66 = c_bot.xor(z[1], t63)
    s_nodes = [
        c_bot.xor(t59, t63),
        None,  # s1 filled below
        None,  # s2 filled below
        c_bot.xor(t53, t66),
        c_bot.xor(t51, t66),
        c_bot.xor(t47, t65),
        c_bot.xnor(t56, t62),
        c_bot.xnor(t48, t60),
    ]
    t67 = c_bot.xor(t64, t65)
    s_nodes[1] = c_bot.xnor(t64, s_nodes[3])  # type: ignore[arg-type]
    s_nodes[2] = c_bot.xnor(t55, t67)
    bot_outs = {f"s{i}": s_nodes[i] for i in range(8)}  # type: ignore[misc]

    zero_ev = c_bot.evaluate({f"z{i}": 0 for i in range(18)}, bot_outs)
    bot_const = [bool(zero_ev[f"s{i}"]) for i in range(8)]
    bot_z_support: List[List[int]] = [[] for _ in range(8)]
    for zk in range(18):
        ev = c_bot.evaluate({f"z{i}": (1 if i == zk else 0) for i in range(18)}, bot_outs)
        for sb in range(8):
            if bool(ev[f"s{sb}"]) ^ bot_const[sb]:
                bot_z_support[sb].append(zk)

    # Map z0..z17 to (e_index 0..8, y_index 1..17 or "u7")
    # e indices: 0:t29, 1:t33, 2:t37, 3:t40, 4:t41, 5:t42, 6:t43, 7:t44, 8:t45
    z_factors: List[Tuple[int, str]] = [
        (7, "y15"), (2, "y6"), (1, "u7"), (6, "y16"), (3, "y1"), (0, "y7"),
        (5, "y11"), (8, "y17"), (4, "y10"), (7, "y12"), (2, "y3"), (1, "y4"),
        (6, "y13"), (3, "y5"), (0, "y2"), (5, "y9"), (8, "y14"), (4, "y8"),
    ]

    # Build 4-byte `SubWord` CSS stages w0, w1, w2, w3
    w0: List[Tuple[str, str]] = []
    w1: List[Tuple[str, str]] = []
    w2: List[Tuple[str, str]] = []
    w3: List[Tuple[str, str]] = []

    for b in range(4):
        u_vars = [f"var(--x{8 * b + i})" for i in range(8)]
        y_var_of = {f"y{idx + 1}": f"var(--y{17 * b + idx})" for idx in range(17)}
        y_var_of["u7"] = u_vars[7]

        # w0: 17 linear basis wires
        for idx in range(17):
            terms = [u_vars[bit] for bit in lin_support[f"y{idx + 1}"]]
            w0.append((f"--y{17 * b + idx}", format_mod2_sum(terms)))

        # w1: 4 GF(2^4) norm wires (t21..t24)
        norm_specs = [
            ([("y12", "y15"), ("y3", "y6"), ("y9", "y11"), ("y14", "y17")], "y20"),
            ([("y12", "y15"), ("y4", "u7"), ("y9", "y11"), ("y8", "y10")], "y19"),
            ([("y13", "y16"), ("y5", "y1"), ("y9", "y11"), ("y14", "y17")], "y21"),
            ([("y13", "y16"), ("y2", "y7"), ("y9", "y11"), ("y8", "y10")], "y18"),
        ]
        for d_idx, (quads, lin_key) in enumerate(norm_specs):
            q_terms = [f"{y_var_of[p]}*{y_var_of[q]}" for p, q in quads]
            l_terms = [u_vars[bit] for bit in lin_support[lin_key]]
            w1.append((f"--d{4 * b + d_idx}", format_mod2_sum(q_terms + l_terms)))

        # w2: 9 GF(2^4) inverse wires via exact 4-bit Mobius ANF
        d_vars = [f"var(--d{4 * b + i})" for i in range(4)]
        for e_idx in range(9):
            w2.append((f"--e{9 * b + e_idx}", mobius_anf_4bit(inv_tts[e_idx], d_vars)))

        # w3: 8 output bits (--o{8*b + 0..7})
        for sb in range(8):
            p_terms = []
            for zk in bot_z_support[sb]:
                e_idx, y_name = z_factors[zk]
                p_terms.append(f"var(--e{9 * b + e_idx})*{y_var_of[y_name]}")
            w3.append((f"--o{8 * b + sb}", format_mod2_sum(p_terms, invert=bot_const[sb])))

    # `ki`: latch --k0..127 -> --q0..127, and load RotWord(W3) [bytes 13,14,15,12] -> --x0..31
    ki: List[Tuple[str, str]] = [(f"--q{i}", f"var(--k{i})") for i in range(128)]
    rot_src = list(range(8 * 13, 8 * 16)) + list(range(8 * 12, 8 * 13))
    for i, src in enumerate(rot_src):
        ki.append((f"--x{i}", f"var(--k{src})"))

    # `p0`: write expanded --k0..127 from --q*, --o*, --rc* AND load state col 0 (--s0..31) -> --x0..31
    p0: List[Tuple[str, str]] = []
    for word in range(4):
        for i in range(32):
            terms = [f"var(--q{32 * w + i})" for w in range(word + 1)]
            terms.append(f"var(--o{i})")
            if i < 8:
                terms.append(f"var(--rc{i})")
            p0.append((f"--k{32 * word + i}", format_mod2_sum(terms)))
    for i in range(32):
        p0.append((f"--x{i}", f"var(--s{i})"))

    # `p1`, `p2`, `p3`: store previous column's --o0..31 into --q* and load next state col into --x0..31
    p1 = [(f"--q{i}", f"var(--o{i})") for i in range(32)] + [
        (f"--x{i}", f"var(--s{32 + i})") for i in range(32)
    ]
    p2 = [(f"--q{32 + i}", f"var(--o{i})") for i in range(32)] + [
        (f"--x{i}", f"var(--s{64 + i})") for i in range(32)
    ]
    p3 = [(f"--q{64 + i}", f"var(--o{i})") for i in range(32)] + [
        (f"--x{i}", f"var(--s{96 + i})") for i in range(32)
    ]

    # Helper to name post-SubBytes bit `(byte_idx, bit_idx)` across `--q0..95` and `--o0..31`
    def sb_var(byte_idx: int, bit_idx: int) -> str:
        if byte_idx < 12:
            return f"var(--q{8 * byte_idx + bit_idx})"
        return f"var(--o{8 * (byte_idx - 12) + bit_idx})"

    # Probe exact linear support of ShiftRows + MixColumns over the 128 SubBytes bits
    c_mc, o_mc = build_aes_mixcolumns_ark()
    # Note: build_aes_mixcolumns_ark expects ShiftRows-ordered `--s0..127`!
    # Map ShiftRows(byte row, col) -> pre-ShiftRows byte index `4 * ((col + row) % 4) + row`
    sr_src_byte = [0] * 16
    for row in range(4):
        for col in range(4):
            sr_src_byte[4 * col + row] = 4 * ((col + row) % 4) + row

    mc: List[Tuple[str, str]] = []
    fn: List[Tuple[str, str]] = []

    # Probe MixColumns linear matrix for one 32-bit column (32 inputs -> 32 outputs, k=0)
    c_col = Circuit()
    col_in = [[c_col.input(f"a_{r}_{b}") for b in range(8)] for r in range(4)]
    a0, a1, a2, a3 = col_in
    t_w = [c_col.xor_many([a0[b], a1[b], a2[b], a3[b]]) for b in range(8)]
    xt01 = build_xtime(c_col, [c_col.xor(a0[b], a1[b]) for b in range(8)])
    xt12 = build_xtime(c_col, [c_col.xor(a1[b], a2[b]) for b in range(8)])
    xt23 = build_xtime(c_col, [c_col.xor(a2[b], a3[b]) for b in range(8)])
    xt30 = build_xtime(c_col, [c_col.xor(a3[b], a0[b]) for b in range(8)])
    col_outs = {}
    for r_idx, (ar, xtr) in enumerate([(a0, xt01), (a1, xt12), (a2, xt23), (a3, xt30)]):
        for b in range(8):
            col_outs[f"m_{r_idx}_{b}"] = c_col.xor_many([ar[b], t_w[b], xtr[b]])

    col_matrix: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
        (r, b): [] for r in range(4) for b in range(8)
    }
    for src_r in range(4):
        for src_b in range(8):
            inp = {
                f"a_{r}_{b}": (1 if (r == src_r and b == src_b) else 0)
                for r in range(4)
                for b in range(8)
            }
            ev = c_col.evaluate(inp, col_outs)
            for out_r in range(4):
                for out_b in range(8):
                    if ev[f"m_{out_r}_{out_b}"]:
                        col_matrix[(out_r, out_b)].append((src_r, src_b))

    for col in range(4):
        for row in range(4):
            out_byte = 4 * col + row
            for b in range(8):
                flat = 8 * out_byte + b
                # MixColumns + ShiftRows + AddRoundKey
                mc_terms = [
                    sb_var(sr_src_byte[4 * col + sr_r], sr_b)
                    for sr_r, sr_b in col_matrix[(row, b)]
                ]
                mc_terms.append(f"var(--k{flat})")
                mc.append((f"--s{flat}", format_mod2_sum(mc_terms)))

                # Final Round: ShiftRows + AddRoundKey
                fn_terms = [sb_var(sr_src_byte[out_byte], b), f"var(--k{flat})"]
                fn.append((f"--s{flat}", format_mod2_sum(fn_terms)))

    return {
        "ki": ki,
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "mc": mc,
        "fn": fn,
    }


def get_round_stage_sequence(round_num: int) -> List[str]:
    """Return the 26 CSS stage keys executed in order for `round_num` (1..10)."""
    tail = "mc" if round_num < 10 else "fn"
    return [
        "ki", "w0", "w1", "w2", "w3",
        "p0", "w0", "w1", "w2", "w3",
        "p1", "w0", "w1", "w2", "w3",
        "p2", "w0", "w1", "w2", "w3",
        "p3", "w0", "w1", "w2", "w3",
        tail,
    ]
