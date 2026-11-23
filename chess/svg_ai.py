#!/usr/bin/env python3
"""
Multi-Stage Linear-VRAM HTML5 SVG `<filter>` 2-Ply CNN Minimax Engine (`chess/svg_ai.py`).

Wires the Stockfish-distilled 3-layer 8-bit quantized Convolutional Value Network (`chess/weights.json`,
2,623 parameters, 5x5 Knight-vision Layer 1, Z0 = 128/255, Pearson r = 0.902 vs Stockfish, 96.4% sibling rank acc)
into a pure HTML5 SVG `<filter>` 2-ply Minimax pipeline with:
  - Exact Ply-1 legal moves (`64` destination columns from CSS `.e12`, including O-O, O-O-O, EP, promos)
  - Exact Ply-2 FIDE King-safety, check-evasion, double-check & pin-axis filtering across all 64 post-move
    boards (`640x10` strip), selecting up to 16 quadrant-optimal MVV-LVA opponent replies per Ply-1 move
  - Parallel evaluation of `64 x 16 = 1,024` depth-2 leaf boards through the 3-layer `<feConvolveMatrix>`
    & `<feColorMatrix>` CNN + lossless linear material skip + calibrated Sigmoid LUT (`[3..253]/255`)
  - Exact `<feMorphology operator="erode">` (MIN over opponent replies, with Mate-in-1 = 254, Stalemate = 128)
    and `<feMorphology operator="dilate">` (MAX over AI legal moves + centrality tie-break)
  - Zero JavaScript, zero `<feImage>` primitives, all `<feConvolveMatrix>` `<= 5x5`, Display-P3 & linearRGB
    lossless neutral-gray `[128..255]` LUTs, `A_out == 1.0` across every primitive, and `<= 72` primitives
    per `<g>` stage for Chrome/Skia & Safari/CoreImage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


CHESS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DEFAULT_PATH = os.path.join(CHESS_DIR, "weights.json")
ASSETS_DIR = os.path.join(CHESS_DIR, "assets")

# Fixed 1:1 SVG Engine Canvas Size in CSS pixels (NEVER CSS-scaled; matches `.ai-stage-shell` 1:1!)
STAGE_W = 664
STAGE_H = 636
VRAM_H = 608   # y = 0..607 holds intermediate VRAM
COMP_H = 628   # y = 0..627 holds all compute VRAM + <foreignObject> + PHYS_LUT_SPECS (332x314 CSS px at 2x!)

# Physical VRAM Coordinates (all even integers so `u = 0.5` on 2x Retina yields exact integer CSS coords!):
# 1. Stages s1a..s1i: Ply-1 & Ply-2 FIDE legality `640 x 10` strips (disjoint from y = 0..571):
VRAM_S1_B0 = (0, 572)      # 640 x 10: post-Ply-1 (P, N, B) in {1, 128, 255}
VRAM_S1_B1 = (0, 584)      # 640 x 10: post-Ply-1 (R, Q, K) in {1, 128, 255}
VRAM_S1_KATT = (0, 596)    # 640 x 10: kingless attack / M2 quadrant winners (R=score, G=128+u2, B=128+pt2)

TMP_S1_PIN = (0, 0)        # 640 x 10: R=allow_H, G=allow_V, B=evade_ok
TMP_S1_PIN2 = (0, 12)      # 640 x 10: R=allow_D1(/), G=allow_D2(\), B=128+mvv_code
TMP_S1_PNK = (0, 24)       # 640 x 10: Pawn / Knight source u2s
TMP_S1_ORTH = (0, 36)      # 640 x 10: Orth & Diag slider rays / s1_in_check

# 2. Stage s2a: Depth-2 `64x16x(10x10) = 640x160` leaf board sheets (live during s2b..s2d):
VRAM_S2_B0 = (0, 0)        # 640 x 160 (y = 0..159)
VRAM_S2_B1 = (0, 160)      # 640 x 160 (y = 160..319)

# 3. Stages s2b..s2e: CNN Layer 1 compacted `384 x 96` tiles (5 or 6 RGB triplets, 100% disjoint from S2_B0/B1!):
VRAM_P1_C6_0 = (0, 320)    # 384 x 96 (s2b)
VRAM_P1_C6_1 = (0, 416)    # 384 x 96 (s2b)
VRAM_P1_C6_2 = (0, 512)    # 384 x 96 (s2c)
VRAM_P1_C6_3L = (384, 320) # 192 x 96 left half of triplet 3 (s2c)
VRAM_P1_C6_3R = (384, 416) # 192 x 96 right half of triplet 3 (s2c)
VRAM_P1_C6_4 = (0, 0)      # 384 x 96 (written in final s2 stage, last reader of S2_B0/B1!)
VRAM_P1_C6_5 = (0, 96)     # 384 x 96 (written in final s2e stage when G1 == 6!)

VRAM_SKIP_C2 = (384, 512)  # 128 x 32: skip_mat 2x2 features
VRAM_META_C2 = (384, 544)  # 128 x 32: R=no_m2, G=s1_in_check, B=is_s2_ai_mated
TMP_S2_MATE  = (0, 320)    # 640 x 160 scratch workspace during s2m0..s2m9 (free before s2b!)

# 4. Stages s3*: CNN Layer 2 compacted `128 x 32` tiles (up to 9 RGB triplets at x=400..527, y=0..287):
VRAM_P2_C2 = [(400, 32 * g) for g in range(9)]

# 5. Stages s4m*: CNN Layer 3a `mix3` `128 x 32` tiles (up to 12 RGB triplets at x=0..127, y=0..383):
VRAM_MIX3 = [(0, 32 * g) for g in range(12)]

# 6. Stages s4d* & s5*: Scalar `64 x 16` feature sheets + Minimax outputs at x=540..603 (y=0..275 < 333):
VRAM_DW3_C1 = [(540, 16 * g) for g in range(12)]      # x=540..603, y=0..191 (up to 12 triplets = 36 ch!)
VRAM_SKIP_C1 = (540, 192)                             # 64 x 16 (y=192..207)
VRAM_NOM2_C1 = (540, 210)                             # 64 x 16 (y=210..225: R=no_m2, G=s1_in_check)
VRAM_VSHEET = (540, 234)                              # 64 x 16 (y=234..249: V_CNN in [3..253]/255, 255=no_m2)
VRAM_QSTRIP = (540, 254)                              # 64 x 2  (y=254..255: Q(v1) in [0..254]/255)
VRAM_WINMASK = (540, 260)                             # 64 x 2  (y=260..261: 1.0 at unique winning v1*)
VRAM_BOARD8 = (540, 268)                              # 8 x 8   (y=268..275: RGB board + FROM/TO highlights)
VRAM_WIN1PX = (560, 268)                              # 2 x 2   (R=128+u1*, G=128+v1*, B=Q_max)

# SourceGraphic DOM `<foreignObject>` & Helper PNG LUT Coordinates (100% Display-P3-immune neutral gray!):
FO_X = 0
FO_Y = 608
FO_W = 640
FO_H = 20
SRC_CH_PHYS = [(12 * c, 0) for c in range(6)]  # 6 neutral-gray 10x10 blocks at x = 0, 12, 24, 36, 48, 60
SRC_META_TURN_PHYS = (72, 0)   # 2 x 2 neutral gray: 128 + 127*turn
SRC_META_CHK_PHYS  = (76, 0)   # 2 x 2 neutral gray: 128 + 127*in_check
SRC_META_TERM_PHYS = (80, 0)   # 2 x 2 neutral gray: 128 + 127*is_terminal
SRC_M1U_PHYS  = (0, 12)        # 640 x 2 neutral gray: 128 + m1_u[v1]
SRC_M1V_PHYS  = (0, 14)        # 640 x 2 neutral gray: 128 + m1_v[v1]
SRC_M1K_PHYS  = (0, 16)        # 640 x 2 neutral gray: 128 + m1_c[v1]
SRC_M1CU_PHYS = (0, 18)        # 64 x 2  neutral gray: 128 + m1_u[v1]
SRC_M1CV_PHYS = (68, 18)       # 64 x 2  neutral gray: 128 + m1_v[v1]
SRC_M1CC_PHYS = (136, 18)      # 64 x 2  neutral gray: 128 + m1_c[v1]

# Right vertical LUT column (`x = 642..663, y = 0..607`):
LUT_M2Y_POS = (642, 0)         # 4 x 160 phys
LUT_C6Y_POS = (642, 164)       # 4 x 96 phys
LUT_C2Y_POS = (642, 264)       # 4 x 32 phys
LUT_C1Y_POS = (642, 300)       # 4 x 16 phys
LUT_QBASE_POS = (642, 320)     # 4 x 160 phys
LUT_C10_2Y_POS = (642, 484)    # 4 x 16 phys (8-cell Y-slice 10->2)
LUT_M2X_POS = (648, 0)         # 10 x 40 phys
LUT_VIS_BY_POS = (648, 44)     # 4 x 224 CSS (8 -> 224 board Y zoom)
LUT_VIS_MY_POS = (654, 44)     # 4 x 192 CSS (16 -> 192 matrix Y zoom)
LUT_VIS_FY_POS = (654, 240)    # 4 x 64 CSS  (16 -> 64 feature Y zoom)

# Bottom horizontal LUT strip (`y = 608..617` at `x >= 104`, and `y = 631..635`):
LUT_CELLSQ_POS = (104, 608)    # 10 x 10 phys
LUT_BMASK10_POS = (116, 608)   # 10 x 10 phys
LUT_BMASK6_POS = (128, 608)    # 6 x 6 phys
LUT_SQ8W_POS = (138, 608)      # 8 x 8 phys
LUT_SQ8B_POS = (148, 608)      # 8 x 8 phys
LUT_CHK8_POS = (158, 608)      # 8 x 8 phys
LUT_TIE_POS = (168, 608)       # 64 x 2 phys
LUT_C6X_POS = (234, 608)       # 132 x 4 phys
LUT_C2X_POS = (368, 608)       # 44 x 4 phys
LUT_C1X_POS = (414, 608)       # 64 x 4 phys
LUT_C10_2X_POS = (480, 608)    # 28 x 4 phys (14-cell X-slice 10->2)
LUT_BAR_POS = (414, 614)       # 224 x 4 CSS (win-prob gauge ramp)
LUT_VIS_BX_POS = (0, 631)      # 224 x 4 CSS (8 -> 224 board X zoom)
LUT_VIS_MX_POS = (228, 631)    # 384 x 4 CSS (64 -> 384 matrix X zoom)

# Visual Dashboard Layout in CSS pixels (rendered inside 664 x 636 by `visa` + `visb`):
VIS_BOARD_X = 22
VIS_BOARD_Y = 54
VIS_BOARD_SZ = 224             # 8 * 28px per square!

VIS_MAT_X = 264
VIS_MAT_Y = 54
VIS_MAT_W = 384                # 64 * 6px per Ply-1 destination column!
VIS_MAT_H = 192                # 16 * 12px per Ply-2 opponent reply quadrant!

VIS_QBAR_Y = 254
VIS_QBAR_H = 18

VIS_WINBAR_X = 22
VIS_WINBAR_Y = 334
VIS_WINBAR_W = 224
VIS_WINBAR_H = 18

VIS_FEAT_X = 264
VIS_FEAT1_Y = 434              # 384 x 64: skip_mat RGB latent sheet
VIS_FEAT2_Y = 528              # 384 x 64: dwconv3[0] RGB latent sheet
VIS_FEAT_W = 384
VIS_FEAT_H = 64

S1_STAGE_IDS: Tuple[str, ...] = (
    "s1a", "s1b", "s1b2", "s1c", "s1c2", "s1d", "s1e", "s1e2", "s1e3", "s1f", "s1f2", "s1g", "s1g2", "s1g3", "s1h", "s1i",
)
S2M_STAGE_IDS: Tuple[str, ...] = (
    "s2m0", "s2m1", "s2m2", "s2m3", "s2m4", "s2q0", "s2m5", "s2m6", "s2m7", "s2m8", "s2m9",
)
S2Q_STAGE_IDS: Tuple[str, ...] = (
    "s2q1", "s2q2", "s2q3", "s2q4", "s2q5", "s2q6", "s2q7",
)


def get_compute_stage_ids(spec: dict) -> Tuple[str, ...]:
    ly = spec["layers"]
    g1 = len(ly["mix1"]["triplets"])
    g2 = len(ly["mix2"]["triplets"])
    g3 = len(ly["mix3"]["triplets"])
    assert g1 in (5, 6) and 1 <= g2 <= 9 and 1 <= g3 <= 12
    s2_tail = ("s2b", "s2c", "s2d") if g1 == 5 else ("s2b", "s2c", "s2e")
    s2_ids = ("s2a",) + S2M_STAGE_IDS + S2Q_STAGE_IDS + s2_tail
    s3_step = 2 if g1 <= 5 else 1
    s3_ids = tuple(f"s3{chr(ord('a') + i)}" for i in range((g2 + s3_step - 1) // s3_step))
    m3_step = 4 if g2 <= 6 else 3
    s4m_ids = tuple(f"s4m{i}" for i in range((g3 + m3_step - 1) // m3_step))
    s4d_ids = tuple(f"s4d{i}" for i in range((g3 + 3) // 4))
    return S1_STAGE_IDS + s2_ids + s3_ids + s4m_ids + s4d_ids + ("s5v", "s5a", "visa", "visb")


def best_dwconv3_triplet(spec: dict) -> int:
    ly = spec["layers"]
    g3 = len(ly["mix3"]["triplets"])
    cms = ly["value_head"]["color_matrices"]
    return max(range(g3), key=lambda g: sum(abs(float(x)) for x in cms[g][:3]))


@dataclass
class SVG2PlyResult:
    best_u_norm: int
    best_v_norm: int
    best_u_real: int
    best_v_real: int
    best_code: int
    best_score_u8: int
    best_win_prob: float
    q_scores_u8: np.ndarray        # (64,) uint8 in [0..254]
    v_sheet_u8: np.ndarray         # (16, 64) uint8


def load_weights(path: str = WEIGHTS_DEFAULT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    assert spec.get("format") == "nojs-svg-chess-value-cnn-v1"
    assert spec.get("input_shape") == [6, 8, 8]
    return spec


def q8(x: np.ndarray | float) -> np.ndarray:
    return np.round(np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * 255.0) / 255.0


def _fmt_num(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _ct_u8(k: int) -> str:
    """Format exact byte `k in [0..255]` for `<feComponentTransfer>` (`table` / `discrete`).

    Chromium/Blink builds `<feComponentTransfer>` `uint8_t[256]` lookup tables via C++ float-to-int
    truncation `static_cast<uint8_t>(255.0f * v)` (`floor`), whereas WebKit/CoreImage and GPU float
    paths round `round(255.0 * v)`. Using `(k + 0.25) / 255.0` for `1 <= k <= 254` places `255 * v`
    at `k + 0.25`, guaranteeing BOTH `int(255 * v) == k` AND `round(255 * v) == k` in `float32` &
    `float64` with a full `0.25`-LSB safety margin on both sides.
    """
    assert 0 <= k <= 255, f"Byte out of [0..255]: {k}"
    if k == 0:
        return "0"
    if k == 255:
        return "1"
    return f"{(k + 0.25) / 255.0:.6f}"


def _fmt_list(vals: Sequence[float]) -> str:
    return " ".join(_fmt_num(float(v)) for v in vals)


# ---------------------------------------------------------------------------
# 1. Display-P3 & linearRGB Lossless PNG LUT Generator (`chess/assets/*.png`)
# ---------------------------------------------------------------------------

def _gray_rgba_from_bytes(vals_2d: np.ndarray) -> Image.Image:
    u8 = np.asarray(vals_2d, dtype=np.uint8)
    assert np.all((u8 == 0) | (u8 >= 128)), f"Dark byte in (1..127) risks linearRGB collision: {np.unique(u8)}"
    rgb = np.stack([u8, u8, u8], axis=-1)
    a = np.full(u8.shape + (1,), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate([rgb, a], axis=-1), mode="RGBA")


def _deltas_to_gray_strip(deltas: Sequence[int], horizontal: bool, thickness: int = 4, scale: float = 230.0) -> Image.Image:
    d = np.asarray(deltas, dtype=np.int32)
    assert d.min() >= 0, f"Negative displacement delta: {d.min()}"
    raw = np.round(127.5 + d * (255.0 / scale)).astype(np.int32)
    assert raw.min() >= 128 and raw.max() <= 255, f"LUT byte out of [128..255]: [{raw.min()}, {raw.max()}]"
    u8 = raw.astype(np.uint8)
    arr = np.broadcast_to(u8[None, :], (thickness, len(d))).copy() if horizontal else (
        np.broadcast_to(u8[:, None], (len(d), thickness)).copy()
    )
    return _gray_rgba_from_bytes(arr)


def build_periodic_zoom_lut(length_phys: int, coords: Sequence[int], cell_zoom_phys: int,
                            dead_coord: int, src_box_phys: int, horizontal: bool, thickness: int = 4) -> Image.Image:
    idx = np.arange(length_phys, dtype=np.int32)
    target = np.full(length_phys, int(dead_coord), dtype=np.int32)
    for ci, sc_c in enumerate(coords):
        i0, i1 = ci * cell_zoom_phys, (ci + 1) * cell_zoom_phys
        if i1 <= length_phys:
            target[i0:i1] = int(sc_c)
    deltas = (target - idx) % int(src_box_phys)
    return _deltas_to_gray_strip(deltas, horizontal=horizontal, thickness=thickness)


def compute_tie_ramp_u8() -> np.ndarray:
    """64 strictly unique bytes in `[128 .. 191]` ordering AI-normalized squares `a = 8*ar + af`
    by file centrality (2.2x weight) + forward rank centrality, breaking remaining ties deterministically.
    """
    scored = []
    for a in range(64):
        ar, af = divmod(a, 8)
        centrality = -2.2 * abs(af - 3.5) - 1.0 * abs(ar - 3.3)
        scored.append((centrality, a))
    scored.sort(key=lambda item: (item[0], -item[1]))
    ramp = np.zeros(64, dtype=np.uint8)
    for rank_idx, (_c, a) in enumerate(scored):
        ramp[a] = 128 + rank_idx  # 128 .. 191 (all 64 strictly distinct & >= 128!)
    return ramp


# Exact physical dimensions `(w_phys, h_phys)` of all resolution-invariant tensor LUT PNGs (all even!):
PHYS_LUT_SPECS: Tuple[Tuple[str, Tuple[int, int], int, int], ...] = (
    ("cell_sq", LUT_CELLSQ_POS, 10, 10),
    ("bmask10", LUT_BMASK10_POS, 10, 10),
    ("bmask6", LUT_BMASK6_POS, 6, 6),
    ("sq8w", LUT_SQ8W_POS, 8, 8),
    ("sq8b", LUT_SQ8B_POS, 8, 8),
    ("chk8", LUT_CHK8_POS, 8, 8),
    ("tie", LUT_TIE_POS, 64, 2),
    ("m2y", LUT_M2Y_POS, 4, 160),
    ("m2x", LUT_M2X_POS, 10, 40),
    ("qbase", LUT_QBASE_POS, 4, 160),
    ("c6x", LUT_C6X_POS, 132, 4),
    ("c6y", LUT_C6Y_POS, 4, 96),
    ("c2x", LUT_C2X_POS, 44, 4),
    ("c2y", LUT_C2Y_POS, 4, 32),
    ("c1x", LUT_C1X_POS, 64, 4),
    ("c1y", LUT_C1Y_POS, 4, 16),
    ("c10_2x", LUT_C10_2X_POS, 28, 4),
    ("c10_2y", LUT_C10_2Y_POS, 4, 16),
)


def write_chess_ai_assets(assets_dir: str = ASSETS_DIR) -> Dict[str, str]:
    os.makedirs(assets_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    def _save(im: Image.Image, key: str, fn: str) -> None:
        im.save(os.path.join(assets_dir, fn), format="PNG", optimize=True)
        paths[key] = f"assets/{fn}"

    # All square indices shifted into `[128 .. 191]` (`250` on moat) for 100% lossless linearRGB & Display-P3!
    csq = np.full((10, 10), 250, dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            csq[y, x] = 128 + 8 * y + x
    _save(_gray_rgba_from_bytes(csq), "cell_sq", "cell_sq.png")

    bm10 = np.zeros((10, 10), dtype=np.uint8)
    bm10[:8, :8] = 255
    _save(_gray_rgba_from_bytes(bm10), "bmask10", "bmask10.png")

    bm6 = np.zeros((6, 6), dtype=np.uint8)
    bm6[:4, :4] = 255
    _save(_gray_rgba_from_bytes(bm6), "bmask6", "bmask6.png")

    # Visual 8x8 board maps (row 0 = Rank 8 top, row 7 = Rank 1 bottom, col 0..7 = File a..h)
    sq8_w = np.zeros((8, 8), dtype=np.uint8)
    sq8_b = np.zeros((8, 8), dtype=np.uint8)
    chk8 = np.zeros((8, 8), dtype=np.uint8)
    for vrow in range(8):
        r = 7 - vrow
        for f in range(8):
            sq8_w[vrow, f] = 128 + 8 * r + f
            sq8_b[vrow, f] = 128 + 8 * (7 - r) + f
            chk8[vrow, f] = 212 if (r + f) % 2 == 1 else 148
    _save(_gray_rgba_from_bytes(sq8_w), "sq8w", "sq8w.png")
    _save(_gray_rgba_from_bytes(sq8_b), "sq8b", "sq8b.png")
    _save(_gray_rgba_from_bytes(sq8_w[::-1, ::-1].copy()), "sq8wf", "sq8w_flip.png")
    _save(_gray_rgba_from_bytes(sq8_b[::-1, ::-1].copy()), "sq8bf", "sq8b_flip.png")
    _save(_gray_rgba_from_bytes(chk8), "chk8", "chk8.png")

    tie_u8 = np.broadcast_to(compute_tie_ramp_u8()[None, :], (2, 64)).copy()
    _save(_gray_rgba_from_bytes(tie_u8), "tie", "tie_ramp.png")

    # M2 quadrant broadcast LUTs (`640x10` -> `640x160`, `scale = 36 * u`, `a_sh = 18 * u` so `640+18 <= 664`!):
    d_m2y = [(2 * ((y // 10) // 4) - y) % 10 for y in range(160)]
    _save(_deltas_to_gray_strip(d_m2y, horizontal=False, thickness=4, scale=36.0), "m2y", "lut_m2_y.png")

    m2x_d = np.zeros((40, 10), dtype=np.int32)
    for ym in range(40):
        qx = ym // 10
        for cx in range(10):
            m2x_d[ym, cx] = 10 + 2 * qx - cx
    m2x_u8 = np.round(127.5 + m2x_d * (255.0 / 36.0)).astype(np.uint8)
    _save(_gray_rgba_from_bytes(m2x_u8), "m2x", "lut_m2_x.png")

    # Quadrant top-left square index `128 + 16*qy + 2*qx` (`4 x 160` phys):
    qb = np.zeros((160, 4), dtype=np.uint8)
    for y in range(160):
        q = y // 10
        qy, qx = divmod(q, 4)
        qb[y, :] = 128 + 16 * qy + 2 * qx
    _save(_gray_rgba_from_bytes(qb), "qbase", "lut_qbase.png")

    # Compaction LUTs (`scale = 230 * u`):
    # 1. `10 -> 6` (`{0, 2, 4, 6, 8, 8}`): X slice (22 cells = 132 px) & Y (16 cells = 96 px)
    t_10_6 = [0, 2, 4, 6, 8, 8]
    _save(_deltas_to_gray_strip([10 * (i // 6) + t_10_6[i % 6] - i for i in range(22 * 6)], True, 4), "c6x", "lut_c6_x.png")
    _save(_deltas_to_gray_strip([10 * (i // 6) + t_10_6[i % 6] - i for i in range(16 * 6)], False, 4), "c6y", "lut_c6_y.png")

    # 2. `6 -> 2` (`{0, 2}`): X slice (22 cells = 44 px) & Y (16 cells = 32 px)
    t_6_2 = [0, 2]
    _save(_deltas_to_gray_strip([6 * (i // 2) + t_6_2[i % 2] - i for i in range(22 * 2)], True, 4), "c2x", "lut_c2_x.png")
    _save(_deltas_to_gray_strip([6 * (i // 2) + t_6_2[i % 2] - i for i in range(16 * 2)], False, 4), "c2y", "lut_c2_y.png")

    # 3. `2 -> 1` (`{0}`): X (64 cells = 64 px) & Y (16 cells = 16 px)
    _save(_deltas_to_gray_strip([i for i in range(64)], True, 4), "c1x", "lut_c1_x.png")
    _save(_deltas_to_gray_strip([i for i in range(16)], False, 4), "c1y", "lut_c1_y.png")

    # 4. `10 -> 2` (`{0, 4}`): X slice (14 cells = 28 px, max delta 107) & Y slice (8 cells = 16 px, max delta 59)
    t_10_2 = [0, 4]
    _save(_deltas_to_gray_strip([10 * (i // 2) + t_10_2[i % 2] - i for i in range(14 * 2)], True, 4), "c10_2x", "lut_c10_2_x.png")
    _save(_deltas_to_gray_strip([10 * (i // 2) + t_10_2[i % 2] - i for i in range(8 * 2)], False, 4), "c10_2y", "lut_c10_2_y.png")

    # DPR-specific visual zoom LUTs (1x and 2x):
    for dpr in (1, 2):
        _save(
            build_periodic_zoom_lut(VIS_BOARD_SZ * dpr, list(range(8)), 28 * dpr, 7, 8, True, 4 * dpr),
            f"vis_bx_{dpr}x", f"lut_vis_bx_{dpr}x.png",
        )
        _save(
            build_periodic_zoom_lut(VIS_BOARD_SZ * dpr, list(range(8)), 28 * dpr, 7, 8, False, 4 * dpr),
            f"vis_by_{dpr}x", f"lut_vis_by_{dpr}x.png",
        )
        _save(
            build_periodic_zoom_lut(VIS_MAT_W * dpr, list(range(64)), 6 * dpr, 63, 64, True, 4 * dpr),
            f"vis_mx_{dpr}x", f"lut_vis_mx_{dpr}x.png",
        )
        _save(
            build_periodic_zoom_lut(VIS_MAT_H * dpr, list(range(16)), 12 * dpr, 15, 16, False, 4 * dpr),
            f"vis_my_{dpr}x", f"lut_vis_my_{dpr}x.png",
        )
        _save(
            build_periodic_zoom_lut(VIS_FEAT_H * dpr, list(range(16)), 4 * dpr, 15, 16, False, 4 * dpr),
            f"vis_fy_{dpr}x", f"lut_vis_fy_{dpr}x.png",
        )
        bw_p = VIS_WINBAR_W * dpr
        bh_p = 4 * dpr
        b_vals = np.broadcast_to(
            np.round(255.0 - 127.0 * np.arange(bw_p, dtype=np.float64) / max(1, bw_p - 1)).astype(np.uint8)[None, :],
            (bh_p, bw_p),
        ).copy()
        _save(_gray_rgba_from_bytes(b_vals), f"bar_{dpr}x", f"lut_winbar_{dpr}x.png")

    return paths


# ---------------------------------------------------------------------------
# 2. Bit-Exact 8-Bit SVG 2-Ply CNN Minimax Simulator
# ---------------------------------------------------------------------------

def svg_lut_eval(x: np.ndarray, table_vals: Sequence[float]) -> np.ndarray:
    tbl = np.asarray(table_vals, dtype=np.float64)
    n = len(tbl) - 1
    pos = np.clip(x, 0.0, 1.0) * n
    idx = np.minimum(np.floor(pos).astype(np.int64), n - 1)
    frac = pos - idx
    return q8(tbl[idx] * (1.0 - frac) + tbl[idx + 1] * frac)


def svg_fe_convolve_batch(
    buf: np.ndarray,
    order: Sequence[int],
    tx: int,
    ty: int,
    kernel_flat: Sequence[float],
    divisor: float,
    bias: float,
    stride: int = 1,
) -> np.ndarray:
    ow, oh = int(order[0]), int(order[1])
    k = np.asarray(kernel_flat, dtype=np.float64).reshape(oh, ow)
    bsz, h, w = buf.shape
    padded = np.pad(buf, ((0, 0), (ty, oh - 1 - ty), (tx, ow - 1 - tx)), mode="constant", constant_values=0.0)
    acc = np.zeros((bsz, h, w), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            kv = k[oh - 1 - i, ow - 1 - j]
            if kv != 0.0:
                acc += kv * padded[:, i : i + h, j : j + w]
    res = q8(acc / float(divisor) + float(bias))
    if stride > 1:
        return res[:, 0::stride, 0::stride]
    if oh == 2 and ow == 2 and tx == 0 and ty == 0 and h == 2 and w == 2:
        return res[:, 0:1, 0:1]
    return res


def svg_pointwise_batch(ch_bufs: np.ndarray, pw_spec: dict) -> np.ndarray:
    bsz, _, h, w = ch_bufs.shape
    z0 = float(pw_spec["zero_point"])
    out = np.zeros((bsz, pw_spec["out_channels"], h, w), dtype=np.float64)
    for trip in pw_spec["triplets"]:
        o = trip["out_triplet"]
        acc = None
        for g, mat20 in enumerate(trip["color_matrices"]):
            m = np.asarray(mat20, dtype=np.float64).reshape(4, 5)[:3, :]
            rgb_in = ch_bufs[:, 3 * g : 3 * (g + 1)]
            part = q8(np.einsum("rc,bchw->brhw", m[:, :3], rgb_in) + m[:, 4].reshape(1, 3, 1, 1))
            acc = part if acc is None else q8(acc + part - z0)
        slopes = np.asarray(trip["rescale_slopes"], dtype=np.float64).reshape(1, 3, 1, 1)
        intercepts = np.asarray(trip["rescale_intercepts"], dtype=np.float64).reshape(1, 3, 1, 1)
        out[:, 3 * o : 3 * (o + 1)] = q8(acc * slopes + intercepts)
    return out


def evaluate_svg_cnn_batch(boards_u8_n688: np.ndarray, spec: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-exact 8-bit SVG `<filter>` CNN Value Network evaluation on `(N, 6, 8, 8)` uint8 boards."""
    ly = spec["layers"]
    x0 = q8(np.asarray(boards_u8_n688, dtype=np.float64) / 255.0)

    sk_spec = ly["skip_mat"]
    z0 = float(sk_spec["zero_point"])
    m0 = np.asarray(sk_spec["color_matrices"][0], dtype=np.float64).reshape(4, 5)[:3, :]
    m1 = np.asarray(sk_spec["color_matrices"][1], dtype=np.float64).reshape(4, 5)[:3, :]
    p0 = q8(np.einsum("rc,bchw->brhw", m0[:, :3], x0[:, 0:3]) + m0[:, 4].reshape(1, 3, 1, 1))
    p1 = q8(np.einsum("rc,bchw->brhw", m1[:, :3], x0[:, 3:6]) + m1[:, 4].reshape(1, 3, 1, 1))
    sk = q8(p0 + p1 - z0)
    for stg in sk_spec["sum_stages"]:
        stride = 2 if sk.shape[-1] > 2 else 1
        sk = np.stack(
            [svg_fe_convolve_batch(sk[:, c], stg["order"], 0, 0, stg["kernelMatrix"], stg["divisor"], stg["bias"], stride=stride)
             for c in range(3)],
            axis=1,
        )

    h = svg_pointwise_batch(x0, ly["mix1"])
    h = np.stack(
        [svg_fe_convolve_batch(h[:, f["channel"]], f["order"], f["targetX"], f["targetY"], f["kernelMatrix"], f["divisor"], f["bias"])
         for f in ly["dwconv1"]["filters"]],
        axis=1,
    )
    p1_s = ly["pool1"]
    h = np.stack(
        [svg_fe_convolve_batch(h[:, c], p1_s["order"], 0, 0, p1_s["kernelMatrix"], p1_s["divisor"], 0.0, stride=2)
         for c in range(h.shape[1])],
        axis=1,
    )

    h = svg_pointwise_batch(h, ly["mix2"])
    h = np.stack(
        [svg_fe_convolve_batch(h[:, f["channel"]], f["order"], f["targetX"], f["targetY"], f["kernelMatrix"], f["divisor"], f["bias"])
         for f in ly["dwconv2"]["filters"]],
        axis=1,
    )
    p2_s = ly["pool2"]
    h = np.stack(
        [svg_fe_convolve_batch(h[:, c], p2_s["order"], 0, 0, p2_s["kernelMatrix"], p2_s["divisor"], 0.0, stride=2)
         for c in range(h.shape[1])],
        axis=1,
    )

    h = svg_pointwise_batch(h, ly["mix3"])
    h = np.stack(
        [svg_fe_convolve_batch(h[:, f["channel"]], f["order"], f["targetX"], f["targetY"], f["kernelMatrix"], f["divisor"], f["bias"])
         for f in ly["dwconv3"]["filters"]],
        axis=1,
    )

    feat = np.concatenate([h[:, :, 0, 0], sk[:, :, 0, 0]], axis=1)  # (N, 27)
    vh = ly["value_head"]
    z0_vh = float(vh["zero_point"])
    acc = None
    for g, mat20 in enumerate(vh["color_matrices"]):
        m = np.asarray(mat20, dtype=np.float64).reshape(4, 5)[0, :]
        part = q8(feat[:, 3 * g : 3 * (g + 1)] @ m[:3] + m[4])
        acc = part if acc is None else q8(acc + part - z0_vh)

    two_m = 2.0 * float(vh["logit_half_range_M"])
    decoded_logits = (acc - z0_vh) * two_m
    v_svg = svg_lut_eval(acc, vh["sigmoid_table_values"])
    v_u8 = np.round(v_svg * 255.0).astype(np.int32)
    u_min = float(vh.get("u8_min", 3))
    u_span = float(vh.get("u8_max", 253) - u_min)
    probs = (v_svg * 255.0 - u_min) / max(1.0, u_span)
    return decoded_logits, probs, v_u8


AI_RAYS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 1), (0, -1, 1),
    (1, 1, 2), (-1, -1, 2),
    (1, -1, 3), (-1, 1, 3),
)
AI_KNIGHT_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, -2), (1, -2), (-2, -1), (2, -1),
    (-2, 1), (2, 1), (-1, 2), (1, 2),
)
AI_KING_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)


def compute_ply2_base_tier(pt_src: int, mvv: int, is_r0: int) -> int:
    """Exact 5-bit tier `0 .. 31` so `score_u8 = 128 + 4 * tier + sub_xy` stays in `[128 .. 255]`!
    Preserves `tier(B) - tier(R) == 1` and `tier(R) - tier(Q) == 1` identically across ALL `mvv in 0..4`.
    """
    if mvv > 0:
        if pt_src == 1 and is_r0:
            return 31 if mvv == 4 else (25 + mvv)
        base = 25 if mvv == 4 else (6 * mvv)
        return base + (6 - pt_src)
    if pt_src == 1 and is_r0:
        return 24
    dev_map = {6: 0, 1: 1, 5: 2, 4: 3, 3: 4, 2: 5}
    return dev_map[pt_src]


def _apply_ply1_on_norm_board(root_u8: np.ndarray, u1: int, v1: int, code: int) -> np.ndarray:
    s1 = root_u8.copy()
    if code <= 0 or u1 >= 64 or v1 >= 64:
        return s1
    uy, ux = divmod(u1, 8)
    vy, vx = divmod(v1, 8)
    s1[:, uy, ux] = 128
    s1[:, vy, vx] = 128
    if code <= 6:
        s1[code - 1, vy, vx] = 255
    elif code == 7:  # O-O (`4 -> 6`, Rook `7 -> 5`)
        s1[5, 0, 6] = 255
        s1[:, 0, 7] = 128
        s1[3, 0, 5] = 255
    elif code == 8:  # O-O-O (`4 -> 2`, Rook `0 -> 3`)
        s1[5, 0, 2] = 255
        s1[:, 0, 0] = 128
        s1[3, 0, 3] = 255
    elif code == 9:  # En Passant (`u1 -> v1`, clear `(vx, 4)`)
        s1[0, vy, vx] = 255
        s1[:, 4, vx] = 128
    return s1


def _select_ply2_quadrant_replies(s1_u8: np.ndarray) -> Tuple[List[Tuple[int, int, int] | None], bool]:
    """Exact Stages `s1b..s1i` Ply-2 FIDE King-safety, pin-axis, check-evasion & 16-quadrant MVV-LVA selector."""
    ai_mask = (s1_u8 == 255)
    opp_mask = (s1_u8 <= 1)
    ai_any = ai_mask.any(axis=0)
    opp_any = opp_mask.any(axis=0)
    empty = ~(ai_any | opp_any)
    opp_k = opp_mask[5]
    onk = empty | opp_k

    katt = np.zeros((8, 8), dtype=bool)
    for y in range(8):
        for x in range(8):
            if ai_mask[0, y, x] and y + 1 < 8:
                if x > 0:
                    katt[y + 1, x - 1] = True
                if x < 7:
                    katt[y + 1, x + 1] = True
            if ai_mask[1, y, x]:
                for dx, dy in AI_KNIGHT_DELTAS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        katt[ny, nx] = True
            if ai_mask[5, y, x]:
                for dx, dy in AI_KING_DELTAS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        katt[ny, nx] = True
            for d_idx, (dx, dy, _ax) in enumerate(AI_RAYS):
                is_sl = (ai_mask[3, y, x] or ai_mask[4, y, x]) if d_idx < 4 else (ai_mask[2, y, x] or ai_mask[4, y, x])
                if not is_sl:
                    continue
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    katt[ny, nx] = True
                    if not onk[ny, nx]:
                        break
                    nx += dx
                    ny += dy

    ky_arr, kx_arr = np.where(opp_k)
    if len(ky_arr) == 0:
        return [None] * 16, False
    ky, kx = int(ky_arr[0]), int(kx_arr[0])

    chk_count = 0
    evade_single = np.zeros((8, 8), dtype=bool)
    pin_axis = np.full((8, 8), -1, dtype=np.int32)

    if ky > 0:
        for px in (kx - 1, kx + 1):
            if 0 <= px < 8 and ai_mask[0, ky - 1, px]:
                chk_count += 1
                evade_single[ky - 1, px] = True
    for dx, dy in AI_KNIGHT_DELTAS:
        nx, ny = kx + dx, ky + dy
        if 0 <= nx < 8 and 0 <= ny < 8 and ai_mask[1, ny, nx]:
            chk_count += 1
            evade_single[ny, nx] = True

    for d_idx, (dx, dy, ax) in enumerate(AI_RAYS):
        ray_sqs: List[Tuple[int, int]] = []
        pinned_cand: Tuple[int, int] | None = None
        nx, ny = kx + dx, ky + dy
        while 0 <= nx < 8 and 0 <= ny < 8:
            if empty[ny, nx]:
                ray_sqs.append((nx, ny))
            elif opp_any[ny, nx]:
                if pinned_cand is None:
                    pinned_cand = (nx, ny)
                else:
                    break
            else:
                is_sl = (ai_mask[3, ny, nx] or ai_mask[4, ny, nx]) if d_idx < 4 else (ai_mask[2, ny, nx] or ai_mask[4, ny, nx])
                if is_sl:
                    if pinned_cand is None:
                        chk_count += 1
                        evade_single[ny, nx] = True
                        for rx, ry in ray_sqs:
                            evade_single[ry, rx] = True
                    else:
                        pin_axis[pinned_cand[1], pinned_cand[0]] = ax
                break
            nx += dx
            ny += dy

    opp_in_check = chk_count > 0
    if chk_count == 0:
        evade_mask = np.ones((8, 8), dtype=bool)
    elif chk_count == 1:
        evade_mask = evade_single
    else:
        evade_mask = np.zeros((8, 8), dtype=bool)

    mvv_grid = np.zeros((8, 8), dtype=np.int32)
    mvv_grid[ai_mask[0]] = 1
    mvv_grid[ai_mask[1] | ai_mask[2]] = 2
    mvv_grid[ai_mask[3]] = 3
    mvv_grid[ai_mask[4] | ai_mask[5]] = 4

    ai_k_adj = np.zeros((8, 8), dtype=bool)
    for ay in range(8):
        for ax in range(8):
            if ai_mask[5, ay, ax]:
                for dx, dy in AI_KING_DELTAS:
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        ai_k_adj[ny, nx] = True

    best_sq_score = np.zeros((8, 8), dtype=np.int32)
    best_sq_u2 = np.full((8, 8), -1, dtype=np.int32)
    best_sq_pt2 = np.zeros((8, 8), dtype=np.int32)

    def _consider_dest(ux: int, uy: int, tx: int, ty: int, pt_src: int) -> None:
        u2 = 8 * uy + ux
        placed_pt = 5 if (pt_src == 1 and ty == 0) else pt_src
        mvv = int(mvv_grid[ty, tx])
        base_tier = compute_ply2_base_tier(pt_src, mvv, 1 if ty == 0 else 0)
        if pt_src == 5 and ai_k_adj[ty, tx]:
            base_tier = max(base_tier, 12)
        sub_xy = 2 * (ty % 2) + (tx % 2)
        sc = 128 + 4 * base_tier + sub_xy
        # Matches `<feBlend mode="lighten">` on `128 + u2`: higher `u2` wins same-piece-type tie!
        if sc > best_sq_score[ty, tx] or (sc == best_sq_score[ty, tx] and u2 > best_sq_u2[ty, tx]):
            best_sq_score[ty, tx] = sc
            best_sq_u2[ty, tx] = u2
            best_sq_pt2[ty, tx] = placed_pt

    for ty in range(8):
        for tx in range(8):
            if opp_any[ty, tx]:
                continue
            if evade_mask[ty, tx] and ty + 1 < 8:
                if ai_any[ty, tx]:
                    if tx > 0 and opp_mask[0, ty + 1, tx - 1] and pin_axis[ty + 1, tx - 1] in (-1, 3):
                        _consider_dest(tx - 1, ty + 1, tx, ty, 1)
                    if tx < 7 and opp_mask[0, ty + 1, tx + 1] and pin_axis[ty + 1, tx + 1] in (-1, 2):
                        _consider_dest(tx + 1, ty + 1, tx, ty, 1)
                else:
                    if opp_mask[0, ty + 1, tx] and pin_axis[ty + 1, tx] in (-1, 1):
                        _consider_dest(tx, ty + 1, tx, ty, 1)
                    if ty == 4 and empty[5, tx] and opp_mask[0, 6, tx] and pin_axis[6, tx] in (-1, 1):
                        _consider_dest(tx, 6, tx, ty, 1)

            if evade_mask[ty, tx]:
                for dx, dy in AI_KNIGHT_DELTAS:
                    ux, uy = tx + dx, ty + dy
                    if 0 <= ux < 8 and 0 <= uy < 8 and opp_mask[1, uy, ux] and pin_axis[uy, ux] == -1:
                        _consider_dest(ux, uy, tx, ty, 2)

            if evade_mask[ty, tx]:
                for d_idx, (dx, dy, ax) in enumerate(AI_RAYS):
                    ux, uy = tx + dx, ty + dy
                    while 0 <= ux < 8 and 0 <= uy < 8:
                        if empty[uy, ux]:
                            ux += dx
                            uy += dy
                            continue
                        if opp_any[uy, ux] and pin_axis[uy, ux] in (-1, ax):
                            if d_idx < 4:
                                if opp_mask[3, uy, ux]:
                                    _consider_dest(ux, uy, tx, ty, 4)
                                elif opp_mask[4, uy, ux]:
                                    _consider_dest(ux, uy, tx, ty, 5)
                            else:
                                if opp_mask[2, uy, ux]:
                                    _consider_dest(ux, uy, tx, ty, 3)
                                elif opp_mask[4, uy, ux]:
                                    _consider_dest(ux, uy, tx, ty, 5)
                        break

            if not katt[ty, tx]:
                for dx, dy in AI_KING_DELTAS:
                    ux, uy = tx + dx, ty + dy
                    if 0 <= ux < 8 and 0 <= uy < 8 and opp_mask[5, uy, ux]:
                        _consider_dest(ux, uy, tx, ty, 6)

    quad_moves: List[Tuple[int, int, int] | None] = []
    for qy in range(4):
        for qx in range(4):
            best_q = (0, 64, 64, 0)
            for dy in (0, 1):
                for dx in (0, 1):
                    tx, ty = 2 * qx + dx, 2 * qy + dy
                    sc = int(best_sq_score[ty, tx])
                    if sc > best_q[0]:
                        best_q = (sc, int(best_sq_u2[ty, tx]), 8 * ty + tx, int(best_sq_pt2[ty, tx]))
            if best_q[0] == 0:
                quad_moves.append(None)
            else:
                quad_moves.append((best_q[1], best_q[2], best_q[3]))

    return quad_moves, opp_in_check


def _apply_ply2_on_norm_board(s1_u8: np.ndarray, u2: int, v2: int, pt2: int) -> np.ndarray:
    s2 = s1_u8.copy()
    uy, ux = divmod(u2, 8)
    vy, vx = divmod(v2, 8)
    s2[:, uy, ux] = 128
    s2[:, vy, vx] = 128
    s2[pt2 - 1, vy, vx] = 1
    return s2


def _shift_8x8(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(arr)
    x0_s, x1_s = max(0, -dx), min(8, 8 - dx)
    y0_s, y1_s = max(0, -dy), min(8, 8 - dy)
    out[y0_s + dy : y1_s + dy, x0_s + dx : x1_s + dx] = arr[y0_s:y1_s, x0_s:x1_s]
    return out


def _ray_prop_8x8(src: np.ndarray, empty: np.ndarray, dx: int, dy: int) -> np.ndarray:
    r = _shift_8x8(src, dx, dy)
    e = _shift_8x8(empty, dx, dy)
    for mul in (1, 2):
        rs = _shift_8x8(r, mul * dx, mul * dy)
        es = _shift_8x8(e, mul * dx, mul * dy)
        r = np.clip(r + e * rs, 0, 1)
        e = e * es
    rs = _shift_8x8(r, 4 * dx, 4 * dy)
    return np.clip(r + e * rs, 0, 1)


def _is_ai_mated_on_s2(s2_u8: np.ndarray) -> bool:
    """Bit-exact mirror of Stages `s2m0..s2m10`: detects when Opponent delivers Checkmate on Ply 2 (`s2`)."""
    ai = (s2_u8 == 255).astype(np.int32)
    op = (s2_u8 <= 1).astype(np.int32)
    ai_P, ai_N, ai_B, ai_R, ai_Q, ai_K = [ai[i] for i in range(6)]
    op_P, op_N, op_B, op_R, op_Q, op_K = [op[i] for i in range(6)]
    ai_any = np.clip(ai.sum(axis=0), 0, 1)
    op_any = np.clip(op.sum(axis=0), 0, 1)
    empty = 1 - np.clip(ai_any + op_any, 0, 1)

    opp_att = np.clip(_shift_8x8(op_P, 1, -1) + _shift_8x8(op_P, -1, -1), 0, 1)
    for dx, dy in AI_KNIGHT_DELTAS:
        opp_att = np.clip(opp_att + _shift_8x8(op_N, dx, dy), 0, 1)
    for dx, dy in AI_KING_DELTAS:
        opp_att = np.clip(opp_att + _shift_8x8(op_K, dx, dy), 0, 1)

    chk_pn_p = op_P * np.clip(_shift_8x8(ai_K, -1, 1) + _shift_8x8(ai_K, 1, 1), 0, 1)
    kn_k = np.zeros((8, 8), dtype=np.int32)
    for dx, dy in AI_KNIGHT_DELTAS:
        kn_k = np.clip(kn_k + _shift_8x8(ai_K, dx, dy), 0, 1)
    chk_sq = np.clip(chk_pn_p + op_N * kn_k, 0, 1)
    line_kz = np.zeros((8, 8), dtype=np.int32)

    axes = (
        ((1, 0), np.clip(op_R + op_Q, 0, 1)),
        ((0, 1), np.clip(op_R + op_Q, 0, 1)),
        ((1, 1), np.clip(op_B + op_Q, 0, 1)),
        ((1, -1), np.clip(op_B + op_Q, 0, 1)),
    )
    for (dx, dy), op_sl in axes:
        rp_r = _ray_prop_8x8(op_sl, empty, dx, dy)
        rp_g = _ray_prop_8x8(ai_K, empty, dx, dy)
        rn_r = _ray_prop_8x8(op_sl, empty, -dx, -dy)
        rn_g = _ray_prop_8x8(ai_K, empty, -dx, -dy)
        xray_p = _shift_8x8(ai_K * rp_r, dx, dy)
        xray_n = _shift_8x8(ai_K * rn_r, -dx, -dy)
        opp_att = np.clip(opp_att + rp_r + rn_r + xray_p + xray_n, 0, 1)
        chk_sq = np.clip(chk_sq + np.clip(rp_g + rn_g, 0, 1) * op_sl, 0, 1)
        line_kz = np.clip(line_kz + np.clip(rp_g * rn_r + rn_g * rp_r, 0, 1), 0, 1)

    k_nb = np.zeros((8, 8), dtype=np.int32)
    for dx, dy in AI_KING_DELTAS:
        k_nb = np.clip(k_nb + _shift_8x8(ai_K, dx, dy), 0, 1)
    kstep_ok = (1 - ai_any) * (1 - opp_att) * k_nb
    n_chk = int(chk_sq.sum())
    if n_chk == 0 or int(kstep_ok.sum()) > 0:
        return False
    if n_chk >= 2:
        return True

    evade_all = np.clip(chk_sq + line_kz * empty, 0, 1)
    unpinned = 1 - (line_kz * ai_any)
    uP, uN, uB, uR, uQ = ai_P * unpinned, ai_N * unpinned, ai_B * unpinned, ai_R * unpinned, ai_Q * unpinned

    p1 = _shift_8x8(uP, 0, 1) * empty
    r2_mask = np.zeros((8, 8), dtype=np.int32)
    r2_mask[2, :] = 1
    p2 = _shift_8x8(p1 * r2_mask, 0, 1) * empty
    pc = np.clip(_shift_8x8(uP, 1, 1) + _shift_8x8(uP, -1, 1), 0, 1) * op_any
    reach = np.clip(p1 + p2 + pc, 0, 1)

    for dx, dy in AI_KNIGHT_DELTAS:
        reach = np.clip(reach + _shift_8x8(uN, dx, dy), 0, 1)
    uRQ = np.clip(uR + uQ, 0, 1)
    uBQ = np.clip(uB + uQ, 0, 1)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        reach = np.clip(reach + _ray_prop_8x8(uRQ, empty, dx, dy), 0, 1)
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        reach = np.clip(reach + _ray_prop_8x8(uBQ, empty, dx, dy), 0, 1)

    return int((reach * evade_all).sum()) == 0


def ply3_tier_and_placed(pt3: int, tc: int) -> Tuple[int, int]:
    """Exact SEE-filtered Ply-3 Quiescence `(tier in 1..15, placed_code in 0..6)` for `(pt3, tc)`."""
    if tc == 0:
        return 0, 0
    if 172 <= tc <= 175:
        if pt3 != 6:
            return 0, 0
        mvv = (tc - 172) + 1
        return {1: 2, 2: 5, 3: 8, 4: 12}[mvv], 6
    if not (140 <= tc <= 171):
        return 0, 0
    idx = tc - 140
    mvv = (idx // 8) + 1
    rem = idx % 8
    def_opp = rem // 4
    is_r7 = (rem // 2) % 2
    is_chk1 = rem % 2

    if def_opp == 0:
        if pt3 == 1 and is_r7 == 1:
            return (15 if mvv == 4 else 14), 5
        hi = pt3 in (5, 6)
        tier_map = {
            1: 2 if hi else 3,
            2: 5 if hi else 6,
            3: 8 if hi else 9,
            4: 12 if hi else 13,
        }
        return tier_map[mvv], pt3

    if pt3 == 6:
        return 0, 0
    val_m = {1: 1, 2: 3, 3: 5, 4: 9}[mvv]
    val_a = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}[pt3]
    gain = val_m - val_a
    if gain <= -1:
        return 0, 0
    if gain == 0:
        return (1, 0) if is_chk1 == 1 else (0, 0)
    return {2: 4, 4: 7, 6: 10, 8: 11}[gain], 0


def _select_and_apply_ply3_quiescence_8x8(
    s2_u8: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int] | None]:
    """Bit-exact mirror of Stages `s2q0..s2q7`: FIDE-legal SEE-filtered Ply-3 Quiescence update."""
    ai = (s2_u8 == 255).astype(np.int32)
    op = (s2_u8 <= 1).astype(np.int32)
    ai_P, ai_N, ai_B, ai_R, ai_Q, ai_K = [ai[i] for i in range(6)]
    op_P, op_N, op_B, op_R, op_Q, op_K = [op[i] for i in range(6)]
    ai_any = np.clip(ai.sum(axis=0), 0, 1)
    op_any = np.clip(op.sum(axis=0), 0, 1)
    empty = 1 - np.clip(ai_any + op_any, 0, 1)

    opp_att = np.clip(_shift_8x8(op_P, 1, -1) + _shift_8x8(op_P, -1, -1), 0, 1)
    for dx, dy in AI_KNIGHT_DELTAS:
        opp_att = np.clip(opp_att + _shift_8x8(op_N, dx, dy), 0, 1)
    for dx, dy in AI_KING_DELTAS:
        opp_att = np.clip(opp_att + _shift_8x8(op_K, dx, dy), 0, 1)

    chk_pn_p = op_P * np.clip(_shift_8x8(ai_K, -1, 1) + _shift_8x8(ai_K, 1, 1), 0, 1)
    kn_k = np.zeros((8, 8), dtype=np.int32)
    for dx, dy in AI_KNIGHT_DELTAS:
        kn_k = np.clip(kn_k + _shift_8x8(ai_K, dx, dy), 0, 1)
    chk_sq = np.clip(chk_pn_p + op_N * kn_k, 0, 1)

    pin_ax = np.zeros((8, 8), dtype=np.int32)
    axes = (
        (1, (1, 0), np.clip(op_R + op_Q, 0, 1)),
        (2, (0, 1), np.clip(op_R + op_Q, 0, 1)),
        (3, (1, 1), np.clip(op_B + op_Q, 0, 1)),
        (4, (1, -1), np.clip(op_B + op_Q, 0, 1)),
    )
    for ax_code, (dx, dy), op_sl in axes:
        rp_r = _ray_prop_8x8(op_sl, empty, dx, dy)
        rp_g = _ray_prop_8x8(ai_K, empty, dx, dy)
        rn_r = _ray_prop_8x8(op_sl, empty, -dx, -dy)
        rn_g = _ray_prop_8x8(ai_K, empty, -dx, -dy)
        xray_p = _shift_8x8(ai_K * rp_r, dx, dy)
        xray_n = _shift_8x8(ai_K * rn_r, -dx, -dy)
        opp_att = np.clip(opp_att + rp_r + rn_r + xray_p + xray_n, 0, 1)
        chk_sq = np.clip(chk_sq + np.clip(rp_g + rn_g, 0, 1) * op_sl, 0, 1)
        line_ax = np.clip(rp_g * rn_r + rn_g * rp_r, 0, 1)
        pin_ax = pin_ax + ax_code * line_ax

    n_chk = int(chk_sq.sum())
    ai_nonk = ai_any * (1 - ai_K)
    src_state = np.zeros((8, 8), dtype=np.int32)
    for y in range(8):
        for x in range(8):
            if not ai_nonk[y, x]:
                continue
            if n_chk >= 2:
                src_state[y, x] = 130
            elif n_chk == 1:
                src_state[y, x] = 131 if pin_ax[y, x] == 0 else 130
            else:
                src_state[y, x] = 131 if pin_ax[y, x] == 0 else (131 + int(pin_ax[y, x]))

    mvv_grid = op_P * 1 + np.clip(op_N + op_B, 0, 1) * 2 + op_R * 3 + op_Q * 4
    tgt_code = np.zeros((8, 8), dtype=np.int32)
    for y in range(8):
        for x in range(8):
            mvv = int(mvv_grid[y, x])
            if mvv == 0:
                continue
            def_opp = 1 if opp_att[y, x] > 0 else 0
            is_r7 = 1 if y == 7 else 0
            if n_chk == 0:
                tgt_code[y, x] = 140 + 8 * (mvv - 1) + 4 * def_opp + 2 * is_r7
            elif n_chk == 1 and chk_sq[y, x] == 1:
                tgt_code[y, x] = 140 + 8 * (mvv - 1) + 4 * def_opp + 2 * is_r7 + 1
            elif def_opp == 0:
                tgt_code[y, x] = 172 + (mvv - 1)

    def _cand(u3: int, vx: int, vy: int, pt3: int) -> Tuple[int, int, int]:
        if u3 < 0:
            return (0, 0, 0)
        tier, placed = ply3_tier_and_placed(pt3, int(tgt_code[vy, vx]))
        if tier == 0:
            return (0, 0, 0)
        return (128 + 8 * (tier - 1) + vy, 128 + u3, 128 + 8 * placed + vx)

    def _mux(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return b if b[0] > a[0] else a

    grid_cand = [[(0, 0, 0) for _ in range(8)] for _ in range(8)]
    for vy in range(8):
        for vx in range(8):
            if tgt_code[vy, vx] == 0:
                continue
            orth_best = 0
            for dx, dy, allow_st in (
                (1, 0, (131, 132)),
                (-1, 0, (131, 132)),
                (0, 1, (131, 133)),
                (0, -1, (131, 133)),
            ):
                ux, uy = vx - dx, vy - dy
                while 0 <= ux < 8 and 0 <= uy < 8:
                    if empty[uy, ux]:
                        ux -= dx
                        uy -= dy
                        continue
                    if int(src_state[uy, ux]) in allow_st:
                        u3 = 8 * uy + ux
                        if ai_R[uy, ux]:
                            orth_best = max(orth_best, 192 + u3)
                        elif ai_Q[uy, ux]:
                            orth_best = max(orth_best, 128 + u3)
                    break

            diag_best = 0
            for dx, dy, allow_st in (
                (1, 1, (131, 134)),
                (-1, -1, (131, 134)),
                (1, -1, (131, 135)),
                (-1, 1, (131, 135)),
            ):
                ux, uy = vx - dx, vy - dy
                while 0 <= ux < 8 and 0 <= uy < 8:
                    if empty[uy, ux]:
                        ux -= dx
                        uy -= dy
                        continue
                    if int(src_state[uy, ux]) in allow_st:
                        u3 = 8 * uy + ux
                        if ai_B[uy, ux]:
                            diag_best = max(diag_best, 192 + u3)
                        elif ai_Q[uy, ux]:
                            diag_best = max(diag_best, 128 + u3)
                    break

            u_b = (diag_best - 192) if diag_best >= 192 else -1
            u_r = (orth_best - 192) if orth_best >= 192 else -1
            q_o = (orth_best - 128) if (128 <= orth_best < 192) else -1
            q_d = (diag_best - 128) if (128 <= diag_best < 192) else -1
            u_q = max(q_o, q_d)

            u_k = -1
            for dx, dy in AI_KING_DELTAS:
                ux, uy = vx - dx, vy - dy
                if 0 <= ux < 8 and 0 <= uy < 8 and ai_K[uy, ux]:
                    u_k = 8 * uy + ux
                    break

            slk = _mux(_mux(_mux(_cand(u_b, vx, vy, 3), _cand(u_r, vx, vy, 4)), _cand(u_q, vx, vy, 5)), _cand(u_k, vx, vy, 6))

            u_p = -1
            for dx, allow_st in ((1, (131, 134)), (-1, (131, 135))):
                ux, uy = vx - dx, vy - 1
                if 0 <= ux < 8 and 0 <= uy < 8 and ai_P[uy, ux] and int(src_state[uy, ux]) in allow_st:
                    u_p = max(u_p, 8 * uy + ux)

            u_n = -1
            for dx, dy in AI_KNIGHT_DELTAS:
                ux, uy = vx - dx, vy - dy
                if 0 <= ux < 8 and 0 <= uy < 8 and ai_N[uy, ux] and int(src_state[uy, ux]) == 131:
                    u_n = max(u_n, 8 * uy + ux)

            pn = _mux(_cand(u_p, vx, vy, 1), _cand(u_n, vx, vy, 2))
            grid_cand[vy][vx] = _mux(pn, slk)

    cur = [[grid_cand[y][x] for x in range(8)] for y in range(8)]
    for step in (1, 2, 4):
        nxt = [[(0, 0, 0) for _ in range(8)] for _ in range(8)]
        for y in range(8):
            for x in range(8):
                b = cur[y][x + step] if x + step < 8 else (0, 0, 0)
                nxt[y][x] = _mux(cur[y][x], b)
        cur = nxt
    for step in (1, 2, 4):
        nxt = [[(0, 0, 0) for _ in range(8)] for _ in range(8)]
        for y in range(8):
            for x in range(8):
                b = cur[y + step][x] if y + step < 8 else (0, 0, 0)
                nxt[y][x] = _mux(cur[y][x], b)
        cur = nxt

    win_r, win_g, win_b = cur[0][0]
    if win_r < 128:
        return s2_u8.copy(), None

    v3y = (win_r - 128) % 8
    u3 = win_g - 128
    placed_code = (win_b - 128) // 8
    v3x = (win_b - 128) % 8
    v3 = 8 * v3y + v3x
    u3y, u3x = divmod(u3, 8)

    out = s2_u8.copy()
    out[:, u3y, u3x] = 128
    out[:, v3y, v3x] = 128
    if 1 <= placed_code <= 6:
        out[placed_code - 1, v3y, v3x] = 255
    return out, (u3, v3, placed_code, (win_r - 128) // 8 + 1)


def extract_root_board_u8_from_css(css_env: Dict[str, int]) -> np.ndarray:
    turn = int(css_env["--turn"])
    arr = np.full((6, 8, 8), 128, dtype=np.uint8)
    for ay in range(8):
        for ax in range(8):
            sq = (8 * ay + ax) if turn == 0 else (8 * (7 - ay) + ax)
            b_val = int(css_env[f"--b{sq}"])
            if b_val == 0:
                continue
            pt = b_val & 7
            if 1 <= pt <= 6:
                is_blk = 1 if b_val >= 8 else 0
                arr[pt - 1, ay, ax] = 255 if is_blk == turn else 1
    return arr


def simulate_svg_2ply_cnn(css_env: Dict[str, int], spec: dict) -> SVG2PlyResult:
    """Bit-exact 8-bit SVG `<filter>` simulation of the full 2-ply CNN Minimax pipeline."""
    turn = int(css_env["--turn"])
    root_u8 = extract_root_board_u8_from_css(css_env)
    tie_ramp = compute_tie_ramp_u8()

    m1_u = [int(css_env[f"--m1_u_{a}"]) for a in range(64)]
    m1_v = [int(css_env[f"--m1_v_{a}"]) for a in range(64)]
    m1_c = [int(css_env[f"--m1_c_{a}"]) for a in range(64)]

    v_sheet_u8 = np.full((16, 64), 255, dtype=np.uint8)
    q_scores = np.zeros(64, dtype=np.uint8)

    leaf_list: List[np.ndarray] = []
    leaf_mated: List[bool] = []
    leaf_coords: List[Tuple[int, int]] = []
    s1_in_check = np.zeros(64, dtype=bool)

    for v1 in range(64):
        code = m1_c[v1]
        if code <= 0:
            continue
        s1 = _apply_ply1_on_norm_board(root_u8, m1_u[v1], m1_v[v1], code)
        quads, opp_chk = _select_ply2_quadrant_replies(s1)
        s1_in_check[v1] = opp_chk
        for q_idx, q_mv in enumerate(quads):
            if q_mv is not None:
                u2, v2, pt2 = q_mv
                s2 = _apply_ply2_on_norm_board(s1, u2, v2, pt2)
                mated = _is_ai_mated_on_s2(s2)
                s2_q, _ = _select_and_apply_ply3_quiescence_8x8(s2)
                leaf_list.append(s2_q)
                leaf_mated.append(mated)
                leaf_coords.append((q_idx, v1))

    if leaf_list:
        _, _, u8_vals = evaluate_svg_cnn_batch(np.stack(leaf_list, axis=0), spec)
        for idx, (q_idx, v1) in enumerate(leaf_coords):
            v_sheet_u8[q_idx, v1] = 2 if leaf_mated[idx] else int(u8_vals[idx])

    is_term = int(css_env.get("--is_terminal", 0))
    if not is_term:
        for v1 in range(64):
            if m1_c[v1] <= 0:
                q_scores[v1] = 0
                continue
            col_min = int(v_sheet_u8[:, v1].min())
            if col_min == 255:
                q_scores[v1] = 254 if s1_in_check[v1] else 128
            else:
                q_scores[v1] = col_min

    max_q = int(q_scores.max())
    if max_q == 0:
        return SVG2PlyResult(64, 64, 64, 64, 0, 0, 0.5, q_scores, v_sheet_u8)

    tied_v1s = [v1 for v1 in range(64) if int(q_scores[v1]) == max_q and m1_c[v1] > 0]
    best_v1 = max(tied_v1s, key=lambda v: int(tie_ramp[v]))
    best_u1 = m1_u[best_v1]
    best_c1 = m1_c[best_v1]

    def _to_real(a: int) -> int:
        ar, af = divmod(a, 8)
        return (8 * ar + af) if turn == 0 else (8 * (7 - ar) + af)

    u_min = float(spec["layers"]["value_head"].get("u8_min", 3))
    u_span = float(spec["layers"]["value_head"].get("u8_max", 253) - u_min)
    win_prob = 1.0 if max_q >= 254 else float(np.clip((max_q - u_min) / max(1.0, u_span), 0.0, 1.0))

    return SVG2PlyResult(
        best_u_norm=best_u1,
        best_v_norm=best_v1,
        best_u_real=_to_real(best_u1),
        best_v_real=_to_real(best_v1),
        best_code=best_c1,
        best_score_u8=max_q,
        best_win_prob=win_prob,
        q_scores_u8=q_scores,
        v_sheet_u8=v_sheet_u8,
    )


# ---------------------------------------------------------------------------
# 3. Multi-Stage Linear-VRAM SVG `<filter>` Compiler (`<= 72` prims/stage)
# ---------------------------------------------------------------------------

def _wrap_filter_raw(fid: str, body: Sequence[str], w: int = STAGE_W, h: int = STAGE_H) -> str:
    return (
        f'<filter id="{fid}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" '
        f'x="0" y="0" width="{w}" height="{h}" color-interpolation-filters="sRGB">'
        f'{"".join(body)}</filter>'
    )


def _pack_rgb(p: List[str], r_in: str, g_in: str, b_in: str, x: int, y: int, w: int, h: int, out: str) -> None:
    p.append(f'<feColorMatrix in="{r_in}" x="{x}" y="{y}" width="{w}" height="{h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{out}_r"/>')
    p.append(f'<feColorMatrix in="{g_in}" x="{x}" y="{y}" width="{w}" height="{h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{out}_g"/>')
    p.append(f'<feColorMatrix in="{b_in}" x="{x}" y="{y}" width="{w}" height="{h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="{out}_b"/>')
    p.append(f'<feComposite in="{out}_r" in2="{out}_g" x="{x}" y="{y}" width="{w}" height="{h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{out}_rg"/>')
    p.append(f'<feComposite in="{out}_rg" in2="{out}_b" x="{x}" y="{y}" width="{w}" height="{h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{out}"/>')


def _emit_passthrough(p: List[str], new_nodes: Sequence[str], keep_vram: bool = True, dpr: int = 1) -> None:
    if keep_vram:
        nodes = ['<feMergeNode in="SourceGraphic"/>'] + [f'<feMergeNode in="{n}"/>' for n in new_nodes]
        p.append(f'<feMerge>{"".join(nodes)}</feMerge>')
    else:
        sc_p = lambda v: v if dpr == 1 else (v // 2)
        rx, rw = sc_p(642), sc_p(STAGE_W - 642)
        vh, cw, lh = sc_p(VRAM_H), sc_p(STAGE_W), sc_p(COMP_H - VRAM_H)
        p.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{rx}" y="0" width="{rw}" height="{vh}" result="pass_rcol"/>')
        p.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="{vh}" width="{cw}" height="{lh}" result="pass_bot"/>')
        nodes = [f'<feMergeNode in="{n}"/>' for n in new_nodes] + ['<feMergeNode in="pass_rcol"/>', '<feMergeNode in="pass_bot"/>']
        p.append(f'<feMerge>{"".join(nodes)}</feMerge>')


def _emit_compact_at_origin(
    p: List[str],
    src_in: str,
    sx: int,
    sy: int,
    sh: int,
    dx: int,
    dy: int,
    dw: int,
    dh: int,
    x_slices: Sequence[Tuple[int, int, int, int]],  # (sx_off, dx_off, src_sl_w, dst_sl_w)
    y_slices: Sequence[Tuple[int, int, int, int]] | None,  # None => single full Y pass
    lutx_tiled: str,
    luty_tiled: str,
    a_sh: float,
    disp_scale: str,
    out_name: str,
) -> None:
    """Executes 2-pass compaction at working origin `(0, 0)` so `dw + pad_a` NEVER exceeds `STAGE_W`!"""
    pad_a = int(a_sh) + 6
    src_ref = src_in
    if len(x_slices) > 1:
        src_full_w = max(sx_off + src_w for sx_off, _, src_w, _ in x_slices)
        src_ref = f"{out_name}_pin"
        p.append(f'<feTile in="{src_in}" x="{sx}" y="{sy}" width="{src_full_w}" height="{sh}" result="{src_ref}"/>')
    x_parts: List[str] = []
    for s_idx, (sx_off, dx_off, src_w, sl_w) in enumerate(x_slices):
        p.append(
            f'<feOffset in="{src_ref}" dx="{dx_off - (sx + sx_off)}" dy="{_fmt_num(-sy + a_sh)}" '
            f'x="{dx_off}" y="0" width="{src_w}" height="{sh + pad_a}" result="{out_name}_xs{s_idx}"/>'
        )
        p.append(
            f'<feDisplacementMap in="{out_name}_xs{s_idx}" in2="{lutx_tiled}" '
            f'x="{dx_off}" y="0" width="{sl_w}" height="{sh}" '
            f'scale="{disp_scale}" xChannelSelector="R" yChannelSelector="A" result="{out_name}_xp{s_idx}"/>'
        )
        x_parts.append(f'<feMergeNode in="{out_name}_xp{s_idx}"/>')

    if len(x_parts) == 1:
        x_merged = f"{out_name}_xp0"
    else:
        x_merged = f"{out_name}_xm"
        p.append(f'<feMerge x="0" y="0" width="{dw}" height="{sh}" result="{x_merged}">{"".join(x_parts)}</feMerge>')

    if y_slices is None:
        p.append(
            f'<feOffset in="{x_merged}" dx="{_fmt_num(a_sh)}" dy="0" '
            f'x="0" y="0" width="{dw + pad_a}" height="{sh}" result="{out_name}_ys"/>'
        )
        raw_res = out_name if (dx == 0 and dy == 0) else f"{out_name}_yr"
        p.append(
            f'<feDisplacementMap in="{out_name}_ys" in2="{luty_tiled}" '
            f'x="0" y="0" width="{dw}" height="{dh}" '
            f'scale="{disp_scale}" xChannelSelector="A" yChannelSelector="R" result="{raw_res}"/>'
        )
    else:
        y_parts: List[str] = []
        for y_idx, (sy_off, dy_off, src_h, sl_h) in enumerate(y_slices):
            p.append(
                f'<feOffset in="{x_merged}" dx="{_fmt_num(a_sh)}" dy="{dy_off - sy_off}" '
                f'x="0" y="{dy_off}" width="{dw + pad_a}" height="{src_h}" result="{out_name}_ys{y_idx}"/>'
            )
            p.append(
                f'<feDisplacementMap in="{out_name}_ys{y_idx}" in2="{luty_tiled}" '
                f'x="0" y="{dy_off}" width="{dw}" height="{sl_h}" '
                f'scale="{disp_scale}" xChannelSelector="A" yChannelSelector="R" result="{out_name}_yp{y_idx}"/>'
            )
            y_parts.append(f'<feMergeNode in="{out_name}_yp{y_idx}"/>')
        raw_res = out_name if (dx == 0 and dy == 0) else f"{out_name}_yr"
        p.append(f'<feMerge x="0" y="0" width="{dw}" height="{dh}" result="{raw_res}">{"".join(y_parts)}</feMerge>')

    if dx != 0 or dy != 0:
        p.append(f'<feOffset in="{raw_res}" dx="{dx}" dy="{dy}" x="{dx}" y="{dy}" width="{dw}" height="{dh}" result="{out_name}"/>')


def _emit_periodic_zoom_rect(
    p: List[str],
    src_in: str,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    dx: int,
    dy: int,
    dw: int,
    dh: int,
    lutx_in: str,
    luty_in: str,
    a_sh: float,
    disp_scale: str,
    out_name: str,
) -> None:
    """Periodic 2-pass visual zoom executed at `(0, 0)` so all intermediates stay well inside `STAGE_W x STAGE_H`."""
    pad_a = int(a_sh) + 6
    a_s = _fmt_num(a_sh)
    xs_w = dw + sw + 4
    xz_h = sh + pad_a
    yz_w = dw + pad_a
    ys_h = dh + sh + 4
    p.append(f'<feTile in="{lutx_in}" x="0" y="0" width="{xs_w}" height="{xz_h}" result="{out_name}_lxt"/>')
    p.append(f'<feTile in="{luty_in}" x="0" y="0" width="{yz_w}" height="{ys_h}" result="{out_name}_lyt"/>')
    p.append(f'<feOffset in="{src_in}" dx="{-sx}" dy="{-sy}" x="0" y="0" width="{sw}" height="{sh}" result="{out_name}_c"/>')
    p.append(f'<feTile in="{out_name}_c" x="0" y="0" width="{xs_w}" height="{sh}" result="{out_name}_ct"/>')
    p.append(f'<feOffset in="{out_name}_ct" dx="0" dy="{a_s}" x="0" y="0" width="{xs_w}" height="{xz_h}" result="{out_name}_xs"/>')
    p.append(f'<feDisplacementMap in="{out_name}_xs" in2="{out_name}_lxt" x="0" y="0" width="{xs_w}" height="{xz_h}" scale="{disp_scale}" xChannelSelector="R" yChannelSelector="A" result="{out_name}_xzr"/>')
    p.append(f'<feOffset in="{out_name}_xzr" dx="0" dy="0" x="0" y="0" width="{dw}" height="{sh}" result="{out_name}_xz"/>')
    p.append(f'<feTile in="{out_name}_xz" x="0" y="0" width="{dw}" height="{ys_h}" result="{out_name}_xzt"/>')
    p.append(f'<feOffset in="{out_name}_xzt" dx="{a_s}" dy="0" x="0" y="0" width="{yz_w}" height="{ys_h}" result="{out_name}_ys"/>')
    p.append(f'<feDisplacementMap in="{out_name}_ys" in2="{out_name}_lyt" x="0" y="0" width="{yz_w}" height="{ys_h}" scale="{disp_scale}" xChannelSelector="A" yChannelSelector="R" result="{out_name}_yzr"/>')
    p.append(f'<feOffset in="{out_name}_yzr" dx="{dx}" dy="{dy}" x="{dx}" y="{dy}" width="{dw}" height="{dh}" result="{out_name}"/>')


def _mux_higher_score(p: List[str], a_in: str, b_in: str, x: int, y: int, w: int, h: int, out: str) -> None:
    """7-primitive exact 3-channel `(R=score, G=u2, B=pt2)` argmax mux with `A == 1.0` on every node!"""
    p.append(f'<feColorMatrix in="{a_in}" x="{x}" y="{y}" width="{w}" height="{h}" type="matrix" values="-1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{out}_ia"/>')
    p.append(f'<feComposite in="{b_in}" in2="{out}_ia" x="{x}" y="{y}" width="{w}" height="{h}" operator="arithmetic" k1="0" k2="255" k3="255" k4="-255" result="{out}_df"/>')
    p.append(f'<feColorMatrix in="{out}_df" x="{x}" y="{y}" width="{w}" height="{h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="{out}_mb"/>')
    p.append(f'<feColorMatrix in="{out}_mb" x="{x}" y="{y}" width="{w}" height="{h}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="{out}_ma"/>')
    p.append(f'<feComposite in="{b_in}" in2="{out}_mb" x="{x}" y="{y}" width="{w}" height="{h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{out}_bm"/>')
    p.append(f'<feComposite in="{a_in}" in2="{out}_ma" x="{x}" y="{y}" width="{w}" height="{h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{out}_am"/>')
    p.append(f'<feComposite in="{out}_am" in2="{out}_bm" x="{x}" y="{y}" width="{w}" height="{h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{out}"/>')


def compile_chess_ai_filters(spec: dict, dpr: int) -> List[str]:
    """Compile all 45 linear-VRAM SVG `<filter>` stages for `dpr in (1, 2)` (`<= 74` prims/stage)."""
    assert dpr in (1, 2)
    u = 1.0 if dpr == 1 else 0.5
    sc = lambda v: v if dpr == 1 else (v // 2 if v % 2 == 0 else v * 0.5)
    a_sh = 115.0 * u
    disp_s = _fmt_num(230.0 * u)
    a_m2 = 18.0 * u
    disp_m2 = _fmt_num(36.0 * u)

    ly = spec["layers"]
    z0_f = float(spec["zero_point"])
    z0_s = _fmt_num(z0_f)
    nz0_s = _fmt_num(-z0_f)

    tbl_eq0 = "1 " + " ".join(["0"] * 255)
    tbl_eq255 = " ".join(["0"] * 255) + " 1"
    tbl_eq1 = "1 1 " + " ".join(["0"] * 254)
    tbl_ge128 = " ".join(["0"] * 128 + ["1"] * 128)
    tbl_gt128 = " ".join(["0"] * 129 + ["1"] * 127)
    # Normalize `<foreignObject>` B0/B1 `{0, 128, 255}` -> `{1/255, 128/255, 255/255}`:
    tbl_norm_root = " ".join(_ct_u8(max(1, c)) for c in range(256))
    z0_ct = _ct_u8(128)
    neg_ct = _ct_u8(1)

    # `<foreignObject>` M1/M1C encodes `128 + code` (`code in 0..9` -> `128..137`):
    def _code128_lut(active_codes: set[int], hi: str = "1", lo: str = z0_ct) -> str:
        return " ".join(hi if (c - 128) in active_codes else lo for c in range(256))

    tbl_ai_p = _code128_lut({1, 9})
    tbl_ai_n = _code128_lut({2})
    tbl_ai_b = _code128_lut({3})
    tbl_ai_r = _code128_lut({4})
    tbl_ai_q = _code128_lut({5})
    tbl_ai_k = _code128_lut({6, 7, 8})
    tbl_is_oo = _code128_lut({7}, "1", "0")
    tbl_is_ooo = _code128_lut({8}, "1", "0")
    tbl_is_ep = _code128_lut({9}, "1", "0")

    tbl_opp_p = _code128_lut({1}, neg_ct, z0_ct)
    tbl_opp_n = _code128_lut({2}, neg_ct, z0_ct)
    tbl_opp_b = _code128_lut({3}, neg_ct, z0_ct)
    tbl_opp_r = _code128_lut({4}, neg_ct, z0_ct)
    tbl_opp_q = _code128_lut({5}, neg_ct, z0_ct)
    tbl_opp_k = _code128_lut({6}, neg_ct, z0_ct)

    # `cell_sq.png` stores `128 + sq` (`128..191`) on `0..7 x 0..7` and `250` on moat:
    tbl_subxy128 = " ".join(
        _ct_u8(128 + 2 * (((c - 128) // 8) % 2) + ((c - 128) % 2)) if (128 <= c < 192) else "0"
        for c in range(256)
    )
    tbl_is_r0 = " ".join("1" if (128 <= c < 136) else "0" for c in range(256))
    tbl_is_r4 = " ".join("1" if (160 <= c < 168) else "0" for c in range(256))
    tbl_csq_valid = " ".join(_ct_u8(c) if (128 <= c < 192) else "0" for c in range(256))

    # `mvv_code128` in `al2.B` is `128 + mvv + 5*is_r0` (`128..137`):
    def _tier128_lut_for_pt(pt_src: int) -> str:
        vals: List[str] = []
        for c in range(256):
            mc = c - 128
            if mc < 0 or mc > 9:
                vals.append("0")
                continue
            is_r0 = 1 if mc >= 5 else 0
            mvv = mc - 5 if mc >= 5 else mc
            bt = compute_ply2_base_tier(pt_src, mvv, is_r0)
            vals.append(_ct_u8(128 + 4 * bt))
        return " ".join(vals)

    tbl_pt_p_placed128 = " ".join(
        _ct_u8(128 + 5) if (133 <= c <= 137) else _ct_u8(128 + 1) for c in range(256)
    )
    tbl_mvv_is_cap = " ".join("1" if (128 <= c < 138 and ((c - 128) % 5) != 0) else "0" for c in range(256))
    tbl_mvv_is_emp = " ".join("1" if (128 <= c < 138 and ((c - 128) % 5) == 0) else "0" for c in range(256))

    out_filters: List[str] = []

    comp_w, comp_h = sc(STAGE_W), sc(COMP_H)

    def _wrap_filter(fid: str, body: List[str], w: int | float = comp_w, h: int | float = comp_h) -> str:
        return _wrap_filter_raw(fid, body, w=w, h=h)

    s1_w, s1_h = sc(640), sc(10)
    b10 = sc(10)
    b6 = sc(6)
    s1_b0x, s1_b0y = sc(VRAM_S1_B0[0]), sc(VRAM_S1_B0[1])
    s1_b1x, s1_b1y = sc(VRAM_S1_B1[0]), sc(VRAM_S1_B1[1])
    s1_kx, s1_ky = sc(VRAM_S1_KATT[0]), sc(VRAM_S1_KATT[1])
    t_pinx, t_piny = sc(TMP_S1_PIN[0]), sc(TMP_S1_PIN[1])
    t_pin2x, t_pin2y = sc(TMP_S1_PIN2[0]), sc(TMP_S1_PIN2[1])
    t_pnkx, t_pnky = sc(TMP_S1_PNK[0]), sc(TMP_S1_PNK[1])
    t_orx, t_ory = sc(TMP_S1_ORTH[0]), sc(TMP_S1_ORTH[1])

    fo_ch = [(sc(FO_X + cx), sc(FO_Y + cy)) for cx, cy in SRC_CH_PHYS]
    fo_m1ux, fo_m1uy = sc(FO_X + SRC_M1U_PHYS[0]), sc(FO_Y + SRC_M1U_PHYS[1])
    fo_m1vx, fo_m1vy = sc(FO_X + SRC_M1V_PHYS[0]), sc(FO_Y + SRC_M1V_PHYS[1])
    fo_m1kx, fo_m1ky = sc(FO_X + SRC_M1K_PHYS[0]), sc(FO_Y + SRC_M1K_PHYS[1])
    fo_m1cux, fo_m1cuy = sc(FO_X + SRC_M1CU_PHYS[0]), sc(FO_Y + SRC_M1CU_PHYS[1])
    fo_m1cvx, fo_m1cvy = sc(FO_X + SRC_M1CV_PHYS[0]), sc(FO_Y + SRC_M1CV_PHYS[1])
    fo_m1ccx, fo_m1ccy = sc(FO_X + SRC_M1CC_PHYS[0]), sc(FO_Y + SRC_M1CC_PHYS[1])
    fo_mt_tx, fo_mt_ty = sc(FO_X + SRC_META_TURN_PHYS[0]), sc(FO_Y + SRC_META_TURN_PHYS[1])
    fo_mt_mx, fo_mt_my = sc(FO_X + SRC_META_TERM_PHYS[0]), sc(FO_Y + SRC_META_TERM_PHYS[1])

    csq_x, csq_y = sc(LUT_CELLSQ_POS[0]), sc(LUT_CELLSQ_POS[1])
    bm10_x, bm10_y = sc(LUT_BMASK10_POS[0]), sc(LUT_BMASK10_POS[1])
    bm6_x, bm6_y = sc(LUT_BMASK6_POS[0]), sc(LUT_BMASK6_POS[1])

    def _emit_packed_ray(p: List[str], pack_in: str, dx_p: int, dy_p: int, wx: int, wy: int, tag: str) -> str:
        sx1, sy1 = sc(dx_p), sc(dy_p)
        p.append(f'<feOffset in="{pack_in}" dx="{sx1}" dy="{sy1}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" result="{tag}_1"/>')
        for mul in (1, 2):
            cur, nxt = f"{tag}_{mul}", f"{tag}_{2 * mul}"
            p.append(f'<feOffset in="{cur}" dx="{mul * sx1}" dy="{mul * sy1}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" result="{cur}s"/>')
            p.append(f'<feColorMatrix in="{cur}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="{cur}e"/>')
            p.append(f'<feComposite in="{cur}e" in2="{cur}s" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{cur}p"/>')
            p.append(f'<feColorMatrix in="{cur}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{cur}rg"/>')
            p.append(f'<feComposite in="{cur}rg" in2="{cur}p" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{nxt}"/>')
        c4 = f"{tag}_4"
        p.append(f'<feOffset in="{c4}" dx="{4 * sx1}" dy="{4 * sy1}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" result="{c4}s"/>')
        p.append(f'<feColorMatrix in="{c4}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="{c4}e"/>')
        p.append(f'<feComposite in="{c4}e" in2="{c4}s" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{c4}p"/>')
        p.append(f'<feComposite in="{c4}" in2="{c4}p" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{tag}"/>')
        return tag

    def _emit_1ch_slider_ray(p: List[str], pack_in: str, dx_p: int, dy_p: int, wx: int, wy: int, tag: str) -> str:
        sx1, sy1 = sc(dx_p), sc(dy_p)
        p.append(f'<feOffset in="{pack_in}" dx="{sx1}" dy="{sy1}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" result="{tag}_1"/>')
        for mul in (1, 2):
            cur, nxt = f"{tag}_{mul}", f"{tag}_{2 * mul}"
            p.append(f'<feOffset in="{cur}" dx="{mul * sx1}" dy="{mul * sy1}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" result="{cur}s"/>')
            p.append(f'<feColorMatrix in="{cur}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="{cur}e"/>')
            p.append(f'<feComposite in="{cur}e" in2="{cur}s" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{cur}p"/>')
            p.append(f'<feColorMatrix in="{cur}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{cur}r"/>')
            p.append(f'<feComposite in="{cur}r" in2="{cur}p" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{nxt}"/>')
        c4 = f"{tag}_4"
        p.append(f'<feOffset in="{c4}" dx="{4 * sx1}" dy="{4 * sy1}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" result="{c4}s"/>')
        p.append(f'<feColorMatrix in="{c4}" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{c4}e"/>')
        p.append(f'<feComposite in="{c4}e" in2="{c4}s" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{c4}p"/>')
        p.append(f'<feComposite in="{c4}" in2="{c4}p" x="{wx}" y="{wy}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{tag}"/>')
        return tag

    # =========================================================================
    # Stage `s1a`: Ply-1 Post-Move Board Strip (`VRAM_S1_B0`, `VRAM_S1_B1`)
    # =========================================================================
    p1a: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{s1_b0x - csq_x}" dy="{s1_b0y - csq_y}" x="{s1_b0x}" y="{s1_b0y}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="csq"/>',
    ]
    for ch_i, (ch_fx, ch_fy) in enumerate(fo_ch):
        p1a.append(
            f'<feOffset in="SourceGraphic" dx="{s1_b0x - ch_fx}" dy="{s1_b0y - ch_fy}" x="{s1_b0x}" y="{s1_b0y}" width="{b10}" height="{b10}" result="rch_{ch_i}"/>'
        )
    _pack_rgb(p1a, "rch_0", "rch_1", "rch_2", s1_b0x, s1_b0y, b10, b10, "rb0_0")
    _pack_rgb(p1a, "rch_3", "rch_4", "rch_5", s1_b0x, s1_b0y, b10, b10, "rb1_0")
    p1a.extend([
        f'<feComponentTransfer in="rb0_0" x="{s1_b0x}" y="{s1_b0y}" width="{b10}" height="{b10}" result="rb0_1"><feFuncR type="table" tableValues="{tbl_norm_root}"/><feFuncG type="table" tableValues="{tbl_norm_root}"/><feFuncB type="table" tableValues="{tbl_norm_root}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feTile in="rb0_1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="rb0"/>',
        f'<feComponentTransfer in="rb1_0" x="{s1_b0x}" y="{s1_b0y}" width="{b10}" height="{b10}" result="rb1_1"><feFuncR type="table" tableValues="{tbl_norm_root}"/><feFuncG type="table" tableValues="{tbl_norm_root}"/><feFuncB type="table" tableValues="{tbl_norm_root}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feTile in="rb1_1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="rb1"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_b0x - fo_m1ux}" dy="{s1_b0y - fo_m1uy}" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{sc(2)}" result="m1u_0"/>',
        f'<feTile in="m1u_0" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="u1_rgb"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_b0x - fo_m1vx}" dy="{s1_b0y - fo_m1vy}" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{sc(2)}" result="m1v_0"/>',
        f'<feTile in="m1v_0" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="v1_rgb"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_b0x - fo_m1kx}" dy="{s1_b0y - fo_m1ky}" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{sc(2)}" result="m1k_0"/>',
        f'<feTile in="m1k_0" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="c1_rgb"/>',
        f'<feBlend in="csq" in2="u1_rgb" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" mode="difference" result="du1"/>',
        f'<feComponentTransfer in="du1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="is_u1"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feBlend in="csq" in2="v1_rgb" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" mode="difference" result="dv1"/>',
        f'<feComponentTransfer in="dv1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="is_v1"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="c1_rgb" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="sp_flags"><feFuncR type="table" tableValues="{tbl_is_oo}"/><feFuncG type="table" tableValues="{tbl_is_ooo}"/><feFuncB type="table" tableValues="{tbl_is_ep}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="is_v1" dx="0" dy="{-sc(1)}" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="v1_dn1"/>',
        f'<feColorMatrix in="sp_flags" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="is_ep_m"/>',
        f'<feComposite in="v1_dn1" in2="is_ep_m" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="clr_ep"/>',
        f'<feComponentTransfer in="csq" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="csq_c70"><feFuncR type="table" tableValues="{_code128_lut({7}, "1", "0")}"/><feFuncG type="table" tableValues="{_code128_lut({0}, "1", "0")}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="csq_c70" in2="sp_flags" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rk_clr_rg"/>',
        f'<feColorMatrix in="rk_clr_rg" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 1" result="clr_rk"/>',
        f'<feComponentTransfer in="csq" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="csq_c53"><feFuncR type="table" tableValues="{_code128_lut({5}, "1", "0")}"/><feFuncG type="table" tableValues="{_code128_lut({3}, "1", "0")}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="csq_c53" in2="sp_flags" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rk_set_rg"/>',
        f'<feColorMatrix in="rk_set_rg" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="set_rk_r"/>',
        # Disjoint masks (`A == 1.0` everywhere!): `clr_nonv = is_u1 + clr_ep + clr_rk`, `all_mod = clr_nonv + is_v1`, `keep_m = 1 - all_mod`
        f'<feComposite in="clr_ep" in2="clr_rk" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="clr_sp"/>',
        f'<feComposite in="is_u1" in2="clr_sp" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="clr_nonv"/>',
        f'<feComposite in="clr_nonv" in2="is_v1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="all_mod"/>',
        f'<feColorMatrix in="all_mod" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="keep_m"/>',
        f'<feComponentTransfer in="c1_rgb" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="pl_b0"><feFuncR type="table" tableValues="{tbl_ai_p}"/><feFuncG type="table" tableValues="{tbl_ai_n}"/><feFuncB type="table" tableValues="{tbl_ai_b}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="c1_rgb" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" result="pl_b1"><feFuncR type="table" tableValues="{tbl_ai_r}"/><feFuncG type="table" tableValues="{tbl_ai_q}"/><feFuncB type="table" tableValues="{tbl_ai_k}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="rb0" in2="keep_m" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b0_kp"/>',
        f'<feComposite in="b0_kp" in2="clr_nonv" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="{z0_s}" k4="0" result="b0_cl"/>',
        f'<feComposite in="pl_b0" in2="is_v1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b0_pl"/>',
        f'<feComposite in="b0_cl" in2="b0_pl" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="s1_b0"/>',
        f'<feComposite in="rb1" in2="keep_m" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b1_kp"/>',
        f'<feComposite in="b1_kp" in2="clr_nonv" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="{z0_s}" k4="0" result="b1_cl"/>',
        f'<feComposite in="pl_b1" in2="is_v1" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b1_pl"/>',
        f'<feComposite in="b1_cl" in2="b1_pl" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="b1_t"/>',
        f'<feComposite in="b1_t" in2="set_rk_r" x="{s1_b0x}" y="{s1_b0y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="{_fmt_num(1.0 - z0_f)}" k4="0" result="b1_u"/>',
        f'<feOffset in="b1_u" dx="0" dy="{s1_b1y - s1_b0y}" x="{s1_b1x}" y="{s1_b1y}" width="{s1_w}" height="{s1_h}" result="s1_b1"/>',
    ])
    _emit_passthrough(p1a, ["s1_b0", "s1_b1"], keep_vram=False, dpr=dpr)
    out_filters.append(_wrap_filter(f"ai-s1a-{dpr}x", p1a))

    # =========================================================================
    # Stage `s1b`: Non-Slider AI Attacks + Orth `ONK` Rays (`d=0,1`) -> `VRAM_S1_KATT`
    # =========================================================================
    p1b: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{s1_kx - bm10_x}" dy="{s1_ky - bm10_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - s1_b0x}" dy="{s1_ky - s1_b0y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - s1_b1x}" dy="{s1_ky - s1_b1y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ab0r"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ab1r"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ob0r"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ob1r"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab0r" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ai_b0"/>',
        f'<feComposite in="ab1r" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ai_b1"/>',
        f'<feComposite in="ob0r" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="opp_b0"/>',
        f'<feComposite in="ob1r" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="opp_b1"/>',
        f'<feColorMatrix in="ai_b0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="a0s"/>',
        f'<feColorMatrix in="ai_b1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="a1s"/>',
        f'<feComposite in="a0s" in2="a1s" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ai_any"/>',
        f'<feColorMatrix in="opp_b0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="o0s"/>',
        f'<feColorMatrix in="opp_b1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="o1s"/>',
        f'<feComposite in="o0s" in2="o1s" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="opp_any"/>',
        f'<feComposite in="ai_any" in2="opp_any" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="occ_any"/>',
        f'<feColorMatrix in="occ_any" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="emp_raw"/>',
        f'<feComposite in="emp_raw" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="emp"/>',
        f'<feColorMatrix in="opp_b1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="opp_k"/>',
        f'<feComposite in="emp" in2="opp_k" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="onk"/>',
        f'<feOffset in="ai_b0" dx="{sc(1)}" dy="{sc(1)}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ap_r"/>',
        f'<feOffset in="ai_b0" dx="{-sc(1)}" dy="{sc(1)}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ap_l"/>',
        f'<feComposite in="ap_r" in2="ap_l" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ap_att"/>',
        f'<feConvolveMatrix in="ai_b0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" order="5 5" targetX="2" targetY="2" divisor="1" bias="0" kernelMatrix="0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0" edgeMode="none" preserveAlpha="true" result="an_att"/>',
        f'<feConvolveMatrix in="ai_b1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" order="3 3" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1 0 1 1 1 1" edgeMode="none" preserveAlpha="true" result="ak_att"/>',
        f'<feColorMatrix in="ap_att" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="apr"/>',
        f'<feColorMatrix in="an_att" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="anr"/>',
        f'<feColorMatrix in="ak_att" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="akr"/>',
        f'<feComposite in="apr" in2="anr" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="apn"/>',
        f'<feComposite in="apn" in2="akr" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="att0"/>',
        f'<feColorMatrix in="ai_b1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ai_rq_q"/>',
        f'<feColorMatrix in="ai_b0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ai_b_g"/>',
        f'<feComposite in="ai_rq_q" in2="ai_b_g" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ai_sl_rg"/>',
        f'<feColorMatrix in="onk" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="onk_b"/>',
        f'<feComposite in="ai_sl_rg" in2="onk_b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_onk"/>',
    ]
    r_d0 = _emit_packed_ray(p1b, "sl_onk", 1, 0, s1_kx, s1_ky, "ka0")
    p1b.append(f'<feComposite in="att0" in2="{r_d0}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="att_p1"/>')
    _pack_rgb(p1b, "att_p1", "onk", "emp", s1_kx, s1_ky, s1_w, s1_h, "katt_s1b")
    _emit_passthrough(p1b, ["katt_s1b"])
    out_filters.append(_wrap_filter(f"ai-s1b-{dpr}x", p1b))

    # =========================================================================
    # Stage `s1b2`: `ONK` Slider Attack Rays (`d=1,2,3`) -> `VRAM_S1_KATT`
    # =========================================================================
    def _emit_onk_setup(p: List[str]) -> None:
        p.extend([
            f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="prev_k"/>',
            f'<feOffset in="SourceGraphic" dx="{s1_kx - bm10_x}" dy="{s1_ky - bm10_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="bm0"/>',
            f'<feTile in="bm0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
            f'<feOffset in="SourceGraphic" dx="{s1_kx - s1_b0x}" dy="{s1_ky - s1_b0y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
            f'<feOffset in="SourceGraphic" dx="{s1_kx - s1_b1x}" dy="{s1_ky - s1_b1y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
            f'<feComponentTransfer in="sb0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ab00"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComponentTransfer in="sb1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComposite in="ab00" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab0"/>',
            f'<feComposite in="ab10" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
            f'<feColorMatrix in="ab1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_q"/>',
            f'<feColorMatrix in="ab0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="b_g"/>',
            f'<feColorMatrix in="prev_k" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="onk_b"/>',
            f'<feComposite in="rq_q" in2="b_g" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_rg"/>',
            f'<feComposite in="sl_rg" in2="onk_b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_onk"/>',
        ])

    p1b2: List[str] = []
    _emit_onk_setup(p1b2)
    r_d1 = _emit_packed_ray(p1b2, "sl_onk", -1, 0, s1_kx, s1_ky, "ka1")
    r_d2 = _emit_packed_ray(p1b2, "sl_onk", 0, 1, s1_kx, s1_ky, "ka2")
    r_d3 = _emit_packed_ray(p1b2, "sl_onk", 0, -1, s1_kx, s1_ky, "ka3")
    p1b2.extend([
        f'<feComposite in="{r_d1}" in2="{r_d2}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ka12"/>',
        f'<feComposite in="ka12" in2="{r_d3}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ka13"/>',
        f'<feColorMatrix in="ka13" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ka13r"/>',
        f'<feComposite in="prev_k" in2="ka13r" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="katt_s1b2"/>',
    ])
    _emit_passthrough(p1b2, ["katt_s1b2"])
    out_filters.append(_wrap_filter(f"ai-s1b2-{dpr}x", p1b2))

    # =========================================================================
    # Stage `s1c`: `ONK` Diag Rays (`d=4,5,6`) -> `VRAM_S1_KATT`
    # =========================================================================
    p1c: List[str] = []
    _emit_onk_setup(p1c)
    r_d4 = _emit_packed_ray(p1c, "sl_onk", 1, 1, s1_kx, s1_ky, "ka4")
    r_d5 = _emit_packed_ray(p1c, "sl_onk", -1, -1, s1_kx, s1_ky, "ka5")
    r_d6 = _emit_packed_ray(p1c, "sl_onk", 1, -1, s1_kx, s1_ky, "ka6")
    p1c.extend([
        f'<feComposite in="{r_d4}" in2="{r_d5}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ka45"/>',
        f'<feComposite in="ka45" in2="{r_d6}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ka46"/>',
        f'<feColorMatrix in="ka46" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ka46r"/>',
        f'<feComposite in="prev_k" in2="ka46r" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="katt_s1c"/>',
    ])
    _emit_passthrough(p1c, ["katt_s1c"])
    out_filters.append(_wrap_filter(f"ai-s1c-{dpr}x", p1c))

    # =========================================================================
    # Stage `s1c2`: Final `ONK` Diag Ray (`d=7`) -> `VRAM_S1_KATT`
    # =========================================================================
    p1c2: List[str] = []
    _emit_onk_setup(p1c2)
    r_d7 = _emit_packed_ray(p1c2, "sl_onk", -1, 1, s1_kx, s1_ky, "ka7")
    p1c2.extend([
        f'<feColorMatrix in="{r_d7}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ka7r"/>',
        f'<feComposite in="prev_k" in2="ka7r" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="katt_fin"/>',
    ])
    _emit_passthrough(p1c2, ["katt_fin"])
    out_filters.append(_wrap_filter(f"ai-s1c2-{dpr}x", p1c2))

    # =========================================================================
    # Stage `s1d`: Axis `H` Pin/Check Scan -> `TMP_S1_PIN`
    # =========================================================================
    p1d: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_pinx - bm10_x}" dy="{t_piny - bm10_y}" x="{t_pinx}" y="{t_piny}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - s1_kx}" dy="{t_piny - s1_ky}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="pk"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - s1_b0x}" dy="{t_piny - s1_b0y}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - s1_b1x}" dy="{t_piny - s1_b1y}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="ab00"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab00" in2="bmask" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab0"/>',
        f'<feComposite in="ab10" in2="bmask" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
        f'<feColorMatrix in="ab1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="k_rq"/>',
        f'<feColorMatrix in="pk" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="empb"/>',
        f'<feComposite in="k_rq" in2="empb" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="krq_e"/>',
    ]
    rh_p = _emit_packed_ray(p1d, "krq_e", 1, 0, t_pinx, t_piny, "hp")
    rh_n = _emit_packed_ray(p1d, "krq_e", -1, 0, t_pinx, t_piny, "hn")
    p1d.extend([
        f'<feColorMatrix in="{rh_n}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="hnsw"/>',
        f'<feComposite in="{rh_p}" in2="hnsw" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="h_pr"/>',
        f'<feComposite in="{rh_p}" in2="{rh_n}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="h_sum"/>',
        f'<feColorMatrix in="h_pr" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="btwh"/>',
        f'<feColorMatrix in="h_sum" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="h_krg"/>',
        f'<feComposite in="h_krg" in2="k_rq" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="chkh_g0"/>',
        f'<feConvolveMatrix in="ab1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" order="5 5" targetX="2" targetY="2" divisor="1" bias="0" kernelMatrix="0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0" edgeMode="none" preserveAlpha="true" result="kn_k"/>',
        f'<feOffset in="ab1" dx="{sc(1)}" dy="{-sc(1)}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="pk_1"/>',
        f'<feOffset in="ab1" dx="{-sc(1)}" dy="{-sc(1)}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="pk_2"/>',
        f'<feComposite in="pk_1" in2="pk_2" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pw_k"/>',
        f'<feColorMatrix in="pw_k" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="pwk_r"/>',
        f'<feColorMatrix in="kn_k" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="knk_g"/>',
        f'<feComposite in="pwk_r" in2="knk_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pkn_rg"/>',
        f'<feComposite in="ab0" in2="pkn_rg" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="chk_pn"/>',
        f'<feColorMatrix in="chk_pn" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ckpn_g"/>',
        f'<feComposite in="chkh_g0" in2="ckpn_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="chkh_g"/>',
        f'<feComposite in="btwh" in2="chkh_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pin_h"/>',
    ])
    _emit_passthrough(p1d, ["pin_h"])
    out_filters.append(_wrap_filter(f"ai-s1d-{dpr}x", p1d))

    # =========================================================================
    # Stage `s1e`: Axis `V` Pin/Check Scan -> `TMP_S1_PIN2` (`R=btw_V, B=chkv`)
    # =========================================================================
    p1e: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - bm10_x}" dy="{t_pin2y - bm10_y}" x="{t_pin2x}" y="{t_pin2y}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - s1_kx}" dy="{t_pin2y - s1_ky}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="pk"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - s1_b1x}" dy="{t_pin2y - s1_b1y}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb1" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab10" in2="bmask" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
        f'<feColorMatrix in="pk" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="empb"/>',
        f'<feColorMatrix in="ab1" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="k_rq"/>',
        f'<feComposite in="k_rq" in2="empb" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="krq_e"/>',
    ]
    rv_p = _emit_packed_ray(p1e, "krq_e", 0, 1, t_pin2x, t_pin2y, "vp")
    rv_n = _emit_packed_ray(p1e, "krq_e", 0, -1, t_pin2x, t_pin2y, "vn")
    p1e.extend([
        f'<feColorMatrix in="{rv_n}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="vnsw"/>',
        f'<feComposite in="{rv_p}" in2="vnsw" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="v_pr"/>',
        f'<feComposite in="{rv_p}" in2="{rv_n}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="v_sum"/>',
        f'<feColorMatrix in="v_pr" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="btwv_r"/>',
        f'<feColorMatrix in="v_sum" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="v_krg"/>',
        f'<feComposite in="v_krg" in2="k_rq" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="chkv_g"/>',
        f'<feColorMatrix in="chkv_g" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="chkv_b"/>',
        f'<feComposite in="btwv_r" in2="chkv_b" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pin_v"/>',
    ])
    _emit_passthrough(p1e, ["pin_v"])
    out_filters.append(_wrap_filter(f"ai-s1e-{dpr}x", p1e))

    # =========================================================================
    # Stage `s1e2`: Axis `/` (`D1`) Pin/Check Scan -> `TMP_S1_PIN2` (`R=btw_V, G=btw_D1, B=chk_V+D1`)
    # =========================================================================
    p1e2_d1: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - bm10_x}" dy="{t_pin2y - bm10_y}" x="{t_pin2x}" y="{t_pin2y}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="pv"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - s1_kx}" dy="{t_pin2y - s1_ky}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="pk"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - s1_b0x}" dy="{t_pin2y - s1_b0y}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pin2x - s1_b1x}" dy="{t_pin2y - s1_b1y}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="ab00"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="ab10"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab00" in2="bmask" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab0"/>',
        f'<feComposite in="ab10" in2="bmask" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
        f'<feColorMatrix in="pk" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="empb"/>',
        f'<feColorMatrix in="ab1" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="k_q"/>',
        f'<feColorMatrix in="ab0" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="b_g"/>',
        f'<feComposite in="k_q" in2="b_g" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="k_bq"/>',
        f'<feComposite in="k_bq" in2="empb" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="kbq_e"/>',
    ]
    rd1_p = _emit_packed_ray(p1e2_d1, "kbq_e", 1, 1, t_pin2x, t_pin2y, "d1p")
    rd1_n = _emit_packed_ray(p1e2_d1, "kbq_e", -1, -1, t_pin2x, t_pin2y, "d1n")
    p1e2_d1.extend([
        f'<feColorMatrix in="{rd1_n}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="d1nsw"/>',
        f'<feComposite in="{rd1_p}" in2="d1nsw" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="d1_pr"/>',
        f'<feComposite in="{rd1_p}" in2="{rd1_n}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="d1_sum"/>',
        f'<feColorMatrix in="d1_pr" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="btwd1_g"/>',
        f'<feColorMatrix in="d1_sum" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="d1_krg"/>',
        f'<feComposite in="d1_krg" in2="k_bq" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="chkd1_g"/>',
        f'<feColorMatrix in="chkd1_g" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="chkd1_b"/>',
        f'<feComposite in="btwd1_g" in2="chkd1_b" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="d1_gb"/>',
        f'<feComposite in="pv" in2="d1_gb" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pin_vd1"/>',
    ])
    _emit_passthrough(p1e2_d1, ["pin_vd1"])
    out_filters.append(_wrap_filter(f"ai-s1e2-{dpr}x", p1e2_d1))

    # =========================================================================
    # Stage `s1e3`: Axis `\` (`D2`) + Checker Count (`{0, 128, 255}`) Broadcast
    # =========================================================================
    tbl_eq128 = " ".join("1" if c == 128 else "0" for c in range(256))
    p1e2: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_pinx - bm10_x}" dy="{t_piny - bm10_y}" x="{t_pinx}" y="{t_piny}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - s1_kx}" dy="{t_piny - s1_ky}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="pk"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="ph"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - t_pin2x}" dy="{t_piny - t_pin2y}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="pvd1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - s1_b0x}" dy="{t_piny - s1_b0y}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pinx - s1_b1x}" dy="{t_piny - s1_b1y}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="ab00"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="ab10"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab00" in2="bmask" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab0"/>',
        f'<feComposite in="ab10" in2="bmask" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
        f'<feColorMatrix in="pk" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="empb"/>',
        f'<feColorMatrix in="ab1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="k_q"/>',
        f'<feColorMatrix in="ab0" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="b_g"/>',
        f'<feComposite in="k_q" in2="b_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="k_bq"/>',
        f'<feComposite in="k_bq" in2="empb" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="kbq_e"/>',
    ]
    rd2_p = _emit_packed_ray(p1e2, "kbq_e", 1, -1, t_pinx, t_piny, "d2p")
    rd2_n = _emit_packed_ray(p1e2, "kbq_e", -1, 1, t_pinx, t_piny, "d2n")
    p1e2.extend([
        f'<feColorMatrix in="{rd2_n}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="d2nsw"/>',
        f'<feComposite in="{rd2_p}" in2="d2nsw" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="d2_pr"/>',
        f'<feComposite in="{rd2_p}" in2="{rd2_n}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="d2_sum"/>',
        f'<feColorMatrix in="d2_pr" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="btwd2_r"/>',
        f'<feColorMatrix in="d2_sum" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="d2_krg"/>',
        f'<feComposite in="d2_krg" in2="k_bq" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="chkd2_g"/>',
        f'<feColorMatrix in="ph" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ck_hr"/>',
        f'<feColorMatrix in="pvd1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ck_vdr"/>',
        f'<feColorMatrix in="chkd2_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ck_d2r"/>',
        f'<feComposite in="ck_hr" in2="ck_vdr" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ck_a"/>',
        f'<feComposite in="ck_a" in2="ck_d2r" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="chk_src"/>',
        f'<feColorMatrix in="ph" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bh_r"/>',
        f'<feColorMatrix in="pvd1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bv_g"/>',
        f'<feComposite in="bh_r" in2="bv_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="out_pin1"/>',
        f'<feColorMatrix in="pvd1" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bd1_r"/>',
        f'<feColorMatrix in="btwd2_r" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bd2_g"/>',
        f'<feColorMatrix in="chk_src" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="cks_b"/>',
        f'<feComposite in="bd1_r" in2="bd2_g" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bd12"/>',
        f'<feComposite in="bd12" in2="cks_b" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="out_pin2_0"/>',
        f'<feOffset in="out_pin2_0" dx="{t_pin2x - t_pinx}" dy="{t_pin2y - t_piny}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="out_pin2"/>',
    ])
    _emit_passthrough(p1e2, ["out_pin1", "out_pin2"])
    out_filters.append(_wrap_filter(f"ai-s1e3-{dpr}x", p1e2))

    # =========================================================================
    # Stage `s1f`: Finalize `ALLOW_AX`, `EVADE_OK`, `128 + MVV_CODE`, `IN_CHECK`
    # =========================================================================
    tbl_cnt_0 = "1 " + " ".join(["0"] * 255)
    tbl_cnt_1 = " ".join("1" if (64 <= c < 192) else "0" for c in range(256))

    p1f: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - bm10_x}" dy="{t_pnky - bm10_y}" x="{t_pnkx}" y="{t_pnky}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - csq_x}" dy="{t_pnky - csq_y}" x="{t_pnkx}" y="{t_pnky}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - s1_kx}" dy="{t_pnky - s1_ky}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pk"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - t_pinx}" dy="{t_pnky - t_piny}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pin1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - t_pin2x}" dy="{t_pnky - t_pin2y}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pin2"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - s1_b0x}" dy="{t_pnky - s1_b0y}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - s1_b1x}" dy="{t_pnky - s1_b1y}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="ab00"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="ob00"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="ob10"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab00" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab0"/>',
        f'<feComposite in="ab10" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
        f'<feComposite in="ob00" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob0"/>',
        f'<feComposite in="ob10" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob1"/>',
        f'<feColorMatrix in="pin1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="p1s"/>',
        f'<feColorMatrix in="pin2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1" result="p2s"/>',
        f'<feComposite in="p1s" in2="p2s" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="btw_all"/>',
        # Count checkers (`pin2.B == chk_src`) per 8x8 board (`divisor="2"` -> 0, 128, 255 in channel B) & broadcast across cell:
        f'<feConvolveMatrix in="pin2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" order="2 2" targetX="0" targetY="0" divisor="2" bias="0" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="cs_s4"/>',
        f'<feConvolveMatrix in="cs_s4" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" order="3 3" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 1 0 0 0 1 0 1" edgeMode="none" preserveAlpha="true" result="cs_s2"/>',
        f'<feConvolveMatrix in="cs_s2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" order="5 5" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="cs_s1"/>',
        f'<feComponentTransfer in="csq" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="is_00"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="cs_s1" in2="is_00" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cnt00"/>',
        f'<feConvolveMatrix in="cnt00" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" order="4 4" targetX="3" targetY="3" divisor="1" bias="0" kernelMatrix="{" ".join(["1"] * 16)}" edgeMode="none" preserveAlpha="true" result="cnt44"/>',
        f'<feConvolveMatrix in="cnt44" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" order="5 5" targetX="4" targetY="4" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="cnt_cell"/>',
        f'<feComponentTransfer in="cnt_cell" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="cnt_f"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_cnt_0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="cnt_cell" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="cnt_1"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_cnt_1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="btw_all" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="btwb"/>',
        f'<feComposite in="pk" in2="btwb" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="emp_btw"/>',
        f'<feComposite in="pin2" in2="emp_btw" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ev1_raw"/>',
        f'<feComposite in="ev1_raw" in2="cnt_1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ev_c1"/>',
        f'<feComposite in="bmask" in2="cnt_f" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ev_c0"/>',
        f'<feComposite in="ev_c0" in2="ev_c1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ev_b"/>',
        f'<feColorMatrix in="ob0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="oo0"/>',
        f'<feColorMatrix in="ob1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="oo1"/>',
        f'<feComposite in="oo0" in2="oo1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="opp_s"/>',
        f'<feColorMatrix in="opp_s" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="no_opp0"/>',
        f'<feComposite in="no_opp0" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="not_opp"/>',
        f'<feComposite in="ev_b" in2="not_opp" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ev_ok_b"/>',
        f'<feColorMatrix in="btw_all" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="-1 0 0 0 1 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 1" result="unp_rg"/>',
        f'<feColorMatrix in="pin1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="p1_rg"/>',
        f'<feComposite in="unp_rg" in2="p1_rg" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="al_hv"/>',
        f'<feComposite in="al_hv" in2="ev_ok_b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="al1_0"/>',
        f'<feOffset in="al1_0" dx="{t_pinx - t_pnkx}" dy="{t_piny - t_pnky}" x="{t_pinx}" y="{t_piny}" width="{s1_w}" height="{s1_h}" result="al1_out"/>',
        # `mvv_code128 = (128 + 1*P + 2*(N+B) + 3*R + 4*(Q+K) + 5*is_r0)/255` on board, `0` on moat:
        f'<feComponentTransfer in="csq" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="r04"><feFuncR type="table" tableValues="{tbl_is_r0}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="ab0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 {_fmt_num(1/255)} {_fmt_num(2/255)} {_fmt_num(2/255)} 0 0 0 0 0 0 1" result="mv0"/>',
        f'<feColorMatrix in="ab1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 {_fmt_num(3/255)} {_fmt_num(4/255)} {_fmt_num(4/255)} 0 0 0 0 0 0 1" result="mv1"/>',
        f'<feColorMatrix in="r04" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 {_fmt_num(5/255)} 0 0 0 {z0_s} 0 0 0 0 1" result="mvr0"/>',
        f'<feColorMatrix in="opp_s" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 {_fmt_num(10/255)} 0 0 0 0 0 0 0 0 1" result="opp_10"/>',
        f'<feComposite in="mv0" in2="mv1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="mv01"/>',
        f'<feComposite in="mv01" in2="mvr0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="mv_b00"/>',
        f'<feComposite in="mv_b00" in2="opp_10" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="mv_b0"/>',
        f'<feComposite in="mv_b0" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="mv_b"/>',
        f'<feColorMatrix in="pin2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="p2_rg"/>',
        f'<feComposite in="unp_rg" in2="p2_rg" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="al_d12"/>',
        f'<feComposite in="al_d12" in2="mv_b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="al2_0"/>',
        f'<feOffset in="al2_0" dx="{t_pin2x - t_pnkx}" dy="{t_pin2y - t_pnky}" x="{t_pin2x}" y="{t_pin2y}" width="{s1_w}" height="{s1_h}" result="al2_out"/>',
        f'<feColorMatrix in="pk" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="kat_r"/>',
        f'<feColorMatrix in="not_opp" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="nop_g"/>',
        f'<feColorMatrix in="cnt_f" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 -1 0 1 0 0 0 0 1" result="ichk_b"/>',
        f'<feComposite in="kat_r" in2="nop_g" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="kn_rg"/>',
        f'<feComposite in="kn_rg" in2="ichk_b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ks_0"/>',
        f'<feOffset in="ks_0" dx="{s1_kx - t_pnkx}" dy="{s1_ky - t_pnky}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ks_out"/>',
    ]
    _emit_passthrough(p1f, ["al1_out", "al2_out", "ks_out"])
    out_filters.append(_wrap_filter(f"ai-s1f-{dpr}x", p1f))

    # =========================================================================
    # Stage `s1f2`: Legal Opponent Pawn & Knight Moves (`128 + u2`) -> `TMP_S1_PNK`
    # =========================================================================
    p1f2: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - bm10_x}" dy="{t_pnky - bm10_y}" x="{t_pnkx}" y="{t_pnky}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - csq_x}" dy="{t_pnky - csq_y}" x="{t_pnkx}" y="{t_pnky}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feComponentTransfer in="csq" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="u2p1"><feFuncR type="table" tableValues="{tbl_csq_valid}"/><feFuncG type="table" tableValues="{tbl_is_r4}"/><feFuncB type="table" tableValues="{tbl_subxy128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - t_pinx}" dy="{t_pnky - t_piny}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="al1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - t_pin2x}" dy="{t_pnky - t_pin2y}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feOffset in="SourceGraphic" dx="{t_pnkx - s1_b0x}" dy="{t_pnky - s1_b0y}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feComponentTransfer in="sb0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="ob00"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob00" in2="bmask" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob0"/>',
        f'<feColorMatrix in="ob0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="op_rgb"/>',
        f'<feColorMatrix in="u2p1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="u2_rgb"/>',
        f'<feComposite in="op_rgb" in2="u2_rgb" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="op_u2"/>',
        f'<feComposite in="op_u2" in2="al1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="op_hv"/>',
        f'<feComposite in="op_u2" in2="al2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="op_d12"/>',
        f'<feColorMatrix in="al2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="mvc"/>',
        f'<feComponentTransfer in="mvc" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="ce0"><feFuncR type="table" tableValues="{tbl_mvv_is_cap}"/><feFuncG type="table" tableValues="{tbl_mvv_is_emp}"/><feFuncB type="table" tableValues="{tbl_mvv_is_emp}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="op_hv" dx="0" dy="{-sc(1)}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pp1_s"/>',
        f'<feOffset in="op_hv" dx="0" dy="{-sc(2)}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pp2_r"/>',
        f'<feOffset in="ce0" dx="0" dy="{-sc(1)}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="emp_y5"/>',
        f'<feComposite in="pp2_r" in2="emp_y5" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pp2_e"/>',
        f'<feComposite in="pp2_e" in2="u2p1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pp2_s"/>',
        f'<feBlend in="pp1_s" in2="pp2_s" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="pp_push"/>',
        f'<feComposite in="pp_push" in2="ce0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pp_okg"/>',
        f'<feOffset in="op_d12" dx="{sc(1)}" dy="{-sc(1)}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pc_l0"/>',
        f'<feColorMatrix in="pc_l0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="pc_lr"/>',
        f'<feOffset in="op_d12" dx="{-sc(1)}" dy="{-sc(1)}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="pc_rr"/>',
        f'<feBlend in="pc_lr" in2="pc_rr" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="pc_both"/>',
        f'<feComposite in="pc_both" in2="ce0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pc_okr"/>',
        f'<feColorMatrix in="pp_okg" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="pp_b"/>',
        f'<feColorMatrix in="pc_okr" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="pc_b"/>',
        f'<feBlend in="pp_b" in2="pc_b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="p_rawb"/>',
        f'<feComposite in="p_rawb" in2="al1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="p_u2b"/>',
        f'<feComposite in="al1" in2="al2" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="al_m"/>',
        f'<feColorMatrix in="al_m" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alm_r"/>',
        f'<feColorMatrix in="al_m" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alm_g"/>',
        f'<feComposite in="alm_r" in2="alm_g" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="unpin"/>',
        f'<feComposite in="ob0" in2="unpin" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="on_unp"/>',
        f'<feComposite in="on_unp" in2="u2_rgb" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="on_u2"/>',
    ]
    kn_cur = ""
    for k_idx, (dx, dy) in enumerate(AI_KNIGHT_DELTAS):
        nm = f"kn_o{k_idx}"
        p1f2.append(f'<feOffset in="on_u2" dx="{sc(dx)}" dy="{sc(dy)}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" result="{nm}"/>')
        if k_idx == 0:
            kn_cur = nm
        else:
            nxt = f"kn_m{k_idx}"
            p1f2.append(f'<feBlend in="{kn_cur}" in2="{nm}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="{nxt}"/>')
            kn_cur = nxt
    p1f2.extend([
        f'<feColorMatrix in="{kn_cur}" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="kn_rawb"/>',
        f'<feComposite in="kn_rawb" in2="al1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="kn_u2b"/>',
        f'<feColorMatrix in="p_u2b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="pu2_r"/>',
        f'<feColorMatrix in="kn_u2b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="knu2_g"/>',
        f'<feColorMatrix in="u2p1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="sub_b"/>',
        f'<feComposite in="pu2_r" in2="knu2_g" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_rg"/>',
        f'<feComposite in="pk_rg" in2="sub_b" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pn_out"/>',
    ])
    _emit_passthrough(p1f2, ["pn_out"])
    out_filters.append(_wrap_filter(f"ai-s1f2-{dpr}x", p1f2))

    # =========================================================================
    # Stages `s1g, s1g2, s1g3`: Opponent Sliders (`R=192+u2_B / 128+u2_R`, `G=128+u2_Q`)
    # =========================================================================
    # `_emit_packed_ray` propagates `R = Rook/Bishop (128+u2)` and `G = Queen (128+u2)` in parallel
    # so `Queen` is never shadowed by `Bishop/Rook` before `s1i` checks `ai_k_adj` Contact-Queen-Check!
    tbl_add64_ge128 = " ".join(_ct_u8(c + 64) if (128 <= c < 192) else "0" for c in range(256))
    tbl_keep_ge128 = " ".join(_ct_u8(c) if (128 <= c < 192) else "0" for c in range(256))

    p1g: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_orx - bm10_x}" dy="{t_ory - bm10_y}" x="{t_orx}" y="{t_ory}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - csq_x}" dy="{t_ory - csq_y}" x="{t_orx}" y="{t_ory}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pinx}" dy="{t_ory - t_piny}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pin2x}" dy="{t_ory - t_pin2y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - s1_b1x}" dy="{t_ory - s1_b1y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="ob10"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob10" in2="bmask" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob1"/>',
        f'<feComponentTransfer in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="emp_b"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_mvv_is_emp}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob1" in2="csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_csq"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alh_rg"/>',
        f'<feComposite in="rq_csq" in2="alh_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_h0"/>',
        f'<feComposite in="rq_h0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_h"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alv_rg"/>',
        f'<feComposite in="rq_csq" in2="alv_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_v0"/>',
        f'<feComposite in="rq_v0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_v"/>',
    ]
    or0 = _emit_packed_ray(p1g, "rq_h", 1, 0, t_orx, t_ory, "or0")
    or1 = _emit_packed_ray(p1g, "rq_h", -1, 0, t_orx, t_ory, "or1")
    or2 = _emit_packed_ray(p1g, "rq_v", 0, 1, t_orx, t_ory, "or2")
    p1g.extend([
        f'<feBlend in="{or0}" in2="{or1}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="or01"/>',
        f'<feBlend in="or01" in2="{or2}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="or012"/>',
    ])
    _emit_passthrough(p1g, ["or012"])
    out_filters.append(_wrap_filter(f"ai-s1g-{dpr}x", p1g))

    p1g2: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_orx - bm10_x}" dy="{t_ory - bm10_y}" x="{t_orx}" y="{t_ory}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - csq_x}" dy="{t_ory - csq_y}" x="{t_orx}" y="{t_ory}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pinx}" dy="{t_ory - t_piny}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pin2x}" dy="{t_ory - t_pin2y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="or_prev"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - s1_b0x}" dy="{t_ory - s1_b0y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - s1_b1x}" dy="{t_ory - s1_b1y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="ob00"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="ob10"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob00" in2="bmask" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob0"/>',
        f'<feComposite in="ob10" in2="bmask" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob1"/>',
        f'<feComponentTransfer in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="emp_b"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_mvv_is_emp}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob1" in2="csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_csq"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alv_rg"/>',
        f'<feComposite in="rq_csq" in2="alv_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_v0"/>',
        f'<feComposite in="rq_v0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_v"/>',
        f'<feComposite in="ob0" in2="csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b_csq"/>',
        f'<feColorMatrix in="b_csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bu_r"/>',
        f'<feColorMatrix in="rq_csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="qu_g"/>',
        f'<feComposite in="bu_r" in2="qu_g" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_rg"/>',
        f'<feColorMatrix in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ald1_rg"/>',
        f'<feComposite in="bq_rg" in2="ald1_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bq_d10"/>',
        f'<feComposite in="bq_d10" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_d1"/>',
    ]
    or3 = _emit_packed_ray(p1g2, "rq_v", 0, -1, t_orx, t_ory, "or3")
    dg0 = _emit_packed_ray(p1g2, "bq_d1", 1, 1, t_orx, t_ory, "dg0")
    p1g2.extend([
        f'<feBlend in="or_prev" in2="{or3}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="or_all"/>',
        f'<feBlend in="or_all" in2="{dg0}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="q_od0"/>',
        f'<feColorMatrix in="or_all" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ora_r"/>',
        f'<feColorMatrix in="{dg0}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="dg0_g"/>',
        f'<feColorMatrix in="q_od0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="q0_b"/>',
        f'<feComposite in="ora_r" in2="dg0_g" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="od_rg"/>',
        f'<feComposite in="od_rg" in2="q0_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="or_g2"/>',
    ])
    _emit_passthrough(p1g2, ["or_g2"])
    out_filters.append(_wrap_filter(f"ai-s1g2-{dpr}x", p1g2))

    p1g3: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{t_orx - csq_x}" dy="{t_ory - csq_y}" x="{t_orx}" y="{t_ory}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="og2"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pinx}" dy="{t_ory - t_piny}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pin2x}" dy="{t_ory - t_pin2y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - s1_b0x}" dy="{t_ory - s1_b0y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - s1_b1x}" dy="{t_ory - s1_b1y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="emp_b"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_mvv_is_emp}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sb0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="s0_r"/>',
        f'<feColorMatrix in="sb1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="s1_g"/>',
        f'<feComposite in="s0_r" in2="s1_g" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="s_bq"/>',
        f'<feComponentTransfer in="s_bq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="ob_rg"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob_rg" in2="csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bq_rg"/>',
        f'<feColorMatrix in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ald1_rg"/>',
        f'<feComposite in="bq_rg" in2="ald1_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bqd1_0"/>',
        f'<feComposite in="bqd1_0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_d1"/>',
        f'<feColorMatrix in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ald2_rg"/>',
        f'<feComposite in="bq_rg" in2="ald2_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bqd2_0"/>',
        f'<feComposite in="bqd2_0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_d2"/>',
    ]
    dg1 = _emit_packed_ray(p1g3, "bq_d1", -1, -1, t_orx, t_ory, "dg1")
    dg2 = _emit_packed_ray(p1g3, "bq_d2", 1, -1, t_orx, t_ory, "dg2")
    dg3 = _emit_packed_ray(p1g3, "bq_d2", -1, 1, t_orx, t_ory, "dg3")
    p1g3.extend([
        f'<feBlend in="{dg1}" in2="{dg2}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="dg12"/>',
        f'<feBlend in="dg12" in2="{dg3}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="dg123"/>',
        f'<feColorMatrix in="og2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="og2_bg"/>',
        f'<feBlend in="og2_bg" in2="dg123" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="fin_bq"/>',
        f'<feComponentTransfer in="fin_bq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="bq_192"><feFuncR type="table" tableValues="{tbl_add64_ge128}"/><feFuncG type="table" tableValues="{tbl_keep_ge128}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="og2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rk_r"/>',
        f'<feBlend in="bq_192" in2="rk_r" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="sl_unm"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ev_rg"/>',
        f'<feComposite in="sl_unm" in2="ev_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sl_out"/>',
    ])
    _emit_passthrough(p1g3, ["sl_out"])
    out_filters.append(_wrap_filter(f"ai-s1g3-{dpr}x", p1g3))

    # =========================================================================
    # Stage `s1h`: Opponent King Moves + `PNK` Best (`{0} U [128..255]`)
    # =========================================================================
    p1h: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{s1_kx - bm10_x}" dy="{s1_ky - bm10_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - csq_x}" dy="{s1_ky - csq_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ks"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_pnkx}" dy="{s1_ky - t_pnky}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="pn"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_orx}" dy="{s1_ky - t_ory}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sl"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_pin2x}" dy="{s1_ky - t_pin2y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - s1_b1x}" dy="{s1_ky - s1_b1y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ok_m0"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ok_m0" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ok_m"/>',
        f'<feComponentTransfer in="csq" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="u2p1_b"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_csq_valid}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ok_m" in2="u2p1_b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ok_u2"/>',
        f'<feConvolveMatrix in="ok_u2" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" order="3 3" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1 0 1 1 1 1" edgeMode="none" preserveAlpha="true" result="kg_nb"/>',
        f'<feColorMatrix in="ks" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 -1 0 0 0 1 0 0 0 0 1" result="no_kat"/>',
        f'<feColorMatrix in="ks" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="nop_b"/>',
        f'<feComposite in="no_kat" in2="nop_b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="kg_ok"/>',
        f'<feComposite in="kg_nb" in2="kg_ok" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="kg_u2b"/>',
        f'<feColorMatrix in="al2" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="mvc"/>',
        # Pawn candidate `(R=128+4*tier_P, G=128+u2, B=128+pt_placed)`:
        f'<feComponentTransfer in="mvc" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tp_rb"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(1)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_pt_p_placed128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="pn" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="pn_r3"/>',
        f'<feComponentTransfer in="pn_r3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="p_m"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="pn_r3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="p_u"/>',
        f'<feComposite in="tp_rb" in2="p_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="p_rb"/>',
        f'<feComposite in="p_rb" in2="p_u" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="cand_p"/>',
        # Knight candidate `(R=128+4*tier_N, G=128+u2, B=130/255)`:
        f'<feColorMatrix in="pn" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="pn_g3"/>',
        f'<feComponentTransfer in="mvc" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tn_rb"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(2)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="{_ct_u8(130)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="pn_g3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="n_m"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="pn_g3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="n_u"/>',
        f'<feComposite in="tn_rb" in2="n_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="n_rb"/>',
        f'<feComposite in="n_rb" in2="n_u" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="cand_n"/>',
        # King candidate `(R=128+4*tier_K, G=128+u2, B=134/255)`:
        f'<feColorMatrix in="kg_u2b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="kg_b3"/>',
        f'<feComponentTransfer in="mvc" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tk_rb"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(6)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="{_ct_u8(134)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="kg_b3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="k_m"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="kg_b3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="k_u"/>',
        f'<feComposite in="tk_rb" in2="k_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="k_rb"/>',
        f'<feComposite in="k_rb" in2="k_u" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="cand_k"/>',
    ]
    _mux_higher_score(p1h, "cand_p", "cand_n", s1_kx, s1_ky, s1_w, s1_h, "cand_pn")
    _mux_higher_score(p1h, "cand_pn", "cand_k", s1_kx, s1_ky, s1_w, s1_h, "pnk_best0")
    p1h.extend([
        # Store `(R = sl.R [orth], G = sl.G [diag], B = ks.B [in_check])` in `TMP_S1_ORTH`:
        f'<feColorMatrix in="sl" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="sl_rg"/>',
        f'<feColorMatrix in="ks" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="ichk_b"/>',
        f'<feComposite in="sl_rg" in2="ichk_b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_meta0"/>',
        f'<feOffset in="sl_meta0" dx="{t_orx - s1_kx}" dy="{t_ory - s1_ky}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="sl_meta"/>',
    ])
    _emit_passthrough(p1h, ["pnk_best0", "sl_meta"])
    out_filters.append(_wrap_filter(f"ai-s1h-{dpr}x", p1h))

    # =========================================================================
    # Stage `s1i`: Merge `B/R` (`slm.R`) & `Q` (`slm.G` w/ `ai_k_adj` Tier 12) vs `PNK` + `2x2` Argmax
    # =========================================================================
    tbl_sl_u2_128 = " ".join(_ct_u8(128 + ((c - 64) % 64)) if c >= 64 else "0" for c in range(256))
    tbl_sl_pt128 = " ".join(
        _ct_u8(131) if c >= 192 else (_ct_u8(132) if c >= 128 else "0")
        for c in range(256)
    )
    tbl_sl_bonus = " ".join(_ct_u8(8) if c >= 192 else (_ct_u8(4) if c >= 128 else "0") for c in range(256))
    tbl_subxy_raw = " ".join(
        _ct_u8(2 * (((c - 128) // 8) % 2) + ((c - 128) % 2)) if (128 <= c < 192) else "0"
        for c in range(256)
    )
    tbl_is_q00 = " ".join(
        "1" if (128 <= c < 192 and (((c - 128) // 8) % 2) == 0 and ((c - 128) % 2) == 0) else "0"
        for c in range(256)
    )
    tbl_eq0_192 = f"{_ct_u8(192)} " + " ".join(["0"] * 255)

    p1i: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{s1_kx - bm10_x}" dy="{s1_ky - bm10_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - csq_x}" dy="{s1_ky - csq_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="pnk"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_orx}" dy="{s1_ky - t_ory}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="slm"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_pin2x}" dy="{s1_ky - t_pin2y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - s1_b1x}" dy="{s1_ky - s1_b1y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="aik0"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="aik0" in2="bmask" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="aik"/>',
        f'<feConvolveMatrix in="aik" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" order="3 3" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1 0 1 1 1 1" edgeMode="none" preserveAlpha="true" result="k_adj"/>',
        f'<feColorMatrix in="al2" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="mvc_r"/>',
        f'<feComponentTransfer in="mvc_r" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tq_r"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(5)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        # `B / R` candidate from `slm.R` (`192+u2=B, 128+u2=R`):
        f'<feColorMatrix in="slm" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="sl_r3"/>',
        f'<feComponentTransfer in="sl_r3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="br_bns"><feFuncR type="table" tableValues="{tbl_sl_bonus}"/><feFuncG type="table" tableValues="{tbl_sl_u2_128}"/><feFuncB type="table" tableValues="{tbl_sl_pt128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sl_r3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="br_msk"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="table" tableValues="{tbl_ge128}"/><feFuncB type="table" tableValues="{tbl_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="tq_r" in2="br_bns" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="br_raw"/>',
        f'<feComposite in="br_raw" in2="br_msk" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cand_br"/>',
        # `Q` candidate from `slm.G` (`128+u2=Q`, boosted to Tier `12` = `176/255` on `k_adj` so contact mates beat `Bxf7+` without masking `N/B/R/Q` captures!):
        f'<feColorMatrix in="k_adj" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 {_fmt_num(176/255)} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="kq_176"/>',
        f'<feBlend in="tq_r" in2="kq_176" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="tq_adj"/>',
        f'<feColorMatrix in="slm" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 {_fmt_num(133/255)} 0 0 0 0 1" result="q_gb"/>',
        f'<feColorMatrix in="slm" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="sl_g3"/>',
        f'<feComponentTransfer in="sl_g3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="q_msk"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="table" tableValues="{tbl_ge128}"/><feFuncB type="table" tableValues="{tbl_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="tq_adj" in2="q_gb" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="q_raw"/>',
        f'<feComposite in="q_raw" in2="q_msk" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cand_q"/>',
    ]
    _mux_higher_score(p1i, "cand_q", "cand_br", s1_kx, s1_ky, s1_w, s1_h, "cand_sl")
    _mux_higher_score(p1i, "pnk", "cand_sl", s1_kx, s1_ky, s1_w, s1_h, "sq_win0")
    p1i.extend([
        f'<feComponentTransfer in="sq_win0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="has_m"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="csq" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sub_r"><feFuncR type="table" tableValues="{tbl_subxy_raw}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="sub_r" in2="has_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sub_ok"/>',
        f'<feComposite in="sq_win0" in2="sub_ok" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sq_w1"/>',
        f'<feTile in="sq_w1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sq_w1t"/>',
        f'<feOffset in="sq_w1t" dx="{-sc(1)}" dy="0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="rx_1"/>',
        f'<feBlend in="sq_w1t" in2="rx_1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="rm_x"/>',
        f'<feOffset in="rm_x" dx="0" dy="{-sc(1)}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="ry_1"/>',
        f'<feBlend in="rm_x" in2="ry_1" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="rm_xy"/>',
        f'<feComponentTransfer in="csq" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="is_q00"><feFuncR type="table" tableValues="{tbl_is_q00}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="rm_xy" in2="is_q00" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rm_00"/>',
        f'<feConvolveMatrix in="rm_00" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" order="2 2" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="rm_22"/>',
        f'<feBlend in="sq_w1t" in2="rm_22" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" mode="difference" result="dr_22"/>',
        f'<feComponentTransfer in="dr_22" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="eq_rm"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="eq_rm" in2="has_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="w_m0"/>',
        f'<feColorMatrix in="w_m0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="w_m"/>',
        f'<feComposite in="sq_w1t" in2="w_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="w_px"/>',
        f'<feConvolveMatrix in="w_px" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" order="2 2" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="qw_raw"/>',
        f'<feColorMatrix in="qw_raw" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="qw_rg"/>',
        f'<feComponentTransfer in="qw_rg" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="no_q"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_eq0_192}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="qw_raw" in2="no_q" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="qw_2x2"/>',
    ])
    _emit_passthrough(p1i, ["qw_2x2"])
    out_filters.append(_wrap_filter(f"ai-s1i-{dpr}x", p1i))

    # =========================================================================
    # Stage `s2a`: `640 x 160` Depth-2 Leaf Board Expansion + `VRAM_META_C2`
    # =========================================================================
    s2_w, s2_h = sc(640), sc(160)
    s2_b1y = sc(VRAM_S2_B1[1])
    pad_m2 = sc(22)  # 22 on 1x (640 + 22 = 662 <= 664), 11 on 2x (320 + 11 = 331 <= 332)!
    m2y_x, m2y_y = sc(LUT_M2Y_POS[0]), sc(LUT_M2Y_POS[1])
    m2x_x, m2x_y = sc(LUT_M2X_POS[0]), sc(LUT_M2X_POS[1])
    qb_x, qb_y = sc(LUT_QBASE_POS[0]), sc(LUT_QBASE_POS[1])
    c10x_x, c10x_y = sc(LUT_C10_2X_POS[0]), sc(LUT_C10_2X_POS[1])
    c10y_x, c10y_y = sc(LUT_C10_2Y_POS[0]), sc(LUT_C10_2Y_POS[1])
    mc2_x, mc2_y = sc(VRAM_META_C2[0]), sc(VRAM_META_C2[1])
    skc2_x, skc2_y = sc(VRAM_SKIP_C2[0]), sc(VRAM_SKIP_C2[1])
    c2_w, c2_h = sc(128), sc(32)

    tbl_dv2_r = " ".join(_ct_u8(8 * ((c // 2) % 2) + (c % 2)) if c >= 128 else "0" for c in range(256))
    tbl_dv2_fb = " ".join("0" if c >= 128 else _ct_u8(192) for c in range(256))

    x_slices_10_2 = [
        (sc(0), sc(0), sc(140), sc(28)),
        (sc(140), sc(28), sc(140), sc(28)),
        (sc(280), sc(56), sc(140), sc(28)),
        (sc(420), sc(84), sc(140), sc(28)),
        (sc(560), sc(112), sc(80), sc(16)),
    ]
    y_slices_10_2 = [
        (sc(0), sc(0), sc(80), sc(16)),
        (sc(80), sc(16), sc(80), sc(16)),
    ]

    p2a: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-m2y_x}" dy="{-m2y_y}" x="0" y="0" width="{sc(4)}" height="{s2_h}" result="ly0"/>',
        f'<feTile in="ly0" x="0" y="0" width="{s2_w + pad_m2}" height="{sc(170)}" result="ly_t"/>',
        f'<feOffset in="SourceGraphic" dx="{-m2x_x}" dy="{-m2x_y}" x="0" y="0" width="{sc(10)}" height="{sc(40)}" result="lx0"/>',
        f'<feTile in="lx0" x="0" y="0" width="{s2_w}" height="{s2_h + pad_m2}" result="lx_t"/>',
        f'<feOffset in="SourceGraphic" dx="{-s1_kx}" dy="{-s1_ky}" x="0" y="0" width="{s2_w}" height="{s1_h}" result="qw0"/>',
        f'<feTile in="qw0" x="0" y="0" width="{s2_w}" height="{sc(170)}" result="qwt"/>',
        f'<feOffset in="qwt" dx="{_fmt_num(a_m2)}" dy="0" x="0" y="0" width="{s2_w + pad_m2}" height="{sc(170)}" result="qw_ys"/>',
        f'<feDisplacementMap in="qw_ys" in2="ly_t" x="0" y="0" width="{s2_w}" height="{s2_h}" scale="{disp_m2}" xChannelSelector="A" yChannelSelector="R" result="qw_y"/>',
        f'<feOffset in="qw_y" dx="{sc(10)}" dy="{_fmt_num(a_m2)}" x="0" y="0" width="{sc(650)}" height="{s2_h + pad_m2}" result="qw_xs"/>',
        f'<feDisplacementMap in="qw_xs" in2="lx_t" x="0" y="0" width="{s2_w}" height="{s2_h}" scale="{disp_m2}" xChannelSelector="R" yChannelSelector="A" result="m2"/>',
        # `m2`: R=score (`0` or `128..255`), G=`128+u2` (`192` if none), B=`128+pt2` (`0` if none)
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="{-qb_x}" dy="{-qb_y}" x="0" y="0" width="{sc(4)}" height="{s2_h}" result="qb0"/>',
        f'<feTile in="qb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="qbase"/>',
        f'<feColorMatrix in="m2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="u2_rgb"/>',
        f'<feBlend in="csq" in2="u2_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="difference" result="du2"/>',
        f'<feComponentTransfer in="du2" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_u2"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="m2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="sc_rgb"/>',
        f'<feComponentTransfer in="sc_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="dv2"><feFuncR type="table" tableValues="{tbl_dv2_r}"/><feFuncG type="table" tableValues="{tbl_ge128}"/><feFuncB type="table" tableValues="{tbl_dv2_fb}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="dv2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="hm2_r"/>',
        f'<feComposite in="qbase" in2="hm2_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="qb_m"/>',
        f'<feComposite in="qb_m" in2="dv2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="v2_s"/>',
        f'<feColorMatrix in="v2_s" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 0 1" result="v2_rgb"/>',
        f'<feBlend in="csq" in2="v2_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="difference" result="dv2_d"/>',
        f'<feComponentTransfer in="dv2_d" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_v2"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        # Apply `(u2, v2, pt2)` onto tiled `S1_B0` & `S1_B1` (`A == 1.0` everywhere!):
        f'<feOffset in="SourceGraphic" dx="{-s1_b0x}" dy="{-s1_b0y}" x="0" y="0" width="{s2_w}" height="{s1_h}" result="sb0_0"/>',
        f'<feTile in="sb0_0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="{-s1_b1x}" dy="{-s1_b1y}" x="0" y="0" width="{s2_w}" height="{s1_h}" result="sb1_0"/>',
        f'<feTile in="sb1_0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComposite in="is_u2" in2="is_v2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="uv2"/>',
        f'<feColorMatrix in="uv2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="keep2"/>',
        f'<feColorMatrix in="m2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="pt2_rgb"/>',
        f'<feComponentTransfer in="pt2_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pl2_0"><feFuncR type="table" tableValues="{tbl_opp_p}"/><feFuncG type="table" tableValues="{tbl_opp_n}"/><feFuncB type="table" tableValues="{tbl_opp_b}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="pt2_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pl2_1"><feFuncR type="table" tableValues="{tbl_opp_r}"/><feFuncG type="table" tableValues="{tbl_opp_q}"/><feFuncB type="table" tableValues="{tbl_opp_k}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="sb0" in2="keep2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="s2b0_k"/>',
        f'<feComposite in="s2b0_k" in2="is_u2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="{z0_s}" k4="0" result="s2b0_u"/>',
        f'<feComposite in="pl2_0" in2="is_v2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="s2b0_v"/>',
        f'<feComposite in="s2b0_u" in2="s2b0_v" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="s2_b0"/>',
        f'<feComposite in="sb1" in2="keep2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="s2b1_k"/>',
        f'<feComposite in="s2b1_k" in2="is_u2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="{z0_s}" k4="0" result="s2b1_u"/>',
        f'<feComposite in="pl2_1" in2="is_v2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="s2b1_v"/>',
        f'<feComposite in="s2b1_u" in2="s2b1_v" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="s2_b1_0"/>',
        f'<feOffset in="s2_b1_0" dx="0" dy="{s2_b1y}" x="0" y="{s2_b1y}" width="{s2_w}" height="{s2_h}" result="s2_b1"/>',
        # Build & compact `META_S2` (`R = 1 - has_m2`, `G = s1_in_check`) -> `VRAM_META_C2` (`(384, 544)`, `128 x 32`)!
        f'<feOffset in="SourceGraphic" dx="{-t_orx}" dy="{-t_ory}" x="0" y="0" width="{s2_w}" height="{s1_h}" result="ich0"/>',
        f'<feTile in="ich0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="icht"/>',
        f'<feColorMatrix in="dv2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 -1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="nom2_r"/>',
        f'<feColorMatrix in="icht" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ich_g"/>',
        f'<feComposite in="nom2_r" in2="ich_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="meta_s2"/>',
        f'<feOffset in="SourceGraphic" dx="{-c10x_x}" dy="{-c10x_y}" x="0" y="0" width="{sc(28)}" height="{sc(4)}" result="c10x0"/>',
        f'<feTile in="c10x0" x="0" y="0" width="{c2_w}" height="{s2_h}" result="c10x_t"/>',
        f'<feOffset in="SourceGraphic" dx="{-c10y_x}" dy="{-c10y_y}" x="0" y="0" width="{sc(4)}" height="{sc(16)}" result="c10y0"/>',
        f'<feTile in="c10y0" x="0" y="0" width="{c2_w}" height="{c2_h}" result="c10y_t"/>',
    ]
    _emit_compact_at_origin(
        p2a, "meta_s2", 0, 0, s2_h, mc2_x, mc2_y, c2_w, c2_h,
        x_slices_10_2, y_slices_10_2, "c10x_t", "c10y_t", a_sh, disp_s, "meta_c2",
    )
    _emit_passthrough(p2a, ["s2_b0", "s2_b1", "meta_c2"], keep_vram=False, dpr=dpr)
    out_filters.append(_wrap_filter(f"ai-s2a-{dpr}x", p2a))

    # =========================================================================
    # Stages `s2m0..s2m10`: Exact Depth-2 Opponent Checkmate Detector (`640 x 160`)
    #                       Writes `is_s2_ai_mated` into `VRAM_META_C2.B` (`128 x 32`)!
    # =========================================================================
    tm_y = sc(TMP_S2_MATE[1])
    tbl_is_r2 = " ".join("1" if (144 <= c < 152) else "0" for c in range(256))
    tbl_chk_r = " ".join("0" if c < 64 else (_ct_u8(128) if c < 192 else "1") for c in range(256))
    tbl_lt64 = " ".join("1" if c < 64 else "0" for c in range(256))
    tbl_64_191 = " ".join("1" if (64 <= c < 192) else "0" for c in range(256))
    tbl_ge192 = " ".join("1" if c >= 192 else "0" for c in range(256))

    def _emit_s2_packed_ray(p: List[str], pack_in: str, dx_p: int, dy_p: int, tag: str) -> str:
        sx1, sy1 = sc(dx_p), sc(dy_p)
        p.append(f'<feOffset in="{pack_in}" dx="{sx1}" dy="{sy1}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_10"/>')
        p.append(f'<feTile in="{tag}_10" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_1"/>')
        for mul in (1, 2):
            cur, nxt_raw = f"{tag}_{mul}", f"{tag}_{2 * mul}r"
            p.append(f'<feOffset in="{cur}" dx="{mul * sx1}" dy="{mul * sy1}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{cur}s"/>')
            p.append(f'<feColorMatrix in="{cur}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="{cur}e"/>')
            p.append(f'<feComposite in="{cur}e" in2="{cur}s" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{cur}p"/>')
            p.append(f'<feColorMatrix in="{cur}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{cur}rg"/>')
            p.append(f'<feComposite in="{cur}rg" in2="{cur}p" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{nxt_raw}"/>')
            p.append(f'<feTile in="{nxt_raw}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_{2 * mul}"/>')
        c4 = f"{tag}_4"
        p.append(f'<feOffset in="{c4}" dx="{4 * sx1}" dy="{4 * sy1}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{c4}s"/>')
        p.append(f'<feColorMatrix in="{c4}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="{c4}e"/>')
        p.append(f'<feComposite in="{c4}e" in2="{c4}s" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{c4}p"/>')
        p.append(f'<feComposite in="{c4}" in2="{c4}p" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{tag}"/>')
        return tag

    def _emit_s2_1ch_ray(p: List[str], pack_in: str, dx_p: int, dy_p: int, tag: str) -> str:
        sx1, sy1 = sc(dx_p), sc(dy_p)
        p.append(f'<feOffset in="{pack_in}" dx="{sx1}" dy="{sy1}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_10"/>')
        p.append(f'<feTile in="{tag}_10" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_1"/>')
        for mul in (1, 2):
            cur, nxt = f"{tag}_{mul}", f"{tag}_{2 * mul}"
            p.append(f'<feOffset in="{cur}" dx="{mul * sx1}" dy="{mul * sy1}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{cur}s"/>')
            p.append(f'<feColorMatrix in="{cur}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="{cur}e"/>')
            p.append(f'<feComposite in="{cur}e" in2="{cur}s" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{cur}p"/>')
            p.append(f'<feColorMatrix in="{cur}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{cur}r"/>')
            p.append(f'<feComposite in="{cur}r" in2="{cur}p" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{nxt}"/>')
        c4 = f"{tag}_4"
        p.append(f'<feOffset in="{c4}" dx="{4 * sx1}" dy="{4 * sy1}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{c4}s"/>')
        p.append(f'<feColorMatrix in="{c4}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{c4}e"/>')
        p.append(f'<feComposite in="{c4}e" in2="{c4}s" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{c4}p"/>')
        p.append(f'<feComposite in="{c4}" in2="{c4}p" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{tag}"/>')
        return tag

    def _emit_s2_empty_b(p: List[str]) -> None:
        p.extend([
            f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="e0"><feFuncR type="table" tableValues="{tbl_eq128}"/><feFuncG type="table" tableValues="{tbl_eq128}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="e1"><feFuncR type="table" tableValues="{tbl_eq128}"/><feFuncG type="table" tableValues="{tbl_eq128}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComposite in="e0" in2="e1" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="e01"/>',
            f'<feColorMatrix in="e01" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 1 1 0 -2 0 0 0 0 1" result="emp0"/>',
            f'<feComposite in="emp0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="emp_b"/>',
        ])

    # `s2m0`: Opponent `P, N, K` attacks (`R`) + `P, N` checkers (`G`) -> `TMP_S2_MATE`
    pm0: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob00"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob00" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob0"/>',
        f'<feColorMatrix in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="sb1_k"/>',
        f'<feComponentTransfer in="sb1_k" x="0" y="0" width="{s2_w}" height="{s2_h}" result="krg0"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="krg0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="krg"/>',
        f'<feOffset in="ob0" dx="{sc(1)}" dy="{-sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="op_pr"/>',
        f'<feOffset in="ob0" dx="{-sc(1)}" dy="{-sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="op_pl"/>',
        f'<feComposite in="op_pr" in2="op_pl" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="op_pa"/>',
        f'<feConvolveMatrix in="ob0" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="2" targetY="2" divisor="1" bias="0" kernelMatrix="0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0" edgeMode="none" preserveAlpha="true" result="on_a"/>',
        f'<feConvolveMatrix in="krg" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1 0 1 1 1 1" edgeMode="none" preserveAlpha="true" result="ok_a"/>',
        f'<feColorMatrix in="op_pa" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="att_pr"/>',
        f'<feComposite in="on_a" in2="ok_a" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="onk_g"/>',
        f'<feColorMatrix in="onk_g" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="att_nkr"/>',
        f'<feComposite in="att_pr" in2="att_nkr" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="att0_r"/>',
        f'<feOffset in="krg" dx="{sc(1)}" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="kp_1"/>',
        f'<feOffset in="krg" dx="{-sc(1)}" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="kp_2"/>',
        f'<feComposite in="kp_1" in2="kp_2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="kp_12"/>',
        f'<feConvolveMatrix in="krg" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="2" targetY="2" divisor="1" bias="0" kernelMatrix="0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0" edgeMode="none" preserveAlpha="true" result="kn_k"/>',
        f'<feColorMatrix in="kp_12" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="kp_r"/>',
        f'<feColorMatrix in="kn_k" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="kn_g"/>',
        f'<feComposite in="kp_r" in2="kn_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="kpn_rg"/>',
        f'<feComposite in="ob0" in2="kpn_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cpn_rg"/>',
        f'<feColorMatrix in="cpn_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="cpn_g"/>',
        f'<feComposite in="att0_r" in2="cpn_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m0_0"/>',
        f'<feOffset in="m0_0" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="m0_out"/>',
    ]
    _emit_passthrough(pm0, ["m0_out"])
    out_filters.append(_wrap_filter(f"ai-s2m0-{dpr}x", pm0))

    # `s2m1..s2m4`: 4 Slider Axes (`H, V, D1, D2`) -> `R = opp_att (+ X-ray)`, `G = chk_sq`, `B = line_kz`
    for m_idx, (dx_a, dy_a, is_diag) in enumerate(((1, 0, False), (0, 1, False), (1, 1, True), (1, -1, True)), 1):
        pma: List[str] = [
            f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
            f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        ]
        _emit_s2_empty_b(pma)
        if not is_diag:
            pma.extend([
                f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob10"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
                f'<feComposite in="ob10" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ob1"/>',
                f'<feColorMatrix in="ob1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="sl_k"/>',
            ])
        else:
            pma.extend([
                f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob00"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
                f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob10"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
                f'<feColorMatrix in="ob00" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="sl_r0"/>',
                f'<feColorMatrix in="ob10" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="sl_rg1"/>',
                f'<feComposite in="sl_r0" in2="sl_rg1" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_k0"/>',
                f'<feComposite in="sl_k0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sl_k"/>',
            ])
        pma.append(f'<feComposite in="sl_k" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_ax0"/>')
        pma.append(f'<feTile in="pk_ax0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pk_ax"/>')
        rp = _emit_s2_packed_ray(pma, "pk_ax", dx_a, dy_a, "rp")
        rn = _emit_s2_packed_ray(pma, "pk_ax", -dx_a, -dy_a, "rn")
        w_ax = _fmt_num((32 * m_idx) / 255.0)
        pma.extend([
            f'<feColorMatrix in="sl_k" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="aik_r"/>',
            f'<feComposite in="aik_r" in2="{rp}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="hit_p"/>',
            f'<feOffset in="hit_p" dx="{sc(dx_a)}" dy="{sc(dy_a)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="xr_p"/>',
            f'<feComposite in="aik_r" in2="{rn}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="hit_n"/>',
            f'<feOffset in="hit_n" dx="{-sc(dx_a)}" dy="{-sc(dy_a)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="xr_n"/>',
            f'<feComposite in="xr_p" in2="xr_n" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="xr_pn"/>',
            f'<feComposite in="{rp}" in2="{rn}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="r_sum"/>',
            f'<feComposite in="r_sum" in2="xr_pn" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="att_ax"/>',
            f'<feColorMatrix in="att_ax" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="att_r"/>',
            f'<feColorMatrix in="r_sum" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="chk_g0"/>',
            f'<feColorMatrix in="sl_k" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="opsl_g"/>',
            f'<feComposite in="chk_g0" in2="opsl_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="chk_g"/>',
            f'<feColorMatrix in="{rn}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rn_sw"/>',
            f'<feComposite in="{rp}" in2="rn_sw" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pr_rg"/>',
            f'<feColorMatrix in="pr_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 {w_ax} {w_ax} 0 0 0 0 0 0 0 1" result="line_b"/>',
            f'<feComposite in="att_r" in2="chk_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ax_rg"/>',
            f'<feComposite in="ax_rg" in2="line_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ax_rgb"/>',
            f'<feComposite in="prev_m" in2="ax_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m_cur"/>',
            f'<feOffset in="m_cur" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="ma_out"/>',
        ])
        _emit_passthrough(pma, ["ma_out"])
        out_filters.append(_wrap_filter(f"ai-s2m{m_idx}-{dpr}x", pma))

    # `s2q0`: Stash disjoint Ply-3 `src_state (130..135)` & `tgt_code (139..175)` into `VRAM_S2_B1.B`!
    tbl_gt0 = "0 " + " ".join(["1"] * 255)
    tbl_ge16 = " ".join("0" if c < 16 else "1" for c in range(256))
    tbl_is_r7 = " ".join("1" if (184 <= c < 192) else "0" for c in range(256))
    tbl_pin_d0 = " ".join(_ct_u8(3 + int(round(c / 32.0))) if c <= 144 else _ct_u8(3) for c in range(256))
    tbl_pin_d1 = " ".join(_ct_u8(3) if c < 16 else _ct_u8(2) for c in range(256))
    tbl_pin_d2 = " ".join(_ct_u8(2) for _ in range(256))

    pq0: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="chk_ggg"/>',
        f'<feConvolveMatrix in="chk_ggg" x="0" y="0" width="{s2_w}" height="{s2_h}" order="2 2" targetX="0" targetY="0" divisor="2" bias="0" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="cg_4"/>',
        f'<feConvolveMatrix in="cg_4" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 1 0 0 0 1 0 1" edgeMode="none" preserveAlpha="true" result="cg_2"/>',
        f'<feConvolveMatrix in="cg_2" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="cg_1"/>',
        f'<feComponentTransfer in="cg_1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="cm_raw"><feFuncR type="table" tableValues="{tbl_lt64}"/><feFuncG type="table" tableValues="{tbl_64_191}"/><feFuncB type="table" tableValues="{tbl_ge192}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_00"><feFuncR type="table" tableValues="{tbl_eq128}"/><feFuncG type="table" tableValues="{tbl_eq128}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="cm_raw" in2="is_00" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cm_00"/>',
        f'<feConvolveMatrix in="cm_00" x="0" y="0" width="{s2_w}" height="{s2_h}" order="4 4" targetX="3" targetY="3" divisor="1" bias="0" kernelMatrix="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1" edgeMode="none" preserveAlpha="true" result="cm_b4"/>',
        f'<feConvolveMatrix in="cm_b4" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="4" targetY="4" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="chk_mode"/>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab00"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab00" in2="ab10" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ab_s"/>',
        f'<feColorMatrix in="ab_s" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1" result="aink0"/>',
        f'<feComposite in="aink0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="aink_b"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="pin_bbb"/>',
        f'<feComponentTransfer in="pin_bbb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pin_ct"><feFuncR type="table" tableValues="{tbl_pin_d0}"/><feFuncG type="table" tableValues="{tbl_pin_d1}"/><feFuncB type="table" tableValues="{tbl_pin_d2}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="pin_ct" in2="chk_mode" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pin_sel"/>',
        f'<feColorMatrix in="pin_sel" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1" result="pin_db0"/>',
        f'<feComposite in="pin_db0" in2="aink_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="delta_ai"/>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob00"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob10"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="ob00" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_num(12/255)} {_fmt_num(20/255)} {_fmt_num(20/255)} 0 0 {_fmt_num(44/255)} {_fmt_num(45/255)} {_fmt_num(45/255)} 0 0 1 1 1 0 0 0 0 0 0 1" result="op_m0"/>',
        f'<feColorMatrix in="ob10" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_num(28/255)} {_fmt_num(36/255)} 0 0 0 {_fmt_num(46/255)} {_fmt_num(47/255)} 0 0 0 1 1 0 0 0 0 0 0 0 1" result="op_m1"/>',
        f'<feComposite in="op_m0" in2="op_m1" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="op_raw"/>',
        f'<feComposite in="op_raw" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="op_base"/>',
        f'<feComponentTransfer in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" result="def_ct"><feFuncR type="table" tableValues="{tbl_gt0}"/><feFuncG type="table" tableValues="{tbl_gt0}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="r7_ct"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_is_r7}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="def_ct" in2="chk_mode" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="c1_g"/>',
        f'<feColorMatrix in="def_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_num(4/255)} 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="d_r"/>',
        f'<feColorMatrix in="r7_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 {_fmt_num(2/255)} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="r7_r"/>',
        f'<feColorMatrix in="c1_g" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 {_fmt_num(1/255)} 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="c1_r"/>',
        f'<feComposite in="d_r" in2="r7_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="dr_sum"/>',
        f'<feComposite in="dr_sum" in2="c1_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="drc_sum"/>',
        f'<feColorMatrix in="op_base" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="opk_m"/>',
        f'<feComposite in="drc_sum" in2="opk_m" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="off_op"/>',
        f'<feComposite in="op_base" in2="off_op" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="std_rg"/>',
        f'<feColorMatrix in="chk_mode" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="eq0_r"/>',
        f'<feColorMatrix in="drc_sum" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ic1_r"/>',
        f'<feComposite in="eq0_r" in2="ic1_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ustd_r"/>',
        f'<feColorMatrix in="ustd_r" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 -1 0 0 0 1 -1 0 0 0 1 0 0 0 0 1" result="nstd_gb"/>',
        f'<feComposite in="ustd_r" in2="nstd_gb" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="u_12"/>',
        f'<feColorMatrix in="drc_sum" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 1 0 0 -1 0 1 0 0 1 0 0 0 0 0 0 1" result="def_gb"/>',
        f'<feComposite in="u_12" in2="def_gb" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="use_rgb"/>',
        f'<feColorMatrix in="std_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 {_fmt_num(11/255)} 0 0 0 0 0 0 1" result="op_c3"/>',
        f'<feComposite in="op_c3" in2="use_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="op_sel"/>',
        f'<feColorMatrix in="op_sel" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1" result="delta_op"/>',
        f'<feComposite in="delta_ai" in2="delta_op" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="delta_all"/>',
        f'<feComposite in="sb1" in2="delta_all" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sb1_u0"/>',
        f'<feOffset in="sb1_u0" dx="0" dy="{s2_b1y}" x="0" y="{s2_b1y}" width="{s2_w}" height="{s2_h}" result="sb1_q0"/>',
    ]
    _emit_passthrough(pq0, ["sb1_q0"])
    out_filters.append(_wrap_filter(f"ai-s2q0-{dpr}x", pq0))

    # `s2m5`: King Steps + Checker Count Gather -> `(R = unpinned + chk_mode@(0,8), G = 0, B = evade_all)`
    pm5: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m0"/>',
        f'<feComponentTransfer in="prev_m0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"><feFuncR type="identity"/><feFuncG type="identity"/><feFuncB type="table" tableValues="{tbl_ge16}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
    ]
    _emit_s2_empty_b(pm5)
    pm5.extend([
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab00"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="ab00" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="a0s"/>',
        f'<feColorMatrix in="ab10" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 1" result="a1s"/>',
        f'<feComposite in="a0s" in2="a1s" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ai_raw"/>',
        f'<feComposite in="ai_raw" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ai_any"/>',
        f'<feComposite in="ab10" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ab1"/>',
        f'<feConvolveMatrix in="ab1" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1 0 1 1 1 1" edgeMode="none" preserveAlpha="true" result="knb0"/>',
        f'<feColorMatrix in="knb0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="knb_g"/>',
        f'<feColorMatrix in="ai_any" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 1" result="nai_g"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 1" result="nat_g"/>',
        f'<feComposite in="knb_g" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ks1"/>',
        f'<feComposite in="ks1" in2="nai_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ks2"/>',
        f'<feComposite in="ks2" in2="nat_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="kst_g"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="chk_r"/>',
        f'<feComposite in="chk_r" in2="kst_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ck_rg"/>',
        f'<feConvolveMatrix in="ck_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" order="2 2" targetX="0" targetY="0" divisor="2" bias="0" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="ck_4"/>',
        f'<feConvolveMatrix in="ck_4" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 1 0 0 0 1 0 1" edgeMode="none" preserveAlpha="true" result="ck_2"/>',
        f'<feConvolveMatrix in="ck_2" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="ck_1"/>',
        f'<feComponentTransfer in="ck_1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="cm_ct"><feFuncR type="table" tableValues="{tbl_chk_r}"/><feFuncG type="table" tableValues="{tbl_lt64}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="cm_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="nok_r"/>',
        f'<feComposite in="cm_ct" in2="nok_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cm0"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_00r"><feFuncR type="table" tableValues="{tbl_eq128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="cm0" in2="is_00r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cm_00"/>',
        f'<feOffset in="cm_00" dx="0" dy="{sc(8)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="cm_08"/>',
        f'<feComposite in="prev_m" in2="ai_any" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pin_b"/>',
        f'<feColorMatrix in="pin_b" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 -1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="unp0"/>',
        f'<feComposite in="unp0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="unp_r"/>',
        f'<feComposite in="unp_r" in2="cm_08" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="r_out"/>',
        f'<feComposite in="prev_m" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="int_b"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="chk_b"/>',
        f'<feComposite in="int_b" in2="chk_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="ev_b"/>',
        f'<feComposite in="r_out" in2="ev_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m5_0"/>',
        f'<feOffset in="m5_0" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="m5_out"/>',
    ])
    _emit_passthrough(pm5, ["m5_out"])
    out_filters.append(_wrap_filter(f"ai-s2m5-{dpr}x", pm5))

    # `s2m6`: Unpinned AI `P` + `N` + `RQ(+1, 0)` Evasion Reach -> `G = def0`
    pm6: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
    ]
    _emit_s2_empty_b(pm6)
    pm6.extend([
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob00"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ob10"><feFuncR type="table" tableValues="{tbl_eq1}"/><feFuncG type="table" tableValues="{tbl_eq1}"/><feFuncB type="table" tableValues="{tbl_eq1}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ob00" in2="ob10" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="o01"/>',
        f'<feColorMatrix in="o01" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="op_r0"/>',
        f'<feComposite in="op_r0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="op_r"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="unp0"/>',
        f'<feComposite in="unp0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="unp"/>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab00"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab00" in2="unp" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="u_pn"/>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab10" in2="unp" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="u_rq0"/>',
        f'<feColorMatrix in="u_rq0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="u_rq"/>',
        f'<feColorMatrix in="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="emp_r"/>',
        f'<feOffset in="u_pn" dx="0" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="p1_0"/>',
        f'<feComposite in="p1_0" in2="emp_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="p1"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_r2"><feFuncR type="table" tableValues="{tbl_is_r2}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="p1" in2="is_r2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="p1_r2"/>',
        f'<feOffset in="p1_r2" dx="0" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="p2_0"/>',
        f'<feComposite in="p2_0" in2="emp_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="p2"/>',
        f'<feOffset in="u_pn" dx="{sc(1)}" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pc_1"/>',
        f'<feOffset in="u_pn" dx="{-sc(1)}" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pc_2"/>',
        f'<feComposite in="pc_1" in2="pc_2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pc_12"/>',
        f'<feComposite in="pc_12" in2="op_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pc"/>',
        f'<feComposite in="p1" in2="p2" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="p_12"/>',
        f'<feComposite in="p_12" in2="pc" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="p_all"/>',
        f'<feConvolveMatrix in="u_pn" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="2" targetY="2" divisor="1" bias="0" kernelMatrix="0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0" edgeMode="none" preserveAlpha="true" result="n_att"/>',
        f'<feColorMatrix in="n_att" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="n_r"/>',
        f'<feComposite in="p_all" in2="n_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pn_r"/>',
        f'<feComposite in="u_rq" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_pk"/>',
    ])
    m6_r0 = _emit_s2_1ch_ray(pm6, "rq_pk", 1, 0, "m6r0")
    pm6.extend([
        f'<feComposite in="pn_r" in2="{m6_r0}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rch0"/>',
        f'<feColorMatrix in="rch0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rch_g"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ev_g"/>',
        f'<feComposite in="rch_g" in2="ev_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="def_g"/>',
        f'<feComposite in="prev_m" in2="def_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m6_0"/>',
        f'<feOffset in="m6_0" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="m6_out"/>',
    ])
    _emit_passthrough(pm6, ["m6_out"])
    out_filters.append(_wrap_filter(f"ai-s2m6-{dpr}x", pm6))

    # `s2m7`: Unpinned AI `RQ` Rays `(-1, 0), (0, +1), (0, -1)` -> `G |= def1`
    pm7: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
    ]
    _emit_s2_empty_b(pm7)
    pm7.extend([
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="unp0"/>',
        f'<feComposite in="unp0" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="unp"/>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab10"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab10" in2="unp" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="u_rq0"/>',
        f'<feColorMatrix in="u_rq0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="u_rq"/>',
        f'<feComposite in="u_rq" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_pk"/>',
    ])
    m7_r1 = _emit_s2_1ch_ray(pm7, "rq_pk", -1, 0, "m7r1")
    m7_r2 = _emit_s2_1ch_ray(pm7, "rq_pk", 0, 1, "m7r2")
    m7_r3 = _emit_s2_1ch_ray(pm7, "rq_pk", 0, -1, "m7r3")
    pm7.extend([
        f'<feComposite in="{m7_r1}" in2="{m7_r2}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="r12"/>',
        f'<feComposite in="r12" in2="{m7_r3}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rch1"/>',
        f'<feColorMatrix in="rch1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rch_g"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ev_g"/>',
        f'<feComposite in="rch_g" in2="ev_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="def_g"/>',
        f'<feComposite in="prev_m" in2="def_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m7_0"/>',
        f'<feOffset in="m7_0" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="m7_out"/>',
    ])
    _emit_passthrough(pm7, ["m7_out"])
    out_filters.append(_wrap_filter(f"ai-s2m7-{dpr}x", pm7))

    # `s2m8`: Unpinned AI `BQ` Diag Rays `(+1,+1), (-1,-1), (+1,-1)` -> `G |= def2`
    tbl_chk_m9 = " ".join("0" if c < 64 else (_ct_u8(64) if c < 192 else _ct_u8(128)) for c in range(256))
    tbl_ndf_m9 = " ".join(_ct_u8(64) if c < 64 else "0" for c in range(256))

    def _emit_s2_ubq_setup(p: List[str]) -> None:
        p.extend([
            f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
            f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        ])
        _emit_s2_empty_b(p)
        p.extend([
            f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab00"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab10"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComposite in="ab00" in2="ab10" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq0"/>',
            f'<feColorMatrix in="bq0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bq_r"/>',
            f'<feComposite in="bq_r" in2="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="u_bq"/>',
            f'<feComposite in="u_bq" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_pk"/>',
        ])

    pm8: List[str] = []
    _emit_s2_ubq_setup(pm8)
    m8_d0 = _emit_s2_1ch_ray(pm8, "bq_pk", 1, 1, "m8d0")
    m8_d1 = _emit_s2_1ch_ray(pm8, "bq_pk", -1, -1, "m8d1")
    m8_d2 = _emit_s2_1ch_ray(pm8, "bq_pk", 1, -1, "m8d2")
    pm8.extend([
        f'<feComposite in="{m8_d0}" in2="{m8_d1}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="d01"/>',
        f'<feComposite in="d01" in2="{m8_d2}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rch2"/>',
        f'<feColorMatrix in="rch2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rch_g"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ev_g"/>',
        f'<feComposite in="rch_g" in2="ev_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="def_g"/>',
        f'<feComposite in="prev_m" in2="def_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m8_0"/>',
        f'<feOffset in="m8_0" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="m8_out"/>',
    ])
    _emit_passthrough(pm8, ["m8_out"])
    out_filters.append(_wrap_filter(f"ai-s2m8-{dpr}x", pm8))

    # `s2m9`: Final `BQ` Diag Ray `(-1, +1)` + Gather `def_all` + Compact `10->2` into `VRAM_META_C2.B`!
    pm9: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
    ]
    _emit_s2_ubq_setup(pm9)
    m9_d3 = _emit_s2_1ch_ray(pm9, "bq_pk", -1, 1, "m9d3")
    pm9.extend([
        f'<feColorMatrix in="{m9_d3}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rch_g"/>',
        f'<feColorMatrix in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ev_g"/>',
        f'<feComposite in="rch_g" in2="ev_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="def_g"/>',
        f'<feComposite in="prev_m" in2="def_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m9_0"/>',
        f'<feColorMatrix in="m9_0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="def_g0"/>',
        f'<feConvolveMatrix in="def_g0" x="0" y="0" width="{s2_w}" height="{s2_h}" order="2 2" targetX="0" targetY="0" divisor="2" bias="0" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="dg_4"/>',
        f'<feConvolveMatrix in="dg_4" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 1 0 0 0 1 0 1" edgeMode="none" preserveAlpha="true" result="dg_2"/>',
        f'<feConvolveMatrix in="dg_2" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="0" targetY="0" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="dg_1"/>',
        f'<feOffset in="prev_m" dx="0" dy="{-sc(8)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="cm_sh"/>',
        f'<feComposite in="cm_sh" in2="dg_1" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="cd_rg"/>',
        f'<feComponentTransfer in="cd_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" result="cd_ct"><feFuncR type="table" tableValues="{tbl_chk_m9}"/><feFuncG type="table" tableValues="{tbl_ndf_m9}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="cd_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 8 8 0 0 -3 0 0 0 0 1" result="mt_raw"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_00b"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="mt_raw" in2="is_00b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="mt_00"/>',
        f'<feConvolveMatrix in="mt_00" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="4" targetY="4" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="mt_sm"/>',
        f'<feOffset in="SourceGraphic" dx="{-c10x_x}" dy="{-c10x_y}" x="0" y="0" width="{sc(28)}" height="{sc(4)}" result="c10x0"/>',
        f'<feTile in="c10x0" x="0" y="0" width="{c2_w}" height="{s2_h}" result="c10x_t"/>',
        f'<feOffset in="SourceGraphic" dx="{-c10y_x}" dy="{-c10y_y}" x="0" y="0" width="{sc(4)}" height="{sc(16)}" result="c10y0"/>',
        f'<feTile in="c10y0" x="0" y="0" width="{c2_w}" height="{c2_h}" result="c10y_t"/>',
    ])
    _emit_compact_at_origin(
        pm9, "mt_sm", 0, 0, s2_h, mc2_x, mc2_y, c2_w, c2_h,
        x_slices_10_2, y_slices_10_2, "c10x_t", "c10y_t", a_sh, disp_s, "mt_c2_0",
    )
    pm9.extend([
        f'<feComposite in="SourceGraphic" in2="mt_c2_0" x="{mc2_x}" y="{mc2_y}" width="{c2_w}" height="{c2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="meta_c2_fin"/>',
    ])
    _emit_passthrough(pm9, ["meta_c2_fin"])
    out_filters.append(_wrap_filter(f"ai-s2m9-{dpr}x", pm9))

    # =========================================================================
    # Stages `s2q1..s2q7`: FIDE-Legal SEE-Filtered Ply-3 Quiescence (`640 x 160`)
    #                      Updates `VRAM_S2_B0` & `VRAM_S2_B1` in-place before `s2b`!
    # =========================================================================
    tbl_allow_h = " ".join("1" if c in (131, 132) else "0" for c in range(256))
    tbl_allow_v = " ".join("1" if c in (131, 133) else "0" for c in range(256))
    tbl_allow_d1 = " ".join("1" if c in (131, 134) else "0" for c in range(256))
    tbl_allow_d2 = " ".join("1" if c in (131, 135) else "0" for c in range(256))
    tbl_allow_unp = " ".join("1" if c == 131 else "0" for c in range(256))
    tbl_hi_sub64 = " ".join(_ct_u8(c - 64) if c >= 192 else "0" for c in range(256))
    tbl_csq_vy = " ".join(_ct_u8((c - 128) // 8) if (128 <= c < 192) else "0" for c in range(256))
    tbl_csq_vx = " ".join(_ct_u8((c - 128) % 8) if (128 <= c < 192) else "0" for c in range(256))

    def _ply3_tc_lut_r(pt3: int) -> str:
        vals: List[str] = []
        for c in range(256):
            tier, _pl = ply3_tier_and_placed(pt3, c)
            vals.append(_ct_u8(128 + 8 * (tier - 1)) if tier > 0 else "0")
        return " ".join(vals)

    def _ply3_tc_lut_b(pt3: int) -> str:
        vals: List[str] = []
        for c in range(256):
            tier, pl = ply3_tier_and_placed(pt3, c)
            vals.append(_ct_u8(128 + 8 * pl) if tier > 0 else "0")
        return " ".join(vals)

    def _emit_score_ply3_pt(p: List[str], u_g_in: str, pt3: int, tag: str) -> str:
        p.extend([
            f'<feComponentTransfer in="tc_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_tc"><feFuncR type="table" tableValues="{_ply3_tc_lut_r(pt3)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{_ply3_tc_lut_b(pt3)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComposite in="{tag}_tc" in2="{u_g_in}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{tag}_rbu"/>',
            f'<feComponentTransfer in="{tag}_rbu" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_act0"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="table" tableValues="{tbl_ge128}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feColorMatrix in="{tag}_act0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 -1 1 1 0 0 -1 1 1 0 0 -1 0 0 0 0 1" result="{tag}_act"/>',
            f'<feComposite in="{tag}_rbu" in2="vy_vx" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{tag}_full"/>',
            f'<feComposite in="{tag}_full" in2="{tag}_act" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{tag}"/>',
        ])
        return tag

    # `s2q1`: Orth `R/Q` Rays `(+1,0), (-1,0), (0,+1)` -> `TMP_S2_MATE.R`
    pq1: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="st_ct"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_allow_h}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="sb1_bb"/>',
        f'<feComponentTransfer in="sb1_bb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="st_ct2"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_allow_v}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="st_ct2" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="ve_m"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="u_rq"><feFuncR type="table" tableValues="{tbl_add64_ge128}"/><feFuncG type="table" tableValues="{tbl_csq_valid}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="st_ct" in2="u_rq" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_uv"/>',
        f'<feColorMatrix in="rq_uv" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_1ch"/>',
        f'<feColorMatrix in="st_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="al_h"/>',
        f'<feComposite in="rq_1ch" in2="al_h" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="src_h"/>',
        f'<feColorMatrix in="ve_m" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="emp_only"/>',
        f'<feComposite in="src_h" in2="emp_only" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_h"/>',
        f'<feComposite in="rq_1ch" in2="ve_m" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="src_v0"/>',
        f'<feColorMatrix in="src_v0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="src_v"/>',
        f'<feComposite in="src_v" in2="emp_only" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_v"/>',
    ]
    q1r0 = _emit_s2_1ch_ray(pq1, "pk_h", 1, 0, "q1r0")
    q1r1 = _emit_s2_1ch_ray(pq1, "pk_h", -1, 0, "q1r1")
    q1r2 = _emit_s2_1ch_ray(pq1, "pk_v", 0, 1, "q1r2")
    pq1.extend([
        f'<feBlend in="{q1r0}" in2="{q1r1}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="or01"/>',
        f'<feBlend in="or01" in2="{q1r2}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="or012_raw"/>',
        f'<feColorMatrix in="or012_raw" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="or012_r"/>',
        f'<feOffset in="or012_r" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="q1_out"/>',
    ])
    _emit_passthrough(pq1, ["q1_out"])
    out_filters.append(_wrap_filter(f"ai-s2q1-{dpr}x", pq1))

    # `s2q2`: Orth `R/Q` Ray `(0,-1)` + Diag `B/Q` `D1` Ray `(+1,+1)` -> `TMP_S2_MATE.RG`
    pq2: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm0"/>',
        f'<feTile in="bm0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bmask"/>',
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="u_sl"><feFuncR type="table" tableValues="{tbl_add64_ge128}"/><feFuncG type="table" tableValues="{tbl_csq_valid}"/><feFuncB type="table" tableValues="{tbl_add64_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="rq_e"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="b_ct"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="st_rg"/>',
        f'<feComponentTransfer in="st_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" result="al_vd1"><feFuncR type="table" tableValues="{tbl_allow_v}"/><feFuncG type="table" tableValues="{tbl_allow_d1}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="rq_e" in2="bmask" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_em"/>',
        f'<feColorMatrix in="rq_em" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="emp_only"/>',
        f'<feComposite in="rq_e" in2="u_sl" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_u"/>',
        f'<feComposite in="b_ct" in2="u_sl" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b_u"/>',
        f'<feColorMatrix in="rq_u" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_rg"/>',
        f'<feColorMatrix in="b_u" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="b_g"/>',
        f'<feComposite in="rq_rg" in2="b_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="v_d1_u"/>',
        f'<feComposite in="v_d1_u" in2="al_vd1" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="vd1_al"/>',
        f'<feColorMatrix in="vd1_al" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="src_v"/>',
        f'<feComposite in="src_v" in2="emp_only" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_v"/>',
        f'<feColorMatrix in="vd1_al" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="src_d1"/>',
        f'<feComposite in="src_d1" in2="emp_only" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_d1"/>',
    ]
    q2r3 = _emit_s2_1ch_ray(pq2, "pk_v", 0, -1, "q2r3")
    q2d0 = _emit_s2_1ch_ray(pq2, "pk_d1", 1, 1, "q2d0")
    pq2.extend([
        f'<feBlend in="prev_m" in2="{q2r3}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="or_raw"/>',
        f'<feColorMatrix in="or_raw" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="orth_r"/>',
        f'<feColorMatrix in="{q2d0}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="d0_g"/>',
        f'<feComposite in="orth_r" in2="d0_g" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="q2_rg"/>',
        f'<feOffset in="q2_rg" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="q2_out"/>',
    ])
    _emit_passthrough(pq2, ["q2_out"])
    out_filters.append(_wrap_filter(f"ai-s2q2-{dpr}x", pq2))

    # `s2q3`: Diag `B/Q` Rays `(-1,-1), (+1,-1), (-1,+1)` -> `TMP_S2_MATE.RG` (`R = orth_best`, `G = diag_best`)
    tbl_lt192 = " ".join("1" if c < 192 else "0" for c in range(256))
    pq3: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="u_bq"><feFuncR type="table" tableValues="{tbl_add64_ge128}"/><feFuncG type="table" tableValues="{tbl_csq_valid}"/><feFuncB type="table" tableValues="{tbl_lt192}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="b_ct"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="b_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="b_r"/>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="q_em"><feFuncR type="discrete" tableValues="0"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="st_rg"/>',
        f'<feComponentTransfer in="st_rg" x="0" y="0" width="{s2_w}" height="{s2_h}" result="al_d12"><feFuncR type="table" tableValues="{tbl_allow_d1}"/><feFuncG type="table" tableValues="{tbl_allow_d2}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="b_r" in2="q_em" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_rge"/>',
        f'<feComposite in="bq_rge" in2="u_bq" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bqu0"/>',
        f'<feColorMatrix in="bqu0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="emp_b"/>',
        f'<feColorMatrix in="bqu0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bqu_rg"/>',
        f'<feComposite in="bqu_rg" in2="al_d12" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="d12_al"/>',
        f'<feColorMatrix in="d12_al" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="src_d1"/>',
        f'<feComposite in="src_d1" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_d1"/>',
        f'<feColorMatrix in="d12_al" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="src_d2"/>',
        f'<feComposite in="src_d2" in2="emp_b" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="pk_d2"/>',
    ]
    q3d1 = _emit_s2_1ch_ray(pq3, "pk_d1", -1, -1, "q3d1")
    q3d2 = _emit_s2_1ch_ray(pq3, "pk_d2", 1, -1, "q3d2")
    q3d3 = _emit_s2_1ch_ray(pq3, "pk_d2", -1, 1, "q3d3")
    pq3.extend([
        f'<feBlend in="{q3d1}" in2="{q3d2}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="d12_raw"/>',
        f'<feBlend in="d12_raw" in2="{q3d3}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="d123_raw"/>',
        f'<feColorMatrix in="d123_raw" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="d123_g"/>',
        f'<feBlend in="prev_m" in2="d123_g" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="q3_rg"/>',
        f'<feOffset in="q3_rg" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="q3_out"/>',
    ])
    _emit_passthrough(pq3, ["q3_out"])
    out_filters.append(_wrap_filter(f"ai-s2q3-{dpr}x", pq3))

    # `s2q4`: Score `B (3), R (4), Q (5), K (6)` & Mux `B -> R -> Q -> K` -> `TMP_S2_MATE`
    pq4: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="vy_vx"><feFuncR type="table" tableValues="{tbl_csq_vy}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_csq_vx}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="tc_rgb"/>',
        f'<feComponentTransfer in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ct_hi"><feFuncR type="table" tableValues="{tbl_hi_sub64}"/><feFuncG type="table" tableValues="{tbl_hi_sub64}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="ct_hi" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ub_g"/>',
        f'<feColorMatrix in="ct_hi" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ur_g"/>',
        f'<feComponentTransfer in="prev_m" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ct_lo"><feFuncR type="table" tableValues="{tbl_keep_ge128}"/><feFuncG type="table" tableValues="{tbl_keep_ge128}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="ct_lo" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="qo_g"/>',
        f'<feColorMatrix in="ct_lo" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="qd_g"/>',
        f'<feBlend in="qo_g" in2="qd_g" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="uq_g"/>',
        f'<feComponentTransfer in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" result="aik_ct"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="aik_ct" in2="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="aik_u0"/>',
        f'<feConvolveMatrix in="aik_u0" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="1" targetY="1" divisor="1" bias="0" kernelMatrix="1 1 1 1 0 1 1 1 1" edgeMode="none" preserveAlpha="true" result="aik_nb"/>',
        f'<feColorMatrix in="aik_nb" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="uk_g"/>',
    ]
    c_b = _emit_score_ply3_pt(pq4, "ub_g", 3, "cb")
    c_r = _emit_score_ply3_pt(pq4, "ur_g", 4, "cr")
    c_q = _emit_score_ply3_pt(pq4, "uq_g", 5, "cq")
    c_k = _emit_score_ply3_pt(pq4, "uk_g", 6, "ck")
    _mux_higher_score(pq4, c_b, c_r, 0, 0, s2_w, s2_h, "mbr")
    _mux_higher_score(pq4, "mbr", c_q, 0, 0, s2_w, s2_h, "mbrq")
    _mux_higher_score(pq4, "mbrq", c_k, 0, 0, s2_w, s2_h, "slk_cand")
    pq4.append(f'<feOffset in="slk_cand" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="q4_out"/>')
    _emit_passthrough(pq4, ["q4_out"])
    out_filters.append(_wrap_filter(f"ai-s2q4-{dpr}x", pq4))

    # `s2q5`: Score `P (1), N (2)` & Mux `P -> N -> slk_cand` -> `TMP_S2_MATE`
    pq5: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="prev_m"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="vy_vx"><feFuncR type="table" tableValues="{tbl_csq_vy}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_csq_vx}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="u_val"><feFuncR type="table" tableValues="{tbl_csq_valid}"/><feFuncG type="table" tableValues="{tbl_csq_valid}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sb1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="tc_rgb"/>',
        f'<feComponentTransfer in="tc_rgb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="st_al"><feFuncR type="table" tableValues="{tbl_allow_d1}"/><feFuncG type="table" tableValues="{tbl_allow_d2}"/><feFuncB type="table" tableValues="{tbl_allow_unp}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sb0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="ab0"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="ab0" in2="u_val" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="pn_u"/>',
        f'<feColorMatrix in="pn_u" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="p_rr"/>',
        f'<feComposite in="p_rr" in2="st_al" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="p_d12"/>',
        f'<feOffset in="p_d12" dx="{sc(1)}" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pc_1"/>',
        f'<feColorMatrix in="pc_1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="pc_1g"/>',
        f'<feOffset in="p_d12" dx="{-sc(1)}" dy="{sc(1)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pc_2"/>',
        f'<feColorMatrix in="pc_2" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="pc_2g"/>',
        f'<feBlend in="pc_1g" in2="pc_2g" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="up_g"/>',
        f'<feColorMatrix in="pn_u" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1" result="n_bb"/>',
        f'<feComposite in="n_bb" in2="st_al" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="n_src0"/>',
        f'<feTile in="n_src0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="n_src"/>',
    ]
    q5_kn = ""
    for k_idx, (dx_k, dy_k) in enumerate(AI_KNIGHT_DELTAS):
        nm = f"q5ko_{k_idx}"
        pq5.append(f'<feOffset in="n_src" dx="{sc(dx_k)}" dy="{sc(dy_k)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{nm}"/>')
        if k_idx == 0:
            q5_kn = nm
        else:
            nxt = f"q5km_{k_idx}"
            pq5.append(f'<feBlend in="{q5_kn}" in2="{nm}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="{nxt}"/>')
            q5_kn = nxt
    pq5.append(f'<feColorMatrix in="{q5_kn}" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="un_g"/>')
    c_p = _emit_score_ply3_pt(pq5, "up_g", 1, "cp")
    c_n = _emit_score_ply3_pt(pq5, "un_g", 2, "cn")
    _mux_higher_score(pq5, c_p, c_n, 0, 0, s2_w, s2_h, "pn_cand")
    _mux_higher_score(pq5, "pn_cand", "prev_m", 0, 0, s2_w, s2_h, "all_cand")
    pq5.append(f'<feOffset in="all_cand" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="q5_out"/>')
    _emit_passthrough(pq5, ["q5_out"])
    out_filters.append(_wrap_filter(f"ai-s2q5-{dpr}x", pq5))

    # `s2q6`: Zero-Mux `dilate(2,2) + lighten` Exact `8x8 -> (0,0)` Argmax + Broadcast Across `8x8`!
    tbl_tie_8x8 = " ".join(_ct_u8(191 - (c - 128)) if (128 <= c < 192) else "0" for c in range(256))

    def _emit_cell8_max_bcast(p: List[str], src_in: str, tag: str, pin_out: bool = True) -> str:
        p.extend([
            f'<feMorphology in="{src_in}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="dilate" radius="{_fmt_num(2 * u)} {_fmt_num(2 * u)}" result="{tag}_dl"/>',
            f'<feTile in="{tag}_dl" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_dlt"/>',
            f'<feOffset in="{tag}_dlt" dx="{-sc(3)}" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_sx"/>',
            f'<feBlend in="{tag}_dlt" in2="{tag}_sx" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="{tag}_mx"/>',
            f'<feOffset in="{tag}_mx" dx="0" dy="{-sc(3)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_sy"/>',
            f'<feBlend in="{tag}_mx" in2="{tag}_sy" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="lighten" result="{tag}_m22"/>',
            f'<feOffset in="{tag}_m22" dx="{-sc(2)}" dy="{-sc(2)}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}_s00"/>',
            f'<feComposite in="{tag}_s00" in2="is_00" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="{tag}_00"/>',
            f'<feConvolveMatrix in="{tag}_00" x="0" y="0" width="{s2_w}" height="{s2_h}" order="4 4" targetX="3" targetY="3" divisor="1" bias="0" kernelMatrix="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1" edgeMode="none" preserveAlpha="true" result="{tag}_b4"/>',
            f'<feConvolveMatrix in="{tag}_b4" x="0" y="0" width="{s2_w}" height="{s2_h}" order="5 5" targetX="4" targetY="4" divisor="1" bias="0" kernelMatrix="1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1" edgeMode="none" preserveAlpha="true" result="{tag}_88"/>',
        ])
        if pin_out:
            p.append(f'<feTile in="{tag}_88" x="0" y="0" width="{s2_w}" height="{s2_h}" result="{tag}"/>')
            return tag
        return f"{tag}_88"

    pq6: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_00"><feFuncR type="table" tableValues="{tbl_eq128}"/><feFuncG type="table" tableValues="{tbl_eq128}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="rx00"/>',
        f'<feTile in="rx00" x="0" y="0" width="{s2_w}" height="{s2_h}" result="rx0"/>',
        f'<feColorMatrix in="rx0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="r_only"/>',
    ]
    rmax_n = _emit_cell8_max_bcast(pq6, "r_only", "rmax", pin_out=True)
    pq6.extend([
        f'<feBlend in="r_only" in2="{rmax_n}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="difference" result="dr_m"/>',
        f'<feComponentTransfer in="dr_m" x="0" y="0" width="{s2_w}" height="{s2_h}" result="eq_rm"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="r_only" x="0" y="0" width="{s2_w}" height="{s2_h}" result="has_r"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="eq_rm" in2="has_r" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="is_rmax"/>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="tie_sq"><feFuncR type="table" tableValues="{tbl_tie_8x8}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="tie_sq" in2="is_rmax" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="t_cand"/>',
    ])
    tmax_n = _emit_cell8_max_bcast(pq6, "t_cand", "tmax", pin_out=True)
    pq6.extend([
        f'<feBlend in="t_cand" in2="{tmax_n}" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="difference" result="dt_m"/>',
        f'<feComponentTransfer in="dt_m" x="0" y="0" width="{s2_w}" height="{s2_h}" result="eq_tm"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="eq_tm" in2="is_rmax" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="is_win_r"/>',
        f'<feColorMatrix in="is_win_r" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="is_win"/>',
        f'<feComposite in="rx0" in2="is_win" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="win_px"/>',
    ])
    win_8x8_n = _emit_cell8_max_bcast(pq6, "win_px", "win_8x8", pin_out=False)
    pq6.append(f'<feOffset in="{win_8x8_n}" dx="0" dy="{tm_y}" x="0" y="{tm_y}" width="{s2_w}" height="{s2_h}" result="q6_out"/>')
    _emit_passthrough(pq6, ["q6_out"])
    out_filters.append(_wrap_filter(f"ai-s2q6-{dpr}x", pq6))

    # `s2q7`: Restore `S2_B1.B` & Apply Ply-3 Quiescence Move `(u3*, v3*, placed_code*)` In-Place!
    tbl_restore_k = " ".join(neg_ct if c <= 1 else ("1" if c == 255 else z0_ct) for c in range(256))
    tbl_u3_match = " ".join(_ct_u8(c) if (128 <= c < 192) else _ct_u8(254) for c in range(256))
    tbl_csq_y128 = " ".join(_ct_u8(128 + ((c - 128) // 8)) if (128 <= c < 192) else _ct_u8(250) for c in range(256))
    tbl_csq_x128 = " ".join(_ct_u8(128 + ((c - 128) % 8)) if (128 <= c < 192) else _ct_u8(250) for c in range(256))
    tbl_win_y128 = " ".join(_ct_u8(128 + ((c - 128) % 8)) if (128 <= c < 248) else _ct_u8(254) for c in range(256))
    tbl_win_x128 = " ".join(_ct_u8(128 + ((c - 128) % 8)) if (128 <= c < 184) else _ct_u8(254) for c in range(256))

    def _pl3_lut(target_pl: int) -> str:
        return " ".join(
            "1" if (128 <= c < 184 and ((c - 128) // 8) == target_pl) else z0_ct
            for c in range(256)
        )

    pq7: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-csq_x}" dy="{-csq_y}" x="0" y="0" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-tm_y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="win"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb0"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1_raw"/>',
        f'<feComponentTransfer in="sb1_raw" x="0" y="0" width="{s2_w}" height="{s2_h}" result="sb1"><feFuncR type="identity"/><feFuncG type="identity"/><feFuncB type="table" tableValues="{tbl_restore_k}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="win" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="win_ggg"/>',
        f'<feComponentTransfer in="win_ggg" x="0" y="0" width="{s2_w}" height="{s2_h}" result="u3_ct"><feFuncR type="table" tableValues="{tbl_u3_match}"/><feFuncG type="table" tableValues="{tbl_u3_match}"/><feFuncB type="table" tableValues="{tbl_u3_match}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feBlend in="csq" in2="u3_ct" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="difference" result="du3"/>',
        f'<feComponentTransfer in="du3" x="0" y="0" width="{s2_w}" height="{s2_h}" result="is_u3"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="csq" x="0" y="0" width="{s2_w}" height="{s2_h}" result="csq_yx"><feFuncR type="table" tableValues="{tbl_csq_y128}"/><feFuncG type="discrete" tableValues="{_ct_u8(250)}"/><feFuncB type="table" tableValues="{tbl_csq_x128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="win" x="0" y="0" width="{s2_w}" height="{s2_h}" result="win_yx"><feFuncR type="table" tableValues="{tbl_win_y128}"/><feFuncG type="discrete" tableValues="{_ct_u8(250)}"/><feFuncB type="table" tableValues="{tbl_win_x128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feBlend in="csq_yx" in2="win_yx" x="0" y="0" width="{s2_w}" height="{s2_h}" mode="difference" result="dv3"/>',
        f'<feComponentTransfer in="dv3" x="0" y="0" width="{s2_w}" height="{s2_h}" result="dv3_eq"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="dv3_eq" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="1 0 1 0 -1 1 0 1 0 -1 1 0 1 0 -1 0 0 0 0 1" result="is_v3"/>',
        f'<feColorMatrix in="win" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="win_bbb"/>',
        f'<feComponentTransfer in="win_bbb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pl3_0"><feFuncR type="table" tableValues="{_pl3_lut(1)}"/><feFuncG type="table" tableValues="{_pl3_lut(2)}"/><feFuncB type="table" tableValues="{_pl3_lut(3)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="win_bbb" x="0" y="0" width="{s2_w}" height="{s2_h}" result="pl3_1"><feFuncR type="table" tableValues="{_pl3_lut(4)}"/><feFuncG type="table" tableValues="{_pl3_lut(5)}"/><feFuncB type="table" tableValues="{_pl3_lut(6)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="is_u3" in2="is_v3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="uv3"/>',
        f'<feColorMatrix in="uv3" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="keep3"/>',
        f'<feComposite in="sb0" in2="keep3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q7b0_k"/>',
        f'<feComposite in="q7b0_k" in2="is_u3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="{z0_s}" k4="0" result="q7b0_u"/>',
        f'<feComposite in="pl3_0" in2="is_v3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q7b0_v"/>',
        f'<feComposite in="q7b0_u" in2="q7b0_v" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="in_b0"/>',
        f'<feComposite in="sb1" in2="keep3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q7b1_k"/>',
        f'<feComposite in="q7b1_k" in2="is_u3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="{z0_s}" k4="0" result="q7b1_u"/>',
        f'<feComposite in="pl3_1" in2="is_v3" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q7b1_v"/>',
        f'<feComposite in="q7b1_u" in2="q7b1_v" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="in_b1"/>',
        f'<feOffset in="in_b1" dx="0" dy="{s2_b1y}" x="0" y="{s2_b1y}" width="{s2_w}" height="{s2_h}" result="q7_b1"/>',
    ]

    sk_spec = ly["skip_mat"]
    sk_cm0, sk_cm1 = sk_spec["color_matrices"]
    sk_st0, sk_st1, sk_st2 = sk_spec["sum_stages"]
    nz0_sk = _fmt_num(-float(sk_spec["zero_point"]))

    def _emit_skip_c2(p: List[str]) -> None:
        p.extend([
            f'<feColorMatrix in="in_b0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_list(sk_cm0)}" result="sk0"/>',
            f'<feColorMatrix in="in_b1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_list(sk_cm1)}" result="sk1"/>',
            f'<feComposite in="sk0" in2="sk1" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="{nz0_sk}" result="sk_8"/>',
            f'<feConvolveMatrix in="sk_8" x="0" y="0" width="{s2_w}" height="{s2_h}" order="2 2" targetX="0" targetY="0" divisor="{_fmt_num(sk_st0["divisor"])}" bias="{_fmt_num(sk_st0["bias"])}" kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="sk_4"/>',
            f'<feConvolveMatrix in="sk_4" x="0" y="0" width="{s2_w}" height="{s2_h}" order="3 3" targetX="0" targetY="0" divisor="{_fmt_num(sk_st1["divisor"])}" bias="{_fmt_num(sk_st1["bias"])}" kernelMatrix="1 0 1 0 0 0 1 0 1" edgeMode="none" preserveAlpha="true" result="sk_2"/>',
            f'<feOffset in="SourceGraphic" dx="{-c10x_x}" dy="{-c10x_y}" x="0" y="0" width="{sc(28)}" height="{sc(4)}" result="c10x0"/>',
            f'<feTile in="c10x0" x="0" y="0" width="{c2_w}" height="{s2_h}" result="c10x_t"/>',
            f'<feOffset in="SourceGraphic" dx="{-c10y_x}" dy="{-c10y_y}" x="0" y="0" width="{sc(4)}" height="{sc(16)}" result="c10y0"/>',
            f'<feTile in="c10y0" x="0" y="0" width="{c2_w}" height="{c2_h}" result="c10y_t"/>',
        ])
        _emit_compact_at_origin(
            p, "sk_2", 0, 0, s2_h, skc2_x, skc2_y, c2_w, c2_h,
            x_slices_10_2, y_slices_10_2, "c10x_t", "c10y_t", a_sh, disp_s, "skip_c2",
        )

    m1_trips = ly["mix1"]["triplets"]
    g1_cnt = len(m1_trips)
    assert g1_cnt in (5, 6)
    if g1_cnt == 6:
        _emit_skip_c2(pq7)
        _emit_passthrough(pq7, ["in_b0", "q7_b1", "skip_c2"])
    else:
        _emit_passthrough(pq7, ["in_b0", "q7_b1"])
    out_filters.append(_wrap_filter(f"ai-s2q7-{dpr}x", pq7))

    # =========================================================================
    # Stages `s2b, s2c, s2d/s2e`: CNN Layer 1 (`mix1 -> bmask10 -> dwconv1 -> pool1 -> C6`)
    # =========================================================================
    c6_w, c6_h = sc(384), sc(96)
    c6x_x, c6x_y = sc(LUT_C6X_POS[0]), sc(LUT_C6X_POS[1])
    c6y_x, c6y_y = sc(LUT_C6Y_POS[0]), sc(LUT_C6Y_POS[1])
    x_slices_10_6 = [
        (sc(0), sc(0), sc(220), sc(132)),
        (sc(220), sc(132), sc(220), sc(132)),
        (sc(440), sc(264), sc(200), sc(120)),
    ]
    flts1 = ly["dwconv1"]["filters"]
    p1_div = _fmt_num(ly["pool1"]["divisor"])
    nz0_m1 = _fmt_num(-float(ly["mix1"]["zero_point"]))

    def _emit_l1_shared(p: List[str]) -> None:
        p.extend([
            f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="in_b0"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-s2_b1y}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="in_b1"/>',
            f'<feOffset in="SourceGraphic" dx="{-bm10_x}" dy="{-bm10_y}" x="0" y="0" width="{b10}" height="{b10}" result="bm10_0"/>',
            f'<feTile in="bm10_0" x="0" y="0" width="{s2_w}" height="{s2_h}" result="bm10"/>',
            f'<feOffset in="SourceGraphic" dx="{-c6x_x}" dy="{-c6x_y}" x="0" y="0" width="{sc(132)}" height="{sc(4)}" result="c6x0"/>',
            f'<feTile in="c6x0" x="0" y="0" width="{c6_w}" height="{s2_h}" result="c6x_t"/>',
            f'<feOffset in="SourceGraphic" dx="{-c6y_x}" dy="{-c6y_y}" x="0" y="0" width="{sc(4)}" height="{c6_h}" result="c6y0"/>',
            f'<feTile in="c6y0" x="0" y="0" width="{c6_w}" height="{c6_h}" result="c6y_t"/>',
        ])

    def _emit_l1_triplet(p: List[str], o: int, dst_xy: Tuple[int, int] | None) -> str:
        trip = m1_trips[o]
        cm0, cm1 = trip["color_matrices"]
        sl, ic = trip["rescale_slopes"], trip["rescale_intercepts"]
        p.extend([
            f'<feColorMatrix in="in_b0" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_list(cm0)}" result="m1p0_{o}"/>',
            f'<feColorMatrix in="in_b1" x="0" y="0" width="{s2_w}" height="{s2_h}" type="matrix" values="{_fmt_list(cm1)}" result="m1p1_{o}"/>',
            f'<feComposite in="m1p0_{o}" in2="m1p1_{o}" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="{nz0_m1}" result="m1a_{o}"/>',
            f'<feComponentTransfer in="m1a_{o}" x="0" y="0" width="{s2_w}" height="{s2_h}" result="m1r_{o}">'
            f'<feFuncR type="linear" slope="{_fmt_num(sl[0])}" intercept="{_fmt_num(ic[0])}"/>'
            f'<feFuncG type="linear" slope="{_fmt_num(sl[1])}" intercept="{_fmt_num(ic[1])}"/>'
            f'<feFuncB type="linear" slope="{_fmt_num(sl[2])}" intercept="{_fmt_num(ic[2])}"/>'
            f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
            f'<feComposite in="m1r_{o}" in2="bm10" x="0" y="0" width="{s2_w}" height="{s2_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="m1m_{o}"/>',
        ])
        for c_off, sfx in enumerate(("r", "g", "b")):
            f = flts1[3 * o + c_off]
            ord_s = f"{int(f['order'][0])} {int(f['order'][1])}"
            p.append(
                f'<feConvolveMatrix in="m1m_{o}" x="0" y="0" width="{s2_w}" height="{s2_h}" '
                f'order="{ord_s}" targetX="{int(f["targetX"])}" targetY="{int(f["targetY"])}" divisor="1" bias="{_fmt_num(f["bias"])}" '
                f'kernelMatrix="{_fmt_list(f["kernelMatrix"])}" edgeMode="none" preserveAlpha="true" result="d1_{o}{sfx}"/>'
            )
        _pack_rgb(p, f"d1_{o}r", f"d1_{o}g", f"d1_{o}b", 0, 0, s2_w, s2_h, f"d1g_{o}")
        p.append(
            f'<feConvolveMatrix in="d1g_{o}" x="0" y="0" width="{s2_w}" height="{s2_h}" '
            f'order="2 2" targetX="0" targetY="0" divisor="{p1_div}" bias="0" '
            f'kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="p1_{o}"/>'
        )
        dx_o = sc(dst_xy[0]) if dst_xy is not None else 0
        dy_o = sc(dst_xy[1]) if dst_xy is not None else 0
        out_nm = f"p1c6_{o}"
        _emit_compact_at_origin(
            p, f"p1_{o}", 0, 0, s2_h, dx_o, dy_o, c6_w, c6_h,
            x_slices_10_6, None, "c6x_t", "c6y_t", a_sh, disp_s, out_nm,
        )
        return out_nm

    p2b: List[str] = []
    _emit_l1_shared(p2b)
    o0 = _emit_l1_triplet(p2b, 0, VRAM_P1_C6_0)
    o1 = _emit_l1_triplet(p2b, 1, VRAM_P1_C6_1)
    _emit_passthrough(p2b, [o0, o1])
    out_filters.append(_wrap_filter(f"ai-s2b-{dpr}x", p2b))

    p2c: List[str] = []
    _emit_l1_shared(p2c)
    o2 = _emit_l1_triplet(p2c, 2, VRAM_P1_C6_2)
    o3_raw = _emit_l1_triplet(p2c, 3, None)
    g3lx, g3ly = sc(VRAM_P1_C6_3L[0]), sc(VRAM_P1_C6_3L[1])
    g3rx, g3ry = sc(VRAM_P1_C6_3R[0]), sc(VRAM_P1_C6_3R[1])
    hw6 = sc(192)
    p2c.extend([
        f'<feOffset in="{o3_raw}" dx="{g3lx}" dy="{g3ly}" x="{g3lx}" y="{g3ly}" width="{hw6}" height="{c6_h}" result="p1c6_3L"/>',
        f'<feOffset in="{o3_raw}" dx="{g3rx - hw6}" dy="{g3ry}" x="{g3rx}" y="{g3ry}" width="{hw6}" height="{c6_h}" result="p1c6_3R"/>',
    ])
    _emit_passthrough(p2c, [o2, "p1c6_3L", "p1c6_3R"])
    out_filters.append(_wrap_filter(f"ai-s2c-{dpr}x", p2c))

    if g1_cnt == 5:
        p2d: List[str] = []
        _emit_l1_shared(p2d)
        o4 = _emit_l1_triplet(p2d, 4, VRAM_P1_C6_4)
        _emit_skip_c2(p2d)
        _emit_passthrough(p2d, [o4, "skip_c2"])
        out_filters.append(_wrap_filter(f"ai-s2d-{dpr}x", p2d))
    else:
        p2e: List[str] = []
        _emit_l1_shared(p2e)
        o4 = _emit_l1_triplet(p2e, 4, VRAM_P1_C6_4)
        o5 = _emit_l1_triplet(p2e, 5, VRAM_P1_C6_5)
        _emit_passthrough(p2e, [o4, o5])
        out_filters.append(_wrap_filter(f"ai-s2e-{dpr}x", p2e))

    # =========================================================================
    # Stages `s3*`: CNN Layer 2 (`mix2 -> bmask6 -> dwconv2 -> pool2 -> C2`)
    # =========================================================================
    c2x_x, c2x_y = sc(LUT_C2X_POS[0]), sc(LUT_C2X_POS[1])
    c2y_x, c2y_y = sc(LUT_C2Y_POS[0]), sc(LUT_C2Y_POS[1])
    x_slices_6_2 = [
        (sc(0), sc(0), sc(132), sc(44)),
        (sc(132), sc(44), sc(132), sc(44)),
        (sc(264), sc(88), sc(120), sc(40)),
    ]
    m2_trips = ly["mix2"]["triplets"]
    flts2 = ly["dwconv2"]["filters"]
    p2_div = _fmt_num(ly["pool2"]["divisor"])
    nz0_m2 = _fmt_num(-float(ly["mix2"]["zero_point"]))
    g2_cnt = len(m2_trips)
    s3_step = 2 if g1_cnt <= 5 else 1

    for s_idx, o0_idx in enumerate(range(0, g2_cnt, s3_step)):
        sfx = chr(ord("a") + s_idx)
        o_pair = range(o0_idx, min(g2_cnt, o0_idx + s3_step))
        p3: List[str] = [
            f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_0[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in0"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_1[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in1"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_2[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in2"/>',
            f'<feOffset in="SourceGraphic" dx="{-g3lx}" dy="{-g3ly}" x="0" y="0" width="{hw6}" height="{c6_h}" result="g3L"/>',
            f'<feOffset in="SourceGraphic" dx="{hw6 - g3rx}" dy="{-g3ry}" x="{hw6}" y="0" width="{hw6}" height="{c6_h}" result="g3R"/>',
            f'<feMerge x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in3"><feMergeNode in="g3L"/><feMergeNode in="g3R"/></feMerge>',
            f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in4"/>',
        ]
        if g1_cnt == 6:
            p3.append(
                f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_5[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in5"/>'
            )
        p3.extend([
            f'<feOffset in="SourceGraphic" dx="{-bm6_x}" dy="{-bm6_y}" x="0" y="0" width="{b6}" height="{b6}" result="bm6_0"/>',
            f'<feTile in="bm6_0" x="0" y="0" width="{c6_w}" height="{c6_h}" result="bm6"/>',
            f'<feOffset in="SourceGraphic" dx="{-c2x_x}" dy="{-c2x_y}" x="0" y="0" width="{sc(44)}" height="{sc(4)}" result="c2x0"/>',
            f'<feTile in="c2x0" x="0" y="0" width="{c2_w}" height="{c6_h}" result="c2x_t"/>',
            f'<feOffset in="SourceGraphic" dx="{-c2y_x}" dy="{-c2y_y}" x="0" y="0" width="{sc(4)}" height="{c2_h}" result="c2y0"/>',
            f'<feTile in="c2y0" x="0" y="0" width="{c2_w}" height="{c2_h}" result="c2y_t"/>',
        ])
        s3_outs: List[str] = []
        for o in o_pair:
            trip = m2_trips[o]
            acc = ""
            for g, mat20 in enumerate(trip["color_matrices"]):
                p3.append(f'<feColorMatrix in="p1_in{g}" x="0" y="0" width="{c6_w}" height="{c6_h}" type="matrix" values="{_fmt_list(mat20)}" result="m2p_{o}_{g}"/>')
                if g == 0:
                    acc = f"m2p_{o}_{g}"
                else:
                    nxt = f"m2a_{o}_{g}"
                    p3.append(f'<feComposite in="{acc}" in2="m2p_{o}_{g}" x="0" y="0" width="{c6_w}" height="{c6_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="{nz0_m2}" result="{nxt}"/>')
                    acc = nxt
            sl, ic = trip["rescale_slopes"], trip["rescale_intercepts"]
            p3.extend([
                f'<feComponentTransfer in="{acc}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="m2r_{o}">'
                f'<feFuncR type="linear" slope="{_fmt_num(sl[0])}" intercept="{_fmt_num(ic[0])}"/>'
                f'<feFuncG type="linear" slope="{_fmt_num(sl[1])}" intercept="{_fmt_num(ic[1])}"/>'
                f'<feFuncB type="linear" slope="{_fmt_num(sl[2])}" intercept="{_fmt_num(ic[2])}"/>'
                f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
                f'<feComposite in="m2r_{o}" in2="bm6" x="0" y="0" width="{c6_w}" height="{c6_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="m2m_{o}"/>',
            ])
            for c_off, csfx in enumerate(("r", "g", "b")):
                f = flts2[3 * o + c_off]
                ord2_s = f"{int(f['order'][0])} {int(f['order'][1])}"
                p3.append(
                    f'<feConvolveMatrix in="m2m_{o}" x="0" y="0" width="{c6_w}" height="{c6_h}" '
                    f'order="{ord2_s}" targetX="{int(f["targetX"])}" targetY="{int(f["targetY"])}" divisor="1" bias="{_fmt_num(f["bias"])}" '
                    f'kernelMatrix="{_fmt_list(f["kernelMatrix"])}" edgeMode="none" preserveAlpha="true" result="d2_{o}{csfx}"/>'
                )
            _pack_rgb(p3, f"d2_{o}r", f"d2_{o}g", f"d2_{o}b", 0, 0, c6_w, c6_h, f"d2g_{o}")
            p3.append(
                f'<feConvolveMatrix in="d2g_{o}" x="0" y="0" width="{c6_w}" height="{c6_h}" '
                f'order="2 2" targetX="0" targetY="0" divisor="{p2_div}" bias="0" '
                f'kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="p2_{o}"/>'
            )
            dx_o, dy_o = sc(VRAM_P2_C2[o][0]), sc(VRAM_P2_C2[o][1])
            out_nm = f"p2c2_{o}"
            _emit_compact_at_origin(
                p3, f"p2_{o}", 0, 0, c6_h, dx_o, dy_o, c2_w, c2_h,
                x_slices_6_2, None, "c2x_t", "c2y_t", a_sh, disp_s, out_nm,
            )
            s3_outs.append(out_nm)
        _emit_passthrough(p3, s3_outs)
        out_filters.append(_wrap_filter(f"ai-s3{sfx}-{dpr}x", p3))

    # =========================================================================
    # Stages `s4m*`: CNN Layer 3a `mix3` (`G2 -> G3` triplets on `128 x 32`)
    # =========================================================================
    m3_trips = ly["mix3"]["triplets"]
    nz0_m3 = _fmt_num(-float(ly["mix3"]["zero_point"]))
    g3_cnt = len(m3_trips)
    m3_step = 4 if g2_cnt <= 6 else 3

    for m_idx, o0_idx in enumerate(range(0, g3_cnt, m3_step)):
        o_range = range(o0_idx, min(g3_cnt, o0_idx + m3_step))
        p4ab: List[str] = [
            f'<feOffset in="SourceGraphic" dx="{-sc(VRAM_P2_C2[g][0])}" dy="{-sc(VRAM_P2_C2[g][1])}" x="0" y="0" width="{c2_w}" height="{c2_h}" result="p2_in{g}"/>'
            for g in range(g2_cnt)
        ]
        s4_outs: List[str] = []
        for o in o_range:
            trip = m3_trips[o]
            acc = ""
            for g, mat20 in enumerate(trip["color_matrices"]):
                p4ab.append(f'<feColorMatrix in="p2_in{g}" x="0" y="0" width="{c2_w}" height="{c2_h}" type="matrix" values="{_fmt_list(mat20)}" result="m3p_{o}_{g}"/>')
                if g == 0:
                    acc = f"m3p_{o}_{g}"
                else:
                    nxt = f"m3a_{o}_{g}"
                    p4ab.append(f'<feComposite in="{acc}" in2="m3p_{o}_{g}" x="0" y="0" width="{c2_w}" height="{c2_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="{nz0_m3}" result="{nxt}"/>')
                    acc = nxt
            sl, ic = trip["rescale_slopes"], trip["rescale_intercepts"]
            dx_o, dy_o = sc(VRAM_MIX3[o][0]), sc(VRAM_MIX3[o][1])
            raw_nm = f"m3g_{o}" if (dx_o == 0 and dy_o == 0) else f"m3r_{o}"
            p4ab.append(
                f'<feComponentTransfer in="{acc}" x="0" y="0" width="{c2_w}" height="{c2_h}" result="{raw_nm}">'
                f'<feFuncR type="linear" slope="{_fmt_num(sl[0])}" intercept="{_fmt_num(ic[0])}"/>'
                f'<feFuncG type="linear" slope="{_fmt_num(sl[1])}" intercept="{_fmt_num(ic[1])}"/>'
                f'<feFuncB type="linear" slope="{_fmt_num(sl[2])}" intercept="{_fmt_num(ic[2])}"/>'
                f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>'
            )
            if dx_o != 0 or dy_o != 0:
                p4ab.append(f'<feOffset in="{raw_nm}" dx="{dx_o}" dy="{dy_o}" x="{dx_o}" y="{dy_o}" width="{c2_w}" height="{c2_h}" result="m3g_{o}"/>')
            s4_outs.append(f"m3g_{o}")
        _emit_passthrough(p4ab, s4_outs)
        out_filters.append(_wrap_filter(f"ai-s4m{m_idx}-{dpr}x", p4ab))

    # =========================================================================
    # Stages `s4d*`: CNN Layer 3b `dwconv3` (`2x2 -> 1x1`) + `META_C1` & `SKIP_C1`
    # =========================================================================
    c1_w, c1_h = sc(64), sc(16)
    c1x_x, c1x_y = sc(LUT_C1X_POS[0]), sc(LUT_C1X_POS[1])
    c1y_x, c1y_y = sc(LUT_C1Y_POS[0]), sc(LUT_C1Y_POS[1])
    x_slices_2_1 = [(sc(0), sc(0), c2_w, c1_w)]
    flts3 = ly["dwconv3"]["filters"]

    for d_idx, g0_idx in enumerate(range(0, g3_cnt, 4)):
        g_range = range(g0_idx, min(g3_cnt, g0_idx + 4))
        p4cd: List[str] = [
            f'<feOffset in="SourceGraphic" dx="{-c1x_x}" dy="{-c1x_y}" x="0" y="0" width="{c1_w}" height="{sc(4)}" result="c1x0"/>',
            f'<feTile in="c1x0" x="0" y="0" width="{c1_w}" height="{c2_h}" result="c1x_t"/>',
            f'<feOffset in="SourceGraphic" dx="{-c1y_x}" dy="{-c1y_y}" x="0" y="0" width="{sc(4)}" height="{c1_h}" result="c1y0"/>',
            f'<feTile in="c1y0" x="0" y="0" width="{c1_w}" height="{c1_h}" result="c1y_t"/>',
        ]
        s4_outs = []
        for g in g_range:
            mx, my = sc(VRAM_MIX3[g][0]), sc(VRAM_MIX3[g][1])
            p4cd.append(f'<feOffset in="SourceGraphic" dx="{-mx}" dy="{-my}" x="0" y="0" width="{c2_w}" height="{c2_h}" result="m3i_{g}"/>')
            for c_off, csfx in enumerate(("r", "g", "b")):
                f = flts3[3 * g + c_off]
                p4cd.append(
                    f'<feConvolveMatrix in="m3i_{g}" x="0" y="0" width="{c2_w}" height="{c2_h}" '
                    f'order="2 2" targetX="0" targetY="0" divisor="1" bias="{_fmt_num(f["bias"])}" '
                    f'kernelMatrix="{_fmt_list(f["kernelMatrix"])}" edgeMode="none" preserveAlpha="true" result="d3_{g}{csfx}"/>'
                )
            _pack_rgb(p4cd, f"d3_{g}r", f"d3_{g}g", f"d3_{g}b", 0, 0, c2_w, c2_h, f"d3g_{g}")
            dx_g, dy_g = sc(VRAM_DW3_C1[g][0]), sc(VRAM_DW3_C1[g][1])
            out_nm = f"dw3c1_{g}"
            _emit_compact_at_origin(
                p4cd, f"d3g_{g}", 0, 0, c2_h, dx_g, dy_g, c1_w, c1_h,
                x_slices_2_1, None, "c1x_t", "c1y_t", a_sh, disp_s, out_nm,
            )
            s4_outs.append(out_nm)

        if d_idx == 0:
            nm_dx, nm_dy = sc(VRAM_NOM2_C1[0]), sc(VRAM_NOM2_C1[1])
            _emit_compact_at_origin(
                p4cd, "SourceGraphic", mc2_x, mc2_y, c2_h, nm_dx, nm_dy, c1_w, c1_h,
                x_slices_2_1, None, "c1x_t", "c1y_t", a_sh, disp_s, "nom2_c1",
            )
            s4_outs.append("nom2_c1")
        elif d_idx == 1:
            p4cd.extend([
                f'<feOffset in="SourceGraphic" dx="{-skc2_x}" dy="{-skc2_y}" x="0" y="0" width="{c2_w}" height="{c2_h}" result="sk_in2"/>',
                f'<feConvolveMatrix in="sk_in2" x="0" y="0" width="{c2_w}" height="{c2_h}" '
                f'order="2 2" targetX="0" targetY="0" divisor="{_fmt_num(sk_st2["divisor"])}" bias="{_fmt_num(sk_st2["bias"])}" '
                f'kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="sk_1"/>',
            ])
            sk1_dx, sk1_dy = sc(VRAM_SKIP_C1[0]), sc(VRAM_SKIP_C1[1])
            _emit_compact_at_origin(
                p4cd, "sk_1", 0, 0, c2_h, sk1_dx, sk1_dy, c1_w, c1_h,
                x_slices_2_1, None, "c1x_t", "c1y_t", a_sh, disp_s, "skip_c1",
            )
            s4_outs.append("skip_c1")

        _emit_passthrough(p4cd, s4_outs)
        out_filters.append(_wrap_filter(f"ai-s4d{d_idx}-{dpr}x", p4cd))

    # =========================================================================
    # Stage `s5v`: Value Head (`(3*G3 + 3) -> 1`) + Sigmoid LUT -> `VRAM_VSHEET`
    # Stage `s5a`: 2-Ply Minimax (`erode` MIN + `dilate` MAX) -> `QSTRIP`, `WINMASK`
    # =========================================================================
    vh = ly["value_head"]
    nz0_vh = _fmt_num(-float(vh["zero_point"]))
    sig_tbl = " ".join(_ct_u8(int(round(float(v) * 255.0))) for v in vh["sigmoid_table_values"])
    nm1_x, nm1_y = sc(VRAM_NOM2_C1[0]), sc(VRAM_NOM2_C1[1])
    vsh_x, vsh_y = sc(VRAM_VSHEET[0]), sc(VRAM_VSHEET[1])
    qs_x, qs_y = sc(VRAM_QSTRIP[0]), sc(VRAM_QSTRIP[1])
    wm_x, wm_y = sc(VRAM_WINMASK[0]), sc(VRAM_WINMASK[1])
    tie_x, tie_y = sc(LUT_TIE_POS[0]), sc(LUT_TIE_POS[1])
    h2 = sc(2)
    b8 = sc(8)
    bd8_x, bd8_y = sc(VRAM_BOARD8[0]), sc(VRAM_BOARD8[1])
    w1_x, w1_y = sc(VRAM_WIN1PX[0]), sc(VRAM_WIN1PX[1])
    sq8w_x, sq8w_y = sc(LUT_SQ8W_POS[0]), sc(LUT_SQ8W_POS[1])
    sq8b_x, sq8b_y = sc(LUT_SQ8B_POS[0]), sc(LUT_SQ8B_POS[1])
    chk8_x, chk8_y = sc(LUT_CHK8_POS[0]), sc(LUT_CHK8_POS[1])

    feat_slots = list(VRAM_DW3_C1[:g3_cnt]) + [VRAM_SKIP_C1]
    p5v: List[str] = []
    vh_acc = ""
    for g, slot in enumerate(feat_slots):
        fx, fy = sc(slot[0]), sc(slot[1])
        mat20 = vh["color_matrices"][g]
        p5v.append(f'<feOffset in="SourceGraphic" dx="{-fx}" dy="{-fy}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="f_{g}"/>')
        p5v.append(f'<feColorMatrix in="f_{g}" x="0" y="0" width="{c1_w}" height="{c1_h}" type="matrix" values="{_fmt_list(mat20)}" result="vp_{g}"/>')
        if g == 0:
            vh_acc = "vp_0"
        else:
            nxt = f"va_{g}"
            p5v.append(f'<feComposite in="{vh_acc}" in2="vp_{g}" x="0" y="0" width="{c1_w}" height="{c1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="{nz0_vh}" result="{nxt}"/>')
            vh_acc = nxt

    p5v.extend([
        f'<feComponentTransfer in="{vh_acc}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="v_sig">'
        f'<feFuncR type="table" tableValues="{sig_tbl}"/><feFuncG type="table" tableValues="{sig_tbl}"/><feFuncB type="table" tableValues="{sig_tbl}"/>'
        f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        # Override Opponent-Checkmate leaves (`nm1.B == 1`) to `2/255`, then force `no_m2` (`nm1.R == 1`) to `255/255`:
        f'<feOffset in="SourceGraphic" dx="{-nm1_x}" dy="{-nm1_y}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="nm1"/>',
        f'<feColorMatrix in="nm1" x="0" y="0" width="{c1_w}" height="{c1_h}" type="matrix" values="0 0 -1 0 1 0 0 -1 0 1 0 0 -1 0 1 0 0 0 0 1" result="not_mt"/>',
        f'<feComposite in="v_sig" in2="not_mt" x="0" y="0" width="{c1_w}" height="{c1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="v_m1"/>',
        f'<feColorMatrix in="nm1" x="0" y="0" width="{c1_w}" height="{c1_h}" type="matrix" values="0 0 {_fmt_num(2/255)} 0 0 0 0 {_fmt_num(2/255)} 0 0 0 0 {_fmt_num(2/255)} 0 0 0 0 0 0 1" result="mt_flr"/>',
        f'<feComposite in="v_m1" in2="mt_flr" x="0" y="0" width="{c1_w}" height="{c1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="v_m2"/>',
        f'<feColorMatrix in="nm1" x="0" y="0" width="{c1_w}" height="{c1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="nm_rgb"/>',
        f'<feComposite in="v_m2" in2="nm_rgb" x="0" y="0" width="{c1_w}" height="{c1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="v_sh0"/>',
        f'<feOffset in="v_sh0" dx="{vsh_x}" dy="{vsh_y}" x="{vsh_x}" y="{vsh_y}" width="{c1_w}" height="{c1_h}" result="v_sheet"/>',
    ])
    _emit_passthrough(p5v, ["v_sheet"])
    out_filters.append(_wrap_filter(f"ai-s5v-{dpr}x", p5v))

    p5a: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-vsh_x}" dy="{-vsh_y}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="v_sh0"/>',
        f'<feOffset in="SourceGraphic" dx="{-nm1_x}" dy="{-nm1_y}" x="0" y="0" width="{c1_w}" height="{h2}" result="nm1"/>',
        # Periodic 3x vertical tile (`height = 3 * c1_h`) -> exact 16-row `erode` with ZERO boundary bleed!
        f'<feTile in="v_sh0" x="0" y="0" width="{c1_w}" height="{3 * c1_h}" result="v_t3"/>',
        f'<feMorphology in="v_t3" x="0" y="0" width="{c1_w}" height="{3 * c1_h}" operator="erode" radius="0 {_fmt_num(8 * u)}" result="v_er"/>',
        f'<feOffset in="v_er" dx="0" dy="{-c1_h}" x="0" y="0" width="{c1_w}" height="{h2}" result="cmin"/>',
        # Checkmate (`254/255`) vs Stalemate (`128/255`) when `cmin == 255/255` (`all_no_m2`):
        f'<feComponentTransfer in="cmin" x="0" y="0" width="{c1_w}" height="{h2}" result="all_nm"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="all_nm" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="has_any_m2"/>',
        f'<feColorMatrix in="nm1" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="0 {_fmt_num(126/255)} 0 0 {z0_s} 0 {_fmt_num(126/255)} 0 0 {z0_s} 0 {_fmt_num(126/255)} 0 0 {z0_s} 0 0 0 0 1" result="term_v"/>',
        f'<feComposite in="cmin" in2="has_any_m2" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q_live"/>',
        f'<feComposite in="term_v" in2="all_nm" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q_term"/>',
        f'<feComposite in="q_live" in2="q_term" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="q_unm"/>',
        # Gate by `valid_m1` (`SRC_M1CC_PHYS > 128`) AND `(1 - is_terminal)` (`SRC_META_TERM_PHYS == 128`):
        f'<feOffset in="SourceGraphic" dx="{-fo_m1ccx}" dy="{-fo_m1ccy}" x="0" y="0" width="{c1_w}" height="{h2}" result="m1c_b"/>',
        f'<feComponentTransfer in="m1c_b" x="0" y="0" width="{c1_w}" height="{h2}" result="val_m10"><feFuncR type="table" tableValues="{tbl_gt128}"/><feFuncG type="table" tableValues="{tbl_gt128}"/><feFuncB type="table" tableValues="{tbl_gt128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="{-fo_mt_mx}" dy="{-fo_mt_my}" x="0" y="0" width="{h2}" height="{h2}" result="mt0"/>',
        f'<feTile in="mt0" x="0" y="0" width="{c1_w}" height="{h2}" result="mt_b"/>',
        f'<feComponentTransfer in="mt_b" x="0" y="0" width="{c1_w}" height="{h2}" result="not_term"><feFuncR type="table" tableValues="{tbl_eq128}"/><feFuncG type="table" tableValues="{tbl_eq128}"/><feFuncB type="table" tableValues="{tbl_eq128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="val_m10" in2="not_term" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="val_m1"/>',
        f'<feComposite in="q_unm" in2="val_m1" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="q_0"/>',
        f'<feOffset in="q_0" dx="{qs_x}" dy="{qs_y}" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" result="q_strip"/>',
        # Periodic 3x horizontal tile -> exact 64-column `dilate` + unique `tie_ramp` argmax!
        f'<feTile in="q_0" x="0" y="0" width="{3 * c1_w}" height="{h2}" result="q_t3"/>',
        f'<feMorphology in="q_t3" x="0" y="0" width="{3 * c1_w}" height="{h2}" operator="dilate" radius="{_fmt_num(32 * u)} 0" result="q_dl"/>',
        f'<feOffset in="q_dl" dx="{-c1_w}" dy="0" x="0" y="0" width="{c1_w}" height="{h2}" result="q_max"/>',
        f'<feBlend in="q_0" in2="q_max" x="0" y="0" width="{c1_w}" height="{h2}" mode="difference" result="dq"/>',
        f'<feComponentTransfer in="dq" x="0" y="0" width="{c1_w}" height="{h2}" result="eq_qm"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="eq_qm" in2="val_m1" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="is_tied"/>',
        f'<feOffset in="SourceGraphic" dx="{-tie_x}" dy="{-tie_y}" x="0" y="0" width="{c1_w}" height="{h2}" result="tie0"/>',
        f'<feComposite in="tie0" in2="is_tied" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="t_tied"/>',
        f'<feTile in="t_tied" x="0" y="0" width="{3 * c1_w}" height="{h2}" result="tt_3"/>',
        f'<feMorphology in="tt_3" x="0" y="0" width="{3 * c1_w}" height="{h2}" operator="dilate" radius="{_fmt_num(32 * u)} 0" result="tt_dl"/>',
        f'<feOffset in="tt_dl" dx="{-c1_w}" dy="0" x="0" y="0" width="{c1_w}" height="{h2}" result="t_max"/>',
        f'<feBlend in="t_tied" in2="t_max" x="0" y="0" width="{c1_w}" height="{h2}" mode="difference" result="dtie"/>',
        f'<feComponentTransfer in="dtie" x="0" y="0" width="{c1_w}" height="{h2}" result="eq_tm"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="eq_tm" in2="is_tied" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="wm_0"/>',
        f'<feOffset in="wm_0" dx="{wm_x}" dy="{wm_y}" x="{wm_x}" y="{wm_y}" width="{c1_w}" height="{h2}" result="win_mask"/>',
        # Fused `s5b`: Extract `VRAM_WIN1PX` + Render `8x8` RGB Move Map (`VRAM_BOARD8`) in the same stage!
        f'<feOffset in="SourceGraphic" dx="{-fo_m1cux}" dy="{-fo_m1cuy}" x="0" y="0" width="{c1_w}" height="{h2}" result="m1cu"/>',
        f'<feOffset in="SourceGraphic" dx="{-fo_m1cvx}" dy="{-fo_m1cvy}" x="0" y="0" width="{c1_w}" height="{h2}" result="m1cv"/>',
        f'<feColorMatrix in="m1cu" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="u_r"/>',
        f'<feColorMatrix in="m1cv" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="v_g"/>',
        f'<feComposite in="u_r" in2="v_g" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="uv_rg"/>',
        f'<feColorMatrix in="q_0" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="q_b"/>',
        f'<feComposite in="uv_rg" in2="q_b" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="uvq"/>',
        f'<feComposite in="uvq" in2="wm_0" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="uvq_w"/>',
        f'<feTile in="uvq_w" x="0" y="0" width="{3 * c1_w}" height="{h2}" result="uw_t3"/>',
        f'<feMorphology in="uw_t3" x="0" y="0" width="{3 * c1_w}" height="{h2}" operator="dilate" radius="{_fmt_num(32 * u)} 0" result="uw_dl"/>',
        f'<feOffset in="uw_dl" dx="{-c1_w}" dy="0" x="0" y="0" width="{h2}" height="{h2}" result="w1_0"/>',
        f'<feOffset in="w1_0" dx="{w1_x}" dy="{w1_y}" x="{w1_x}" y="{w1_y}" width="{h2}" height="{h2}" result="win_1px"/>',
        f'<feTile in="w1_0" x="0" y="0" width="{b8}" height="{b8}" result="wt8"/>',
        f'<feOffset in="SourceGraphic" dx="{-fo_mt_tx}" dy="{-fo_mt_ty}" x="0" y="0" width="{h2}" height="{h2}" result="mt_t0"/>',
        f'<feTile in="mt_t0" x="0" y="0" width="{b8}" height="{b8}" result="mt8"/>',
        f'<feComponentTransfer in="mt8" x="0" y="0" width="{b8}" height="{b8}" result="is_blk"><feFuncR type="table" tableValues="{tbl_eq255}"/><feFuncG type="table" tableValues="{tbl_eq255}"/><feFuncB type="table" tableValues="{tbl_eq255}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="is_blk" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="is_wht"/>',
        f'<feOffset in="SourceGraphic" dx="{-sq8w_x}" dy="{-sq8w_y}" x="0" y="0" width="{b8}" height="{b8}" result="sqw"/>',
        f'<feOffset in="SourceGraphic" dx="{-sq8b_x}" dy="{-sq8b_y}" x="0" y="0" width="{b8}" height="{b8}" result="sqb"/>',
        f'<feComposite in="sqw" in2="is_wht" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sq_wm"/>',
        f'<feComposite in="sqb" in2="is_blk" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sq_bm"/>',
        f'<feComposite in="sq_wm" in2="sq_bm" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sq8"/>',
        f'<feColorMatrix in="wt8" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="u1s"/>',
        f'<feColorMatrix in="wt8" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="v1s"/>',
        f'<feBlend in="sq8" in2="u1s" x="0" y="0" width="{b8}" height="{b8}" mode="difference" result="du_8"/>',
        f'<feComponentTransfer in="du_8" x="0" y="0" width="{b8}" height="{b8}" result="m_from"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feBlend in="sq8" in2="v1s" x="0" y="0" width="{b8}" height="{b8}" mode="difference" result="dv_8"/>',
        f'<feComponentTransfer in="dv_8" x="0" y="0" width="{b8}" height="{b8}" result="m_to"><feFuncR type="table" tableValues="{tbl_eq0}"/><feFuncG type="table" tableValues="{tbl_eq0}"/><feFuncB type="table" tableValues="{tbl_eq0}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="{-chk8_x}" dy="{-chk8_y}" x="0" y="0" width="{b8}" height="{b8}" result="chk0"/>',
        f'<feColorMatrix in="chk0" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="0.16 0 0 0 -0.02 0 0.22 0 0 -0.02 0 0 0.33 0 -0.01 0 0 0 0 1" result="bg8"/>',
        f'<feComposite in="m_from" in2="m_to" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="m_ft"/>',
        f'<feColorMatrix in="m_ft" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="m_bg"/>',
        f'<feComposite in="bg8" in2="m_bg" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bd_bg"/>',
        f'<feColorMatrix in="m_from" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="0.659 0 0 0 0 0 0.333 0 0 0 0 0 0.969 0 0 0 0 0 0 1" result="c_from"/>',
        f'<feColorMatrix in="m_to" x="0" y="0" width="{b8}" height="{b8}" type="matrix" values="0.063 0 0 0 0 0 0.725 0 0 0 0 0 0.506 0 0 0 0 0 0 1" result="c_to"/>',
        f'<feComposite in="bd_bg" in2="c_from" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bd_1"/>',
        f'<feComposite in="bd_1" in2="c_to" x="0" y="0" width="{b8}" height="{b8}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bd_2"/>',
        f'<feOffset in="bd_2" dx="{bd8_x}" dy="{bd8_y}" x="{bd8_x}" y="{bd8_y}" width="{b8}" height="{b8}" result="board8"/>',
    ]
    _emit_passthrough(p5a, ["q_strip", "win_mask", "win_1px", "board8"])
    out_filters.append(_wrap_filter(f"ai-s5a-{dpr}x", p5a))

    # =========================================================================
    # Stage `visa`: Visual Zoom of `8x8` Move Map + `64x16` Minimax Value Sheet + `Q(v1)` Bar
    #               (Preserves VRAM `x=540..664` + bottom bar LUT for `visb`!)
    # =========================================================================
    vbx_x, vbx_y = LUT_VIS_BX_POS
    vby_x, vby_y = LUT_VIS_BY_POS
    vmx_x, vmx_y = LUT_VIS_MX_POS
    vmy_x, vmy_y = LUT_VIS_MY_POS
    vfy_x, vfy_y = LUT_VIS_FY_POS
    bar_lx, bar_ly = LUT_BAR_POS

    # Colorize `V_CNN` (`[3..253]` -> Crimson `<128` -> Slate `128` -> Emerald `>128`, `254`=Gold, `255`=Dark Navy `#070d1a`!):
    hm_r, hm_g, hm_b = [], [], []
    for c in range(256):
        if c == 255:
            r, g, b = 0.035, 0.055, 0.110
        elif c == 254:
            r, g, b = 0.980, 0.800, 0.080
        elif c == 2:
            r, g, b = 0.960, 0.100, 0.240
        elif c == 0:
            r, g, b = 0.030, 0.045, 0.085
        else:
            p_val = np.clip((c - 3.0) / 250.0, 0.0, 1.0)
            if p_val < 0.5:
                t = p_val / 0.5
                r = 0.88 * (1 - t) + 0.16 * t
                g = 0.18 * (1 - t) + 0.23 * t
                b = 0.30 * (1 - t) + 0.35 * t
            else:
                t = (p_val - 0.5) / 0.5
                r = 0.16 * (1 - t) + 0.06 * t
                g = 0.23 * (1 - t) + 0.80 * t
                b = 0.35 * (1 - t) + 0.52 * t
        hm_r.append(_fmt_num(r))
        hm_g.append(_fmt_num(g))
        hm_b.append(_fmt_num(b))
    tbl_hm_r, tbl_hm_g, tbl_hm_b = " ".join(hm_r), " ".join(hm_g), " ".join(hm_b)

    sk1_x, sk1_y = sc(VRAM_SKIP_C1[0]), sc(VRAM_SKIP_C1[1])
    best_d3 = best_dwconv3_triplet(spec)
    d37_x, d37_y = sc(VRAM_DW3_C1[best_d3][0]), sc(VRAM_DW3_C1[best_d3][1])

    tbl_q_to_prob = " ".join(
        "0" if c == 0 else ("1" if c >= 253 else _fmt_num(max(0.0, (c - 3.0) / 250.0))) for c in range(256)
    )

    pva2: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-vmx_x}" dy="{-vmx_y}" x="0" y="0" width="{VIS_FEAT_W}" height="4" result="vmx"/>',
        f'<feOffset in="SourceGraphic" dx="{-vfy_x}" dy="{-vfy_y}" x="0" y="0" width="4" height="{VIS_FEAT_H}" result="vfy"/>',
        f'<feFlood x="{VIS_WINBAR_X - 1}" y="{VIS_WINBAR_Y - 1}" width="{VIS_WINBAR_W + 2}" height="{VIS_WINBAR_H + 2}" flood-color="#1e293b" flood-opacity="1" result="wb_bdr"/>',
        f'<feFlood x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" flood-color="#09101f" flood-opacity="1" result="wb_trk"/>',
        f'<feOffset in="SourceGraphic" dx="{VIS_WINBAR_X - w1_x}" dy="{VIS_WINBAR_Y - w1_y}" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="1" height="1" result="w1_1"/>',
        f'<feTile in="w1_1" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" result="w1_t"/>',
        f'<feColorMatrix in="w1_t" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="qmax_t"/>',
        f'<feComponentTransfer in="qmax_t" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" result="prob_t"><feFuncR type="table" tableValues="{tbl_q_to_prob}"/><feFuncG type="table" tableValues="{tbl_q_to_prob}"/><feFuncB type="table" tableValues="{tbl_q_to_prob}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="{VIS_WINBAR_X - bar_lx}" dy="{VIS_WINBAR_Y - bar_ly}" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="1" result="br_1"/>',
        f'<feTile in="br_1" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" result="br_t"/>',
        f'<feComposite in="prob_t" in2="br_t" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" operator="arithmetic" k1="0" k2="180" k3="361.42" k4="-361.32" result="wb_m"/>',
        f'<feColorMatrix in="wb_m" x="{VIS_WINBAR_X}" y="{VIS_WINBAR_Y}" width="{VIS_WINBAR_W}" height="{VIS_WINBAR_H}" type="matrix" values="0.063 0 0 0 0.035 0 0.725 0 0 0.063 0 0 0.506 0 0.122 0 0 0 0 1" result="wb_fill"/>',
        f'<feFlood x="{VIS_FEAT_X - 1}" y="{VIS_FEAT1_Y - 1}" width="{VIS_FEAT_W + 2}" height="{VIS_FEAT_H + 2}" flood-color="#1e293b" flood-opacity="1" result="f1_bdr"/>',
        f'<feFlood x="{VIS_FEAT_X - 1}" y="{VIS_FEAT2_Y - 1}" width="{VIS_FEAT_W + 2}" height="{VIS_FEAT_H + 2}" flood-color="#1e293b" flood-opacity="1" result="f2_bdr"/>',
        f'<feComponentTransfer in="SourceGraphic" x="{sk1_x}" y="{sk1_y}" width="{c1_w}" height="{c1_h}" result="sk_vis"><feFuncR type="linear" slope="4" intercept="-1.5"/><feFuncG type="linear" slope="4" intercept="-1.5"/><feFuncB type="linear" slope="4" intercept="-1.5"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="SourceGraphic" x="{d37_x}" y="{d37_y}" width="{c1_w}" height="{c1_h}" result="d3_vis"><feFuncR type="linear" slope="3.2" intercept="0.03"/><feFuncG type="linear" slope="3.6" intercept="0.03"/><feFuncB type="linear" slope="3.6" intercept="0.05"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
    ]
    _emit_periodic_zoom_rect(
        pva2, "sk_vis", sk1_x, sk1_y, c1_w, c1_h,
        VIS_FEAT_X, VIS_FEAT1_Y, VIS_FEAT_W, VIS_FEAT_H,
        "vmx", "vfy", a_sh, disp_s, "vis_f1",
    )
    _emit_periodic_zoom_rect(
        pva2, "d3_vis", d37_x, d37_y, c1_w, c1_h,
        VIS_FEAT_X, VIS_FEAT2_Y, VIS_FEAT_W, VIS_FEAT_H,
        "vmx", "vfy", a_sh, disp_s, "vis_f2",
    )
    _emit_passthrough(pva2, ["wb_bdr", "wb_trk", "wb_fill", "f1_bdr", "vis_f1", "f2_bdr", "vis_f2"], keep_vram=True)
    out_filters.append(_wrap_filter(f"ai-visa-{dpr}x", pva2, w=STAGE_W, h=STAGE_H))

    pvb2: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-vbx_x}" dy="{-vbx_y}" x="0" y="0" width="{VIS_BOARD_SZ}" height="4" result="vbx"/>',
        f'<feOffset in="SourceGraphic" dx="{-vby_x}" dy="{-vby_y}" x="0" y="0" width="4" height="{VIS_BOARD_SZ}" result="vby"/>',
        f'<feOffset in="SourceGraphic" dx="{-vmx_x}" dy="{-vmx_y}" x="0" y="0" width="{VIS_MAT_W}" height="4" result="vmx"/>',
        f'<feOffset in="SourceGraphic" dx="{-vmy_x}" dy="{-vmy_y}" x="0" y="0" width="4" height="{VIS_MAT_H}" result="vmy"/>',
        f'<feFlood x="{VIS_BOARD_X - 1}" y="{VIS_BOARD_Y - 1}" width="{VIS_BOARD_SZ + 2}" height="{VIS_BOARD_SZ + 2}" flood-color="#334155" flood-opacity="1" result="bd_bdr"/>',
        f'<feFlood x="{VIS_MAT_X - 1}" y="{VIS_MAT_Y - 1}" width="{VIS_MAT_W + 2}" height="{VIS_MAT_H + 2}" flood-color="#334155" flood-opacity="1" result="mt_bdr"/>',
        f'<feFlood x="{VIS_MAT_X - 1}" y="{VIS_QBAR_Y - 1}" width="{VIS_MAT_W + 2}" height="{VIS_QBAR_H + 2}" flood-color="#334155" flood-opacity="1" result="qb_bdr"/>',
    ]
    _emit_periodic_zoom_rect(
        pvb2, "SourceGraphic", bd8_x, bd8_y, b8, b8,
        VIS_BOARD_X, VIS_BOARD_Y, VIS_BOARD_SZ, VIS_BOARD_SZ,
        "vbx", "vby", a_sh, disp_s, "vis_bd",
    )
    pvb2.append(
        f'<feComponentTransfer in="SourceGraphic" x="{vsh_x}" y="{vsh_y}" width="{c1_w}" height="{c1_h}" result="vsh_col">'
        f'<feFuncR type="table" tableValues="{tbl_hm_r}"/><feFuncG type="table" tableValues="{tbl_hm_g}"/><feFuncB type="table" tableValues="{tbl_hm_b}"/>'
        f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>'
    )
    _emit_periodic_zoom_rect(
        pvb2, "vsh_col", vsh_x, vsh_y, c1_w, c1_h,
        VIS_MAT_X, VIS_MAT_Y, VIS_MAT_W, VIS_MAT_H,
        "vmx", "vmy", a_sh, disp_s, "vis_mat",
    )
    pvb2.extend([
        f'<feComponentTransfer in="SourceGraphic" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" result="qs_col">'
        f'<feFuncR type="table" tableValues="{tbl_hm_r}"/><feFuncG type="table" tableValues="{tbl_hm_g}"/><feFuncB type="table" tableValues="{tbl_hm_b}"/>'
        f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="{qs_x - wm_x}" dy="{qs_y - wm_y}" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" result="wm_s"/>',
        f'<feColorMatrix in="wm_s" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="wm_inv"/>',
        f'<feComposite in="qs_col" in2="wm_inv" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="qs_k"/>',
        f'<feColorMatrix in="wm_s" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" type="matrix" values="0.984 0 0 0 0 0 0.749 0 0 0 0 0 0.141 0 0 0 0 0 0 1" result="wm_gold"/>',
        f'<feComposite in="qs_k" in2="wm_gold" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="qbar_src"/>',
        f'<feTile in="vmx" x="0" y="0" width="{VIS_MAT_W + c1_w + 4}" height="{h2 + int(a_sh) + 6}" result="qbx_t"/>',
        f'<feOffset in="qbar_src" dx="{-qs_x}" dy="{-qs_y}" x="0" y="0" width="{c1_w}" height="{h2}" result="qb_c"/>',
        f'<feTile in="qb_c" x="0" y="0" width="{VIS_MAT_W + c1_w + 4}" height="{h2}" result="qb_ct"/>',
        f'<feOffset in="qb_ct" dx="0" dy="{_fmt_num(a_sh)}" x="0" y="0" width="{VIS_MAT_W + c1_w + 4}" height="{h2 + int(a_sh) + 6}" result="qb_xs"/>',
        f'<feDisplacementMap in="qb_xs" in2="qbx_t" x="0" y="0" width="{VIS_MAT_W}" height="{h2}" scale="{disp_s}" xChannelSelector="R" yChannelSelector="A" result="qb_1r"/>',
        f'<feTile in="qb_1r" x="0" y="0" width="{VIS_MAT_W}" height="{VIS_QBAR_H}" result="qb_t"/>',
        f'<feOffset in="qb_t" dx="{VIS_MAT_X}" dy="{VIS_QBAR_Y}" x="{VIS_MAT_X}" y="{VIS_QBAR_Y}" width="{VIS_MAT_W}" height="{VIS_QBAR_H}" result="vis_qbar"/>',
        # Keep ONLY the clean rendered `y = 330..596` visual strip from `visa`, stripping ALL VRAM & LUT helpers!
        '<feOffset in="SourceGraphic" dx="0" dy="0" x="18" y="330" width="634" height="266" result="vis_bot"/>',
        '<feMerge><feMergeNode in="vis_bot"/><feMergeNode in="bd_bdr"/><feMergeNode in="vis_bd"/><feMergeNode in="mt_bdr"/><feMergeNode in="vis_mat"/><feMergeNode in="qb_bdr"/><feMergeNode in="vis_qbar"/></feMerge>',
    ])
    out_filters.append(_wrap_filter(f"ai-visb-{dpr}x", pvb2, w=STAGE_W, h=STAGE_H))

    return out_filters
