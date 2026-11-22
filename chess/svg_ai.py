#!/usr/bin/env python3
"""
Multi-Stage Linear-VRAM HTML5 SVG `<filter>` 2-Ply CNN Minimax Engine (`chess/svg_ai.py`).

Wires the Stockfish-distilled 3-layer 8-bit quantized Convolutional Value Network (`chess/weights.json`,
1,345 parameters, 0 BatchNorm, Z0 = 128/255, Pearson r = 0.9033 vs Stockfish) into a pure HTML5 SVG
`<filter>` 2-ply Minimax pipeline with:
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
VRAM_H = 608  # y = 0..607 holds intermediate VRAM; x = 642..663 & y = 608..635 hold SourceGraphic & LUT PNGs

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

# 3. Stages s2b..s2d: CNN Layer 1 compacted `384 x 96` tiles (5 RGB triplets, 100% disjoint from S2_B0/B1!):
VRAM_P1_C6_0 = (0, 320)    # 384 x 96 (s2b)
VRAM_P1_C6_1 = (0, 416)    # 384 x 96 (s2b)
VRAM_P1_C6_2 = (0, 512)    # 384 x 96 (s2c)
VRAM_P1_C6_3L = (384, 320) # 192 x 96 left half of triplet 3 (s2c)
VRAM_P1_C6_3R = (384, 416) # 192 x 96 right half of triplet 3 (s2c)
VRAM_P1_C6_4 = (0, 0)      # 384 x 96 (written at end of s2d, last reader of S2_B0/B1!)

VRAM_SKIP_C2 = (384, 512)  # 128 x 32: skip_mat 2x2 features (s2d -> s4d)
VRAM_META_C2 = (384, 544)  # 128 x 32: R=no_m2, G=s1_in_check (s2a -> s4c)

# 4. Stages s3a..s3c: CNN Layer 2 compacted `128 x 32` tiles (6 RGB triplets at x=400..527, y=0..191):
VRAM_P2_C2 = [(400, 32 * g) for g in range(6)]

# 5. Stages s4a..s4b: CNN Layer 3a `mix3` `128 x 32` tiles (8 RGB triplets at x=0..127, y=0..255):
VRAM_MIX3 = [(0, 32 * g) for g in range(8)]

# 6. Stages s4c..s4d & s5a..s5b: Scalar `64 x 16` feature sheets + Minimax outputs at x=540..603:
VRAM_DW3_C1 = [(540, 16 * g) for g in range(8)]       # x=540..603, y=0..127
VRAM_SKIP_C1 = (540, 128)                             # 64 x 16 (y=128..143)
VRAM_NOM2_C1 = (540, 146)                             # 64 x 16 (y=146..161: R=no_m2, G=s1_in_check)
VRAM_VSHEET = (540, 170)                              # 64 x 16 (y=170..185: V_CNN in [3..253]/255, 255=no_m2)
VRAM_QSTRIP = (540, 190)                              # 64 x 2  (y=190..191: Q(v1) in [0..254]/255)
VRAM_WINMASK = (540, 196)                             # 64 x 2  (y=196..197: 1.0 at unique winning v1*)
VRAM_BOARD8 = (540, 204)                              # 8 x 8   (y=204..211: RGB board + FROM/TO highlights)
VRAM_WIN1PX = (560, 204)                              # 2 x 2   (R=128+u1*, G=128+v1*, B=Q_max)

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

COMPUTE_STAGE_IDS: Tuple[str, ...] = (
    "s1a", "s1b", "s1b2", "s1c", "s1c2", "s1d", "s1e", "s1e2", "s1e3", "s1f", "s1f2", "s1g", "s1g2", "s1g3", "s1h", "s1i",
    "s2a", "s2b", "s2c", "s2d",
    "s3a", "s3b", "s3c",
    "s4a", "s4b", "s4c", "s4d",
    "s5a", "s5b",
    "visa", "visb",
)


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

    best_sq_score = np.zeros((8, 8), dtype=np.int32)
    best_sq_u2 = np.full((8, 8), -1, dtype=np.int32)
    best_sq_pt2 = np.zeros((8, 8), dtype=np.int32)

    def _consider_dest(ux: int, uy: int, tx: int, ty: int, pt_src: int) -> None:
        u2 = 8 * uy + ux
        placed_pt = 5 if (pt_src == 1 and ty == 0) else pt_src
        mvv = int(mvv_grid[ty, tx])
        base_tier = compute_ply2_base_tier(pt_src, mvv, 1 if ty == 0 else 0)
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


def extract_root_board_u8_from_css(css_env: Dict[str, int]) -> np.ndarray:
    turn = int(css_env["--turn"])
    arr = np.full((6, 8, 8), 128, dtype=np.uint8)
    roles = ("p", "n", "b", "r", "q", "k")
    for ay in range(8):
        for ax in range(8):
            sq = (8 * ay + ax) if turn == 0 else (8 * (7 - ay) + ax)
            for ch, r_sfx in enumerate(roles):
                own_v = int(css_env[f"--own_{r_sfx}{sq}"])
                opp_v = int(css_env[f"--opp_{r_sfx}{sq}"])
                arr[ch, ay, ax] = 128 + 127 * (own_v - opp_v)
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
                leaf_list.append(_apply_ply2_on_norm_board(s1, u2, v2, pt2))
                leaf_coords.append((q_idx, v1))

    if leaf_list:
        _, _, u8_vals = evaluate_svg_cnn_batch(np.stack(leaf_list, axis=0), spec)
        for idx, (q_idx, v1) in enumerate(leaf_coords):
            v_sheet_u8[q_idx, v1] = int(u8_vals[idx])

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

def _wrap_filter(fid: str, body: Sequence[str], w: int = STAGE_W, h: int = STAGE_H) -> str:
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


def _emit_passthrough(p: List[str], new_nodes: Sequence[str], keep_vram: bool = True) -> None:
    if keep_vram:
        nodes = ['<feMergeNode in="SourceGraphic"/>'] + [f'<feMergeNode in="{n}"/>' for n in new_nodes]
        p.append(f'<feMerge>{"".join(nodes)}</feMerge>')
    else:
        low_h = STAGE_H - VRAM_H
        p.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="642" y="0" width="{STAGE_W - 642}" height="{VRAM_H}" result="pass_rcol"/>')
        p.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="{VRAM_H}" width="{STAGE_W}" height="{low_h}" result="pass_bot"/>')
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
    x_parts: List[str] = []
    for s_idx, (sx_off, dx_off, src_w, sl_w) in enumerate(x_slices):
        p.append(
            f'<feOffset in="{src_in}" dx="{dx_off - (sx + sx_off)}" dy="{_fmt_num(-sy + a_sh)}" '
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
    """Compile all 28 linear-VRAM SVG `<filter>` stages for `dpr in (1, 2)` (`<= 72` prims/stage)."""
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
    tbl_norm_root = " ".join(_fmt_num(max(1, c) / 255.0) for c in range(256))

    # `<foreignObject>` M1/M1C encodes `128 + code` (`code in 0..9` -> `128..137`):
    def _code128_lut(active_codes: set[int], hi: str = "1", lo: str = z0_s) -> str:
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

    neg_s = _fmt_num(1.0 / 255.0)
    tbl_opp_p = _code128_lut({1}, neg_s, z0_s)
    tbl_opp_n = _code128_lut({2}, neg_s, z0_s)
    tbl_opp_b = _code128_lut({3}, neg_s, z0_s)
    tbl_opp_r = _code128_lut({4}, neg_s, z0_s)
    tbl_opp_q = _code128_lut({5}, neg_s, z0_s)
    tbl_opp_k = _code128_lut({6}, neg_s, z0_s)

    # `cell_sq.png` stores `128 + sq` (`128..191`) on `0..7 x 0..7` and `250` on moat:
    tbl_subxy128 = " ".join(
        _fmt_num((128 + 2 * (((c - 128) // 8) % 2) + ((c - 128) % 2)) / 255.0) if (128 <= c < 192) else "0"
        for c in range(256)
    )
    tbl_is_r0 = " ".join("1" if (128 <= c < 136) else "0" for c in range(256))
    tbl_is_r4 = " ".join("1" if (160 <= c < 168) else "0" for c in range(256))
    tbl_csq_valid = " ".join(_fmt_num(c / 255.0) if (128 <= c < 192) else "0" for c in range(256))

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
            vals.append(_fmt_num((128 + 4 * bt) / 255.0))
        return " ".join(vals)

    tbl_pt_p_placed128 = " ".join(
        _fmt_num((128 + 5) / 255.0) if (133 <= c <= 137) else _fmt_num((128 + 1) / 255.0) for c in range(256)
    )
    tbl_mvv_is_cap = " ".join("1" if (128 <= c < 138 and ((c - 128) % 5) != 0) else "0" for c in range(256))
    tbl_mvv_is_emp = " ".join("1" if (128 <= c < 138 and ((c - 128) % 5) == 0) else "0" for c in range(256))

    out_filters: List[str] = []

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

    fo_ch = [(FO_X + sc(cx), FO_Y + sc(cy)) for cx, cy in SRC_CH_PHYS]
    fo_m1ux, fo_m1uy = FO_X + sc(SRC_M1U_PHYS[0]), FO_Y + sc(SRC_M1U_PHYS[1])
    fo_m1vx, fo_m1vy = FO_X + sc(SRC_M1V_PHYS[0]), FO_Y + sc(SRC_M1V_PHYS[1])
    fo_m1kx, fo_m1ky = FO_X + sc(SRC_M1K_PHYS[0]), FO_Y + sc(SRC_M1K_PHYS[1])
    fo_m1cux, fo_m1cuy = FO_X + sc(SRC_M1CU_PHYS[0]), FO_Y + sc(SRC_M1CU_PHYS[1])
    fo_m1cvx, fo_m1cvy = FO_X + sc(SRC_M1CV_PHYS[0]), FO_Y + sc(SRC_M1CV_PHYS[1])
    fo_m1ccx, fo_m1ccy = FO_X + sc(SRC_M1CC_PHYS[0]), FO_Y + sc(SRC_M1CC_PHYS[1])
    fo_mt_tx, fo_mt_ty = FO_X + sc(SRC_META_TURN_PHYS[0]), FO_Y + sc(SRC_META_TURN_PHYS[1])
    fo_mt_mx, fo_mt_my = FO_X + sc(SRC_META_TERM_PHYS[0]), FO_Y + sc(SRC_META_TERM_PHYS[1])

    csq_x, csq_y = LUT_CELLSQ_POS
    bm10_x, bm10_y = LUT_BMASK10_POS
    bm6_x, bm6_y = LUT_BMASK6_POS

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
    _emit_passthrough(p1a, ["s1_b0", "s1_b1"], keep_vram=False)
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
        f'<feComposite in="mv0" in2="mv1" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="mv01"/>',
        f'<feComposite in="mv01" in2="mvr0" x="{t_pnkx}" y="{t_pnky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="mv_b0"/>',
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
    # Stages `s1g, s1g2, s1g3`: Opponent Sliders (`192+u2` = R/B, `128+u2` = Q!)
    # =========================================================================
    # `csq` is ALREADY `128+u2` on `0..7 x 0..7`! Adding `+64/255` for `R` (orth) / `B` (diag) puts `R/B` in
    # `192..255` and `Q` in `128..191` (`0` if empty) — 100% inside `{0} U [128..255]`!
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
        f'<feColorMatrix in="ob1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="{_fmt_num(64/255)} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_bns"/>',
        f'<feColorMatrix in="rq_csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_u"/>',
        f'<feComposite in="rq_bns" in2="rq_u" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_pk"/>',
        f'<feComposite in="rq_pk" in2="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_h0"/>',
        f'<feComposite in="rq_h0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_h"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alv_r"/>',
        f'<feComposite in="rq_pk" in2="alv_r" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_v0"/>',
        f'<feComposite in="rq_v0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_v"/>',
    ]
    or0 = _emit_1ch_slider_ray(p1g, "rq_h", 1, 0, t_orx, t_ory, "or0")
    or1 = _emit_1ch_slider_ray(p1g, "rq_h", -1, 0, t_orx, t_ory, "or1")
    or2 = _emit_1ch_slider_ray(p1g, "rq_v", 0, 1, t_orx, t_ory, "or2")
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
        f'<feColorMatrix in="ob1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="{_fmt_num(64/255)} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_bns"/>',
        f'<feColorMatrix in="rq_csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="rq_u"/>',
        f'<feComposite in="rq_bns" in2="rq_u" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_pk"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="alv_r"/>',
        f'<feComposite in="rq_pk" in2="alv_r" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="rq_v0"/>',
        f'<feComposite in="rq_v0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rq_v"/>',
        f'<feComposite in="ob0" in2="csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="b_csq"/>',
        f'<feColorMatrix in="ob0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 {_fmt_num(64/255)} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="b_bns"/>',
        f'<feColorMatrix in="b_csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bu_r"/>',
        f'<feColorMatrix in="rq_csq" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="qu_r"/>',
        f'<feComposite in="bu_r" in2="qu_r" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bqu_r"/>',
        f'<feComposite in="b_bns" in2="bqu_r" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_pk"/>',
        f'<feComposite in="bq_pk" in2="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bq_d10"/>',
        f'<feComposite in="bq_d10" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_d1"/>',
    ]
    or3 = _emit_1ch_slider_ray(p1g2, "rq_v", 0, -1, t_orx, t_ory, "or3")
    dg0 = _emit_1ch_slider_ray(p1g2, "bq_d1", 1, 1, t_orx, t_ory, "dg0")
    p1g2.extend([
        f'<feBlend in="or_prev" in2="{or3}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="or_all"/>',
        f'<feColorMatrix in="or_all" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="ora_r"/>',
        f'<feColorMatrix in="{dg0}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="dg0_g"/>',
        f'<feColorMatrix in="bq_pk" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="bqpk_b"/>',
        f'<feComposite in="ora_r" in2="dg0_g" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="od_rg"/>',
        f'<feComposite in="od_rg" in2="bqpk_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="or_g2"/>',
    ])
    _emit_passthrough(p1g2, ["or_g2"])
    out_filters.append(_wrap_filter(f"ai-s1g2-{dpr}x", p1g2))

    p1g3: List[str] = [
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="og2"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pinx}" dy="{t_ory - t_piny}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al1"/>',
        f'<feOffset in="SourceGraphic" dx="{t_orx - t_pin2x}" dy="{t_ory - t_pin2y}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feComponentTransfer in="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" result="emp_b"><feFuncR type="discrete" tableValues="0"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_mvv_is_emp}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="og2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="bq_rg"/>',
        f'<feComposite in="bq_rg" in2="al2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="bq_12"/>',
        f'<feColorMatrix in="bq_12" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bqd1_0"/>',
        f'<feComposite in="bqd1_0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_d1"/>',
        f'<feColorMatrix in="bq_12" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="bqd2_0"/>',
        f'<feComposite in="bqd2_0" in2="emp_b" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="bq_d2"/>',
    ]
    dg1 = _emit_1ch_slider_ray(p1g3, "bq_d1", -1, -1, t_orx, t_ory, "dg1")
    dg2 = _emit_1ch_slider_ray(p1g3, "bq_d2", 1, -1, t_orx, t_ory, "dg2")
    dg3 = _emit_1ch_slider_ray(p1g3, "bq_d2", -1, 1, t_orx, t_ory, "dg3")
    p1g3.extend([
        f'<feColorMatrix in="og2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="dg0_r"/>',
        f'<feBlend in="dg0_r" in2="{dg1}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="dg01"/>',
        f'<feBlend in="{dg2}" in2="{dg3}" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="dg23"/>',
        f'<feBlend in="dg01" in2="dg23" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" mode="lighten" result="dg_all"/>',
        f'<feColorMatrix in="al1" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1" result="ev_rg"/>',
        f'<feColorMatrix in="og2" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="or_r"/>',
        f'<feColorMatrix in="dg_all" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="dg_g"/>',
        f'<feComposite in="or_r" in2="dg_g" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_rg0"/>',
        f'<feComposite in="sl_rg0" in2="ev_rg" x="{t_orx}" y="{t_ory}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sl_out"/>',
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
        f'<feComponentTransfer in="mvc" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tn_rb"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(2)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="{_fmt_num(130/255)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="pn_g3" x="{s1_kx}" y="{t_ory + (s1_ky - t_ory)}" width="{s1_w}" height="{s1_h}" result="n_m"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="table" tableValues="{tbl_ge128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="pn_g3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="n_u"/>',
        f'<feComposite in="tn_rb" in2="n_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="n_rb"/>',
        f'<feComposite in="n_rb" in2="n_u" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="cand_n"/>',
        # King candidate `(R=128+4*tier_K, G=128+u2, B=134/255)`:
        f'<feColorMatrix in="kg_u2b" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1" result="kg_b3"/>',
        f'<feComponentTransfer in="mvc" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tk_rb"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(6)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="{_fmt_num(134/255)}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
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
    # Stage `s1i`: Merge `Orth(R/Q)` & `Diag(B/Q)` vs `PNK` + `2x2` Quadrant Argmax
    # =========================================================================
    # Inside `s1i` (single `<filter>`, no `<g>` boundary!), map `slm.G` (`diag`: `192+u2=B, 128+u2=Q`) -> `192+u2 (B), 64+u2 (Q)`
    # and `slm.R` (`orth`: `192+u2=R, 128+u2=Q`) -> `128+u2 (R), 64+u2 (Q)`, so `<feBlend lighten>` picks `B > R > Q`!
    tbl_diag_remap = " ".join(_fmt_num((c if c >= 192 else (c - 64)) / 255.0) if c >= 128 else "0" for c in range(256))
    tbl_orth_remap = " ".join(_fmt_num((c - 64) / 255.0) if c >= 128 else "0" for c in range(256))
    tbl_sl_u2_128 = " ".join(_fmt_num((128 + ((c - 64) % 64)) / 255.0) if c >= 64 else "0" for c in range(256))
    tbl_sl_pt128 = " ".join(
        _fmt_num(131 / 255) if c >= 192 else (_fmt_num(132 / 255) if c >= 128 else (_fmt_num(133 / 255) if c >= 64 else "0"))
        for c in range(256)
    )
    tbl_sl_has = " ".join("1" if c >= 64 else "0" for c in range(256))
    tbl_sl_bonus = " ".join(_fmt_num(8 / 255) if c >= 192 else (_fmt_num(4 / 255) if c >= 128 else "0") for c in range(256))
    tbl_subxy_raw = " ".join(
        _fmt_num((2 * (((c - 128) // 8) % 2) + ((c - 128) % 2)) / 255.0) if (128 <= c < 192) else "0"
        for c in range(256)
    )

    p1i: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{s1_kx - csq_x}" dy="{s1_ky - csq_y}" x="{s1_kx}" y="{s1_ky}" width="{b10}" height="{b10}" result="csq0"/>',
        f'<feTile in="csq0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="csq"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="pnk"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_orx}" dy="{s1_ky - t_ory}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="slm"/>',
        f'<feOffset in="SourceGraphic" dx="{s1_kx - t_pin2x}" dy="{s1_ky - t_pin2y}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="al2"/>',
        f'<feComponentTransfer in="slm" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sl_rm"><feFuncR type="table" tableValues="{tbl_orth_remap}"/><feFuncG type="table" tableValues="{tbl_diag_remap}"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feColorMatrix in="sl_rm" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="slo_3"/>',
        f'<feColorMatrix in="sl_rm" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1" result="sld_3"/>',
        f'<feBlend in="sld_3" in2="slo_3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" mode="lighten" result="sl_r3"/>',
        f'<feColorMatrix in="al2" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="mvc_r"/>',
        f'<feComponentTransfer in="mvc_r" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="tq_r"><feFuncR type="table" tableValues="{_tier128_lut_for_pt(5)}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sl_r3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sl_bns"><feFuncR type="table" tableValues="{tbl_sl_bonus}"/><feFuncG type="table" tableValues="{tbl_sl_u2_128}"/><feFuncB type="table" tableValues="{tbl_sl_pt128}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="sl_r3" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sl_msk"><feFuncR type="table" tableValues="{tbl_sl_has}"/><feFuncG type="discrete" tableValues="1"/><feFuncB type="discrete" tableValues="1"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="tq_r" in2="sl_bns" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sl_raw"/>',
        f'<feComposite in="sl_raw" in2="sl_msk" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cand_sl"/>',
    ]
    _mux_higher_score(p1i, "pnk", "cand_sl", s1_kx, s1_ky, s1_w, s1_h, "sq_win0")
    p1i.extend([
        f'<feComponentTransfer in="sq_win0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="has_m"><feFuncR type="table" tableValues="{tbl_ge128}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComponentTransfer in="csq" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="sub_r"><feFuncR type="table" tableValues="{tbl_subxy_raw}"/><feFuncG type="discrete" tableValues="0"/><feFuncB type="discrete" tableValues="0"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feComposite in="sub_r" in2="has_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sub_ok"/>',
        f'<feColorMatrix in="has_m" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" type="matrix" values="0 0 0 0 0 {_fmt_num(-192/255)} 0 0 0 {_fmt_num(192/255)} 0 0 0 0 0 0 0 0 0 1" result="u192_def"/>',
        f'<feComposite in="sq_win0" in2="sub_ok" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sq_w1"/>',
        f'<feComposite in="sq_w1" in2="u192_def" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="sq_win"/>',
        f'<feOffset in="sq_win" dx="{-sc(1)}" dy="0" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="qw_x1"/>',
    ])
    _mux_higher_score(p1i, "sq_win", "qw_x1", s1_kx, s1_ky, s1_w, s1_h, "qw_h")
    p1i.append(f'<feOffset in="qw_h" dx="0" dy="{-sc(1)}" x="{s1_kx}" y="{s1_ky}" width="{s1_w}" height="{s1_h}" result="qw_y1"/>')
    _mux_higher_score(p1i, "qw_h", "qw_y1", s1_kx, s1_ky, s1_w, s1_h, "qw_2x2")
    _emit_passthrough(p1i, ["qw_2x2"])
    out_filters.append(_wrap_filter(f"ai-s1i-{dpr}x", p1i))

    # =========================================================================
    # Stage `s2a`: `640 x 160` Depth-2 Leaf Board Expansion + `VRAM_META_C2`
    # =========================================================================
    s2_w, s2_h = sc(640), sc(160)
    s2_b1y = sc(VRAM_S2_B1[1])
    pad_m2 = int(a_m2) + 4  # 22 on 1x (`640 + 22 = 662 <= 664`!)
    m2y_x, m2y_y = LUT_M2Y_POS
    m2x_x, m2x_y = LUT_M2X_POS
    qb_x, qb_y = LUT_QBASE_POS
    c10x_x, c10x_y = LUT_C10_2X_POS
    c10y_x, c10y_y = LUT_C10_2Y_POS
    mc2_x, mc2_y = sc(VRAM_META_C2[0]), sc(VRAM_META_C2[1])
    skc2_x, skc2_y = sc(VRAM_SKIP_C2[0]), sc(VRAM_SKIP_C2[1])
    c2_w, c2_h = sc(128), sc(32)

    tbl_dv2_r = " ".join(_fmt_num((8 * ((c // 2) % 2) + (c % 2)) / 255.0) if c >= 128 else "0" for c in range(256))
    tbl_dv2_fb = " ".join("0" if c >= 128 else _fmt_num(192 / 255) for c in range(256))

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
    _emit_passthrough(p2a, ["s2_b0", "s2_b1", "meta_c2"], keep_vram=False)
    out_filters.append(_wrap_filter(f"ai-s2a-{dpr}x", p2a))

    # =========================================================================
    # Stages `s2b, s2c, s2d`: CNN Layer 1 (`mix1 -> bmask10 -> dwconv1 -> pool1 -> C6`) + `skip_mat`
    # =========================================================================
    c6_w, c6_h = sc(384), sc(96)
    c6x_x, c6x_y = LUT_C6X_POS
    c6y_x, c6y_y = LUT_C6Y_POS
    x_slices_10_6 = [
        (sc(0), sc(0), sc(220), sc(132)),
        (sc(220), sc(132), sc(220), sc(132)),
        (sc(440), sc(264), sc(200), sc(120)),
    ]
    m1_trips = ly["mix1"]["triplets"]
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
            p.append(
                f'<feConvolveMatrix in="m1m_{o}" x="0" y="0" width="{s2_w}" height="{s2_h}" '
                f'order="3 3" targetX="1" targetY="1" divisor="1" bias="{_fmt_num(f["bias"])}" '
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

    sk_spec = ly["skip_mat"]
    sk_cm0, sk_cm1 = sk_spec["color_matrices"]
    sk_st0, sk_st1, sk_st2 = sk_spec["sum_stages"]
    nz0_sk = _fmt_num(-float(sk_spec["zero_point"]))

    p2d: List[str] = []
    _emit_l1_shared(p2d)
    o4 = _emit_l1_triplet(p2d, 4, VRAM_P1_C6_4)
    p2d.extend([
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
        p2d, "sk_2", 0, 0, s2_h, skc2_x, skc2_y, c2_w, c2_h,
        x_slices_10_2, y_slices_10_2, "c10x_t", "c10y_t", a_sh, disp_s, "skip_c2",
    )
    _emit_passthrough(p2d, [o4, "skip_c2"])
    out_filters.append(_wrap_filter(f"ai-s2d-{dpr}x", p2d))

    # =========================================================================
    # Stages `s3a, s3b, s3c`: CNN Layer 2 (`mix2 -> bmask6 -> dwconv2 -> pool2 -> C2`)
    # =========================================================================
    c2x_x, c2x_y = LUT_C2X_POS
    c2y_x, c2y_y = LUT_C2Y_POS
    x_slices_6_2 = [
        (sc(0), sc(0), sc(132), sc(44)),
        (sc(132), sc(44), sc(132), sc(44)),
        (sc(264), sc(88), sc(120), sc(40)),
    ]
    m2_trips = ly["mix2"]["triplets"]
    flts2 = ly["dwconv2"]["filters"]
    p2_div = _fmt_num(ly["pool2"]["divisor"])
    nz0_m2 = _fmt_num(-float(ly["mix2"]["zero_point"]))

    for sfx, o_pair in (("a", (0, 1)), ("b", (2, 3)), ("c", (4, 5))):
        p3: List[str] = [
            f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_0[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in0"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_1[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in1"/>',
            f'<feOffset in="SourceGraphic" dx="0" dy="{-sc(VRAM_P1_C6_2[1])}" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in2"/>',
            f'<feOffset in="SourceGraphic" dx="{-g3lx}" dy="{-g3ly}" x="0" y="0" width="{hw6}" height="{c6_h}" result="g3L"/>',
            f'<feOffset in="SourceGraphic" dx="{hw6 - g3rx}" dy="{-g3ry}" x="{hw6}" y="0" width="{hw6}" height="{c6_h}" result="g3R"/>',
            f'<feMerge x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in3"><feMergeNode in="g3L"/><feMergeNode in="g3R"/></feMerge>',
            f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{c6_w}" height="{c6_h}" result="p1_in4"/>',
            f'<feOffset in="SourceGraphic" dx="{-bm6_x}" dy="{-bm6_y}" x="0" y="0" width="{b6}" height="{b6}" result="bm6_0"/>',
            f'<feTile in="bm6_0" x="0" y="0" width="{c6_w}" height="{c6_h}" result="bm6"/>',
            f'<feOffset in="SourceGraphic" dx="{-c2x_x}" dy="{-c2x_y}" x="0" y="0" width="{sc(44)}" height="{sc(4)}" result="c2x0"/>',
            f'<feTile in="c2x0" x="0" y="0" width="{c2_w}" height="{c6_h}" result="c2x_t"/>',
            f'<feOffset in="SourceGraphic" dx="{-c2y_x}" dy="{-c2y_y}" x="0" y="0" width="{sc(4)}" height="{c2_h}" result="c2y0"/>',
            f'<feTile in="c2y0" x="0" y="0" width="{c2_w}" height="{c2_h}" result="c2y_t"/>',
        ]
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
                p3.append(
                    f'<feConvolveMatrix in="m2m_{o}" x="0" y="0" width="{c6_w}" height="{c6_h}" '
                    f'order="3 3" targetX="1" targetY="1" divisor="1" bias="{_fmt_num(f["bias"])}" '
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
    # Stages `s4a, s4b`: CNN Layer 3a `mix3` (`6 -> 8` triplets on `128 x 32`)
    # =========================================================================
    m3_trips = ly["mix3"]["triplets"]
    nz0_m3 = _fmt_num(-float(ly["mix3"]["zero_point"]))
    for sfx, o_range in (("a", range(0, 4)), ("b", range(4, 8))):
        p4ab: List[str] = [
            f'<feOffset in="SourceGraphic" dx="{-sc(VRAM_P2_C2[g][0])}" dy="{-sc(VRAM_P2_C2[g][1])}" x="0" y="0" width="{c2_w}" height="{c2_h}" result="p2_in{g}"/>'
            for g in range(6)
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
        out_filters.append(_wrap_filter(f"ai-s4{sfx}-{dpr}x", p4ab))

    # =========================================================================
    # Stages `s4c, s4d`: CNN Layer 3b `dwconv3` (`2x2 -> 1x1`) + `META_C1` & `SKIP_C1`
    # =========================================================================
    c1_w, c1_h = sc(64), sc(16)
    c1x_x, c1x_y = LUT_C1X_POS
    c1y_x, c1y_y = LUT_C1Y_POS
    x_slices_2_1 = [(sc(0), sc(0), c2_w, c1_w)]
    flts3 = ly["dwconv3"]["filters"]

    for sfx, g_range in (("c", range(0, 4)), ("d", range(4, 8))):
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

        if sfx == "c":
            nm_dx, nm_dy = sc(VRAM_NOM2_C1[0]), sc(VRAM_NOM2_C1[1])
            _emit_compact_at_origin(
                p4cd, "SourceGraphic", mc2_x, mc2_y, c2_h, nm_dx, nm_dy, c1_w, c1_h,
                x_slices_2_1, None, "c1x_t", "c1y_t", a_sh, disp_s, "nom2_c1",
            )
            s4_outs.append("nom2_c1")
        else:
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
        out_filters.append(_wrap_filter(f"ai-s4{sfx}-{dpr}x", p4cd))

    # =========================================================================
    # Stage `s5a`: Value Head (`27 -> 1`) + Sigmoid LUT + 2-Ply Minimax (`erode` + `dilate`)
    # =========================================================================
    vh = ly["value_head"]
    nz0_vh = _fmt_num(-float(vh["zero_point"]))
    sig_tbl = _fmt_list(vh["sigmoid_table_values"])
    nm1_x, nm1_y = sc(VRAM_NOM2_C1[0]), sc(VRAM_NOM2_C1[1])
    vsh_x, vsh_y = sc(VRAM_VSHEET[0]), sc(VRAM_VSHEET[1])
    qs_x, qs_y = sc(VRAM_QSTRIP[0]), sc(VRAM_QSTRIP[1])
    wm_x, wm_y = sc(VRAM_WINMASK[0]), sc(VRAM_WINMASK[1])
    tie_x, tie_y = LUT_TIE_POS
    h2 = sc(2)

    feat_slots = list(VRAM_DW3_C1) + [VRAM_SKIP_C1]
    p5a: List[str] = []
    vh_acc = ""
    for g, slot in enumerate(feat_slots):
        fx, fy = sc(slot[0]), sc(slot[1])
        mat20 = vh["color_matrices"][g]
        p5a.append(f'<feOffset in="SourceGraphic" dx="{-fx}" dy="{-fy}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="f_{g}"/>')
        p5a.append(f'<feColorMatrix in="f_{g}" x="0" y="0" width="{c1_w}" height="{c1_h}" type="matrix" values="{_fmt_list(mat20)}" result="vp_{g}"/>')
        if g == 0:
            vh_acc = "vp_0"
        else:
            nxt = f"va_{g}"
            p5a.append(f'<feComposite in="{vh_acc}" in2="vp_{g}" x="0" y="0" width="{c1_w}" height="{c1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="{nz0_vh}" result="{nxt}"/>')
            vh_acc = nxt

    p5a.extend([
        f'<feComponentTransfer in="{vh_acc}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="v_sig">'
        f'<feFuncR type="table" tableValues="{sig_tbl}"/><feFuncG type="table" tableValues="{sig_tbl}"/><feFuncB type="table" tableValues="{sig_tbl}"/>'
        f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        # Force `no_m2` quadrants to `255/255 = 1.0`:
        f'<feOffset in="SourceGraphic" dx="{-nm1_x}" dy="{-nm1_y}" x="0" y="0" width="{c1_w}" height="{c1_h}" result="nm1"/>',
        f'<feColorMatrix in="nm1" x="0" y="0" width="{c1_w}" height="{c1_h}" type="matrix" values="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="nm_rgb"/>',
        f'<feComposite in="v_sig" in2="nm_rgb" x="0" y="0" width="{c1_w}" height="{c1_h}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="v_sh0"/>',
        f'<feOffset in="v_sh0" dx="{vsh_x}" dy="{vsh_y}" x="{vsh_x}" y="{vsh_y}" width="{c1_w}" height="{c1_h}" result="v_sheet"/>',
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
    ])
    _emit_passthrough(p5a, ["v_sheet", "q_strip", "win_mask"])
    out_filters.append(_wrap_filter(f"ai-s5a-{dpr}x", p5a))

    # =========================================================================
    # Stage `s5b`: Extract `VRAM_WIN1PX` + Render `8x8` RGB Move Map (`VRAM_BOARD8`)
    # =========================================================================
    b8 = sc(8)
    bd8_x, bd8_y = sc(VRAM_BOARD8[0]), sc(VRAM_BOARD8[1])
    w1_x, w1_y = sc(VRAM_WIN1PX[0]), sc(VRAM_WIN1PX[1])
    sq8w_x, sq8w_y = LUT_SQ8W_POS
    sq8b_x, sq8b_y = LUT_SQ8B_POS
    chk8_x, chk8_y = LUT_CHK8_POS

    p5b: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-fo_m1cux}" dy="{-fo_m1cuy}" x="0" y="0" width="{c1_w}" height="{h2}" result="m1cu"/>',
        f'<feOffset in="SourceGraphic" dx="{-fo_m1cvx}" dy="{-fo_m1cvy}" x="0" y="0" width="{c1_w}" height="{h2}" result="m1cv"/>',
        f'<feOffset in="SourceGraphic" dx="{-qs_x}" dy="{-qs_y}" x="0" y="0" width="{c1_w}" height="{h2}" result="qs"/>',
        f'<feOffset in="SourceGraphic" dx="{-wm_x}" dy="{-wm_y}" x="0" y="0" width="{c1_w}" height="{h2}" result="wm"/>',
        f'<feColorMatrix in="m1cu" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="u_r"/>',
        f'<feColorMatrix in="m1cv" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="v_g"/>',
        f'<feComposite in="u_r" in2="v_g" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="uv_rg"/>',
        f'<feColorMatrix in="qs" x="0" y="0" width="{c1_w}" height="{h2}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1" result="q_b"/>',
        f'<feComposite in="uv_rg" in2="q_b" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="uvq"/>',
        f'<feComposite in="uvq" in2="wm" x="0" y="0" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="uvq_w"/>',
        f'<feTile in="uvq_w" x="0" y="0" width="{3 * c1_w}" height="{h2}" result="uw_t3"/>',
        f'<feMorphology in="uw_t3" x="0" y="0" width="{3 * c1_w}" height="{h2}" operator="dilate" radius="{_fmt_num(32 * u)} 0" result="uw_dl"/>',
        f'<feOffset in="uw_dl" dx="{-c1_w}" dy="0" x="0" y="0" width="{h2}" height="{h2}" result="w1_0"/>',
        f'<feOffset in="w1_0" dx="{w1_x}" dy="{w1_y}" x="{w1_x}" y="{w1_y}" width="{h2}" height="{h2}" result="win_1px"/>',
        # Tile `w1_0` to `8x8` & orient `sq8` by `turn` (`SRC_META_TURN_PHYS == 255` when Black to move!):
        f'<feTile in="w1_0" x="0" y="0" width="{b8}" height="{b8}" result="wt8"/>',
        f'<feOffset in="SourceGraphic" dx="{-fo_mt_tx}" dy="{-fo_mt_ty}" x="0" y="0" width="{h2}" height="{h2}" result="mt0"/>',
        f'<feTile in="mt0" x="0" y="0" width="{b8}" height="{b8}" result="mt8"/>',
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
        # Paint dark-slate checkerboard + Violet FROM (`#a855f7`: 0.659, 0.333, 0.969) + Emerald TO (`#10b981`: 0.063, 0.725, 0.506):
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
    _emit_passthrough(p5b, ["win_1px", "board8"])
    out_filters.append(_wrap_filter(f"ai-s5b-{dpr}x", p5b))

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

    pva: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-vbx_x}" dy="{-vbx_y}" x="0" y="0" width="{VIS_BOARD_SZ}" height="4" result="vbx"/>',
        f'<feOffset in="SourceGraphic" dx="{-vby_x}" dy="{-vby_y}" x="0" y="0" width="4" height="{VIS_BOARD_SZ}" result="vby"/>',
        f'<feOffset in="SourceGraphic" dx="{-vmx_x}" dy="{-vmx_y}" x="0" y="0" width="{VIS_MAT_W}" height="4" result="vmx"/>',
        f'<feOffset in="SourceGraphic" dx="{-vmy_x}" dy="{-vmy_y}" x="0" y="0" width="4" height="{VIS_MAT_H}" result="vmy"/>',
    ]
    _emit_periodic_zoom_rect(
        pva, "SourceGraphic", bd8_x, bd8_y, b8, b8,
        VIS_BOARD_X, VIS_BOARD_Y, VIS_BOARD_SZ, VIS_BOARD_SZ,
        "vbx", "vby", a_sh, disp_s, "vis_bd",
    )
    pva.append(
        f'<feComponentTransfer in="SourceGraphic" x="{vsh_x}" y="{vsh_y}" width="{c1_w}" height="{c1_h}" result="vsh_col">'
        f'<feFuncR type="table" tableValues="{tbl_hm_r}"/><feFuncG type="table" tableValues="{tbl_hm_g}"/><feFuncB type="table" tableValues="{tbl_hm_b}"/>'
        f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>'
    )
    _emit_periodic_zoom_rect(
        pva, "vsh_col", vsh_x, vsh_y, c1_w, c1_h,
        VIS_MAT_X, VIS_MAT_Y, VIS_MAT_W, VIS_MAT_H,
        "vmx", "vmy", a_sh, disp_s, "vis_mat",
    )
    # Colorize `Q(v1)` + highlight winning column `win_mask` in bright Gold (`#fbbf24`):
    pva.extend([
        f'<feComponentTransfer in="SourceGraphic" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" result="qs_col">'
        f'<feFuncR type="table" tableValues="{tbl_hm_r}"/><feFuncG type="table" tableValues="{tbl_hm_g}"/><feFuncB type="table" tableValues="{tbl_hm_b}"/>'
        f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>',
        f'<feOffset in="SourceGraphic" dx="{qs_x - wm_x}" dy="{qs_y - wm_y}" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" result="wm_s"/>',
        f'<feColorMatrix in="wm_s" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" type="matrix" values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0 1" result="wm_inv"/>',
        f'<feComposite in="qs_col" in2="wm_inv" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="qs_k"/>',
        f'<feColorMatrix in="wm_s" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" type="matrix" values="0.984 0 0 0 0 0 0.749 0 0 0 0 0 0.141 0 0 0 0 0 0 1" result="wm_gold"/>',
        f'<feComposite in="qs_k" in2="wm_gold" x="{qs_x}" y="{qs_y}" width="{c1_w}" height="{h2}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="qbar_src"/>',
        # 1D X-zoom of `qbar_src` (`64 -> 384`) + vertical `<feTile>` to `VIS_QBAR_H` (`8` prims!):
        f'<feTile in="vmx" x="0" y="0" width="{VIS_MAT_W + c1_w + 4}" height="{h2 + int(a_sh) + 6}" result="qbx_t"/>',
        f'<feOffset in="qbar_src" dx="{-qs_x}" dy="{-qs_y}" x="0" y="0" width="{c1_w}" height="{h2}" result="qb_c"/>',
        f'<feTile in="qb_c" x="0" y="0" width="{VIS_MAT_W + c1_w + 4}" height="{h2}" result="qb_ct"/>',
        f'<feOffset in="qb_ct" dx="0" dy="{_fmt_num(a_sh)}" x="0" y="0" width="{VIS_MAT_W + c1_w + 4}" height="{h2 + int(a_sh) + 6}" result="qb_xs"/>',
        f'<feDisplacementMap in="qb_xs" in2="qbx_t" x="0" y="0" width="{VIS_MAT_W}" height="{h2}" scale="{disp_s}" xChannelSelector="R" yChannelSelector="A" result="qb_1r"/>',
        f'<feTile in="qb_1r" x="0" y="0" width="{VIS_MAT_W}" height="{VIS_QBAR_H}" result="qb_t"/>',
        f'<feOffset in="qb_t" dx="{VIS_MAT_X}" dy="{VIS_QBAR_Y}" x="{VIS_MAT_X}" y="{VIS_QBAR_Y}" width="{VIS_MAT_W}" height="{VIS_QBAR_H}" result="vis_qbar"/>',
        # Pass through ONLY the VRAM & LUT slices needed by `visb` (`x=538..664, y=0..310` and `y=612..635`)!
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="538" y="0" width="{STAGE_W - 538}" height="310" result="keep_vr"/>',
        f'<feOffset in="SourceGraphic" dx="0" dy="0" x="220" y="612" width="430" height="24" result="keep_bl"/>',
        '<feMerge><feMergeNode in="keep_vr"/><feMergeNode in="keep_bl"/><feMergeNode in="vis_bd"/><feMergeNode in="vis_mat"/><feMergeNode in="vis_qbar"/></feMerge>',
    ])
    out_filters.append(_wrap_filter(f"ai-visa-{dpr}x", pva))

    # =========================================================================
    # Stage `visb`: Win-Probability Bar + Neural Latent Sheets (`skip_mat` & `dwconv3[0]`)
    #               (Strips all remaining VRAM & LUTs -> crisp final dashboard!)
    # =========================================================================
    sk1_x, sk1_y = sc(VRAM_SKIP_C1[0]), sc(VRAM_SKIP_C1[1])
    d37_x, d37_y = sc(VRAM_DW3_C1[7][0]), sc(VRAM_DW3_C1[7][1])

    tbl_q_to_prob = " ".join(
        "0" if c == 0 else ("1" if c >= 253 else _fmt_num(max(0.0, (c - 3.0) / 250.0))) for c in range(256)
    )
    out_filters.pop()

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
    out_filters.append(_wrap_filter(f"ai-visa-{dpr}x", pva2))

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
    out_filters.append(_wrap_filter(f"ai-visb-{dpr}x", pvb2))

    return out_filters
