#!/usr/bin/env python3
"""
Multi-Stage Linear VRAM SVG <feConvolveMatrix> CNN Compiler & Simulator (`cnn/svg_cnn.py`).

To avoid Blink/Skia's exponential O(2^depth) DAG re-evaluation on multi-layer diamond filter
graphs, the CNN pipeline is partitioned across 7 nested HTML `<div>` stages (`s1..s7`).
Each `<div>`'s CSS `filter: url(#...)` materializes a flat cached RGBA8888 texture (`y=0..47`
acting as a Neural VRAM Register File), reducing per-stage filter depth to <= 6 and total
HTML size to ~180 KB while keeping 100% zero-JavaScript HTML5 + CSS + SVG execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


CNN_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DEFAULT_PATH = os.path.join(CNN_DIR, "weights.json")
ASSETS_DIR = os.path.join(CNN_DIR, "assets")

# VRAM Register Coordinates in CSS pixels (identical on 1x and 2x!):
# Row 0 (y = 0, step = 30px, max height = 28px < 30px):
VRAM_X0 = (0, 0)
VRAM_CONV1 = [(30 + 30 * g, 0) for g in range(4)]      # 30, 60, 90, 120
VRAM_MIX2 = [(150 + 30 * g, 0) for g in range(5)]      # 150, 180, 210, 240, 270
VRAM_DW2 = [(300 + 30 * g, 0) for g in range(5)]       # 300, 330, 360, 390, 420
VRAM_P2C6 = [(460 + 10 * g, 0) for g in range(5)]      # 460..500 (6x6 compacted)

# Row 1 (y = 30, step = 10px, max height = 6px; 30 + 6 = 36 < 48px):
VRAM_MIX3 = [(10 * g, 30) for g in range(8)]           # 0..70
VRAM_DW3 = [(90 + 10 * g, 30) for g in range(8)]       # 90..160
VRAM_DW4 = [(180 + 10 * g, 30) for g in range(8)]      # 180..250
VRAM_SM_PR = (280, 30)                                 # 10x1 strip
VRAM_SM_WIN = (300, 30)                                # 10x1 strip
VRAM_SM_NWIN = (320, 30)                               # 10x1 strip

VRAM_H = 48  # y = 0..47 reserved for VRAM registers; y = 48..303 holds SourceGraphic helpers

# Layer radio options:
#   (key, short_title, badge_shape, vram_slots, src_coords, src_box_phys, num_rgb_triplets, (cols, rows, cell_zoom_px, gap_px))
LAYER_VIEWS: List[Tuple[str, str, str, List[Tuple[int, int]], List[int], int, int, Tuple[int, int, int, int]]] = [
    ("conv1",   "Conv1 (5×5)",        "12 ch → 4 RGB · 24×24", VRAM_CONV1, list(range(2, 26)),    28, 4, (2, 2, 3, 10)),
    ("mix2",    "Pool1 + Mix2 (1×1)", "15 ch → 5 RGB · 12×12", VRAM_MIX2,  list(range(2, 26, 2)), 28, 5, (3, 2, 5, 8)),
    ("dwconv2", "DWConv2 (3×3)",      "15 ch → 5 RGB · 10×10", VRAM_DW2,   list(range(4, 24, 2)), 28, 5, (3, 2, 6, 8)),
    ("mix3",    "Pool2 + Mix3 (1×1)", "24 ch → 8 RGB · 5×5",   VRAM_MIX3,  [0, 1, 2, 3, 4],        6, 8, (4, 2, 9, 7)),
    ("dwconv3", "DWConv3 (3×3)",      "24 ch → 8 RGB · 3×3",   VRAM_DW3,   [1, 2, 3],              6, 8, (4, 2, 15, 7)),
    ("dwconv4", "DWConv4 (Global)",   "24 ch → 8 RGB · 1×1",   VRAM_DW4,   [2],                    6, 8, (4, 2, 42, 8)),
]

STAGE_W = 720
STAGE_H = 322

INPUT_PREVIEW_X = 18
INPUT_PREVIEW_Y = 52
INPUT_PREVIEW_SZ = 140  # 28 * 5
INPUT_SRC_Y = 20        # Place 28x28 / 14x14 <img> above y=52 drop box so 100% of drops append at end!

ACT_PANEL_X = 186
ACT_PANEL_Y = 52
ACT_PANEL_W = 216
ACT_PANEL_H = 176

BAR_LABEL_X = 434
BAR_X = 462
BAR_W = 224
BAR_Y0 = 52
BAR_ROW_H = 23
BAR_H = 15

# Non-overlapping helper asset slots inside y = 48..321 (neutral-gray 1D strips, Display-P3 immune!):
C5_LUTX_POS = (18, 276)      # 6x6
C5_LUTY_POS = (28, 276)      # 6x6
ANCHOR_MASK_X = 40
ANCHOR_MASK_Y = 276          # 6x6
TILE_LUTX_POS = (56, 260)    # W x 4 strip
TILE_LUTY_POS = (160, 236)   # 4 x H strip
IN_LUTX_POS = (18, 292)      # 140 x 4 strip
IN_LUTY_POS = (166, 52)      # 4 x 140 strip


@dataclass
class SVGSimResult:
    x0: np.ndarray
    is_light_bg: int
    layer_rgb_maps: Dict[str, np.ndarray]
    u_logits: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    pred: int


def load_weights(path: str = WEIGHTS_DEFAULT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    assert spec.get("format") == "nojs-svg-feconvolvematrix-cnn-v1"
    assert spec.get("input_size") == [28, 28]
    return spec


def q8(x: np.ndarray | float) -> np.ndarray:
    return np.round(np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * 255.0) / 255.0


def _fmt_num(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    s = f"{v:.5f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _fmt_list(vals: Sequence[float]) -> str:
    return " ".join(_fmt_num(float(v)) for v in vals)


def _flip_dense_3x3(weights_unflipped: Sequence[Sequence[float]]) -> List[float]:
    arr = np.asarray(weights_unflipped, dtype=np.float64)[::-1, ::-1]
    return [round(float(v), 6) for v in arr.reshape(-1)]


# ---------------------------------------------------------------------------
# Bit-Exact 8-Bit SVG Filter Reference Simulator
# ---------------------------------------------------------------------------

def svg_fe_convolve_2d(
    buf: np.ndarray,
    order: Sequence[int],
    tx: int,
    ty: int,
    kernel_flat: Sequence[float],
    divisor: float,
    bias: float,
) -> np.ndarray:
    ow, oh = int(order[0]), int(order[1])
    k = np.asarray(kernel_flat, dtype=np.float64).reshape(oh, ow)
    h, w = buf.shape[0], buf.shape[1]
    if buf.ndim == 2:
        padded = np.pad(buf, ((ty, oh - 1 - ty), (tx, ow - 1 - tx)), mode="constant", constant_values=0.0)
    else:
        padded = np.pad(buf, ((ty, oh - 1 - ty), (tx, ow - 1 - tx), (0, 0)), mode="constant", constant_values=0.0)
    acc = np.zeros_like(buf, dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            kv = k[oh - 1 - i, ow - 1 - j]
            if kv != 0.0:
                acc += kv * padded[i : i + h, j : j + w]
    return q8(acc / float(divisor) + float(bias))


def svg_lut_eval(x: np.ndarray, table_vals: Sequence[float]) -> np.ndarray:
    tbl = np.asarray(table_vals, dtype=np.float64)
    n = len(tbl) - 1
    pos = np.clip(x, 0.0, 1.0) * n
    idx = np.minimum(np.floor(pos).astype(np.int64), n - 1)
    frac = pos - idx
    return q8(tbl[idx] * (1.0 - frac) + tbl[idx + 1] * frac)


def preprocess_rgba_to_x0(rgba_28x28: np.ndarray, auto_polarity: bool = True) -> Tuple[np.ndarray, int]:
    arr = np.asarray(rgba_28x28, dtype=np.float64)
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
        alpha = np.ones((28, 28, 1), dtype=np.float64)
    elif arr.shape[-1] == 3:
        rgb = arr
        alpha = np.ones((28, 28, 1), dtype=np.float64)
    else:
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]

    if arr.max() > 1.001:
        rgb = rgb / 255.0
        if arr.ndim == 3 and arr.shape[-1] == 4:
            alpha = alpha / 255.0

    comp_rgb = q8(rgb * alpha)
    gray_raw = q8(0.2126 * comp_rgb[..., 0] + 0.7152 * comp_rgb[..., 1] + 0.0722 * comp_rgb[..., 2])
    if not auto_polarity:
        return gray_raw, 0

    corner_avg = svg_fe_convolve_2d(gray_raw, [5, 5], 2, 2, [1.0] * 25, 25.0, 0.0)
    is_light = 1 if float(corner_avg[2, 2]) >= 0.5 else 0
    x0 = q8(np.abs(gray_raw - is_light))
    return x0, is_light


def simulate_svg_cnn(rgba_28x28: np.ndarray, spec: dict, auto_polarity: bool = True) -> SVGSimResult:
    x0, is_light = preprocess_rgba_to_x0(rgba_28x28, auto_polarity=auto_polarity)
    layers = spec["layers"]

    c1_ch = [
        svg_fe_convolve_2d(x0, f["order"], f["targetX"], f["targetY"], f["kernelMatrix"], f["divisor"], f["bias"])
        for f in layers["conv1"]["filters"]
    ]
    c1_rgb = np.stack([np.stack([c1_ch[3 * g], c1_ch[3 * g + 1], c1_ch[3 * g + 2]], axis=-1) for g in range(4)], axis=0)

    p1_s = layers["pool1"]
    p1_rgb = np.stack(
        [svg_fe_convolve_2d(c1_rgb[g], p1_s["order"], p1_s["targetX"], p1_s["targetY"], p1_s["kernelMatrix"], p1_s["divisor"], 0.0)
         for g in range(4)],
        axis=0,
    )

    def run_pointwise_rgb(in_rgb: np.ndarray, pw_spec: dict) -> np.ndarray:
        out_groups = []
        for trip in pw_spec["triplets"]:
            acc = None
            for g, mat20 in enumerate(trip["color_matrices"]):
                m = np.asarray(mat20, dtype=np.float64).reshape(4, 5)[:3, :]
                part = q8(np.einsum("rc,hwc->hwr", m[:, :3], in_rgb[g]) + m[:, 4].reshape(1, 1, 3))
                acc = part if acc is None else q8(acc + part - 0.5)
            slopes = np.asarray(trip["rescale_slopes"], dtype=np.float64).reshape(1, 1, 3)
            intercepts = np.asarray(trip["rescale_intercepts"], dtype=np.float64).reshape(1, 1, 3)
            out_groups.append(q8(acc * slopes + intercepts))
        return np.stack(out_groups, axis=0)

    def run_depthwise_rgb(in_rgb: np.ndarray, dw_spec: dict, force_dense_3x3: bool) -> np.ndarray:
        n_groups = dw_spec["num_channels"] // 3
        flts = dw_spec["filters"]
        out_groups = []
        for g in range(n_groups):
            ch_outs = []
            for c_off in range(3):
                f = flts[3 * g + c_off]
                if force_dense_3x3:
                    k_flat = _flip_dense_3x3(f["weights_unflipped_3x3_or_5x5"])
                    order, tx, ty = [3, 3], 1, 1
                else:
                    k_flat = f["kernelMatrix"]
                    order, tx, ty = f["order"], f["targetX"], f["targetY"]
                conv_res = svg_fe_convolve_2d(in_rgb[g], order, tx, ty, k_flat, f["divisor"], f["bias"])
                ch_outs.append(conv_res[..., c_off])
            out_groups.append(np.stack(ch_outs, axis=-1))
        return np.stack(out_groups, axis=0)

    m2_rgb = run_pointwise_rgb(p1_rgb, layers["mix2"])
    c2_rgb = run_depthwise_rgb(m2_rgb, layers["dwconv2"], force_dense_3x3=False)
    p2_s = layers["pool2"]
    p2_rgb_28 = np.stack(
        [svg_fe_convolve_2d(c2_rgb[g], p2_s["order"], p2_s["targetX"], p2_s["targetY"], p2_s["kernelMatrix"], p2_s["divisor"], 0.0)
         for g in range(5)],
        axis=0,
    )

    p2_c6 = np.zeros((5, 6, 6, 3), dtype=np.float64)
    src_5 = [4, 8, 12, 16, 20]
    p2_c6[:, :5, :5, :] = p2_rgb_28[np.ix_(range(5), src_5, src_5, range(3))]

    m3_c6 = run_pointwise_rgb(p2_c6, layers["mix3"])
    c3_c6 = run_depthwise_rgb(m3_c6, layers["dwconv3"], force_dense_3x3=True)
    c4_c6 = run_depthwise_rgb(c3_c6, layers["dwconv4"], force_dense_3x3=True)

    raw_buffers = {"conv1": c1_rgb, "mix2": m2_rgb, "dwconv2": c2_rgb, "mix3": m3_c6, "dwconv3": c3_c6, "dwconv4": c4_c6}
    layer_rgb_maps: Dict[str, np.ndarray] = {}
    for key, _, _, _, coords, _, _, _ in LAYER_VIEWS:
        ix = np.ix_(range(raw_buffers[key].shape[0]), coords, coords, range(3))
        layer_rgb_maps[key] = raw_buffers[key][ix]

    h_anchor = c4_c6[:, 2, 2, :]
    u_pad = np.zeros(12, dtype=np.float64)
    for trip in layers["fc"]["triplets"]:
        o = trip["out_triplet"]
        acc = None
        for g, mat20 in enumerate(trip["color_matrices"]):
            m = np.asarray(mat20, dtype=np.float64).reshape(4, 5)[:3, :]
            part = q8(m[:, :3] @ h_anchor[g] + m[:, 4])
            acc = part if acc is None else q8(acc + part - 0.5)
        u_pad[3 * o : 3 * (o + 1)] = acc

    u_logits = u_pad[:10]
    decoded_logits = (u_logits - 0.5) * (2.0 * float(layers["fc"]["logit_half_range_M"]))

    sm = layers["softmax"]
    u_max = np.max(u_logits)
    r = float(sm["diff_scale_r"])
    d = q8(r * u_logits - r * u_max + 1.0)
    e = svg_lut_eval(d, sm["exp_table_values"])
    bar_e = float(q8(np.sum(e) / 10.0))
    v_full = float(svg_lut_eval(np.array([bar_e]), sm["logsum_table_values"])[0])
    p_log = q8(d - float(sm["logsum_sub_k"]) * v_full)
    probs = svg_lut_eval(p_log, sm["exp_table_values"])

    return SVGSimResult(x0=x0, is_light_bg=is_light, layer_rgb_maps=layer_rgb_maps,
                        u_logits=u_logits, logits=decoded_logits, probs=probs, pred=int(np.argmax(probs)))


# ---------------------------------------------------------------------------
# External Asset PNG Generators (`cnn/assets/*.png`)
# ---------------------------------------------------------------------------

def build_1d_gray_lut(
    length_phys: int,
    coords: Sequence[int],
    cell_zoom_phys: int,
    dead_coord: int,
    src_box_phys: int,
    horizontal: bool,
    thickness_phys: int = 8,
) -> Image.Image:
    """Build a Display-P3-immune & linear-light-quantization-immune neutral grayscale (`R=G=B`, `A=255`)
    1D periodic LUT strip with all byte values in `[128 .. 128 + src_box_phys - 1]` (`>= 128` bright sRGB!).
    Because `<feTile>` repeats the `src_box_phys` source patch periodically, `(target - idx) % src_box_phys`
    is always in `[0 .. src_box_phys - 1]`, avoiding dark sRGB values (`< 124`) that CoreImage linear-light
    intermediate buffers quantize across multi-stage filter chains.
    """
    idx = np.arange(length_phys, dtype=np.int32)
    target = np.full(length_phys, int(dead_coord), dtype=np.int32)
    for ci, sc in enumerate(coords):
        i0, i1 = ci * cell_zoom_phys, (ci + 1) * cell_zoom_phys
        if i1 <= length_phys:
            target[i0:i1] = int(sc)

    delta = (target - idx) % int(src_box_phys)
    raw = np.round(127.5 + delta * (255.0 / 230.0)).astype(np.int32)
    assert raw.min() >= 128 and raw.max() <= 255, f"1D LUT out of [128..255]: [{raw.min()}, {raw.max()}]"
    vals = raw.astype(np.uint8)

    if horizontal:
        strip = np.broadcast_to(vals[None, :, None], (thickness_phys, length_phys, 3)).copy()
        alpha = np.full((thickness_phys, length_phys, 1), 255, dtype=np.uint8)
    else:
        strip = np.broadcast_to(vals[:, None, None], (length_phys, thickness_phys, 3)).copy()
        alpha = np.full((length_phys, thickness_phys, 1), 255, dtype=np.uint8)

    return Image.fromarray(np.concatenate([strip, alpha], axis=-1), mode="RGBA")


def center_shift_css(out_w_css: int, src_box_phys: int, dpr: int) -> int:
    src_css = src_box_phys // dpr
    return max(0, (out_w_css - src_css) // 2)


def write_filter_lut_assets(assets_dir: str = ASSETS_DIR) -> Dict[str, str]:
    os.makedirs(assets_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for dpr in (1, 2):
        for axis, horiz in (("x", True), ("y", False)):
            im = build_1d_gray_lut(
                INPUT_PREVIEW_SZ * dpr, list(range(28)), 5 * dpr, 27, 28, horizontal=horiz, thickness_phys=4 * dpr
            )
            fn = f"lut_in_{axis}_{dpr}x.png"
            im.save(os.path.join(assets_dir, fn), format="PNG", optimize=True)
            paths[f"in_{axis}_{dpr}x"] = f"assets/{fn}"

        for axis, horiz in (("x", True), ("y", False)):
            im_c5 = build_1d_gray_lut(6, [4, 8, 12, 16, 20], 1, 27, 28, horizontal=horiz, thickness_phys=6)
            fn_c5 = f"lut_c5_{axis}_{dpr}x.png"
            im_c5.save(os.path.join(assets_dir, fn_c5), format="PNG", optimize=True)
            paths[f"c5_{axis}_{dpr}x"] = f"assets/{fn_c5}"

        am = np.zeros((6, 6, 4), dtype=np.uint8)
        am[..., 3] = 255
        am[2, 2, :3] = 255
        fn_am = f"mask_anchor_{dpr}x.png"
        Image.fromarray(am, mode="RGBA").save(os.path.join(assets_dir, fn_am), format="PNG", optimize=True)
        paths[f"anchor_{dpr}x"] = f"assets/{fn_am}"

        bw_phys = BAR_W * dpr
        bh_phys = 4 * dpr
        b_vals = np.round(255.0 - 127.0 * np.arange(bw_phys, dtype=np.float64) / max(1, bw_phys - 1)).astype(np.uint8)
        b_rgb = np.broadcast_to(b_vals[None, :, None], (bh_phys, bw_phys, 3)).copy()
        b_a = np.full((bh_phys, bw_phys, 1), 255, dtype=np.uint8)
        fn_bar = f"lut_bar_{dpr}x.png"
        Image.fromarray(np.concatenate([b_rgb, b_a], axis=-1), mode="RGBA").save(
            os.path.join(assets_dir, fn_bar), format="PNG", optimize=True
        )
        paths[f"bar_{dpr}x"] = f"assets/{fn_bar}"

        for key, _, _, _, coords, src_box, _, (_, _, cell_zoom, _) in LAYER_VIEWS:
            if key == "dwconv4":
                continue  # dwconv4 uses exact <feTile> solid color swatches!
            tw_css = len(coords) * cell_zoom
            for axis, horiz in (("x", True), ("y", False)):
                t_im = build_1d_gray_lut(
                    tw_css * dpr, coords, cell_zoom * dpr, src_box - 1, src_box, horizontal=horiz, thickness_phys=4 * dpr
                )
                fn_t = f"lut_tile_{key}_{axis}_{dpr}x.png"
                t_im.save(os.path.join(assets_dir, fn_t), format="PNG", optimize=True)
                paths[f"tile_{key}_{axis}_{dpr}x"] = f"assets/{fn_t}"
    return paths


# ---------------------------------------------------------------------------
# Multi-Stage O(N) Linear VRAM SVG <filter> Compiler (<= 75 prims/stage for WebKit & Blink!)
# ---------------------------------------------------------------------------

COMPUTE_STAGE_IDS: Tuple[str, ...] = (
    "s1", "s2", "s3a", "s3b", "s4a", "s4b", "s5a", "s5b", "s6a", "s6b", "s6c", "visbarsa", "visbarsb",
)

SPLIT_VISUAL_LAYERS: Tuple[str, ...] = ("mix2", "dwconv2", "mix3", "dwconv3")


def _wrap_filter(
    fid: str,
    body: Sequence[str],
    x: int = 0,
    y: int = 0,
    w: int = STAGE_W,
    h: int = STAGE_H,
) -> str:
    return (
        f'<filter id="{fid}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" '
        f'x="{x}" y="{y}" width="{w}" height="{h}" color-interpolation-filters="sRGB">'
        f'{"".join(body)}</filter>'
    )


def compile_pre_filters() -> List[str]:
    """Optional SVG <feComponentTransfer> contrast boost on the <foreignObject> OCR input box.
    Using an SVG <g> filter avoids creating an HTML RenderLayer inside <foreignObject> (WebKit Bug 23113).
    """
    src_h = INPUT_PREVIEW_Y - INPUT_SRC_Y
    con = (
        f'<feComponentTransfer in="SourceGraphic" x="{INPUT_PREVIEW_X}" y="{INPUT_SRC_Y}" '
        f'width="{INPUT_PREVIEW_SZ}" height="{src_h}">'
        '<feFuncR type="linear" slope="1.65" intercept="-0.325"/>'
        '<feFuncG type="linear" slope="1.65" intercept="-0.325"/>'
        '<feFuncB type="linear" slope="1.65" intercept="-0.325"/>'
        '<feFuncA type="identity"/></feComponentTransfer>'
    )
    return [_wrap_filter("cnn-pre-con", [con], x=INPUT_PREVIEW_X, y=INPUT_SRC_Y, w=INPUT_PREVIEW_SZ, h=src_h)]


def _emit_2pass_gray_zoom(
    p: List[str],
    src_in: str,
    src_x: int,
    src_y: int,
    src_sz: int,
    dst_x: int,
    dst_y: int,
    dst_w: int,
    dst_h: int,
    sh_css: int,
    lutx_in: str,
    luty_in: str,
    u: float,
    disp_scale: str,
    out_name: str,
) -> None:
    """Display-P3-immune, WebKit-safe & linear-light-quantization-immune 2-pass periodic zoom.
    Tiles the `src_sz x src_sz` source patch across `(dst_x, dst_y)` before displacing so every LUT byte
    stays in `[128 .. 155]` (`>= 128` bright sRGB, zero multi-stage CoreImage quantization).
    """
    _ = sh_css
    pad_a = int(118 * u) + 4
    a_shift = 115.0 * u
    xs_w = dst_w + src_sz + 4
    xz_h = src_sz + pad_a
    yz_w = dst_w + pad_a
    ys_h = dst_h + src_sz + 4
    p.append(f'<feTile in="{lutx_in}" x="{dst_x}" y="{dst_y}" width="{xs_w}" height="{xz_h}" result="{out_name}_lxt"/>')
    p.append(f'<feTile in="{luty_in}" x="{dst_x}" y="{dst_y}" width="{yz_w}" height="{ys_h}" result="{out_name}_lyt"/>')
    p.append(f'<feOffset in="{src_in}" dx="{dst_x - src_x}" dy="{dst_y - src_y}" x="{dst_x}" y="{dst_y}" width="{src_sz}" height="{src_sz}" result="{out_name}_c"/>')
    p.append(f'<feTile in="{out_name}_c" x="{dst_x}" y="{dst_y}" width="{xs_w}" height="{src_sz}" result="{out_name}_ct"/>')
    p.append(
        f'<feOffset in="{out_name}_ct" dx="0" dy="{_fmt_num(a_shift)}" '
        f'x="{dst_x}" y="{dst_y}" width="{xs_w}" height="{xz_h}" result="{out_name}_xs"/>'
    )
    p.append(
        f'<feDisplacementMap in="{out_name}_xs" in2="{out_name}_lxt" x="{dst_x}" y="{dst_y}" width="{xs_w}" height="{xz_h}" '
        f'scale="{disp_scale}" xChannelSelector="R" yChannelSelector="A" result="{out_name}_xzr"/>'
    )
    p.append(f'<feOffset in="{out_name}_xzr" dx="0" dy="0" x="{dst_x}" y="{dst_y}" width="{dst_w}" height="{src_sz}" result="{out_name}_xz"/>')
    p.append(f'<feTile in="{out_name}_xz" x="{dst_x}" y="{dst_y}" width="{dst_w}" height="{ys_h}" result="{out_name}_xzt"/>')
    p.append(
        f'<feOffset in="{out_name}_xzt" dx="{_fmt_num(a_shift)}" dy="0" '
        f'x="{dst_x}" y="{dst_y}" width="{yz_w}" height="{ys_h}" result="{out_name}_ys"/>'
    )
    p.append(
        f'<feDisplacementMap in="{out_name}_ys" in2="{out_name}_lyt" x="{dst_x}" y="{dst_y}" width="{yz_w}" height="{ys_h}" '
        f'scale="{disp_scale}" xChannelSelector="A" yChannelSelector="R" result="{out_name}_yzr"/>'
    )
    p.append(f'<feOffset in="{out_name}_yzr" dx="0" dy="0" x="{dst_x}" y="{dst_y}" width="{dst_w}" height="{dst_h}" result="{out_name}"/>')


def _pack_rgb_at(p: List[str], r_in: str, g_in: str, b_in: str, x: int, y: int, sz: int, out_name: str) -> None:
    p.append(f'<feColorMatrix in="{r_in}" x="{x}" y="{y}" width="{sz}" height="{sz}" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{out_name}r"/>')
    p.append(f'<feColorMatrix in="{g_in}" x="{x}" y="{y}" width="{sz}" height="{sz}" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1" result="{out_name}g"/>')
    p.append(f'<feColorMatrix in="{b_in}" x="{x}" y="{y}" width="{sz}" height="{sz}" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1" result="{out_name}b"/>')
    p.append(f'<feComposite in="{out_name}r" in2="{out_name}g" x="{x}" y="{y}" width="{sz}" height="{sz}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{out_name}rg"/>')
    p.append(f'<feComposite in="{out_name}rg" in2="{out_name}b" x="{x}" y="{y}" width="{sz}" height="{sz}" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="{out_name}"/>')


def _emit_passthrough_merge(p: List[str], new_nodes: Sequence[str], keep_vram: bool = True) -> None:
    low_h = STAGE_H - VRAM_H
    p.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="{VRAM_H}" width="{STAGE_W}" height="{low_h}" result="pass_low"/>')
    nodes = []
    if keep_vram:
        p.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{STAGE_W}" height="{VRAM_H}" result="vram_prev"/>')
        nodes.append('<feMergeNode in="vram_prev"/>')
    nodes.extend(f'<feMergeNode in="{n}"/>' for n in new_nodes)
    nodes.append('<feMergeNode in="pass_low"/>')
    p.append(f'<feMerge>{"".join(nodes)}</feMerge>')


def compile_compute_stages(spec: dict, dpr: int) -> List[str]:
    """Compile compute & Softmax bar stages for a given dpr (1 or 2).
    Every stage reads ONLY `SourceGraphic` from its child SVG `<g>` stage and stays <= 75 primitives
    so both Chrome/Skia and Safari/CoreImage execute in O(N) time without hitting WebKit's filter caps.
    """
    u = 1.0 if dpr == 1 else 0.5
    t_sz = int(28 * u)
    c6_sz = int(6 * u)
    disp_scale = _fmt_num(230.0 * u)
    layers = spec["layers"]
    out_filters: List[str] = []

    # --- Stage 1 (normal & polarity-flipped): Grayscale + Auto-Polarity XOR -> x0 + Conv1 -> VRAM_CONV1 ---
    for inv_sfx, tbl_pol in (("", "0 1"), ("i", "1 0")):
        p1: List[str] = []
        p1.append(f'<feOffset in="SourceGraphic" dx="{-INPUT_PREVIEW_X}" dy="{-INPUT_SRC_Y}" x="0" y="0" width="{t_sz}" height="{t_sz}" result="s0"/>')
        p1.append(f'<feFlood x="0" y="0" width="{t_sz}" height="{t_sz}" flood-color="#000000" flood-opacity="1" result="bg0"/>')
        p1.append(f'<feComposite in="s0" in2="bg0" x="0" y="0" width="{t_sz}" height="{t_sz}" operator="over" result="so"/>')
        p1.append(f'<feColorMatrix in="so" x="0" y="0" width="{t_sz}" height="{t_sz}" type="matrix" values="0.2126 0.7152 0.0722 0 0 0.2126 0.7152 0.0722 0 0 0.2126 0.7152 0.0722 0 0 0 0 0 0 1" result="gr"/>')
        p1.append(f'<feConvolveMatrix in="gr" x="0" y="0" width="{t_sz}" height="{t_sz}" order="5 5" targetX="2" targetY="2" divisor="25" bias="0" kernelMatrix="{" ".join(["1"] * 25)}" edgeMode="none" preserveAlpha="true" result="cbg"/>')
        p1.append(f'<feOffset in="SourceGraphic" dx="{-ANCHOR_MASK_X}" dy="{-ANCHOR_MASK_Y}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" result="amask"/>')
        p1.append(f'<feComposite in="cbg" in2="amask" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="cbgm"/>')
        p1.append(f'<feMorphology in="cbgm" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="dilate" radius="2" result="cbgd"/>')
        p1.append('<feOffset in="cbgd" dx="0" dy="0" x="1" y="1" width="1" height="1" result="cbg1"/>')
        p1.append(f'<feTile in="cbg1" x="0" y="0" width="{t_sz}" height="{t_sz}" result="ctile"/>')
        p1.append(f'<feComponentTransfer in="ctile" x="0" y="0" width="{t_sz}" height="{t_sz}" result="isl"><feFuncR type="discrete" tableValues="{tbl_pol}"/><feFuncG type="discrete" tableValues="{tbl_pol}"/><feFuncB type="discrete" tableValues="{tbl_pol}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>')
        p1.append(f'<feBlend in="gr" in2="isl" x="0" y="0" width="{t_sz}" height="{t_sz}" mode="difference" result="x0"/>')

        s1_outs = ["x0"]
        for f in layers["conv1"]["filters"]:
            ch = f["channel"]
            g, c_off = ch // 3, ch % 3
            vx, vy = VRAM_CONV1[g]
            if c_off == 0:
                p1.append(f'<feOffset in="x0" dx="{vx}" dy="{vy}" x="{vx}" y="{vy}" width="{t_sz}" height="{t_sz}" result="x0_{ch}"/>')
            p1.append(
                f'<feConvolveMatrix in="x0_{3*g}" x="{vx}" y="{vy}" width="{t_sz}" height="{t_sz}" '
                f'order="5 5" targetX="2" targetY="2" divisor="1" bias="{_fmt_num(f["bias"])}" '
                f'kernelMatrix="{_fmt_list(f["kernelMatrix"])}" edgeMode="none" preserveAlpha="true" result="c1_{ch}"/>'
            )
        for g in range(4):
            vx, vy = VRAM_CONV1[g]
            _pack_rgb_at(p1, f"c1_{3*g}", f"c1_{3*g+1}", f"c1_{3*g+2}", vx, vy, t_sz, f"c1g_{g}")
            s1_outs.append(f"c1g_{g}")
        _emit_passthrough_merge(p1, s1_outs, keep_vram=False)
        out_filters.append(_wrap_filter(f"cnn-s1{inv_sfx}-{dpr}x", p1))

    # --- Stage 2: Pool1 + Mix2 -> VRAM_MIX2 (64 prims) ---
    p2: List[str] = []
    p2.append(
        f'<feConvolveMatrix in="SourceGraphic" x="30" y="0" width="120" height="{t_sz}" '
        f'order="2 2" targetX="0" targetY="0" divisor="4" bias="0" '
        f'kernelMatrix="1 1 1 1" edgeMode="none" preserveAlpha="true" result="p1_strip"/>'
    )
    s2_outs = []
    for trip in layers["mix2"]["triplets"]:
        o = trip["out_triplet"]
        ox, oy = VRAM_MIX2[o]
        acc = ""
        for g, mat20 in enumerate(trip["color_matrices"]):
            gx, gy = VRAM_CONV1[g]
            p2.append(f'<feOffset in="p1_strip" dx="{ox - gx}" dy="{oy - gy}" x="{ox}" y="{oy}" width="{t_sz}" height="{t_sz}" result="m2i_{o}_{g}"/>')
            p2.append(f'<feColorMatrix in="m2i_{o}_{g}" x="{ox}" y="{oy}" width="{t_sz}" height="{t_sz}" type="matrix" values="{_fmt_list(mat20)}" result="m2p_{o}_{g}"/>')
            if g == 0:
                acc = f"m2p_{o}_{g}"
            else:
                nxt = f"m2a_{o}_{g}"
                p2.append(f'<feComposite in="{acc}" in2="m2p_{o}_{g}" x="{ox}" y="{oy}" width="{t_sz}" height="{t_sz}" operator="arithmetic" k1="0" k2="1" k3="1" k4="-0.5" result="{nxt}"/>')
                acc = nxt
        sl, ic = trip["rescale_slopes"], trip["rescale_intercepts"]
        p2.append(
            f'<feComponentTransfer in="{acc}" x="{ox}" y="{oy}" width="{t_sz}" height="{t_sz}" result="m2g_{o}">'
            f'<feFuncR type="linear" slope="{_fmt_num(sl[0])}" intercept="{_fmt_num(ic[0])}"/>'
            f'<feFuncG type="linear" slope="{_fmt_num(sl[1])}" intercept="{_fmt_num(ic[1])}"/>'
            f'<feFuncB type="linear" slope="{_fmt_num(sl[2])}" intercept="{_fmt_num(ic[2])}"/>'
            f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>'
        )
        s2_outs.append(f"m2g_{o}")
    _emit_passthrough_merge(p2, s2_outs, keep_vram=True)
    out_filters.append(_wrap_filter(f"cnn-s2-{dpr}x", p2))

    # --- Stages 3a & 3b: DWConv2 -> VRAM_DW2 + Pool2 + 6x6 Compaction -> VRAM_P2C6 ---
    flts2 = layers["dwconv2"]["filters"]
    p24_sz = int(24 * u)
    c5x_x, c5x_y = C5_LUTX_POS
    c5y_x, c5y_y = C5_LUTY_POS
    for part_sfx, g_range in (("a", range(0, 3)), ("b", range(3, 5))):
        p3: List[str] = []
        s3_outs = []
        for g in g_range:
            mx, my = VRAM_MIX2[g]
            dx_dst, dy_dst = VRAM_DW2[g]
            p3.append(f'<feOffset in="SourceGraphic" dx="{dx_dst - mx}" dy="{dy_dst - my}" x="{dx_dst}" y="{dy_dst}" width="{t_sz}" height="{t_sz}" result="d2in_{g}"/>')
            for c_off, sfx in enumerate(("r", "g", "b")):
                f = flts2[3 * g + c_off]
                p3.append(
                    f'<feConvolveMatrix in="d2in_{g}" x="{dx_dst}" y="{dy_dst}" width="{t_sz}" height="{t_sz}" '
                    f'order="5 5" targetX="2" targetY="2" divisor="1" bias="{_fmt_num(f["bias"])}" '
                    f'kernelMatrix="{_fmt_list(f["kernelMatrix"])}" edgeMode="none" preserveAlpha="true" result="d2_{g}{sfx}"/>'
                )
            _pack_rgb_at(p3, f"d2_{g}r", f"d2_{g}g", f"d2_{g}b", dx_dst, dy_dst, t_sz, f"d2g_{g}")
            s3_outs.append(f"d2g_{g}")

            cx, cy = VRAM_P2C6[g]
            p3.append(
                f'<feConvolveMatrix in="d2g_{g}" x="{dx_dst}" y="{dy_dst}" width="{p24_sz}" height="{p24_sz}" '
                f'order="3 3" targetX="0" targetY="0" divisor="4" bias="0" '
                f'kernelMatrix="1 0 1 0 0 0 1 0 1" edgeMode="none" preserveAlpha="true" result="p2r_{g}"/>'
            )
            p3.append(f'<feOffset in="SourceGraphic" dx="{cx - c5x_x}" dy="{cy - c5x_y}" x="{cx}" y="{cy}" width="{c6_sz}" height="{c6_sz}" result="c5x_{g}"/>')
            p3.append(f'<feOffset in="SourceGraphic" dx="{cx - c5y_x}" dy="{cy - c5y_y}" x="{cx}" y="{cy}" width="{c6_sz}" height="{c6_sz}" result="c5y_{g}"/>')
            _emit_2pass_gray_zoom(p3, f"p2r_{g}", dx_dst, dy_dst, t_sz, cx, cy, c6_sz, c6_sz, 0, f"c5x_{g}", f"c5y_{g}", u, disp_scale, f"p2c_{g}")
            s3_outs.append(f"p2c_{g}")
        _emit_passthrough_merge(p3, s3_outs, keep_vram=True)
        out_filters.append(_wrap_filter(f"cnn-s3{part_sfx}-{dpr}x", p3))

    # --- Stages 4a & 4b: Mix3 (5 -> 8 triplets on 6x6 patches, 4 triplets/stage = 63 prims) -> VRAM_MIX3 ---
    m3_trips = layers["mix3"]["triplets"]
    for part_sfx, trip_slice in (("a", m3_trips[:4]), ("b", m3_trips[4:])):
        p4: List[str] = []
        s4_outs = []
        for trip in trip_slice:
            o = trip["out_triplet"]
            ox, oy = VRAM_MIX3[o]
            acc = ""
            for g, mat20 in enumerate(trip["color_matrices"]):
                gx, gy = VRAM_P2C6[g]
                p4.append(f'<feOffset in="SourceGraphic" dx="{ox - gx}" dy="{oy - gy}" x="{ox}" y="{oy}" width="{c6_sz}" height="{c6_sz}" result="m3i_{o}_{g}"/>')
                p4.append(f'<feColorMatrix in="m3i_{o}_{g}" x="{ox}" y="{oy}" width="{c6_sz}" height="{c6_sz}" type="matrix" values="{_fmt_list(mat20)}" result="m3p_{o}_{g}"/>')
                if g == 0:
                    acc = f"m3p_{o}_{g}"
                else:
                    nxt = f"m3a_{o}_{g}"
                    p4.append(f'<feComposite in="{acc}" in2="m3p_{o}_{g}" x="{ox}" y="{oy}" width="{c6_sz}" height="{c6_sz}" operator="arithmetic" k1="0" k2="1" k3="1" k4="-0.5" result="{nxt}"/>')
                    acc = nxt
            sl, ic = trip["rescale_slopes"], trip["rescale_intercepts"]
            p4.append(
                f'<feComponentTransfer in="{acc}" x="{ox}" y="{oy}" width="{c6_sz}" height="{c6_sz}" result="m3g_{o}">'
                f'<feFuncR type="linear" slope="{_fmt_num(sl[0])}" intercept="{_fmt_num(ic[0])}"/>'
                f'<feFuncG type="linear" slope="{_fmt_num(sl[1])}" intercept="{_fmt_num(ic[1])}"/>'
                f'<feFuncB type="linear" slope="{_fmt_num(sl[2])}" intercept="{_fmt_num(ic[2])}"/>'
                f'<feFuncA type="discrete" tableValues="1"/></feComponentTransfer>'
            )
            s4_outs.append(f"m3g_{o}")
        _emit_passthrough_merge(p4, s4_outs, keep_vram=True)
        out_filters.append(_wrap_filter(f"cnn-s4{part_sfx}-{dpr}x", p4))

    # --- Stages 5a & 5b: DWConv3 -> VRAM_DW3 & DWConv4 -> VRAM_DW4 (4 groups/stage = 75 prims) ---
    flts3 = layers["dwconv3"]["filters"]
    flts4 = layers["dwconv4"]["filters"]
    for part_sfx, g_range in (("a", range(0, 4)), ("b", range(4, 8))):
        p5: List[str] = []
        s5_outs = []
        for g in g_range:
            m3x, m3y = VRAM_MIX3[g]
            d3x, d3y = VRAM_DW3[g]
            d4x, d4y = VRAM_DW4[g]
            p5.append(f'<feOffset in="SourceGraphic" dx="{d3x - m3x}" dy="{d3y - m3y}" x="{d3x}" y="{d3y}" width="{c6_sz}" height="{c6_sz}" result="d3in_{g}"/>')
            for c_off, sfx in enumerate(("r", "g", "b")):
                f3 = flts3[3 * g + c_off]
                k3 = _fmt_list(_flip_dense_3x3(f3["weights_unflipped_3x3_or_5x5"]))
                p5.append(
                    f'<feConvolveMatrix in="d3in_{g}" x="{d3x}" y="{d3y}" width="{c6_sz}" height="{c6_sz}" '
                    f'order="3 3" targetX="1" targetY="1" divisor="1" bias="{_fmt_num(f3["bias"])}" '
                    f'kernelMatrix="{k3}" edgeMode="none" preserveAlpha="true" result="d3_{g}{sfx}"/>'
                )
            _pack_rgb_at(p5, f"d3_{g}r", f"d3_{g}g", f"d3_{g}b", d3x, d3y, c6_sz, f"d3g_{g}")
            s5_outs.append(f"d3g_{g}")

            p5.append(f'<feOffset in="d3g_{g}" dx="{d4x - d3x}" dy="{d4y - d3y}" x="{d4x}" y="{d4y}" width="{c6_sz}" height="{c6_sz}" result="d4in_{g}"/>')
            for c_off, sfx in enumerate(("r", "g", "b")):
                f4 = flts4[3 * g + c_off]
                k4 = _fmt_list(_flip_dense_3x3(f4["weights_unflipped_3x3_or_5x5"]))
                p5.append(
                    f'<feConvolveMatrix in="d4in_{g}" x="{d4x}" y="{d4y}" width="{c6_sz}" height="{c6_sz}" '
                    f'order="3 3" targetX="1" targetY="1" divisor="1" bias="{_fmt_num(f4["bias"])}" '
                    f'kernelMatrix="{k4}" edgeMode="none" preserveAlpha="true" result="d4_{g}{sfx}"/>'
                )
            _pack_rgb_at(p5, f"d4_{g}r", f"d4_{g}g", f"d4_{g}b", d4x, d4y, c6_sz, f"d4g_{g}")
            s5_outs.append(f"d4g_{g}")
        _emit_passthrough_merge(p5, s5_outs, keep_vram=True)
        out_filters.append(_wrap_filter(f"cnn-s5{part_sfx}-{dpr}x", p5))

    # --- Stages 6a & 6b: FC (24 -> 10, 2 triplets/stage = 56 prims) -> temp VRAM (260+5*o, 30) ---
    fc_trips = layers["fc"]["triplets"]
    for part_sfx, trip_slice in (("a", fc_trips[:2]), ("b", fc_trips[2:])):
        p6ab: List[str] = [
            f'<feOffset in="SourceGraphic" dx="{-ANCHOR_MASK_X}" dy="{-ANCHOR_MASK_Y}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" result="am6"/>'
        ]
        outs6: List[str] = []
        for trip in trip_slice:
            o = trip["out_triplet"]
            tx = 260 + 5 * o
            acc = ""
            for g, mat20 in enumerate(trip["color_matrices"]):
                gx, gy = VRAM_DW4[g]
                p6ab.append(f'<feOffset in="SourceGraphic" dx="{-gx}" dy="{-gy}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" result="fci_{o}_{g}"/>')
                p6ab.append(f'<feColorMatrix in="fci_{o}_{g}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" type="matrix" values="{_fmt_list(mat20)}" result="fcp_{o}_{g}"/>')
                if g == 0:
                    acc = f"fcp_{o}_{g}"
                else:
                    nxt = f"fca_{o}_{g}"
                    p6ab.append(f'<feComposite in="{acc}" in2="fcp_{o}_{g}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="arithmetic" k1="0" k2="1" k3="1" k4="-0.5" result="{nxt}"/>')
                    acc = nxt
            p6ab.append(f'<feComposite in="{acc}" in2="am6" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="fcm_{o}"/>')
            p6ab.append(f'<feMorphology in="fcm_{o}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="dilate" radius="2" result="fcd0_{o}"/>')
            p6ab.append(f'<feOffset in="fcd0_{o}" dx="{tx - 1}" dy="29" x="{tx}" y="30" width="1" height="1" result="fcs_{o}"/>')
            outs6.append(f"fcs_{o}")
        _emit_passthrough_merge(p6ab, outs6, keep_vram=True)
        out_filters.append(_wrap_filter(f"cnn-s6{part_sfx}-{dpr}x", p6ab))

    # --- Stage 6c: 10-pixel Logit Strip Assembly + Exact 8-Bit Softmax -> VRAM_SM_PR / WIN / NWIN (40 prims) ---
    p6c: List[str] = [
        f'<feOffset in="SourceGraphic" dx="{-ANCHOR_MASK_X}" dy="{-ANCHOR_MASK_Y}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" result="am6"/>'
    ]
    ch_extract = [
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1",
        "0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1",
        "0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1",
    ]
    sx_pr, sy_pr = VRAM_SM_PR
    sx_w, sy_w = VRAM_SM_WIN
    sx_nw, sy_nw = VRAM_SM_NWIN
    strip_nodes = []
    for d in range(10):
        o, c_idx = d // 3, d % 3
        tx = 260 + 5 * o
        p6c.append(f'<feColorMatrix in="SourceGraphic" x="{tx}" y="30" width="1" height="1" type="matrix" values="{ch_extract[c_idx]}" result="uch_{d}"/>')
        p6c.append(f'<feOffset in="uch_{d}" dx="{sx_pr + d - tx}" dy="{sy_pr - 30}" x="{sx_pr + d}" y="{sy_pr}" width="1" height="1" result="upx_{d}"/>')
        strip_nodes.append(f'<feMergeNode in="upx_{d}"/>')
    p6c.append(f'<feMerge x="{sx_pr}" y="{sy_pr}" width="10" height="1" result="u_strip">{"".join(strip_nodes)}</feMerge>')

    sm = layers["softmax"]
    r_val = _fmt_num(sm["diff_scale_r"])
    nr_val = _fmt_num(-sm["diff_scale_r"])
    pk_val = _fmt_num(sm["logsum_sub_k"])
    nk_val = _fmt_num(-sm["logsum_sub_k"])
    exp_tbl = _fmt_list(sm["exp_table_values"])
    inv_lse_tbl = _fmt_list([1.0 - float(v) for v in sm["logsum_table_values"]])
    sum_w = 10 * dpr

    p6c.append(f'<feMorphology in="u_strip" x="{sx_pr}" y="{sy_pr}" width="10" height="1" operator="dilate" radius="10 1" result="u_max"/>')
    p6c.append(f'<feComposite in="u_strip" in2="u_max" x="{sx_pr}" y="{sy_pr}" width="10" height="1" operator="arithmetic" k1="0" k2="{r_val}" k3="{nr_val}" k4="1" result="sm_d"/>')
    p6c.append(f'<feComponentTransfer in="sm_d" x="{sx_pr}" y="{sy_pr}" width="10" height="1" result="sm_e"><feFuncR type="table" tableValues="{exp_tbl}"/><feFuncG type="table" tableValues="{exp_tbl}"/><feFuncB type="table" tableValues="{exp_tbl}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>')
    p6c.append(f'<feConvolveMatrix in="sm_e" x="{sx_pr}" y="{sy_pr}" width="10" height="1" order="{sum_w} 1" targetX="0" targetY="0" divisor="{sum_w}" bias="0" kernelMatrix="{" ".join(["1"] * sum_w)}" edgeMode="none" preserveAlpha="true" result="sm_m0"/>')
    p6c.append(f'<feOffset in="am6" dx="{_fmt_num(sx_pr - 2 * u)}" dy="{_fmt_num(sy_pr - 2 * u)}" x="{sx_pr}" y="{sy_pr}" width="1" height="1" result="m0m"/>')
    p6c.append(f'<feComposite in="sm_m0" in2="m0m" x="{sx_pr}" y="{sy_pr}" width="1" height="1" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="sm_mm"/>')
    p6c.append(f'<feMorphology in="sm_mm" x="{sx_pr}" y="{sy_pr}" width="1" height="1" operator="dilate" radius="1" result="sm_mp"/>')
    p6c.append(f'<feTile in="sm_mp" x="{sx_pr}" y="{sy_pr}" width="10" height="1" result="sm_be"/>')
    p6c.append(f'<feComponentTransfer in="sm_be" x="{sx_pr}" y="{sy_pr}" width="10" height="1" result="sm_iv"><feFuncR type="table" tableValues="{inv_lse_tbl}"/><feFuncG type="table" tableValues="{inv_lse_tbl}"/><feFuncB type="table" tableValues="{inv_lse_tbl}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>')
    p6c.append(f'<feComposite in="sm_d" in2="sm_iv" x="{sx_pr}" y="{sy_pr}" width="10" height="1" operator="arithmetic" k1="0" k2="1" k3="{pk_val}" k4="{nk_val}" result="sm_pl"/>')
    p6c.append(f'<feComponentTransfer in="sm_pl" x="{sx_pr}" y="{sy_pr}" width="10" height="1" result="sm_pr"><feFuncR type="table" tableValues="{exp_tbl}"/><feFuncG type="table" tableValues="{exp_tbl}"/><feFuncB type="table" tableValues="{exp_tbl}"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>')

    p6c.append(f'<feOffset in="sm_d" dx="{sx_w - sx_pr}" dy="0" x="{sx_w}" y="{sy_w}" width="10" height="1" result="smd_w"/>')
    p6c.append(f'<feComponentTransfer in="smd_w" x="{sx_w}" y="{sy_w}" width="10" height="1" result="sm_win"><feFuncR type="linear" slope="255" intercept="-252.5"/><feFuncG type="linear" slope="255" intercept="-252.5"/><feFuncB type="linear" slope="255" intercept="-252.5"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>')

    p6c.append(f'<feOffset in="sm_d" dx="{sx_nw - sx_pr}" dy="0" x="{sx_nw}" y="{sy_nw}" width="10" height="1" result="smd_nw"/>')
    p6c.append(f'<feComponentTransfer in="smd_nw" x="{sx_nw}" y="{sy_nw}" width="10" height="1" result="sm_nwin"><feFuncR type="linear" slope="-255" intercept="253.5"/><feFuncG type="linear" slope="-255" intercept="253.5"/><feFuncB type="linear" slope="-255" intercept="253.5"/><feFuncA type="discrete" tableValues="1"/></feComponentTransfer>')

    _emit_passthrough_merge(p6c, ["sm_pr", "sm_win", "sm_nwin"], keep_vram=True)
    out_filters.append(_wrap_filter(f"cnn-s6c-{dpr}x", p6c))

    # --- Shared Softmax Bar Stages: visbarsa (digits 0..4 = 60 prims) & visbarsb (digits 5..9 = 60 prims) ---
    h_part = 5 * BAR_ROW_H
    for part_sfx, d_range in (("a", range(0, 5)), ("b", range(5, 10))):
        pb: List[str] = []
        bar_val_nodes, bar_win_nodes, bar_nwin_nodes, bar_trk_nodes, dot_nodes = [], [], [], [], []
        y_top = BAR_Y0 + d_range[0] * BAR_ROW_H
        for d in d_range:
            yb = BAR_Y0 + d * BAR_ROW_H
            pb.append(f'<feFlood x="{BAR_X}" y="{yb}" width="{BAR_W}" height="{BAR_H}" flood-color="#0b1324" flood-opacity="1" result="btk_{d}"/>')
            bar_trk_nodes.append(f'<feMergeNode in="btk_{d}"/>')

            pb.append(f'<feOffset in="SourceGraphic" dx="{BAR_X - (sx_pr + d)}" dy="{yb - sy_pr}" x="{BAR_X}" y="{yb}" width="1" height="1" result="bp_{d}"/>')
            pb.append(f'<feTile in="bp_{d}" x="{BAR_X}" y="{yb}" width="{BAR_W}" height="{BAR_H}" result="bpt_{d}"/>')
            bar_val_nodes.append(f'<feMergeNode in="bpt_{d}"/>')

            pb.append(f'<feOffset in="SourceGraphic" dx="{BAR_X - (sx_w + d)}" dy="{yb - sy_w}" x="{BAR_X}" y="{yb}" width="1" height="1" result="bw_{d}"/>')
            pb.append(f'<feTile in="bw_{d}" x="{BAR_X}" y="{yb}" width="{BAR_W}" height="{BAR_H}" result="bwt_{d}"/>')
            bar_win_nodes.append(f'<feMergeNode in="bwt_{d}"/>')

            pb.append(f'<feOffset in="SourceGraphic" dx="{BAR_X - (sx_nw + d)}" dy="{yb - sy_nw}" x="{BAR_X}" y="{yb}" width="1" height="1" result="bn_{d}"/>')
            pb.append(f'<feTile in="bn_{d}" x="{BAR_X}" y="{yb}" width="{BAR_W}" height="{BAR_H}" result="bnt_{d}"/>')
            bar_nwin_nodes.append(f'<feMergeNode in="bnt_{d}"/>')

            dx_dot = (BAR_LABEL_X - 12) - (sx_w + d)
            pb.append(f'<feOffset in="SourceGraphic" dx="{dx_dot}" dy="{(yb + 4) - sy_w}" x="{BAR_LABEL_X - 12}" y="{yb + 4}" width="1" height="1" result="wd_{d}"/>')
            pb.append(f'<feTile in="wd_{d}" x="{BAR_LABEL_X - 12}" y="{yb + 4}" width="7" height="7" result="wdt_{d}"/>')
            dot_nodes.append(f'<feMergeNode in="wdt_{d}"/>')

        pb.append(f'<feMerge x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" result="b_val">{"".join(bar_val_nodes)}</feMerge>')
        pb.append(f'<feMerge x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" result="b_win">{"".join(bar_win_nodes)}</feMerge>')
        pb.append(f'<feMerge x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" result="b_nwin">{"".join(bar_nwin_nodes)}</feMerge>')
        pb.append(f'<feMerge x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" result="b_trk">{"".join(bar_trk_nodes)}</feMerge>')

        # Clean 1-row tiled bright ramp (y=292 strip, [128..255] bright sRGB -> zero linear-light quantization or 2D gradient dither!)
        pb.append(f'<feOffset in="SourceGraphic" dx="0" dy="{y_top - 292}" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="1" result="bg_1r"/>')
        pb.append(f'<feTile in="bg_1r" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" result="bg_clean"/>')
        pb.append(f'<feComposite in="b_val" in2="bg_clean" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" operator="arithmetic" k1="0" k2="180" k3="361.42" k4="-361.32" result="b_mraw"/>')
        pb.append(f'<feComposite in="b_mraw" in2="b_trk" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" operator="in" result="b_mask"/>')
        pb.append(f'<feComposite in="b_mask" in2="b_win" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="m_win"/>')
        pb.append(f'<feComposite in="b_mask" in2="b_nwin" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="m_oth"/>')
        pb.append(f'<feColorMatrix in="m_win" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" type="matrix" values="0.204 0 0 0 0 0 0.827 0 0 0 0 0 0.600 0 0 1 0 0 0 0" result="c_win"/>')
        pb.append(f'<feColorMatrix in="m_oth" x="{BAR_X}" y="{y_top}" width="{BAR_W}" height="{h_part}" type="matrix" values="0.220 0 0 0 0 0 0.741 0 0 0 0 0 0.973 0 0 1 0 0 0 0" result="c_oth"/>')
        pb.append(f'<feMerge x="{BAR_LABEL_X - 12}" y="{y_top}" width="7" height="{h_part}" result="w_dotsr">{"".join(dot_nodes)}</feMerge>')
        pb.append(f'<feColorMatrix in="w_dotsr" x="{BAR_LABEL_X - 12}" y="{y_top}" width="7" height="{h_part}" type="matrix" values="0.204 0 0 0 0 0 0.827 0 0 0 0 0 0.600 0 0 1 0 0 0 0" result="w_dots"/>')
        pb.append('<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="420" height="322" result="pass_l"/>')
        if part_sfx == "a":
            pb.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{BAR_X}" y="292" width="{BAR_W}" height="4" result="pass_r"/>')
        else:
            pb.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="420" y="{BAR_Y0}" width="270" height="{h_part}" result="pass_r"/>')
        pb.append('<feMerge><feMergeNode in="pass_l"/><feMergeNode in="pass_r"/><feMergeNode in="b_trk"/><feMergeNode in="c_oth"/><feMergeNode in="c_win"/><feMergeNode in="w_dots"/></feMerge>')
        out_filters.append(_wrap_filter(f"cnn-visbars{part_sfx}-{dpr}x", pb))

    return out_filters


def compile_visual_stage(layer_view: str, dpr: int) -> List[str]:
    """Compile outermost layer-inspector visual filter(s) (`<= 69` prims each!).
    Reads `x0` and the selected layer's RGB tiles from `SourceGraphic` while passing through
    the rendered Softmax bars (`x=416..720, y=48..322`) and stripping off-screen VRAM/LUT helpers.
    """
    u = 1.0 if dpr == 1 else 0.5
    disp_scale = _fmt_num(230.0 * u)
    in_sh_css = center_shift_css(INPUT_PREVIEW_SZ, 28, dpr)
    t_sz = int(28 * u)
    c6_sz = int(6 * u)
    ix_x, ix_y = IN_LUTX_POS
    iy_x, iy_y = IN_LUTY_POS

    lv_map = {k: (vslots, coords, src_box, n_rgb, layout) for k, _, _, vslots, coords, src_box, n_rgb, layout in LAYER_VIEWS}
    vslots, coords, src_box, n_rgb, (cols, rows, cell_zoom, gap) = lv_map[layer_view]
    tile_w = len(coords) * cell_zoom
    tile_h = len(coords) * cell_zoom
    t_sh_css = center_shift_css(tile_w, src_box, dpr)
    src_sz = int(src_box * u)
    grid_w = cols * tile_w + (cols - 1) * gap
    grid_h = rows * tile_h + (rows - 1) * gap
    start_x = ACT_PANEL_X + (ACT_PANEL_W - grid_w) // 2
    start_y = ACT_PANEL_Y + (ACT_PANEL_H - grid_h) // 2
    tlx_x, tlx_y = TILE_LUTX_POS
    tly_x, tly_y = TILE_LUTY_POS

    def _emit_input_preview(p: List[str]) -> None:
        p.append(f'<feFlood x="{INPUT_PREVIEW_X}" y="{INPUT_PREVIEW_Y}" width="{INPUT_PREVIEW_SZ}" height="{INPUT_PREVIEW_SZ}" flood-color="#000000" flood-opacity="1" result="in_bg"/>')
        p.append(f'<feOffset in="SourceGraphic" dx="{INPUT_PREVIEW_X - ix_x}" dy="{INPUT_PREVIEW_Y - ix_y}" x="{INPUT_PREVIEW_X}" y="{INPUT_PREVIEW_Y}" width="{INPUT_PREVIEW_SZ}" height="4" result="in_lx0"/>')
        p.append(f'<feOffset in="SourceGraphic" dx="{INPUT_PREVIEW_X - iy_x}" dy="{INPUT_PREVIEW_Y - iy_y}" x="{INPUT_PREVIEW_X}" y="{INPUT_PREVIEW_Y}" width="4" height="{INPUT_PREVIEW_SZ}" result="in_ly0"/>')
        _emit_2pass_gray_zoom(p, "SourceGraphic", 0, 0, t_sz, INPUT_PREVIEW_X, INPUT_PREVIEW_Y, INPUT_PREVIEW_SZ, INPUT_PREVIEW_SZ, in_sh_css, "in_lx0", "in_ly0", u, disp_scale, "vis_input")

    def _emit_tiles(p: List[str], g_range: Sequence[int]) -> List[str]:
        if layer_view == "dwconv4":
            p.append(f'<feOffset in="SourceGraphic" dx="{-ANCHOR_MASK_X}" dy="{-ANCHOR_MASK_Y}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" result="am_vis"/>')
        act_nodes: List[str] = []
        for g in g_range:
            col, row = g % cols, g // cols
            tx = start_x + col * (tile_w + gap)
            ty = start_y + row * (tile_h + gap)
            vx, vy = vslots[g]
            p.append(f'<feFlood x="{tx - 1}" y="{ty - 1}" width="{tile_w + 2}" height="{tile_h + 2}" flood-color="#26354d" flood-opacity="1" result="tbg_{g}"/>')
            p.append(f'<feFlood x="{tx}" y="{ty}" width="{tile_w}" height="{tile_h}" flood-color="#050811" flood-opacity="1" result="tfl_{g}"/>')
            if layer_view == "dwconv4":
                p.append(f'<feOffset in="SourceGraphic" dx="{-vx}" dy="{-vy}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" result="d4c_{g}"/>')
                p.append(f'<feComposite in="d4c_{g}" in2="am_vis" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="arithmetic" k1="1" k2="0" k3="0" k4="0" result="d4m_{g}"/>')
                p.append(f'<feMorphology in="d4m_{g}" x="0" y="0" width="{c6_sz}" height="{c6_sz}" operator="dilate" radius="2" result="d4d_{g}"/>')
                p.append(f'<feOffset in="d4d_{g}" dx="{tx - 1}" dy="{ty - 1}" x="{tx}" y="{ty}" width="1" height="1" result="d4p_{g}"/>')
                p.append(f'<feTile in="d4p_{g}" x="{tx}" y="{ty}" width="{tile_w}" height="{tile_h}" result="atl_{g}"/>')
            else:
                p.append(f'<feOffset in="SourceGraphic" dx="{tx - tlx_x}" dy="{ty - tlx_y}" x="{tx}" y="{ty}" width="{tile_w}" height="4" result="tlx0_{g}"/>')
                p.append(f'<feOffset in="SourceGraphic" dx="{tx - tly_x}" dy="{ty - tly_y}" x="{tx}" y="{ty}" width="4" height="{tile_h}" result="tly0_{g}"/>')
                _emit_2pass_gray_zoom(p, "SourceGraphic", vx, vy, src_sz, tx, ty, tile_w, tile_h, t_sh_css, f"tlx0_{g}", f"tly0_{g}", u, disp_scale, f"atl_{g}")
            act_nodes.append(f'<feMergeNode in="tbg_{g}"/><feMergeNode in="tfl_{g}"/><feMergeNode in="atl_{g}"/>')
        return act_nodes

    if layer_view not in SPLIT_VISUAL_LAYERS:
        p_single: List[str] = []
        _emit_input_preview(p_single)
        nodes_all = _emit_tiles(p_single, range(n_rgb))
        p_single.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="416" y="{VRAM_H}" width="{STAGE_W - 416}" height="{STAGE_H - VRAM_H}" result="pass_bars"/>')
        p_single.append(
            '<feMerge><feMergeNode in="in_bg"/><feMergeNode in="vis_input"/>'
            f'{"".join(nodes_all)}<feMergeNode in="pass_bars"/></feMerge>'
        )
        return [_wrap_filter(f"cnn-vis-{layer_view}-{dpr}x", p_single)]

    # Split 2-row zoom layers into Row 0 (`visa`) + Row 1 (`visb`) so each filter stays <= 66 prims!
    pa: List[str] = []
    _emit_input_preview(pa)
    nodes_r0 = _emit_tiles(pa, range(0, cols))
    pa.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="0" y="0" width="{STAGE_W}" height="{STAGE_H}" result="pass_all"/>')
    pa.append(
        '<feMerge><feMergeNode in="pass_all"/><feMergeNode in="in_bg"/><feMergeNode in="vis_input"/>'
        f'{"".join(nodes_r0)}</feMerge>'
    )

    pb_vis: List[str] = []
    nodes_r1 = _emit_tiles(pb_vis, range(cols, n_rgb))
    pb_vis.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{INPUT_PREVIEW_X}" y="{INPUT_PREVIEW_Y}" width="{INPUT_PREVIEW_SZ}" height="{INPUT_PREVIEW_SZ}" result="pass_in"/>')
    pb_vis.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="{ACT_PANEL_X}" y="{start_y - 2}" width="{ACT_PANEL_W}" height="{tile_h + 4}" result="pass_r0"/>')
    pb_vis.append(f'<feOffset in="SourceGraphic" dx="0" dy="0" x="416" y="{VRAM_H}" width="{STAGE_W - 416}" height="{STAGE_H - VRAM_H}" result="pass_bars"/>')
    pb_vis.append(
        '<feMerge><feMergeNode in="pass_in"/><feMergeNode in="pass_r0"/>'
        f'{"".join(nodes_r1)}<feMergeNode in="pass_bars"/></feMerge>'
    )
    return [
        _wrap_filter(f"cnn-visa-{layer_view}-{dpr}x", pa),
        _wrap_filter(f"cnn-visb-{layer_view}-{dpr}x", pb_vis),
    ]
