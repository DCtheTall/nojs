#!/usr/bin/env python3
"""
Standalone Pure HTML5 + CSS + Inline SVG Generator for MNIST CNN OCR (`cnn/index.html`).

Generates a fast, zero-JavaScript HTML5 application executing our trained 3,004-parameter
BatchNorm-free Convolutional Neural Network (`weights.json`, 98.65% exact 8-bit SVG MNIST test
accuracy, 99.1% printed/typed font OCR accuracy) across nested `<svg><g>` filter stages:
  - O(N) linear texture caching between `<g>` stages (`<= 76` primitives/filter for Safari/WebKit & Chrome/Skia)
  - Stage 1 Alpha-neutral multi-edge polarity detection + automatic 8-bit SVG Auto-Levels
  - External PNG assets in `cnn/assets/` loaded via relative `<img>` and `<image href="assets/...">`
  - Zero `<feImage>` tags inside `<filter>` and every `<feConvolveMatrix>` `<= 5x5` (GPU fast path)
  - WebKit-safe `<foreignObject>` `contenteditable="true"` drag-and-drop / paste / type OCR input slot
  - Radio selector to inspect intermediate convolutional layer activations as RGB color images
"""

from __future__ import annotations

import html
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svg_cnn import (
    ACT_PANEL_H,
    ACT_PANEL_W,
    ACT_PANEL_X,
    ACT_PANEL_Y,
    ANCHOR_MASK_X,
    ANCHOR_MASK_Y,
    ASSETS_DIR,
    BAR_H,
    BAR_LABEL_X,
    BAR_ROW_H,
    BAR_W,
    BAR_X,
    BAR_Y0,
    C5_LUTX_POS,
    C5_LUTY_POS,
    CNN_DIR,
    COMPUTE_STAGE_IDS,
    IN_LUTX_POS,
    IN_LUTY_POS,
    INPUT_PREVIEW_SZ,
    INPUT_PREVIEW_X,
    INPUT_PREVIEW_Y,
    INPUT_SRC_Y,
    LAYER_VIEWS,
    STAGE_H,
    STAGE_W,
    THREE_ROW_VISUAL_LAYERS,
    TILE_LUTX_POS,
    TILE_LUTY_POS,
    WEIGHTS_DEFAULT_PATH,
    compile_compute_stages,
    compile_pre_filters,
    compile_visual_stage,
    load_weights,
    simulate_svg_cnn,
    write_filter_lut_assets,
)


LAYER_DESCRIPTIONS: Dict[str, str] = {
    "conv1":   "Stage 1 · 15 learned 5×5 feConvolveMatrix spatial filters packed into 5 RGB color images (24×24).",
    "mix2":    "Stage 2a · 2×2 feConvolveMatrix AvgPool + signed 1×1 feColorMatrix mixing (15→24 ch = 8 RGB maps, 12×12).",
    "dwconv2": "Stage 2b · 24 depthwise dilated 3×3 feConvolveMatrix filters capturing curves & corners (8 RGB maps, 10×10).",
    "mix3":    "Stage 3a · 2×2 AvgPool + signed 1×1 feColorMatrix mixing (24→36 ch = 12 RGB color images, 5×5 part detectors).",
    "dwconv3": "Stage 3b · 36 depthwise 3×3 feConvolveMatrix filters on compacted 5×5 grid (12 RGB color maps, 3×3).",
    "dwconv4": "Stage 4 · 36 global 3×3 feConvolveMatrix aggregators collapsing 3×3 → 1×1 bottleneck code (12 RGB color swatches).",
}

PAPER_PALETTES: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]] = [
    ((248, 245, 236), (22, 28, 45),   "Cream Paper"),
    ((238, 246, 255), (15, 52, 115),  "Blue Ink"),
    ((252, 242, 238), (125, 24, 28),  "Red Stamp"),
    ((240, 250, 244), (16, 78, 48),   "Green Ticket"),
    ((244, 244, 246), (30, 32, 36),   "Thermal Scan"),
    ((253, 248, 228), (68, 42, 16),   "Sepia Ledger"),
    ((243, 238, 252), (58, 22, 108),  "Violet Tag"),
    ((236, 248, 250), (14, 68, 82),   "Teal Check"),
    ((250, 246, 240), (40, 40, 40),   "Newsprint"),
    ((245, 247, 250), (18, 24, 38),   "Matte Card"),
]


def minify_css(raw: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([{};:,>~+])\s*", r"\1", s)
    return s.replace(";}", "}").strip()


def _save_png(rgb_arr: np.ndarray, rel_name: str, out_size: int = 84) -> str:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    img = Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB")
    if out_size != rgb_arr.shape[0]:
        img = img.resize((out_size, out_size), resample=Image.Resampling.NEAREST)
    img.save(os.path.join(ASSETS_DIR, rel_name), format="PNG", optimize=True)
    return f"assets/{rel_name}"


def write_sample_digit_assets(spec: dict) -> Tuple[str, List[Tuple[int, str, int]], List[Tuple[int, str, str, int]]]:
    by_digit: Dict[int, List[dict]] = {d: [] for d in range(10)}
    for gs in spec["golden_samples"]:
        by_digit[int(gs["label"])].append(gs)

    mnist_cards: List[Tuple[int, str, int]] = []
    paper_cards: List[Tuple[int, str, str, int]] = []
    default_rel = ""
    rng = np.random.default_rng(2026)

    for d in range(10):
        candidates = sorted(by_digit[d], key=lambda g: float(g["svg_probs"][d]), reverse=True)
        gs0 = candidates[0]
        u8_0 = np.array(gs0["pixels_u8"], dtype=np.uint8)
        rgb0 = np.stack([u8_0, u8_0, u8_0], axis=-1)
        rel0 = _save_png(rgb0, f"mnist_{d}.png", out_size=84)
        pct0 = int(round(float(gs0["svg_probs"][d]) * 100))
        mnist_cards.append((d, rel0, pct0))
        if d == 7:
            default_rel = rel0

        bg_rgb, ink_rgb, style_lbl = PAPER_PALETTES[d]
        noise = rng.normal(0.0, 3.5, size=(28, 28, 1))
        chosen_paper = None
        chosen_sim = None
        for cand in candidates[1:] + candidates[:1]:
            alpha_s = np.array(cand["pixels_u8"], dtype=np.float64) / 255.0
            paper = np.zeros((28, 28, 3), dtype=np.float64)
            for c in range(3):
                paper[..., c] = bg_rgb[c] * (1.0 - alpha_s) + ink_rgb[c] * alpha_s
            paper_u8 = np.clip(np.round(paper + noise), 0, 255).astype(np.uint8)
            sim = simulate_svg_cnn(paper_u8, spec, auto_polarity=True)
            if sim.pred == d and float(sim.probs[d]) >= 0.85:
                chosen_paper = paper_u8
                chosen_sim = sim
                break
        if chosen_paper is None:
            alpha_s = np.array(candidates[0]["pixels_u8"], dtype=np.float64) / 255.0
            paper = np.zeros((28, 28, 3), dtype=np.float64)
            for c in range(3):
                paper[..., c] = bg_rgb[c] * (1.0 - alpha_s) + ink_rgb[c] * alpha_s
            chosen_paper = np.clip(np.round(paper), 0, 255).astype(np.uint8)
            chosen_sim = simulate_svg_cnn(chosen_paper, spec, auto_polarity=True)

        rel_p = _save_png(chosen_paper, f"paper_{d}.png", out_size=84)
        pct_p = int(round(float(chosen_sim.probs[chosen_sim.pred]) * 100))
        paper_cards.append((d, style_lbl, rel_p, pct_p))

    return default_rel, mnist_cards, paper_cards


def generate_html(weights_path: str = WEIGHTS_DEFAULT_PATH) -> str:
    spec = load_weights(weights_path)
    metrics = spec.get("metrics", {})
    lut_paths = write_filter_lut_assets(ASSETS_DIR)
    default_rel, mnist_cards, paper_cards = write_sample_digit_assets(spec)

    svg_defs: List[str] = list(compile_pre_filters())
    for dpr in (1, 2):
        svg_defs.extend(compile_compute_stages(spec, dpr))

    dyn_css: List[str] = []
    hidpi_css: List[str] = []
    tile_luts_1x: List[str] = []
    tile_luts_2x: List[str] = []

    for sid in COMPUTE_STAGE_IDS:
        dyn_css.append(f".cnn-{sid}{{filter:url(#cnn-{sid}-1x)}}")
        hidpi_css.append(f".cnn-{sid}{{filter:url(#cnn-{sid}-2x)}}")

    dyn_css.append(".cnn-app:has(#opt-invert:checked) .cnn-s1{filter:url(#cnn-s1i-1x)}")
    hidpi_css.append(".cnn-app:has(#opt-invert:checked) .cnn-s1{filter:url(#cnn-s1i-2x)}")
    dyn_css.append(".cnn-app:has(#opt-contrast:checked) .cnn-pre{filter:url(#cnn-pre-con)}")

    lyr_active_lbl = ",".join(f".cnn-app:has(#lyr-{k}:checked) label[for=lyr-{k}]" for k, *_ in LAYER_VIEWS)
    lyr_active_meta = ",".join(f".cnn-app:has(#lyr-{k}:checked) label[for=lyr-{k}] .lyr-meta" for k, *_ in LAYER_VIEWS)
    dyn_css.append(f"{lyr_active_lbl}{{border-color:#38bdf8;background:#0c2136;box-shadow:0 0 0 1px rgba(56,189,248,.3)}}")
    dyn_css.append(f"{lyr_active_meta}{{color:#7dd3fc}}")

    tlx_x, tlx_y = TILE_LUTX_POS
    tly_x, tly_y = TILE_LUTY_POS

    for key, _, _, _, coords, _, _, (_, _, cell_zoom, _, _) in LAYER_VIEWS:
        for dpr in (1, 2):
            svg_defs.extend(compile_visual_stage(key, dpr))

        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visa{{filter:url(#cnn-visa-{key}-1x)}}")
        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visb{{filter:url(#cnn-visb-{key}-1x)}}")
        hidpi_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visa{{filter:url(#cnn-visa-{key}-2x)}}")
        hidpi_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visb{{filter:url(#cnn-visb-{key}-2x)}}")
        if key in THREE_ROW_VISUAL_LAYERS:
            dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visc{{filter:url(#cnn-visc-{key}-1x)}}")
            hidpi_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visc{{filter:url(#cnn-visc-{key}-2x)}}")
        else:
            dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visc{{filter:none}}")
            hidpi_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-visc{{filter:none}}")

        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .ldesc-{key}{{display:block}}")
        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .tgrid-{key}{{display:block}}")

        if key != "dwconv4":
            tw = len(coords) * cell_zoom
            dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .tlut-{key}{{display:inline}}")
            tile_luts_1x.append(
                f'<image class="tlut tlut-{key}" href="{lut_paths[f"tile_{key}_x_1x"]}" x="{tlx_x}" y="{tlx_y}" width="{tw}" height="4" preserveAspectRatio="none"/>'
                f'<image class="tlut tlut-{key}" href="{lut_paths[f"tile_{key}_y_1x"]}" x="{tly_x}" y="{tly_y}" width="4" height="{tw}" preserveAspectRatio="none"/>'
            )
            tile_luts_2x.append(
                f'<image class="tlut tlut-{key}" href="{lut_paths[f"tile_{key}_x_2x"]}" x="{tlx_x}" y="{tlx_y}" width="{tw}" height="4" preserveAspectRatio="none"/>'
                f'<image class="tlut tlut-{key}" href="{lut_paths[f"tile_{key}_y_2x"]}" x="{tly_x}" y="{tly_y}" width="4" height="{tw}" preserveAspectRatio="none"/>'
            )

    dyn_css.append(f"@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{{''.join(hidpi_css)}}}")

    bars_total_h = 10 * BAR_ROW_H

    ui_css = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0b0f17;color:#e2e8f0;line-height:1.5;padding:1.4rem .9rem 3.5rem
}}
.cnn-app{{max-width:820px;margin:0 auto;display:flex;flex-direction:column;gap:1.1rem}}
header.hero{{
  border:1px solid #1e293b;background:linear-gradient(180deg,#111827,#0f172a);
  border-radius:14px;padding:1.2rem 1.45rem;box-shadow:0 12px 32px rgba(0,0,0,.35)
}}
.back-row{{margin-bottom:.6rem}}
.back-link{{
  display:inline-flex;align-items:center;gap:.35rem;font-size:.74rem;font-weight:600;
  color:#94a3b8;text-decoration:none;padding:.2rem .58rem;border-radius:999px;
  background:#0b1120;border:1px solid #1e293b;transition:color .14s,border-color .14s,background .14s
}}
.back-link:hover{{color:#38bdf8;border-color:#38bdf8;background:#0f1d32}}
.badge-row{{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.55rem}}
.badge{{
  font-size:.68rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  padding:.16rem .5rem;border-radius:999px;background:#1e293b;color:#38bdf8;border:1px solid #334155
}}
.badge.green{{color:#4ade80;border-color:#14532d;background:#052e16}}
.badge.purple{{color:#c084fc;border-color:#4c1d95;background:#1e1035}}
.badge.amber{{color:#fbbf24;border-color:#78350f;background:#271705}}
header.hero h1{{font-size:1.42rem;font-weight:700;color:#f8fafc;letter-spacing:-.02em}}
header.hero p{{margin-top:.35rem;color:#94a3b8;font-size:.86rem}}
header.hero code{{color:#e2e8f0;background:#1e293b;padding:.08rem .3rem;border-radius:4px;font-size:.84em}}

.panel{{
  border:1px solid #1e293b;background:#111827;border-radius:14px;
  padding:1.05rem 1.2rem;box-shadow:0 8px 24px rgba(0,0,0,.25)
}}
.panel-head{{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:.45rem;margin-bottom:.68rem}}
.panel-title{{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8}}
.panel-hint{{font-size:.75rem;color:#64748b}}

.ctrl-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.42rem}}
@media (max-width:620px){{.ctrl-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.lyr-lbl{{
  display:flex;flex-direction:column;gap:.12rem;padding:.46rem .58rem;border-radius:8px;
  background:#090d16;border:1px solid #263349;cursor:pointer;user-select:none;transition:all .12s
}}
.lyr-lbl:hover{{border-color:#475569;background:#0f172a}}
.lyr-top{{display:flex;align-items:center;gap:.36rem;font-size:.75rem;font-weight:700;color:#e2e8f0}}
.lyr-top input{{accent-color:#38bdf8;cursor:pointer}}
.lyr-meta{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.65rem;color:#64748b}}

.sub-bar{{
  margin-top:.6rem;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.55rem;
  padding:.45rem .7rem;border-radius:8px;background:#090d16;border:1px solid #1e293b
}}
.ldesc{{display:none;font-size:.74rem;color:#bae6fd;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.pre-toggles{{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}}
.chk-pill{{
  display:inline-flex;align-items:center;gap:.35rem;font-size:.73rem;color:#cbd5e1;
  cursor:pointer;user-select:none;padding:.2rem .5rem;border-radius:6px;background:#111827;border:1px solid #334155
}}
.chk-pill:hover{{border-color:#475569}}
.chk-pill input{{accent-color:#38bdf8;cursor:pointer}}

.stage-scroll{{overflow-x:auto}}
.stage-shell{{
  position:relative;width:{STAGE_W}px;height:{STAGE_H}px;margin:0 auto;
  border-radius:12px;background:#070b14;border:1px solid #1e3a5f;
  box-shadow:0 0 32px rgba(14,165,233,.12);overflow:hidden
}}
.fstage{{position:absolute;inset:0;width:{STAGE_W}px;height:{STAGE_H}px;display:block}}
.fstage image{{image-rendering:pixelated;pointer-events:none}}
.cnn-visa{{filter:url(#cnn-visa-conv1-1x)}}
.cnn-visb{{filter:url(#cnn-visb-conv1-1x)}}
.cnn-visc{{filter:none}}
.luts-2x,.tlut{{display:none}}
@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{
  .cnn-visa{{filter:url(#cnn-visa-conv1-2x)}}
  .cnn-visb{{filter:url(#cnn-visb-conv1-2x)}}
  .cnn-visc{{filter:none}}
  .luts-1x{{display:none}}
  .luts-2x{{display:inline}}
}}

/* Static-flow contenteditable="true" OCR Drop Target inside <foreignObject>
   (no position:absolute / overflow:hidden / z-index / CSS filter — avoids WebKit Bug 23113!).
   Starts at y=INPUT_SRC_Y (20px) so the 28x28 / 14x14 active <img> sits above the y=52..192 drop box,
   guaranteeing every drop anywhere in the box appends at the end and hides all older images! */
.ocr-drop{{
  width:{INPUT_PREVIEW_SZ}px;height:{INPUT_PREVIEW_SZ + (INPUT_PREVIEW_Y - INPUT_SRC_Y)}px;
  outline:none;cursor:copy;padding-left:7px;
  background:#000;color:#fff;caret-color:transparent;
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Arial Rounded MT Bold",sans-serif;
  font-variant-numeric:tabular-nums;
  font-weight:700;font-size:21px;line-height:28px;text-align:left
}}
.ocr-drop::selection,.ocr-drop *::selection{{background:transparent!important;color:inherit!important}}
.ocr-drop img{{
  width:28px;height:28px;object-fit:contain;display:block;image-rendering:pixelated;
  background:transparent;border:0;margin:0;padding:0;pointer-events:none;user-select:none
}}
.ocr-drop br,.ocr-drop meta{{display:none!important}}
.ocr-drop > :not(:nth-last-child(1 of :not(br):not(meta))),
.ocr-drop > * img:not(:last-of-type){{display:none!important}}
.ocr-drop:has(img){{background:transparent;padding:0;font-size:0;line-height:0;color:transparent;caret-color:transparent}}
.ocr-drop:empty{{background:transparent}}
.cnn-s1:has(.ocr-drop:not(:empty)) .starter-svg{{display:none}}

@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{
  .ocr-drop{{font-size:10.5px;line-height:14px;padding-left:3.5px}}
  .ocr-drop img{{width:14px;height:14px}}
}}

/* Vector HUD Overlay */
.stage-hud{{position:absolute;inset:0;pointer-events:none;z-index:4}}
.hud-col-title{{
  position:absolute;top:14px;font-size:.69rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.07em;color:#94a3b8
}}
.hud-col-sub{{
  position:absolute;top:30px;font-size:.64rem;color:#64748b;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace
}}
.drop-frame{{
  position:absolute;left:{INPUT_PREVIEW_X - 2}px;top:{INPUT_PREVIEW_Y - 2}px;
  width:{INPUT_PREVIEW_SZ + 4}px;height:{INPUT_PREVIEW_SZ + 4}px;
  border:2px dashed #38bdf8;border-radius:6px;
  box-shadow:0 0 0 1px rgba(15,23,42,.85),0 0 16px rgba(56,189,248,.18)
}}
.fstage:has(.ocr-drop:focus) ~ .stage-hud .drop-frame{{
  border-style:solid;border-color:#4ade80;box-shadow:0 0 20px rgba(74,222,128,.35)
}}
.drop-caption{{
  position:absolute;left:{INPUT_PREVIEW_X - 2}px;top:{INPUT_PREVIEW_Y + INPUT_PREVIEW_SZ + 10}px;
  width:{INPUT_PREVIEW_SZ + 4}px;font-size:.68rem;color:#94a3b8;line-height:1.4
}}
.drop-caption strong{{color:#e2e8f0}}
.drop-empty-hint{{
  display:none;position:absolute;left:{INPUT_PREVIEW_X + 8}px;top:{INPUT_PREVIEW_Y + 48}px;
  width:{INPUT_PREVIEW_SZ - 16}px;text-align:center;font-size:.7rem;color:#7dd3fc;
  background:rgba(9,13,22,.85);padding:.32rem;border-radius:6px;border:1px solid #1e3a5f
}}
.fstage:has(.ocr-drop:focus:empty) ~ .stage-hud .drop-empty-hint{{display:block}}

.divider-v{{position:absolute;top:12px;bottom:12px;width:1px;background:#1e293b}}

.rgb-legend{{
  position:absolute;left:{ACT_PANEL_X}px;top:{ACT_PANEL_Y + ACT_PANEL_H + 18}px;
  width:{ACT_PANEL_W}px;display:flex;flex-direction:column;gap:4px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.63rem;color:#94a3b8
}}
.rgb-pills{{display:flex;gap:5px;align-items:center}}
.rp{{padding:1px 5px;border-radius:4px;font-weight:700;font-size:.61rem}}
.rp.r{{background:#3b0d15;color:#fca5a5;border:1px solid #7f1d1d}}
.rp.g{{background:#052e16;color:#86efac;border:1px solid #14532d}}
.rp.b{{background:#0c1e3d;color:#93c5fd;border:1px solid #1e3a8a}}

.tgrid{{display:none;position:absolute;inset:0}}
.tlbl{{
  position:absolute;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.55rem;color:#64748b;text-align:center;letter-spacing:-.02em
}}

.d-row-lbl{{
  position:absolute;left:{BAR_LABEL_X}px;width:20px;text-align:right;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;font-weight:700;color:#e2e8f0
}}
.axis-line{{
  position:absolute;top:{BAR_Y0 - 3}px;height:{bars_total_h - 4}px;
  width:1px;background:rgba(148,163,184,.14)
}}
.axis-tick{{
  position:absolute;top:{BAR_Y0 + bars_total_h}px;transform:translateX(-50%);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.61rem;color:#64748b
}}

.gal-section{{display:flex;flex-direction:column;gap:.75rem}}
.sample-strip{{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:.42rem}}
@media (max-width:700px){{.sample-strip{{grid-template-columns:repeat(5,minmax(0,1fr))}}}}
.s-card{{
  display:flex;flex-direction:column;align-items:center;gap:.24rem;
  padding:.36rem;border-radius:8px;background:#090d16;border:1px solid #263349;
  transition:transform .12s,border-color .12s
}}
.s-card:hover{{border-color:#38bdf8;transform:translateY(-1px)}}
.s-card img{{
  width:48px;height:48px;border-radius:4px;border:1px solid #334155;
  image-rendering:pixelated;cursor:grab;display:block;-webkit-user-drag:element
}}
.s-card img:active{{cursor:grabbing}}
.s-lbl{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;font-weight:700;color:#e2e8f0}}
.s-sub{{font-size:.58rem;color:#64748b;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}}
"""

    css_all = minify_css(ui_css) + "".join(dyn_css)

    radio_items, desc_items, tile_overlays = [], [], []
    for idx, (key, title, badge_shape, _, coords, _, n_rgb, (cols, rows, cell_zoom, gap_x, gap_y)) in enumerate(LAYER_VIEWS):
        checked_attr = " checked" if idx == 0 else ""
        radio_items.append(
            f'<label class="lyr-lbl" for="lyr-{key}">'
            f'<span class="lyr-top"><input type="radio" name="cnn-layer" id="lyr-{key}"{checked_attr}>{html.escape(title)}</span>'
            f'<span class="lyr-meta">{html.escape(badge_shape)}</span>'
            f'</label>'
        )
        desc_items.append(f'<div class="ldesc ldesc-{key}">{html.escape(LAYER_DESCRIPTIONS[key])}</div>')

        tile_w = len(coords) * cell_zoom
        tile_h = len(coords) * cell_zoom
        grid_w = cols * tile_w + (cols - 1) * gap_x
        grid_h = rows * tile_h + (rows - 1) * gap_y
        sx = ACT_PANEL_X + (ACT_PANEL_W - grid_w) // 2
        sy = ACT_PANEL_Y + (ACT_PANEL_H - grid_h) // 2

        tlbls = []
        for g in range(n_rgb):
            col, row = g % cols, g // cols
            tx = sx + col * (tile_w + gap_x)
            ty = sy + row * (tile_h + gap_y)
            ly = ty - 11
            tlbls.append(
                f'<span class="tlbl" style="left:{tx - 3}px;top:{ly}px;width:{tile_w + 6}px">'
                f'ch {3*g}–{3*g+2}'
                f'</span>'
            )
        tile_overlays.append(f'<div class="tgrid tgrid-{key}">{"".join(tlbls)}</div>')

    hud_bars = []
    for d in range(10):
        yb = BAR_Y0 + d * BAR_ROW_H
        hud_bars.append(f'<span class="d-row-lbl" style="top:{yb - 1}px">{d}</span>')

    for pct in (0, 20, 40, 60, 80, 100):
        ax_x = BAR_X + int(round(BAR_W * pct / 100.0))
        hud_bars.append(f'<span class="axis-line" style="left:{ax_x}px"></span>')
        hud_bars.append(f'<span class="axis-tick" style="left:{ax_x}px">{pct}%</span>')

    mnist_html = "".join(
        f'<div class="s-card">'
        f'<img src="{rel}" draggable="true" alt="MNIST test digit {d}" title="Drag digit {d} into the OCR input box">'
        f'<span class="s-lbl">Digit {d}</span>'
        f'<span class="s-sub">MNIST · {pct}%</span>'
        f'</div>'
        for d, rel, pct in mnist_cards
    )
    paper_html = "".join(
        f'<div class="s-card">'
        f'<img src="{rel}" draggable="true" alt="{html.escape(style)} digit {d}" title="Drag {html.escape(style)} digit {d} into the OCR input box">'
        f'<span class="s-lbl">Digit {d}</span>'
        f'<span class="s-sub">{html.escape(style)}</span>'
        f'</div>'
        for d, style, rel, pct in paper_cards
    )

    svg_acc = metrics.get("test_accuracy_svg_uint8", 0.9865) * 100.0
    font_acc = metrics.get("font_ocr_accuracy_svg_uint8", 0.991) * 100.0
    n_params = metrics.get("total_parameters", 3004)

    stage_chain = ("visc", "visb", "visa") + tuple(reversed(COMPUTE_STAGE_IDS))
    open_gs = "".join(f'<g class="cnn-{sid}">' for sid in stage_chain)
    close_gs = "</g>" * len(stage_chain)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoJS MNIST CNN OCR — Pure HTML5 SVG feConvolveMatrix + CSS</title>
<style>{css_all}</style>
</head>
<body>
<svg aria-hidden="true" width="0" height="0" style="position:absolute;width:0;height:0;overflow:hidden;pointer-events:none">
<defs>{"".join(svg_defs)}</defs>
</svg>

<main class="cnn-app">
  <header class="hero">
    <nav class="back-row" aria-label="Site navigation">
      <a class="back-link" href="../">&larr; All NoJS Demos</a>
    </nav>
    <div class="badge-row">
      <span class="badge green">Zero JavaScript</span>
      <span class="badge">SVG feConvolveMatrix</span>
      <span class="badge purple">contenteditable=&quot;true&quot; OCR</span>
      <span class="badge amber">{svg_acc:.2f}% Exact 8-Bit SVG Test Acc ({font_acc:.1f}% Font OCR)</span>
      <span class="badge">{n_params:,} Params · 0 BatchNorm</span>
    </div>
    <h1>Convolutional Neural Network OCR in Pure HTML5 SVG &amp; CSS</h1>
    <p>
      Drag any number image (from the sample trays below or another tab), paste a screenshot, or type <code>0–9</code>
      into the <code>contenteditable=&quot;true&quot;</code> input box. Linearly cached SVG <code>&lt;filter&gt;</code>
      stages auto-grayscale, polarity-normalize &amp; contrast-stretch the input, run 4 stages of
      <code>&lt;feConvolveMatrix&gt;</code> and <code>&lt;feColorMatrix&gt;</code> layers, and render RGB layer
      activations alongside Softmax probabilities.
    </p>
  </header>

  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Convolutional Layer Activation Inspector (RGB Color Feature Maps)</span>
      <span class="panel-hint">Select a layer to view its intermediate activations represented as RGB color images</span>
    </div>
    <div class="ctrl-grid">
      {"".join(radio_items)}
    </div>
    <div class="sub-bar">
      <div>{"".join(desc_items)}</div>
      <div class="pre-toggles">
        <label class="chk-pill" for="opt-invert">
          <input type="checkbox" id="opt-invert">Flip Input Polarity
        </label>
        <label class="chk-pill" for="opt-contrast">
          <input type="checkbox" id="opt-contrast">OCR Contrast Boost
        </label>
      </div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Live SVG Filter Neural Inference Pipeline</span>
      <span class="panel-hint">Emerald bar &amp; dot = argmax predicted digit · Sky-blue = runner-up Softmax probabilities</span>
    </div>

    <div class="stage-scroll">
      <div class="stage-shell">
        <svg class="fstage" width="{STAGE_W}" height="{STAGE_H}" viewBox="0 0 {STAGE_W} {STAGE_H}">
          {open_gs}
            <rect x="0" y="0" width="{STAGE_W}" height="{STAGE_H}" fill="#000" fill-opacity="0"/>
            <g class="luts-1x">
              <image href="{lut_paths['in_x_1x']}" x="{IN_LUTX_POS[0]}" y="{IN_LUTX_POS[1]}" width="{INPUT_PREVIEW_SZ}" height="4" preserveAspectRatio="none"/>
              <image href="{lut_paths['in_y_1x']}" x="{IN_LUTY_POS[0]}" y="{IN_LUTY_POS[1]}" width="4" height="{INPUT_PREVIEW_SZ}" preserveAspectRatio="none"/>
              <image href="{lut_paths['c5_x_1x']}" x="{C5_LUTX_POS[0]}" y="{C5_LUTX_POS[1]}" width="6" height="6" preserveAspectRatio="none"/>
              <image href="{lut_paths['c5_y_1x']}" x="{C5_LUTY_POS[0]}" y="{C5_LUTY_POS[1]}" width="6" height="6" preserveAspectRatio="none"/>
              <image href="{lut_paths['anchor_1x']}" x="{ANCHOR_MASK_X}" y="{ANCHOR_MASK_Y}" width="6" height="6" preserveAspectRatio="none"/>
              <image href="{lut_paths['bar_1x']}" x="{BAR_X}" y="292" width="{BAR_W}" height="4" preserveAspectRatio="none"/>
              <image class="starter-svg" href="{default_rel}" x="{INPUT_PREVIEW_X}" y="{INPUT_SRC_Y}" width="28" height="28" preserveAspectRatio="none"/>
              {"".join(tile_luts_1x)}
            </g>
            <g class="luts-2x">
              <image href="{lut_paths['in_x_2x']}" x="{IN_LUTX_POS[0]}" y="{IN_LUTX_POS[1]}" width="{INPUT_PREVIEW_SZ}" height="4" preserveAspectRatio="none"/>
              <image href="{lut_paths['in_y_2x']}" x="{IN_LUTY_POS[0]}" y="{IN_LUTY_POS[1]}" width="4" height="{INPUT_PREVIEW_SZ}" preserveAspectRatio="none"/>
              <image href="{lut_paths['c5_x_2x']}" x="{C5_LUTX_POS[0]}" y="{C5_LUTX_POS[1]}" width="3" height="3" preserveAspectRatio="none"/>
              <image href="{lut_paths['c5_y_2x']}" x="{C5_LUTY_POS[0]}" y="{C5_LUTY_POS[1]}" width="3" height="3" preserveAspectRatio="none"/>
              <image href="{lut_paths['anchor_2x']}" x="{ANCHOR_MASK_X}" y="{ANCHOR_MASK_Y}" width="3" height="3" preserveAspectRatio="none"/>
              <image href="{lut_paths['bar_2x']}" x="{BAR_X}" y="292" width="{BAR_W}" height="4" preserveAspectRatio="none"/>
              <image class="starter-svg" href="{default_rel}" x="{INPUT_PREVIEW_X}" y="{INPUT_SRC_Y}" width="14" height="14" preserveAspectRatio="none"/>
              {"".join(tile_luts_2x)}
            </g>
            <g class="cnn-pre">
              <foreignObject x="{INPUT_PREVIEW_X}" y="{INPUT_SRC_Y}" width="{INPUT_PREVIEW_SZ}" height="{INPUT_PREVIEW_SZ + (INPUT_PREVIEW_Y - INPUT_SRC_Y)}">
                <div class="ocr-drop" id="ocr-input" contenteditable="true" spellcheck="false" aria-label="Drag or paste a number image here"></div>
              </foreignObject>
            </g>
          {close_gs}
        </svg>

        <div class="stage-hud" aria-hidden="true">
          <span class="hud-col-title" style="left:{INPUT_PREVIEW_X}px">1. OCR Input Box</span>
          <span class="hud-col-sub" style="left:{INPUT_PREVIEW_X}px">contenteditable · 28×28</span>
          <div class="drop-frame"></div>
          <div class="drop-empty-hint">Drop or paste image<br>(or type 0–9)</div>
          <div class="drop-caption">
            <strong>Drag &amp; drop</strong> any digit card here, <strong>paste</strong> an image, or type <code>0–9</code>.
          </div>

          <div class="divider-v" style="left:172px"></div>

          <span class="hud-col-title" style="left:{ACT_PANEL_X}px">2. Layer RGB Activations</span>
          <span class="hud-col-sub" style="left:{ACT_PANEL_X}px">3 conv channels → (R, G, B)</span>
          {"".join(tile_overlays)}
          <div class="rgb-legend">
            <div class="rgb-pills">
              <span>Channels:</span>
              <span class="rp r">R = 3k</span>
              <span class="rp g">G = 3k+1</span>
              <span class="rp b">B = 3k+2</span>
            </div>
            <span>Overlapping activations blend in RGB.</span>
          </div>

          <div class="divider-v" style="left:416px"></div>

          <span class="hud-col-title" style="left:{BAR_LABEL_X - 12}px">3. Per-Digit Probabilities (Softmax)</span>
          <span class="hud-col-sub" style="left:{BAR_LABEL_X - 12}px">Logits → LogSumExp Softmax</span>
          {"".join(hud_bars)}
        </div>
      </div>
    </div>
  </section>

  <section class="panel gal-section">
    <div>
      <div class="panel-head">
        <span class="panel-title">Draggable Unseen MNIST Test Digits (White-on-Black)</span>
        <span class="panel-hint">Drag any image into &ldquo;1. OCR Input Box&rdquo; above</span>
      </div>
      <div class="sample-strip">
        {mnist_html}
      </div>
    </div>

    <div>
      <div class="panel-head">
        <span class="panel-title">Draggable Colored &amp; Light-Paper OCR Numbers (Auto-Grayscaled &amp; Contrast-Normalized)</span>
        <span class="panel-hint">Tests live BT.709 RGB→Grayscale + Alpha-neutral polarity + SVG Auto-Levels inside the filter</span>
      </div>
      <div class="sample-strip">
        {paper_html}
      </div>
    </div>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    out_path = os.path.join(CNN_DIR, "index.html")
    doc = generate_html()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Generated {out_path} ({len(doc) / 1024:.1f} KB) + external assets in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
