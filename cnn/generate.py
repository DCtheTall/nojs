#!/usr/bin/env python3
"""
Standalone Pure HTML5 + CSS + Inline SVG Generator for MNIST CNN OCR (`cnn/index.html`).

Generates a fast, compact (~415 KB), zero-JavaScript HTML5 application executing our trained
1,771-parameter Convolutional Neural Network (`weights.json`) across 7 nested `<div>` SVG
`<filter>` stages (`s1..s6` compute stages + `vis` layer inspector & Softmax bar stage):
  - O(N) linear texture caching between stages (eliminates Blink/Skia exponential DAG blowup)
  - External PNG assets in `cnn/assets/` loaded via relative `<img>` and CSS `url(assets/...)`
  - Zero `<feImage>` tags inside `<filter>` and every `<feConvolveMatrix>` `<= 5x5` (GPU fast path)
  - `contenteditable="true"` drag-and-drop / paste / type OCR input slot
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
    IN_LUTX_POS,
    IN_LUTY_POS,
    INPUT_PREVIEW_SZ,
    INPUT_PREVIEW_X,
    INPUT_PREVIEW_Y,
    LAYER_VIEWS,
    STAGE_H,
    STAGE_W,
    TILE_LUTX_POS,
    TILE_LUTY_POS,
    WEIGHTS_DEFAULT_PATH,
    compile_compute_stages,
    compile_visual_stage,
    load_weights,
    simulate_svg_cnn,
    write_filter_lut_assets,
)


LAYER_DESCRIPTIONS: Dict[str, str] = {
    "conv1":   "Stage 1 · 12 learned 5×5 feConvolveMatrix spatial filters packed into 4 RGB color images (24×24).",
    "mix2":    "Stage 2a · 2×2 feConvolveMatrix AvgPool + signed 1×1 feColorMatrix mixing (12→15 ch = 5 RGB maps, 12×12).",
    "dwconv2": "Stage 2b · 15 depthwise dilated 3×3 feConvolveMatrix filters capturing curves & corners (5 RGB maps, 10×10).",
    "mix3":    "Stage 3a · 2×2 AvgPool + signed 1×1 feColorMatrix mixing (15→24 ch = 8 RGB color images, 5×5 part detectors).",
    "dwconv3": "Stage 3b · 24 depthwise 3×3 feConvolveMatrix filters on compacted 5×5 grid (8 RGB color maps, 3×3).",
    "dwconv4": "Stage 4 · 24 global 3×3 feConvolveMatrix aggregators collapsing 3×3 → 1×1 bottleneck code (8 RGB color swatches).",
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
        # Sort unseen test samples for digit d by confidence so MNIST card uses crispest test digit
        candidates = sorted(by_digit[d], key=lambda g: float(g["svg_probs"][d]), reverse=True)
        gs0 = candidates[0]
        u8_0 = np.array(gs0["pixels_u8"], dtype=np.uint8)
        rgb0 = np.stack([u8_0, u8_0, u8_0], axis=-1)
        rel0 = _save_png(rgb0, f"mnist_{d}.png", out_size=84)
        pct0 = int(round(float(gs0["svg_probs"][d]) * 100))
        mnist_cards.append((d, rel0, pct0))
        if d == 7:
            default_rel = rel0

        # Distinct unseen test digit rendered as colored ink on light paper
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
            if sim.pred == d and float(sim.probs[d]) >= 0.80:
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

    svg_filters: List[str] = []
    for dpr in (1, 2):
        svg_filters.extend(compile_compute_stages(spec, dpr))

    dyn_css: List[str] = []
    hidpi_css: List[str] = []
    tile_lut_divs: List[str] = []

    for s_idx in range(1, 7):
        dyn_css.append(f".cnn-s{s_idx}{{filter:url(#cnn-s{s_idx}-1x)}}")
        hidpi_css.append(f".cnn-s{s_idx}{{filter:url(#cnn-s{s_idx}-2x)}}")

    # Lock Display-P3-immune neutral-gray 1D LUTs to the exact same CSS @media query as 1x/2x SVG filters!
    dyn_css.append(f".lut-inx{{background-image:url({lut_paths['in_x_1x']})}}")
    hidpi_css.append(f".lut-inx{{background-image:url({lut_paths['in_x_2x']})}}")
    dyn_css.append(f".lut-iny{{background-image:url({lut_paths['in_y_1x']})}}")
    hidpi_css.append(f".lut-iny{{background-image:url({lut_paths['in_y_2x']})}}")
    dyn_css.append(f".lut-c5x{{background-image:url({lut_paths['c5_x_1x']})}}")
    hidpi_css.append(f".lut-c5x{{background-image:url({lut_paths['c5_x_2x']})}}")
    dyn_css.append(f".lut-c5y{{background-image:url({lut_paths['c5_y_1x']})}}")
    hidpi_css.append(f".lut-c5y{{background-image:url({lut_paths['c5_y_2x']})}}")
    dyn_css.append(f".lut-am{{background-image:url({lut_paths['anchor_1x']})}}")
    hidpi_css.append(f".lut-am{{background-image:url({lut_paths['anchor_2x']})}}")

    tlx_x, tlx_y = TILE_LUTX_POS
    tly_x, tly_y = TILE_LUTY_POS

    for key, _, _, _, coords, _, _, (_, _, cell_zoom, _) in LAYER_VIEWS:
        for dpr in (1, 2):
            svg_filters.append(compile_visual_stage(key, dpr))

        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-vis{{filter:url(#cnn-vis-{key}-1x)}}")
        hidpi_css.append(f".cnn-app:has(#lyr-{key}:checked) .cnn-vis{{filter:url(#cnn-vis-{key}-2x)}}")
        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .ldesc-{key}{{display:block}}")
        dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .tgrid-{key}{{display:block}}")

        if key != "dwconv4":
            tw = len(coords) * cell_zoom
            dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .tlutx-{key}{{display:block}}")
            dyn_css.append(f".cnn-app:has(#lyr-{key}:checked) .tluty-{key}{{display:block}}")
            dyn_css.append(f".tlutx-{key}{{background-image:url({lut_paths[f'tile_{key}_x_1x']});width:{tw}px;height:4px}}")
            hidpi_css.append(f".tlutx-{key}{{background-image:url({lut_paths[f'tile_{key}_x_2x']})}}")
            dyn_css.append(f".tluty-{key}{{background-image:url({lut_paths[f'tile_{key}_y_1x']});width:4px;height:{tw}px}}")
            hidpi_css.append(f".tluty-{key}{{background-image:url({lut_paths[f'tile_{key}_y_2x']})}}")
            tile_lut_divs.append(f'<div class="lut-box tlut-box tlutx-{key}" style="left:{tlx_x}px;top:{tlx_y}px"></div>')
            tile_lut_divs.append(f'<div class="lut-box tlut-box tluty-{key}" style="left:{tly_x}px;top:{tly_y}px"></div>')

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
.lyr-lbl:has(input:checked){{border-color:#38bdf8;background:#0c2136;box-shadow:0 0 0 1px rgba(56,189,248,.3)}}
.lyr-lbl:has(input:checked) .lyr-meta{{color:#7dd3fc}}

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
.fstage{{position:absolute;inset:0;width:{STAGE_W}px;height:{STAGE_H}px}}
.cnn-vis{{filter:url(#cnn-vis-conv1-1x)}}
@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{
  .cnn-vis{{filter:url(#cnn-vis-conv1-2x)}}
}}

/* SourceGraphic helper LUT & ramp elements (locked to 1x/2x via CSS @media) */
.lut-box{{
  position:absolute;pointer-events:none;user-select:none;
  background-size:100% 100%;background-repeat:no-repeat;image-rendering:pixelated
}}
.tlut-box{{display:none}}
.c6-slot{{width:6px;height:6px}}
@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{
  .c6-slot{{width:3px;height:3px}}
}}
.bar-ramp-src{{
  position:absolute;left:{BAR_X}px;top:{BAR_Y0}px;width:{BAR_W}px;height:{bars_total_h}px;
  background:linear-gradient(90deg,#fff 0%,#000 100%);pointer-events:none
}}

/* contenteditable="true" OCR Drop Target */
.ocr-drop{{
  position:absolute;left:{INPUT_PREVIEW_X}px;top:{INPUT_PREVIEW_Y}px;
  width:{INPUT_PREVIEW_SZ}px;height:{INPUT_PREVIEW_SZ}px;
  overflow:hidden;outline:none;cursor:copy;z-index:2;
  background:#000;color:#fff;
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Arial Rounded MT Bold",sans-serif;
  font-weight:700;font-size:22px;line-height:28px;text-align:center
}}
.ocr-drop::selection,.ocr-drop *::selection{{background:transparent!important;color:inherit!important}}
.ocr-drop img{{
  position:absolute;left:0;top:0;
  width:28px;height:28px;object-fit:contain;display:block;
  background:transparent;border:0;margin:0;padding:0;pointer-events:none;user-select:none
}}
.ocr-drop:has(img:not(.starter-img)) img.starter-img{{display:none!important}}
.ocr-drop img:not(.starter-img) ~ img:not(.starter-img){{display:none!important}}
.ocr-drop:has(img){{font-size:0;line-height:0;color:transparent;caret-color:transparent}}

@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{
  .ocr-drop{{font-size:11px;line-height:14px}}
  .ocr-drop img{{width:14px;height:14px}}
}}

.cnn-app:has(#opt-invert:checked) .ocr-drop img{{filter:invert(1)}}
.cnn-app:has(#opt-contrast:checked) .ocr-drop img{{filter:contrast(1.65)}}
.cnn-app:has(#opt-invert:checked):has(#opt-contrast:checked) .ocr-drop img{{filter:invert(1) contrast(1.65)}}

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
.cnn-vis:has(.ocr-drop:focus) ~ .stage-hud .drop-frame{{
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
.cnn-vis:has(.ocr-drop:empty) ~ .stage-hud .drop-empty-hint{{display:block}}

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
  font-size:.56rem;color:#64748b;text-align:center;letter-spacing:-.02em
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
    for idx, (key, title, badge_shape, _, coords, _, n_rgb, (cols, rows, cell_zoom, gap)) in enumerate(LAYER_VIEWS):
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
        grid_w = cols * tile_w + (cols - 1) * gap
        grid_h = rows * tile_h + (rows - 1) * gap
        sx = ACT_PANEL_X + (ACT_PANEL_W - grid_w) // 2
        sy = ACT_PANEL_Y + (ACT_PANEL_H - grid_h) // 2

        tlbls = []
        for g in range(n_rgb):
            col, row = g % cols, g // cols
            tx = sx + col * (tile_w + gap)
            ty = sy + row * (tile_h + gap)
            ly = (ty - 12) if row == 0 else (ty + tile_h + 2)
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

    svg_acc = metrics.get("test_accuracy_svg_uint8", 0.9586) * 100.0
    qat_acc = metrics.get("test_accuracy_qat", 0.9786) * 100.0
    n_params = metrics.get("total_parameters", 1771)

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
<defs>{"".join(svg_filters)}</defs>
</svg>

<main class="cnn-app">
  <header class="hero">
    <div class="badge-row">
      <span class="badge green">Zero JavaScript</span>
      <span class="badge">SVG feConvolveMatrix</span>
      <span class="badge purple">contenteditable=&quot;true&quot; OCR</span>
      <span class="badge amber">{svg_acc:.1f}% Exact 8-Bit SVG Test Acc ({qat_acc:.1f}% QAT)</span>
      <span class="badge">{n_params:,} Params · 0 BatchNorm</span>
    </div>
    <h1>Convolutional Neural Network OCR in Pure HTML5 SVG &amp; CSS</h1>
    <p>
      Drag any number image (from the sample trays below or another tab) or paste an image into the
      <code>contenteditable=&quot;true&quot;</code> input box. Seven nested SVG <code>&lt;filter&gt;</code> stages
      grayscale &amp; polarity-normalize the image, run 4 stages of <code>&lt;feConvolveMatrix&gt;</code>
      and <code>&lt;feColorMatrix&gt;</code> layers, and render RGB layer activations alongside Softmax probabilities.
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
        <div class="fstage cnn-vis">
          <div class="fstage cnn-s6">
            <div class="fstage cnn-s5">
              <div class="fstage cnn-s4">
                <div class="fstage cnn-s3">
                  <div class="fstage cnn-s2">
                    <div class="fstage cnn-s1">
                      <div class="lut-box lut-inx" style="left:{IN_LUTX_POS[0]}px;top:{IN_LUTX_POS[1]}px;width:{INPUT_PREVIEW_SZ}px;height:4px"></div>
                      <div class="lut-box lut-iny" style="left:{IN_LUTY_POS[0]}px;top:{IN_LUTY_POS[1]}px;width:4px;height:{INPUT_PREVIEW_SZ}px"></div>
                      <div class="lut-box lut-c5x c6-slot" style="left:{C5_LUTX_POS[0]}px;top:{C5_LUTX_POS[1]}px"></div>
                      <div class="lut-box lut-c5y c6-slot" style="left:{C5_LUTY_POS[0]}px;top:{C5_LUTY_POS[1]}px"></div>
                      <div class="lut-box lut-am c6-slot" style="left:{ANCHOR_MASK_X}px;top:{ANCHOR_MASK_Y}px"></div>
                      {"".join(tile_lut_divs)}
                      <div class="bar-ramp-src"></div>
                      <div class="ocr-drop" id="ocr-input" contenteditable="true" spellcheck="false" aria-label="Drag or paste a number image here"><img class="starter-img" src="{default_rel}" alt="Default MNIST digit 7"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

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
        <span class="panel-title">Draggable Colored &amp; Light-Paper OCR Numbers (Auto-Grayscaled &amp; Polarity-Normalized)</span>
        <span class="panel-hint">Tests live BT.709 RGB→Grayscale + corner-background XOR inversion inside the SVG filter</span>
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
