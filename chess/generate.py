#!/usr/bin/env python3
"""
Static Pure HTML5 + CSS + SVG `<filter>` Generator for 1P / 2P FIDE Chess (`chess/index.html`).

Generates a standalone zero-JavaScript HTML5 application combining:
  1. Pure CSS `@property <integer>` FIDE Chess Referee (`.m0`, `.m1`, `.e0a .. .e12`):
     100-ply state machine with Radix-3 Kogge-Stone ray legality, pins, single & double checks,
     castling, en passant (including horizontal discovered-check pins), underpromotions,
     checkmate, stalemate, insufficient material, 50-move rule, LIFO Undo, and `.e12` Ply-1 bus.
  2. Pure HTML5 SVG `<filter>` 2-Ply CNN Minimax AI Opponent (`chess/svg_ai.py` + `chess/weights.json`):
     Stockfish-distilled 3-layer 8-bit quantized CNN Value Network (`2,623` params, `r = 0.9186`)
     evaluating `64 x 16 = 1,024` depth-2 FIDE-legal leaf boards in parallel (with exact depth-2
     checkmate detection) via `<feConvolveMatrix>`, `<feColorMatrix>`, `<feDisplacementMap>`, and
     `<feMorphology>` (`erode` + `dilate`).
"""

from __future__ import annotations

import argparse
import html
import os
import re
from typing import List

from chess_ref import (
    PRESETS,
    START_FEN,
    parse_fen,
    sq_name,
)
from css_chess import (
    DEFAULT_MAX_PLIES,
    build_chess_css_spec,
    css_eq,
    state_to_css_vars,
)
from svg_ai import (
    ASSETS_DIR,
    COMP_H,
    FO_H,
    FO_W,
    FO_X,
    FO_Y,
    LUT_BAR_POS,
    LUT_SQ8B_POS,
    LUT_SQ8W_POS,
    LUT_VIS_BX_POS,
    LUT_VIS_BY_POS,
    LUT_VIS_FY_POS,
    LUT_VIS_MX_POS,
    LUT_VIS_MY_POS,
    PHYS_LUT_SPECS,
    STAGE_H,
    STAGE_W,
    VIS_BOARD_SZ,
    VIS_BOARD_X,
    VIS_BOARD_Y,
    VIS_FEAT1_Y,
    VIS_FEAT2_Y,
    VIS_FEAT_H,
    VIS_FEAT_W,
    VIS_FEAT_X,
    VIS_MAT_H,
    VIS_MAT_W,
    VIS_MAT_X,
    VIS_MAT_Y,
    VIS_QBAR_H,
    VIS_QBAR_Y,
    VIS_WINBAR_H,
    VIS_WINBAR_W,
    VIS_WINBAR_X,
    VIS_WINBAR_Y,
    WEIGHTS_DEFAULT_PATH,
    best_dwconv3_triplet,
    compile_chess_ai_filters,
    get_compute_stage_ids,
    load_weights,
    write_chess_ai_assets,
)


# Standard White-perspective visual order: Rank 8 (a8..h8 = 56..63) down to Rank 1 (a1..h1 = 0..7)
VISUAL_SQ_ORDER: List[int] = [8 * r + f for r in range(7, -1, -1) for f in range(8)]

GAME_MODES = (
    ("1pw", "1P · Play White vs SVG AI", "Human = White · 2-Ply CNN plays Black"),
    ("1pb", "1P · Play Black vs SVG AI", "Human = Black · 2-Ply CNN plays White"),
    ("2p",  "2P · Local Pass & Play",    "Pure CSS FIDE referee for two humans"),
)


def minify_ui_css(raw: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([{};:,>~+])\s*", r"\1", s)
    return s.replace(";}", "}").strip()


def _chan_b_expr(pt_code: int, sq_w: int, sq_b: int) -> str:
    """Lossless `{0, 128, 255}` CSS channel formula for AI-normalized square `(ax, ay)`:
    `0` = Opponent piece (`s1a` maps `0 -> 1/255`), `128` = Empty (`Z0`), `255` = AI piece!
    """
    bw = f"var(--b{sq_w})"
    bb = f"var(--b{sq_b})"
    dw = f"({css_eq(bw, pt_code)} - {css_eq(bw, pt_code + 8)})"
    db = f"({css_eq(bb, pt_code + 8)} - {css_eq(bb, pt_code)})"
    return f"calc(128 + 128*((1 - var(--turn))*{dw} + var(--turn)*{db}))"


def generate_html(max_plies: int = DEFAULT_MAX_PLIES, weights_path: str = WEIGHTS_DEFAULT_PATH) -> str:
    spec = build_chess_css_spec()
    w_spec = load_weights(weights_path)
    metrics = w_spec.get("metrics", {})
    lut_paths = write_chess_ai_assets(ASSETS_DIR)

    svg_defs: List[str] = []
    for dpr in (1, 2):
        svg_defs.extend(compile_chess_ai_filters(w_spec, dpr))

    css: List[str] = []

    # -----------------------------------------------------------------------
    # A. @counter-style definitions & @property <integer> registrations
    # -----------------------------------------------------------------------
    piece_syms = '"" "♟\\fe0e" "♞\\fe0e" "♝\\fe0e" "♜\\fe0e" "♛\\fe0e" "♚\\fe0e" "" "" "♟\\fe0e" "♞\\fe0e" "♝\\fe0e" "♜\\fe0e" "♛\\fe0e" "♚\\fe0e"'
    css.append(f"@counter-style chess-sym{{system:fixed 0;symbols:{piece_syms}}}")
    css.append('@counter-style turn-lbl{system:fixed 0;symbols:"White" "Black"}')
    css.append('@counter-style promo-lbl{system:fixed 0;symbols:"" "" "Knight (♞)" "Bishop (♝)" "Rook (♜)" "Queen (♛)"}')
    css.append(
        '@counter-style game-status{system:fixed 0;symbols:'
        '"White to Move" '
        '"Black to Move" '
        '"Check — White must defend King!" '
        '"Check — Black must defend King!" '
        '"Checkmate — Black Wins! (0–1)" '
        '"Checkmate — White Wins! (1–0)" '
        '"Draw — Stalemate (½–½)" '
        '"Draw — Insufficient Material (½–½)" '
        '"Draw — 50-Move Rule (½–½)"}'
    )
    sq_sym_list = " ".join(f'"{sq_name(sq)}"' for sq in range(64)) + ' "—"'
    css.append(f"@counter-style sq-coord{{system:fixed 0;symbols:{sq_sym_list}}}")

    extra_ui_props = ("--ai_side", "--ai_en", "--ai_peek", "--is_ai_turn", "--ai_show")
    for prop in tuple(spec.registered_props) + extra_ui_props:
        css.append(f'@property {prop}{{syntax:"<integer>";initial-value:0;inherits:true}}')

    # -----------------------------------------------------------------------
    # B. UI Theme, Board, Controls & Fixed-1:1 SVG AI Stage Styling
    # -----------------------------------------------------------------------
    STAGE_VIEW_W = 666  # Exact multiple of 3 (666 * 2/3 = 444px, 636 * 2/3 = 424px) for 0-error 3x DPR scaling!
    ui_css = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0b0f17;color:#e2e8f0;line-height:1.5;padding:1.45rem .9rem 3.8rem;min-height:100vh
}}
.chess-app{{
  max-width:1244px;margin:0 auto;display:flex;flex-direction:column;gap:1.15rem
}}
.top-inp{{display:none}}

header.hero{{
  border:1px solid #1e293b;background:linear-gradient(180deg,#111827,#0f172a);
  border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 12px 32px rgba(0,0,0,.35)
}}
.back-row{{margin-bottom:.55rem}}
.back-link{{
  display:inline-flex;align-items:center;gap:.35rem;font-size:.75rem;font-weight:600;
  color:#94a3b8;text-decoration:none;padding:.2rem .6rem;border-radius:999px;
  background:#0b1120;border:1px solid #1e293b;transition:color .14s,border-color .14s,background .14s
}}
.back-link:hover{{color:#38bdf8;border-color:#38bdf8;background:#0f1d32}}
.badge-row{{display:flex;flex-wrap:wrap;gap:.42rem;margin-bottom:.55rem}}
.badge{{
  font-size:.69rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  padding:.18rem .52rem;border-radius:999px;background:#1e293b;color:#38bdf8;border:1px solid #334155
}}
.badge.green{{color:#4ade80;border-color:#14532d;background:#052e16}}
.badge.purple{{color:#c084fc;border-color:#4c1d95;background:#1e1035}}
.badge.amber{{color:#fbbf24;border-color:#78350f;background:#271705}}
header.hero h1{{font-size:1.52rem;font-weight:700;color:#f8fafc;letter-spacing:-.02em}}
header.hero p{{margin-top:.35rem;color:#94a3b8;font-size:.88rem}}
header.hero code{{color:#e2e8f0;background:#1e293b;padding:.08rem .3rem;border-radius:4px;font-size:.84em}}

.top-bar-grid{{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1.45fr);gap:1rem}}
@media (max-width:1040px){{.top-bar-grid{{grid-template-columns:1fr}}}}

.panel{{
  border:1px solid #1e293b;background:#111827;border-radius:14px;
  padding:1rem 1.2rem;box-shadow:0 8px 24px rgba(0,0,0,.25)
}}
.panel-head{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.45rem;margin-bottom:.68rem}}
.panel-title{{font-size:.77rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8}}
.pill-row{{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center}}

.mode-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.45rem}}
@media (max-width:620px){{.mode-grid{{grid-template-columns:1fr}}}}

.preset-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.45rem}}
@media (max-width:760px){{.preset-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}

.pr-lbl,.md-lbl{{
  display:flex;flex-direction:column;gap:.16rem;padding:.5rem .62rem;border-radius:9px;
  background:#090d16;border:1px solid #263349;cursor:pointer;user-select:none;transition:all .12s
}}
.pr-lbl:hover,.md-lbl:hover{{border-color:#475569;background:#0f172a}}
.pr-top{{display:flex;align-items:center;gap:.38rem;font-size:.75rem;font-weight:700;color:#e2e8f0}}
.ui-dot{{
  display:inline-block;width:11px;height:11px;flex:0 0 11px;border-radius:50%;
  border:2px solid #475569;background:#090d16;transition:all .12s
}}
.ui-chk{{
  display:inline-block;width:11px;height:11px;flex:0 0 11px;border-radius:3px;
  border:1.5px solid #475569;background:#090d16;transition:all .12s
}}
.pr-sub{{font-size:.67rem;color:#64748b;line-height:1.32}}

.flip-pill{{
  display:inline-flex;align-items:center;gap:.4rem;padding:.28rem .62rem;border-radius:8px;
  background:#090d16;border:1px solid #334155;font-size:.74rem;font-weight:600;color:#cbd5e1;
  cursor:pointer;user-select:none;transition:border-color .12s,background .12s
}}
.flip-pill:hover{{border-color:#38bdf8;background:#0f172a}}
#ai-hint:checked~.top-bar-grid label[for=ai-hint]{{border-color:#a855f7;background:#1e1035;color:#e9d5ff}}
#ai-hint:checked~.top-bar-grid label[for=ai-hint] .ui-chk{{border-color:#c084fc;background:#a855f7;box-shadow:inset 0 0 0 2px #1e1035}}
#flip-bd:checked~.top-bar-grid label[for=flip-bd]{{border-color:#38bdf8;background:#0c2136;color:#bae6fd}}
#flip-bd:checked~.top-bar-grid label[for=flip-bd] .ui-chk{{border-color:#7dd3fc;background:#38bdf8;box-shadow:inset 0 0 0 2px #0c2136}}

#mv{{display:block;--ai_side:1;--ai_en:1;--ai_peek:0}}
#mode-1pb:checked~#mv{{--ai_side:0;--ai_en:1}}
#mode-2p:checked~#mv{{--ai_en:0}}
#ai-hint:checked~#mv{{--ai_peek:1}}
.st,.ui-ctrl{{display:contents}}
#mv input{{display:none}}

.arena{{
  display:grid;grid-template-columns:minmax(300px,496px) minmax(0,1fr);
  gap:1.1rem;align-items:start
}}
@media (max-width:1160px){{.arena{{grid-template-columns:1fr;justify-items:center}}}}

.play-col{{width:100%;max-width:496px;display:flex;flex-direction:column;gap:.85rem}}

.status-card{{
  border-radius:14px;padding:.9rem 1.1rem;contain:layout paint;
  background:linear-gradient(180deg,#0f1c2e,#0b1320);
  border:1px solid #1e3a5f;box-shadow:0 8px 24px rgba(0,0,0,.3)
}}
.status-top{{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:.4rem}}
.status-kicker{{font-size:.69rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#7dd3fc}}
.status-main{{font-size:1.12rem;font-weight:800;color:#f8fafc;letter-spacing:-.01em}}
.status-main i{{font-style:normal}}
.status-main i::after{{content:counter(st,game-status)}}

.ai-turn-Callout{{
  margin-top:.55rem;padding:.48rem .72rem;border-radius:8px;
  background:linear-gradient(90deg,rgba(168,85,247,.18),rgba(16,185,129,.18));
  border:1px solid rgba(168,85,247,.55);font-size:.77rem;font-weight:600;color:#f1f5f9;
  display:none;align-items:center;gap:.45rem
}}
.sw-dot{{display:inline-block;width:10px;height:10px;border-radius:3px;vertical-align:-1px}}
.sw-from{{background:#a855f7;box-shadow:0 0 6px #a855f7}}
.sw-to{{background:#10b981;box-shadow:0 0 6px #10b981}}

.stat-grid{{
  margin-top:.65rem;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.42rem
}}
@media (max-width:520px){{.stat-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.stat-box{{
  background:#080c14;border:1px solid #1e293b;border-radius:8px;padding:.4rem .55rem
}}
.stat-lbl{{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#64748b}}
.stat-val{{
  margin-top:.06rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.84rem;font-weight:700;color:#e2e8f0
}}
.stat-val i{{font-style:normal}}
.sv-turn i::after{{content:counter(tn,turn-lbl)}}
.sv-ply i::after{{content:counter(pl)}}
.sv-hmc i::after{{content:counter(hm) "/100"}}
.sv-last i::after{{content:counter(lf,sq-coord) "→" counter(lt,sq-coord)}}

.board-shell{{
  position:relative;width:100%;max-width:496px;padding:11px;border-radius:16px;contain:layout paint;
  background:linear-gradient(180deg,#162032,#0e1524);border:1px solid #28364d;
  box-shadow:0 16px 40px rgba(0,0,0,.45)
}}
.board-frame{{
  position:relative;width:100%;aspect-ratio:1/1;border-radius:8px;overflow:hidden;
  border:2px solid #334155;transition:transform .28s cubic-bezier(.22,1,.36,1)
}}
#flip-bd:checked~#mv .board-frame{{transform:rotate(180deg)}}

.board-grid{{
  position:absolute;inset:0;display:grid;
  grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr)
}}
.sq{{
  position:relative;display:flex;align-items:center;justify-content:center;
  user-select:none;overflow:hidden
}}
.sq.lgt{{background:#dce4ee}}
.sq.drk{{background:#475a74}}
.sq .lm{{position:absolute;inset:0;background:rgba(250,204,21,.36);pointer-events:none}}
.sq .sl{{position:absolute;inset:0;background:rgba(56,189,248,.42);box-shadow:inset 0 0 0 3px #0284c7;pointer-events:none}}
.sq .ck{{
  position:absolute;inset:0;
  background:radial-gradient(circle,rgba(239,68,68,.9) 18%,rgba(220,38,38,.55) 62%,transparent 100%);
  box-shadow:inset 0 0 0 3px #ef4444;pointer-events:none
}}
.sq-in{{
  position:relative;z-index:2;width:100%;height:100%;
  display:flex;align-items:center;justify-content:center;
  transition:transform .28s cubic-bezier(.22,1,.36,1);pointer-events:none
}}
#flip-bd:checked~#mv .sq-in{{transform:rotate(180deg)}}
.pc{{
  font-style:normal;font-size:clamp(1.85rem,5vw,2.85rem);line-height:1;
  font-variant-emoji:text;transform:translateY(-1px)
}}
.pc::after{{content:counter(p,chess-sym)}}
.crd{{
  position:absolute;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.63rem;font-weight:800;line-height:1;pointer-events:none;opacity:.82
}}
.sq.lgt .crd{{color:#3b4d66}}
.sq.drk .crd{{color:#cbd5e1}}
.crd-r{{top:3px;left:4px}}
.crd-f{{bottom:3px;right:4px}}
.mv-dot{{
  position:absolute;z-index:3;width:30%;height:30%;border-radius:50%;
  background:rgba(16,185,129,.88);box-shadow:0 1px 4px rgba(0,0,0,.35);pointer-events:none
}}
.cp-ring{{
  position:absolute;inset:5%;z-index:3;border-radius:50%;
  border:4px solid rgba(239,68,68,.88);box-shadow:0 0 8px rgba(239,68,68,.35);pointer-events:none
}}

.ov-stack{{position:absolute;inset:0;z-index:10}}
.ov{{display:none;position:absolute;inset:0;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr)}}
.ov-bg{{position:absolute;inset:0;z-index:1;cursor:default}}
.ov label.lf,.ov label.lt{{position:relative;align-self:start;justify-self:start;display:block;cursor:pointer;pointer-events:auto}}
.ov label.lf{{z-index:2}}
.ov label.lt{{z-index:3}}
.ov label.lf:hover{{background:rgba(56,189,248,.16)}}
.ov label.lt:hover{{background:rgba(16,185,129,.22)}}

.term-banner{{
  position:absolute;inset:auto 6% 6% 6%;z-index:20;
  padding:.72rem .95rem;border-radius:11px;text-align:center;
  background:rgba(9,13,22,.93);border:1px solid #38bdf8;
  box-shadow:0 12px 32px rgba(0,0,0,.65);backdrop-filter:blur(6px);pointer-events:none
}}
.term-banner strong{{display:block;font-size:.96rem;color:#f8fafc}}
.term-banner strong::after{{content:counter(st,game-status)}}
.term-banner span{{font-size:.74rem;color:#94a3b8}}

.ctrl-card{{
  border:1px solid #1e293b;background:#111827;border-radius:14px;padding:.85rem 1.05rem;
  display:flex;flex-direction:column;gap:.68rem;contain:layout paint
}}
.btn-row{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.5rem}}
.btn-grp{{display:flex;flex-wrap:wrap;gap:.5rem}}
.btn{{
  display:inline-flex;align-items:center;justify-content:center;gap:.38rem;
  padding:.48rem .85rem;border-radius:8px;font-size:.79rem;font-weight:700;
  cursor:pointer;user-select:none;border:1px solid #334155;background:#162032;color:#f8fafc;
  transition:background .13s,border-color .13s,transform .1s
}}
.btn:hover{{background:#1e293b;border-color:#38bdf8;color:#7dd3fc}}
.btn:active{{transform:translateY(1px)}}
.btn.rst{{border-color:#7f1d1d;background:#231115;color:#fca5a5}}
.btn.rst:hover{{border-color:#ef4444;background:#35151b;color:#fecaca}}
.undo-btn{{display:none;border-color:#0369a1;background:#0c2438;color:#7dd3fc}}
.undo-disabled{{
  display:none;align-items:center;gap:.38rem;padding:.48rem .85rem;border-radius:8px;
  font-size:.79rem;font-weight:600;border:1px solid #1e293b;background:#0b0f17;color:#475569
}}
#t0n:checked~.arena .undo-disabled{{display:inline-flex}}
.sel-pill{{font-size:.73rem;color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.sel-pill i{{font-style:normal;color:#38bdf8;font-weight:700}}
.sel-pill i::after{{content:counter(slc,sq-coord)}}

.promo-box{{
  padding:.62rem .78rem;border-radius:9px;background:#090d16;border:1px solid #1e293b;
  transition:border-color .15s,box-shadow .15s
}}
.promo-head{{display:flex;justify-content:space-between;align-items:center;gap:.4rem;margin-bottom:.42rem}}
.promo-title{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#94a3b8}}
.promo-cur{{font-size:.73rem;font-weight:700;color:#fbbf24}}
.promo-cur i{{font-style:normal}}
.promo-cur i::after{{content:counter(pr,promo-lbl)}}
.pr-bar{{display:none;grid-template-columns:repeat(4,1fr);gap:.38rem}}
.pr-opt{{
  display:flex;align-items:center;justify-content:center;gap:.28rem;padding:.34rem .25rem;
  border-radius:7px;background:#111827;border:1px solid #28364d;cursor:pointer;
  color:#cbd5e1;font-size:.72rem;font-weight:600;user-select:none;transition:all .12s
}}
.pr-opt:hover{{border-color:#fbbf24;color:#fef08a}}
.pr-opt b{{font-size:1.08rem;line-height:1;font-variant-emoji:text}}

/* Exact 1:1 Device-Pixel SVG 2-Ply + Q3 CNN Minimax Engine Stage (`666 x 636` user units; `444 x 424` on 3x DPR!) */
.ai-col{{width:100%;max-width:{STAGE_VIEW_W + 24}px}}
.ai-panel{{
  border:1px solid #1e3a5f;background:#0b111e;border-radius:14px;
  padding:.85rem .72rem .9rem;box-shadow:0 12px 32px rgba(0,0,0,.38)
}}
.ai-head{{
  display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;
  gap:.4rem;padding:0 .35rem .6rem
}}
.ai-title{{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#c084fc}}
.ai-sub{{font-size:.72rem;color:#64748b;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.ai-scroll{{overflow-x:auto}}
.ai-stage-shell{{
  position:relative;width:{STAGE_VIEW_W}px;height:{STAGE_H}px;margin:0 auto;contain:layout paint;
  border-radius:11px;background:#060a13;border:1px solid #1e293b;overflow:hidden
}}
.ai-fstage{{position:absolute;inset:0;width:{STAGE_VIEW_W}px;height:{STAGE_H}px;display:block}}
.ai-fstage image{{image-rendering:pixelated;pointer-events:none}}
.ai-luts-2x,.ai-vis-luts-2x,.sq8-flip{{display:none}}
#flip-bd:checked~#mv .sq8-norm{{display:none}}
#flip-bd:checked~#mv .sq8-flip{{display:inline}}

/* Bypass GPU filter chain in 2-Player mode unless `Peek AI Hint` is checked! */
#mode-2p:checked~#ai-hint:not(:checked)~#mv .ai-fstage g{{filter:none!important}}
#mode-2p:checked~#ai-hint:not(:checked)~#mv .ai-fstage{{display:none}}

/* Static-flow neutral-gray (`R == G == B`, Display-P3-immune!) `<foreignObject>` CSS-to-SVG pixel bridge */
.ai-src-wrap{{width:{FO_W}px;height:{FO_H}px;background:transparent}}
.ai-src-wrap i,.ai-meta{{background:rgb(var(--g),var(--g),var(--g))}}
.ai-src-r1{{display:flex;align-items:flex-start;height:12px}}
.ai-bch{{
  display:grid;grid-template-columns:repeat(8,1px);grid-template-rows:repeat(8,1px);
  width:10px;height:10px;padding:0 2px 2px 0;margin-right:2px;background:rgb(128,128,128)
}}
.ai-bch i{{display:block;width:1px;height:1px}}
.ai-meta{{width:2px;height:2px;flex:0 0 2px;margin-right:2px}}
.ai-m1w{{display:flex;width:640px;height:2px}}
.ai-m1w i{{flex:0 0 10px;height:2px;display:block}}
.ai-m1c-row{{display:flex;align-items:flex-start;height:2px}}
.ai-m1c{{display:flex;width:64px;height:2px;margin-right:4px}}
.ai-m1c i{{flex:0 0 1px;height:2px;display:block}}

@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{
  .ai-luts-1x,.ai-vis-luts-1x{{display:none}}
  .ai-luts-2x,.ai-vis-luts-2x{{display:inline}}
}}
/* Exact 3x DPR iPhone / Super-Retina Mobile Scaling (`444/666 = 2/3` * `3` DPR = `2.000000` device px/unit!) */
@media (-webkit-min-device-pixel-ratio:2.5),(min-resolution:2.5dppx){{
  .ai-col{{max-width:468px}}
  .ai-stage-shell,.ai-fstage{{width:{STAGE_VIEW_W * 2 // 3}px;height:{STAGE_H * 2 // 3}px}}
  .ai-hud{{inset:auto;left:0;top:0;width:{STAGE_VIEW_W}px;height:{STAGE_H}px;transform:scale(calc(2/3));transform-origin:0 0}}
}}

/* HTML HUD Overlay inside `.ai-stage-shell` (titles, coordinates, legend, spoiler shield) */
.ai-hud{{position:absolute;inset:0;pointer-events:none;font-family:ui-sans-serif,-apple-system,sans-serif}}
.hud-ttl{{
  position:absolute;font-size:.7rem;font-weight:800;text-transform:uppercase;
  letter-spacing:.07em;color:#e2e8f0
}}
.hud-sub{{
  position:absolute;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.63rem;color:#64748b
}}
.bd-files{{
  position:absolute;left:{VIS_BOARD_X}px;top:{VIS_BOARD_Y + VIS_BOARD_SZ + 4}px;width:{VIS_BOARD_SZ}px;
  display:grid;grid-template-columns:repeat(8,1fr);text-align:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.62rem;font-weight:700;color:#64748b
}}
.bd-ranks{{
  position:absolute;left:{VIS_BOARD_X - 14}px;top:{VIS_BOARD_Y}px;height:{VIS_BOARD_SZ}px;
  display:grid;grid-template-rows:repeat(8,1fr);align-items:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.62rem;font-weight:700;color:#64748b
}}
#flip-bd:checked~#mv .bd-files{{direction:rtl}}
#flip-bd:checked~#mv .bd-ranks{{display:flex;flex-direction:column-reverse;justify-content:space-around}}

/* Idle / Spoiler shield over `.ai-stage-shell` when it is the Human's turn (unless `Peek AI Hint` is checked!) */
.ai-shield{{
  position:absolute;inset:0;z-index:5;
  background:rgba(6,10,19,.96);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:2rem;text-align:center;gap:.55rem
}}
.ai-shield b{{font-size:.95rem;color:#f8fafc}}
.ai-shield span{{font-size:.78rem;color:#94a3b8;max-width:440px;line-height:1.45}}

.ai-2p-cover{{
  position:absolute;inset:0;z-index:6;background:rgba(6,10,19,.97);
  display:none;flex-direction:column;align-items:center;justify-content:center;
  padding:2rem;text-align:center;gap:.55rem
}}
#mode-2p:checked~#ai-hint:not(:checked)~#mv .ai-2p-cover{{display:flex}}
.ai-2p-cover b{{font-size:.95rem;color:#f8fafc}}
.ai-2p-cover span{{font-size:.78rem;color:#94a3b8;max-width:420px}}

.wb-ticks{{
  position:absolute;left:{VIS_WINBAR_X}px;top:{VIS_WINBAR_Y + VIS_WINBAR_H + 4}px;width:{VIS_WINBAR_W}px;
  display:flex;justify-content:space-between;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.61rem;color:#64748b
}}
.hm-legend{{
  position:absolute;left:{VIS_MAT_X}px;top:{VIS_WINBAR_Y - 2}px;width:{VIS_MAT_W}px;
  display:flex;flex-wrap:wrap;gap:.55rem;align-items:center;
  font-size:.67rem;color:#94a3b8
}}
.lg-sw{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}}

.arch-card{{
  position:absolute;left:{VIS_BOARD_X}px;top:{VIS_FEAT1_Y}px;width:{VIS_BOARD_SZ}px;height:{VIS_FEAT2_Y + VIS_FEAT_H - VIS_FEAT1_Y}px;
  border-radius:8px;background:#09101d;border:1px solid #1e293b;padding:.56rem .72rem;
  display:flex;flex-direction:column;justify-content:space-between
}}
.arch-card h4{{font-size:.67rem;text-transform:uppercase;letter-spacing:.06em;color:#7dd3fc}}
.arch-list{{list-style:none;display:flex;flex-direction:column;gap:.2rem;font-size:.63rem;color:#94a3b8;line-height:1.28}}
.arch-list code{{color:#e2e8f0;font-size:.61rem}}
"""
    css.append(minify_ui_css(ui_css))

    # SVG stage filter bindings (1x and 2x/3x Retina)
    stage_ids = get_compute_stage_ids(w_spec)
    assert stage_ids[-2:] == ("visa", "visb"), f"Expected tail (visa, visb), got {stage_ids[-2:]}"
    hidpi_sels: List[str] = []
    for sid in stage_ids:
        css.append(f".ai-{sid}{{filter:url(#ai-{sid}-1x)}}")
        hidpi_sels.append(f".ai-{sid}{{filter:url(#ai-{sid}-2x)}}")
    css.append(f"@media (-webkit-min-device-pixel-ratio:1.5),(min-resolution:1.5dppx){{{''.join(hidpi_sels)}}}")

    # -----------------------------------------------------------------------
    # C. Initial Position, Modes, Presets, & Scoped O(1) Ply Input Rules
    # -----------------------------------------------------------------------
    std_vars = state_to_css_vars(parse_fen(START_FEN))
    init_decls = ";".join(f"{k}:{v}" for k, v in sorted(std_vars.items()))
    css.append(f".init{{{init_decls}}}")

    md_card_sels = ",".join(f"#mode-{mid}:checked~.top-bar-grid label[for=mode-{mid}]" for mid, *_ in GAME_MODES)
    md_dot_sels = ",".join(f"#mode-{mid}:checked~.top-bar-grid label[for=mode-{mid}] .ui-dot" for mid, *_ in GAME_MODES)
    md_sub_sels = ",".join(f"#mode-{mid}:checked~.top-bar-grid label[for=mode-{mid}] .pr-sub" for mid, *_ in GAME_MODES)
    css.append(f"{md_card_sels}{{border-color:#c084fc;background:#1b1030;box-shadow:0 0 0 1px rgba(192,132,252,.28)}}")
    css.append(f"{md_dot_sels}{{border-color:#c084fc;background:#a855f7;box-shadow:inset 0 0 0 2px #1b1030}}")
    css.append(f"{md_sub_sels}{{color:#d8b4fe}}")

    pr_card_sels = ",".join(f"#pr-{pid}:checked~.top-bar-grid label[for=pr-{pid}]" for pid, *_ in PRESETS)
    pr_dot_sels = ",".join(f"#pr-{pid}:checked~.top-bar-grid label[for=pr-{pid}] .ui-dot" for pid, *_ in PRESETS)
    pr_sub_sels = ",".join(f"#pr-{pid}:checked~.top-bar-grid label[for=pr-{pid}] .pr-sub" for pid, *_ in PRESETS)
    css.append(f"{pr_card_sels}{{border-color:#38bdf8;background:#0c2136;box-shadow:0 0 0 1px rgba(56,189,248,.28)}}")
    css.append(f"{pr_dot_sels}{{border-color:#7dd3fc;background:#38bdf8;box-shadow:inset 0 0 0 2px #0c2136}}")
    css.append(f"{pr_sub_sels}{{color:#7dd3fc}}")

    for pid, _title, fen, _sub in PRESETS:
        if pid == "std":
            continue
        p_vars = state_to_css_vars(parse_fen(fen))
        p_decls = ";".join(f"{k}:{v}" for k, v in sorted(p_vars.items()) if p_vars[k] != std_vars.get(k))
        if p_decls:
            css.append(f"#pr-{pid}:checked~#mv .init{{{p_decls}}}")

    css.append(".p{--in_f:64;--in_t:64;--in_c:0;--in_pr:5}")
    for sq in range(64):
        css.append(f".f{sq}:checked~.p{{--in_f:{sq}}}")
        css.append(f".t{sq}:checked~.p{{--in_t:{sq};--in_c:1}}")
    css.append(".pn:checked~.p{--in_pr:2}")
    css.append(".pb:checked~.p{--in_pr:3}")
    css.append(".pr:checked~.p{--in_pr:4}")

    css.append("#t0n:checked~.arena .ov0,#t0n:checked~.arena .pb0{display:grid}")
    even_ply_sels: List[str] = ["#t0n:checked~.arena"]
    odd_ply_sels: List[str] = []
    for k in range(1, max_plies):
        sel_k = f"#t{k - 1}n:not(:checked)+#t{k}n:checked~.arena"
        css.append(f"{sel_k} .ov{k},{sel_k} .pb{k}{{display:grid}}")
        css.append(f"{sel_k} .ub{k}{{display:inline-flex}}")
        if k % 2 == 0:
            even_ply_sels.append(sel_k)
        else:
            odd_ply_sels.append(sel_k)
    tail_sel = f"#t{max_plies - 1}n:not(:checked)~.arena"
    css.append(f"{tail_sel} .ub{max_plies}{{display:inline-flex}}")
    if max_plies % 2 == 0:
        even_ply_sels.append(tail_sel)
    else:
        odd_ply_sels.append(tail_sel)

    # Pause `.ai-fstage` (`display:none` + `filter:none!important` => 0 GPU surfaces & 0 Skia/CoreImage passes!)
    # whenever it is the Human's turn and `Peek AI Hint` is unchecked!
    pause_1pw_pfx = "#mode-1pw:checked~#ai-hint:not(:checked)~#mv "
    pause_1pb_pfx = "#mode-1pb:checked~#ai-hint:not(:checked)~#mv "
    pause_sels = [f"{pause_1pw_pfx}{s}" for s in even_ply_sels] + [f"{pause_1pb_pfx}{s}" for s in odd_ply_sels]
    css.append(",".join(f"{s} .ai-fstage" for s in pause_sels) + "{display:none}")
    css.append(",".join(f"{s} .ai-fstage g" for s in pause_sels) + "{filter:none!important}")

    for code, cls_s in ((5, "pq"), (2, "pn"), (4, "pr"), (3, "pb")):
        eq_pr = css_eq("var(--cur_pr)", code)
        css.append(
            f".pr-opt.{cls_s}{{"
            f"border-color:rgb(calc(40 + 210*{eq_pr}),calc(54 + 137*{eq_pr}),calc(77 - 41*{eq_pr}));"
            f"background:rgb(calc(17 + 28*{eq_pr}),calc(24 + 12*{eq_pr}),calc(39 - 25*{eq_pr}));"
            f"color:rgb(calc(203 + 51*{eq_pr}),calc(213 + 27*{eq_pr}),calc(225 - 87*{eq_pr}))}}"
        )
    css.append(
        ".promo-box{"
        "border-color:rgb(calc(30 + 220*var(--sel_can_promo)),calc(41 + 150*var(--sel_can_promo)),calc(59 - 23*var(--sel_can_promo)));"
        "box-shadow:0 0 calc(16px*var(--sel_can_promo)) rgba(251,191,36,.28)}"
    )

    # -----------------------------------------------------------------------
    # D. Compiled Chess Pipeline Stages (`.m0`, `.m1`, `.e0a .. .e12`)
    # -----------------------------------------------------------------------
    for cls_name, rules in spec.stage_defs.items():
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".{cls_name}{{{body}}}")

    # -----------------------------------------------------------------------
    # E. Per-Square Board, Hit-Targets, & `<foreignObject>` CSS-to-SVG Bridge
    # -----------------------------------------------------------------------
    for sq in range(64):
        bu = f"var(--b{sq})"
        is_blk = f"clamp(0,{bu} - 8,1)"
        is_last = f"min(1,{css_eq('var(--last_f)', sq)} + {css_eq('var(--last_t)', sq)})"
        quiet_dot = f"calc(var(--leg_{sq})*(1 - var(--cap_{sq})))"

        css.append(
            f".sq{sq} .pc{{"
            f"counter-reset:p {bu};"
            f"color:rgb(calc(255 - 239*{is_blk}),calc(252 - 230*{is_blk}),calc(244 - 212*{is_blk}));"
            f"filter:drop-shadow(0 1px 1px rgba(0,0,0,calc(.82 - .45*{is_blk}))) "
            f"drop-shadow(0 0 1px rgba(255,255,255,calc(.65*{is_blk})))}}"
        )
        css.append(f".sq{sq} .lm{{opacity:{is_last}}}")
        css.append(f".sq{sq} .sl{{opacity:var(--isel{sq})}}")
        css.append(f".sq{sq} .ck{{opacity:var(--chk_k_{sq})}}")
        css.append(f".sq{sq} .mv-dot{{opacity:{quiet_dot};transform:scale({quiet_dot})}}")
        css.append(f".sq{sq} .cp-ring{{opacity:var(--cap_{sq});transform:scale(var(--cap_{sq}))}}")

        r_sq, f_sq = divmod(sq, 8)
        grid_pos = f"grid-row:{8 - r_sq};grid-column:{f_sq + 1};"
        css.append(
            f".lf{sq}{{{grid_pos}"
            f"transform:translateX(calc(-9999px*(1 - var(--can_sel_{sq}))));"
            f"width:calc(100%*var(--can_sel_{sq}));"
            f"height:calc(100%*var(--can_sel_{sq}))}}"
        )
        css.append(
            f".lt{sq}{{{grid_pos}"
            f"transform:translateX(calc(-9999px*(1 - var(--leg_{sq}))));"
            f"width:calc(100%*var(--leg_{sq}));"
            f"height:calc(100%*var(--leg_{sq}))}}"
        )

    # `<foreignObject>` Display-P3-immune neutral-gray (`R == G == B`) root board & `.e12` Ply-1 move bus
    for ay in range(8):
        for ax in range(8):
            a = 8 * ay + ax
            sq_w = 8 * ay + ax
            sq_b = 8 * (7 - ay) + ax
            for ch_idx in range(6):
                css.append(f".ach{ch_idx}_{a}{{--g:{_chan_b_expr(ch_idx + 1, sq_w, sq_b)}}}")

    for v1 in range(64):
        css.append(f".amu{v1}{{--g:calc(128 + var(--m1_u_{v1}))}}")
        css.append(f".amv{v1}{{--g:calc(128 + var(--m1_v_{v1}))}}")
        css.append(f".amk{v1}{{--g:calc(128 + var(--m1_c_{v1}))}}")
    css.append(".ai-mt-turn{--g:calc(128 + 127*var(--turn))}")
    css.append(".ai-mt-chk{--g:calc(128 + 127*var(--in_check))}")
    css.append(".ai-mt-term{--g:calc(128 + 127*var(--is_terminal))}")

    # CSS inheritance firewall on `.ai-col`: block piece-selection (`f_k`) & promo (`p_k`) style invalidation
    # from ever propagating into `<foreignObject>` or re-running the 58-stage SVG filter pipeline in Blink!
    firewall_vars = {
        "--cur_from", "--cur_pr", "--in_f", "--in_t", "--in_c", "--in_pr", "--sel_can_promo",
        *(f"--isel{sq}" for sq in range(64)),
        *(f"--leg_{sq}" for sq in range(64)),
        *(f"--cap_{sq}" for sq in range(64)),
        *(f"--can_sel_{sq}" for sq in range(64)),
    }
    for stg_nm in ("e10", "e11", "e12"):
        for lhs, _rhs in spec.stage_defs.get(stg_nm, []):
            if lhs.startswith("--t"):
                firewall_vars.add(lhs)
    fw_decls = ";".join(f"{v}:0" for v in sorted(firewall_vars))

    # Live AI-turn detection (`--is_ai_turn == 1` iff 1P mode & live game & `--turn == --ai_side`)
    # Note: `--ai_show` is computed on child `.ai-col` so `.arena` has zero intra-element `@property` dependencies!
    css.append(
        ".arena{"
        "--is_ai_turn:calc(var(--ai_en)*(1 - var(--is_terminal))*(1 - (var(--turn) - var(--ai_side))*(var(--turn) - var(--ai_side))))}"
        f".ai-col{{--ai_show:min(1,var(--is_ai_turn) + var(--ai_peek));{fw_decls}}}"
        ".play-col{"
        "counter-reset:st var(--status) tn var(--turn) pl var(--ply) hm var(--hmc) "
        "slc var(--cur_from) lf var(--last_f) lt var(--last_t) pr var(--cur_pr)}"
    )
    css.append(
        ".ai-shield{opacity:calc(1 - var(--ai_show));transform:scale(calc(1 - var(--ai_show)))}"
        ".ai-turn-Callout{display:flex;opacity:var(--is_ai_turn);max-height:calc(80px*var(--is_ai_turn));"
        "padding:calc(.48rem*var(--is_ai_turn)) .72rem;margin-top:calc(.55rem*var(--is_ai_turn));"
        "border-width:calc(1px*var(--is_ai_turn));overflow:hidden}"
    )
    css.append(
        ".term-banner{"
        "opacity:var(--is_terminal);"
        "transform:translateY(calc(12px*(1 - var(--is_terminal)))) scale(calc(.95 + .05*var(--is_terminal)))}"
    )
    css.append(
        ".status-card{"
        "border-color:rgb(calc(30 + 138*var(--is_ai_turn) + 209*min(1,var(--in_check) + var(--checkmate))),"
        "calc(58 + 27*var(--is_ai_turn) + 10*var(--is_terminal)),"
        "calc(95 + 152*var(--is_ai_turn) - 27*var(--in_check)))}"
    )

    # -----------------------------------------------------------------------
    # F. HTML Markup Assembly (Modes, Presets, 100-Ply Pipeline, Board & SVG AI)
    # -----------------------------------------------------------------------
    top_inputs_html: List[str] = []
    mode_cards_html: List[str] = []
    for idx, (mid, title, sub) in enumerate(GAME_MODES):
        chk = " checked" if idx == 0 else ""
        top_inputs_html.append(f'<input class=top-inp type=radio name=gmode id=mode-{mid}{chk}>')
        mode_cards_html.append(
            f'<label class=md-lbl for=mode-{mid}>'
            f'<span class=pr-top><i class=ui-dot></i>'
            f'<span>{html.escape(title)}</span></span>'
            f'<span class=pr-sub>{html.escape(sub)}</span></label>'
        )

    preset_cards_html: List[str] = []
    for idx, (pid, title, _fen, sub) in enumerate(PRESETS):
        chk = " checked" if idx == 0 else ""
        top_inputs_html.append(f'<input class=top-inp type=radio name=preset id=pr-{pid}{chk}>')
        preset_cards_html.append(
            f'<label class=pr-lbl for=pr-{pid}>'
            f'<span class=pr-top><i class=ui-dot></i>'
            f'<span>{html.escape(title)}</span></span>'
            f'<span class=pr-sub>{html.escape(sub)}</span></label>'
        )
    top_inputs_html.append('<input class=top-inp type=checkbox id=ai-hint>')
    top_inputs_html.append('<input class=top-inp type=checkbox id=flip-bd>')

    board_cells_html: List[str] = []
    for sq in VISUAL_SQ_ORDER:
        r, f = divmod(sq, 8)
        shade = "lgt" if (r + f) % 2 == 1 else "drk"
        coord_bits: List[str] = []
        if f == 0:
            coord_bits.append(f'<span class="crd crd-r">{r + 1}</span>')
        if r == 0:
            coord_bits.append(f'<span class="crd crd-f">{chr(ord("a") + f)}</span>')
        board_cells_html.append(
            f'<div class="sq {shade} sq{sq}">'
            f'<span class=lm></span><span class=sl></span><span class=ck></span>'
            f'<span class=mv-dot></span><span class=cp-ring></span>'
            f'<div class=sq-in>{"".join(coord_bits)}<i class=pc></i></div>'
            f'</div>'
        )

    overlays_html: List[str] = []
    promo_bars_html: List[str] = []
    undo_btns_html: List[str] = ['<span class=undo-disabled>&#x21B6; Undo Move</span>']

    for k in range(max_plies):
        cells_k = "".join(
            f'<label class="lf lf{sq}" for=f{k}_{sq}></label>'
            f'<label class="lt lt{sq}" for=t{k}_{sq}></label>'
            for sq in VISUAL_SQ_ORDER
        )
        overlays_html.append(
            f'<div class="ov ov{k}"><label class=ov-bg for=f{k}n></label>{cells_k}</div>'
        )
        promo_bars_html.append(
            f'<div class="pr-bar pb{k}">'
            f'<label class="pr-opt pq" for=p{k}q><b>&#x265B;&#xFE0E;</b><span>Queen</span></label>'
            f'<label class="pr-opt pn" for=p{k}n><b>&#x265E;&#xFE0E;</b><span>Knight</span></label>'
            f'<label class="pr-opt pr" for=p{k}r><b>&#x265C;&#xFE0E;</b><span>Rook</span></label>'
            f'<label class="pr-opt pb" for=p{k}b><b>&#x265D;&#xFE0E;</b><span>Bishop</span></label>'
            f'</div>'
        )
        undo_btns_html.append(
            f'<label class="btn undo-btn ub{k + 1}" for=t{k}n>&#x21B6; Undo Move</label>'
        )

    # Assemble compute LUTs inside `s1a` (`0..332 x 0..314` at 2x DPR!) and visual LUTs inside `visa`
    luts_1x_imgs: List[str] = []
    luts_2x_imgs: List[str] = []
    for key, (lx, ly), pw, ph in PHYS_LUT_SPECS:
        if key in ("sq8w", "sq8b"):
            fk = key + "f"
            luts_1x_imgs.append(
                f'<image class=sq8-norm href="{lut_paths[key]}" x="{lx}" y="{ly}" width="{pw}" height="{ph}" preserveAspectRatio="none"/>'
                f'<image class=sq8-flip href="{lut_paths[fk]}" x="{lx}" y="{ly}" width="{pw}" height="{ph}" preserveAspectRatio="none"/>'
            )
            luts_2x_imgs.append(
                f'<image class=sq8-norm href="{lut_paths[key]}" x="{lx // 2}" y="{ly // 2}" width="{pw // 2}" height="{ph // 2}" preserveAspectRatio="none"/>'
                f'<image class=sq8-flip href="{lut_paths[fk]}" x="{lx // 2}" y="{ly // 2}" width="{pw // 2}" height="{ph // 2}" preserveAspectRatio="none"/>'
            )
        else:
            luts_1x_imgs.append(
                f'<image href="{lut_paths[key]}" x="{lx}" y="{ly}" width="{pw}" height="{ph}" preserveAspectRatio="none"/>'
            )
            luts_2x_imgs.append(
                f'<image href="{lut_paths[key]}" x="{lx // 2}" y="{ly // 2}" width="{pw // 2}" height="{ph // 2}" preserveAspectRatio="none"/>'
            )

    vis_luts_1x_imgs: List[str] = []
    vis_luts_2x_imgs: List[str] = []
    for dpr, dst_lst in ((1, vis_luts_1x_imgs), (2, vis_luts_2x_imgs)):
        dst_lst.extend([
            f'<image href="{lut_paths[f"vis_bx_{dpr}x"]}" x="{LUT_VIS_BX_POS[0]}" y="{LUT_VIS_BX_POS[1]}" width="{VIS_BOARD_SZ}" height="4" preserveAspectRatio="none"/>',
            f'<image href="{lut_paths[f"vis_by_{dpr}x"]}" x="{LUT_VIS_BY_POS[0]}" y="{LUT_VIS_BY_POS[1]}" width="4" height="{VIS_BOARD_SZ}" preserveAspectRatio="none"/>',
            f'<image href="{lut_paths[f"vis_mx_{dpr}x"]}" x="{LUT_VIS_MX_POS[0]}" y="{LUT_VIS_MX_POS[1]}" width="{VIS_MAT_W}" height="4" preserveAspectRatio="none"/>',
            f'<image href="{lut_paths[f"vis_my_{dpr}x"]}" x="{LUT_VIS_MY_POS[0]}" y="{LUT_VIS_MY_POS[1]}" width="4" height="{VIS_MAT_H}" preserveAspectRatio="none"/>',
            f'<image href="{lut_paths[f"vis_fy_{dpr}x"]}" x="{LUT_VIS_FY_POS[0]}" y="{LUT_VIS_FY_POS[1]}" width="4" height="{VIS_FEAT_H}" preserveAspectRatio="none"/>',
            f'<image href="{lut_paths[f"bar_{dpr}x"]}" x="{LUT_BAR_POS[0]}" y="{LUT_BAR_POS[1]}" width="{VIS_WINBAR_W}" height="4" preserveAspectRatio="none"/>',
        ])

    ch_grids_html = "".join(
        f'<div class=ai-bch>{"".join(f"<i class=ach{ch_i}_{a}></i>" for a in range(64))}</div>'
        for ch_i in range(6)
    )
    m1u_cells = "".join(f"<i class=amu{v1}></i>" for v1 in range(64))
    m1v_cells = "".join(f"<i class=amv{v1}></i>" for v1 in range(64))
    m1k_cells = "".join(f"<i class=amk{v1}></i>" for v1 in range(64))

    comp_chain = tuple(reversed(stage_ids[:-2]))
    open_comp_gs = "".join(f'<g class=ai-{sid}>' for sid in comp_chain)
    close_comp_gs = "</g>" * len(comp_chain)

    corr_r = metrics.get("test_stockfish_corr_svg_uint8", metrics.get("test_pearson_r_svg", 0.9186))
    rank_acc_pct = 100.0 * float(metrics.get("val_sibling_rank_acc", 0.9614))
    n_params = metrics.get("total_parameters", 2623)
    c1_ch = int(w_spec["layers"]["mix1"]["out_channels"])
    k1_sz = int(w_spec["layers"]["dwconv1"]["svg_order"][0])
    c2_ch = int(w_spec["layers"]["mix2"]["out_channels"])
    k2_sz = int(w_spec["layers"]["dwconv2"]["svg_order"][0])
    c3_ch = int(w_spec["layers"]["mix3"]["out_channels"])
    k3_sz = int(w_spec["layers"]["dwconv3"]["svg_order"][0])
    skip_ch0 = c3_ch
    vh_in = int(w_spec["layers"]["value_head"]["in_features"])
    d3_g = best_dwconv3_triplet(w_spec)

    files_lbls = "".join(f"<span>{c}</span>" for c in "abcdefgh")
    ranks_lbls = "".join(f"<span>{r}</span>" for r in range(8, 0, -1))

    fo_inner_html = (
        f'<foreignObject x="0" y="0" width="{FO_W}" height="{FO_H}">'
        f'<div class=ai-src-wrap>'
        f'<div class=ai-src-r1>'
        f'{ch_grids_html}'
        f'<div class="ai-meta ai-mt-turn"></div>'
        f'<div class="ai-meta ai-mt-chk"></div>'
        f'<div class="ai-meta ai-mt-term"></div>'
        f'</div>'
        f'<div class=ai-m1w>{m1u_cells}</div>'
        f'<div class=ai-m1w>{m1v_cells}</div>'
        f'<div class=ai-m1w>{m1k_cells}</div>'
        f'<div class=ai-m1c-row>'
        f'<div class=ai-m1c>{m1u_cells}</div>'
        f'<div class=ai-m1c>{m1v_cells}</div>'
        f'<div class=ai-m1c>{m1k_cells}</div>'
        f'</div></div></foreignObject>'
    )

    svg_panel_html = (
        '<section class="ai-col" aria-label="SVG 2-Ply + Quiescence CNN Minimax AI Engine">'
        '<div class=ai-panel>'
        '<div class=ai-head>'
        '<span class=ai-title>HTML5 SVG 2-Ply + Q3 CNN Minimax Engine</span>'
        f'<span class=ai-sub>{n_params:,} params &middot; 6&rarr;{c1_ch}({k1_sz}&times;{k1_sz})&rarr;{c2_ch}({k2_sz}&times;{k2_sz})&rarr;{c3_ch}({k3_sz}&times;{k3_sz})+3 skip &middot; Stockfish r={corr_r:.3f}</span>'
        '</div>'
        '<div class=ai-scroll>'
        '<div class=ai-stage-shell>'
        f'<svg class=ai-fstage width="{STAGE_VIEW_W}" height="{STAGE_H}" viewBox="0 0 {STAGE_VIEW_W} {STAGE_H}">'
        f'<g class=ai-visb><g class=ai-visa>'
        f'{open_comp_gs}'
        f'<rect x="0" y="0" width="{STAGE_W // 2}" height="{COMP_H // 2}" fill="#000" fill-opacity="0"/>'
        f'<g class=ai-luts-1x><g transform="translate({FO_X},{FO_Y})">{fo_inner_html}</g>{"".join(luts_1x_imgs)}</g>'
        f'<g class=ai-luts-2x><g transform="translate({FO_X // 2},{FO_Y // 2}) scale(0.5)">{fo_inner_html}</g>{"".join(luts_2x_imgs)}</g>'
        f'{close_comp_gs}'
        f'<g class=ai-vis-luts-1x>{"".join(vis_luts_1x_imgs)}</g>'
        f'<g class=ai-vis-luts-2x>{"".join(vis_luts_2x_imgs)}</g>'
        f'</g></g></svg>'
        '<div class=ai-hud aria-hidden="true">'
        f'<span class=hud-ttl style="left:{VIS_BOARD_X}px;top:16px">1. AI Best-Move Map (8&times;8)</span>'
        f'<span class=hud-sub style="left:{VIS_BOARD_X}px;top:33px">Violet = FROM &middot; Emerald = TO</span>'
        f'<div class=bd-files>{files_lbls}</div>'
        f'<div class=bd-ranks>{ranks_lbls}</div>'
        '<div class=ai-shield>'
        '<b>Your Turn to Move &mdash; SVG AI Engine Idle (0 GPU Load)</b>'
        '<span>Play any legal move on the chessboard &mdash; the 58-stage SVG 2-Ply + Q3 CNN engine wakes automatically on the AI&rsquo;s turn (or check <strong>Peek AI Hint</strong> above to run live neural search).</span>'
        '</div>'
        f'<span class=hud-ttl style="left:{VIS_MAT_X}px;top:16px">2. 2-Ply + Q3 Value Sheet (64 &times; 16 = 1,024 Leaves)</span>'
        f'<span class=hud-sub style="left:{VIS_MAT_X}px;top:33px">Cols = 64 Ply-1 targets &middot; Rows = 16 MVV-LVA replies + SEE Q3</span>'
        f'<span class=hud-sub style="left:{VIS_MAT_X}px;top:{VIS_QBAR_Y + VIS_QBAR_H + 5}px">Minimax Q(v&#x2081;) = min over opponent replies &middot; Gold bar = unique argmax move v&#x2081;*</span>'
        f'<span class=hud-ttl style="left:{VIS_WINBAR_X}px;top:{VIS_WINBAR_Y - 22}px">3. AI Win Probability (WDL)</span>'
        '<div class=wb-ticks><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>'
        '<div class=hm-legend>'
        '<span><i class=lg-sw style="background:#a855f7"></i>AI FROM</span>'
        '<span><i class=lg-sw style="background:#10b981"></i>AI TO / Winning</span>'
        '<span><i class=lg-sw style="background:#293b59"></i>50% Draw</span>'
        '<span><i class=lg-sw style="background:#e02e4d"></i>Losing</span>'
        '<span><i class=lg-sw style="background:#fbbf24"></i>Mate / Argmax</span>'
        '</div>'
        '<div class=arch-card>'
        '<h4>Zero-JS 2-Ply + Q3 CNN Architecture</h4>'
        '<ul class=arch-list>'
        '<li><code>.e12&rarr;s1</code> 64 Ply-1 &times; 16 FIDE-legal replies (1,024 leaves)</li>'
        '<li><code>s2m+s2q</code> Exact depth-2 mate check + SEE Ply-3 Quiescence</li>'
        f'<li><code>s2b&ndash;s4</code> <code>6&rarr;{c1_ch}({k1_sz}&times;{k1_sz})&rarr;{c2_ch}({k2_sz}&times;{k2_sz})&rarr;{c3_ch}({k3_sz}&times;{k3_sz})</code> + 3-ch skip</li>'
        f'<li><code>s5v&ndash;s5a</code> <code>{vh_in}&rarr;1</code> WDL head + <code>feMorphology</code> min/max</li>'
        '</ul></div>'
        f'<span class=hud-ttl style="left:{VIS_FEAT_X}px;top:{VIS_FEAT1_Y - 20}px">4. Live Skip-Material RGB Latent (ch {skip_ch0}&ndash;{skip_ch0 + 2}, 64&times;16)</span>'
        f'<span class=hud-ttl style="left:{VIS_FEAT_X}px;top:{VIS_FEAT2_Y - 20}px">5. Deep Tactical Conv3 RGB Latent (ch {3 * d3_g}&ndash;{3 * d3_g + 2}, 64&times;16)</span>'
        '<div class=ai-2p-cover>'
        '<b>2-Player Pass &amp; Play Active</b>'
        '<span>SVG 2-Ply CNN Minimax filters are paused for zero GPU overhead. Switch to <strong>1P vs SVG AI</strong> or check <strong>Peek AI Hint</strong> above to run live neural search.</span>'
        '</div>'
        '</div></div></div></div></section>'
    )

    tn_sentinels_html = "".join(
        f"<input type=radio name=t{k} id=t{k}n checked>" for k in range(max_plies)
    )
    arena_html = (
        f'<div class=ui-ctrl>{tn_sentinels_html}'
        '<div class=arena>'
        '<div class=play-col>'
        '<div class=status-card>'
        '<div class=status-top>'
        '<span class=status-kicker>Live CSS Cascade Referee</span>'
        '<div class=status-main id=chess-status><i></i></div>'
        '</div>'
        '<div class=ai-turn-Callout>'
        '<span>&#x26A1; <strong>SVG AI Turn:</strong> Execute the AI move shown on the 8&times;8 map &mdash; click <i class="sw-dot sw-from"></i> <strong>Violet (FROM)</strong> then <i class="sw-dot sw-to"></i> <strong>Emerald (TO)</strong>!</span>'
        '</div>'
        '<div class=stat-grid>'
        '<div class=stat-box><div class=stat-lbl>Turn</div><div class="stat-val sv-turn" id=chess-turn><i></i></div></div>'
        '<div class=stat-box><div class=stat-lbl>Plies</div><div class="stat-val sv-ply" id=chess-ply><i></i></div></div>'
        '<div class=stat-box><div class=stat-lbl>Last Move</div><div class="stat-val sv-last" id=chess-last><i></i></div></div>'
        '<div class=stat-box><div class=stat-lbl>50-Move</div><div class="stat-val sv-hmc" id=chess-hmc><i></i></div></div>'
        '</div></div>'
        '<div class=board-shell>'
        '<div class=board-frame>'
        f'<div class=board-grid>{"".join(board_cells_html)}</div>'
        f'<div class=ov-stack>{"".join(overlays_html)}</div>'
        '</div>'
        '<div class=term-banner><strong></strong><span>Use Undo Move or Reset Moves to continue playing</span></div>'
        '</div>'
        '<div class=ctrl-card>'
        '<div class=btn-row>'
        '<div class=btn-grp>'
        f'{"".join(undo_btns_html)}'
        '<button type=reset class="btn rst">&#x21BA; Reset Moves</button>'
        '</div>'
        '<span class=sel-pill>Selected: <i id=chess-sel></i></span>'
        '</div>'
        '<div class=promo-box>'
        '<div class=promo-head>'
        '<span class=promo-title>Pawn Promotion Piece</span>'
        '<span class=promo-cur><i></i></span>'
        '</div>'
        f'{"".join(promo_bars_html)}'
        '</div></div></div>'
        f'{svg_panel_html}'
        '</div></div>'
    )

    pipe_open: List[str] = ['<div class="st init">']
    open_divs = 1
    for k in range(max_plies):
        f_inputs = f"<input type=radio name=f{k} id=f{k}n checked>" + "".join(
            f"<input type=radio name=f{k} id=f{k}_{sq} class=f{sq}>" for sq in range(64)
        )
        t_inputs = "".join(
            f"<input type=radio name=t{k} id=t{k}_{sq} class=t{sq}>" for sq in range(64)
        )
        p_inputs = (
            f"<input type=radio name=p{k} id=p{k}q checked>"
            f"<input type=radio name=p{k} id=p{k}n class=pn>"
            f"<input type=radio name=p{k} id=p{k}r class=pr>"
            f"<input type=radio name=p{k} id=p{k}b class=pb>"
        )
        pipe_open.append(f'{f_inputs}{t_inputs}{p_inputs}<div class="st p"><div class="st m0"><div class="st m1">')
        open_divs += 3

    for eval_cls in spec.eval_classes:
        pipe_open.append(f'<div class="st {eval_cls}">')
        open_divs += 1

    pipe_close = "</div>" * open_divs

    return (
        "<!DOCTYPE html>\n<html lang=en>\n<head>\n<meta charset=utf-8>\n"
        "<meta name=viewport content='width=device-width,initial-scale=1'>\n"
        "<title>NoJS Chess — Pure HTML + CSS FIDE Chess &amp; SVG 2-Ply CNN Minimax AI</title>\n"
        f"<style>{''.join(css)}</style>\n</head>\n<body>\n"
        '<svg aria-hidden="true" width="0" height="0" style="position:absolute;width:0;height:0;overflow:hidden;pointer-events:none">'
        f'<defs>{"".join(svg_defs)}</defs></svg>\n'
        "<main class=chess-app>\n"
        f"{''.join(top_inputs_html)}\n"
        "<header class=hero>"
        "<nav class=back-row aria-label='Site navigation'>"
        "<a class=back-link href='../'>&larr; All NoJS Demos</a>"
        "</nav>"
        "<div class=badge-row>"
        "<span class='badge green'>0 Bytes JavaScript</span>"
        "<span class='badge purple'>SVG 2-Ply + Ply-3 Quiescence AI</span>"
        f"<span class=badge>{n_params:,}-Param Stockfish Value Net (r={corr_r:.3f} &middot; {rank_acc_pct:.1f}% Rank Acc)</span>"
        f"<span class='badge amber'>6&rarr;{c1_ch} ({k1_sz}&times;{k1_sz})&rarr;{c2_ch} ({k2_sz}&times;{k2_sz})&rarr;{c3_ch} ({k3_sz}&times;{k3_sz}) + 3-Ch Skip</span>"
        "</div>"
        "<h1>Pure HTML + CSS Chess with SVG 2-Ply + Quiescence CNN AI</h1>"
        "<p>Full FIDE legal-move referee inside CSS <code>@property &lt;integer&gt;</code> style recalculation "
        "paired with a 1-Player <strong>HTML5 SVG <code>&lt;filter&gt;</code> 2-Ply + Ply-3 Quiescence Minimax AI</strong> powered by "
        f"a {n_params:,}-parameter Stockfish-distilled 8-bit CNN Value Network "
        f"(<code>6&rarr;{c1_ch} ({k1_sz}&times;{k1_sz} Knight-vision)&rarr;{c2_ch} ({k2_sz}&times;{k2_sz})&rarr;{c3_ch} ({k3_sz}&times;{k3_sz}) + 3-ch material skip</code>, "
        f"<code>r={corr_r:.3f}</code>, <code>{rank_acc_pct:.1f}%</code> sibling move-ranking accuracy) "
        "evaluating 1,024 FIDE-legal leaves in parallel with exact Ply-1/Ply-2 checkmate detection &amp; SEE-filtered Ply-3 capture resolution.</p></header>\n"
        "<div class=top-bar-grid>"
        '<section class=panel aria-label="Game Mode and AI Controls">'
        "<div class=panel-head>"
        "<span class=panel-title>Game Mode (1P vs SVG AI / 2P)</span>"
        "<div class=pill-row>"
        "<label class=flip-pill for=ai-hint><i class=ui-chk></i><span>&#x1F441; Peek AI Hint</span></label>"
        "<label class=flip-pill for=flip-bd><i class=ui-chk></i><span>&#x21C5; Flip Board</span></label>"
        "</div></div>"
        f"<div class=mode-grid>{''.join(mode_cards_html)}</div>"
        "</section>"
        '<section class=panel aria-label="Position Presets">'
        "<div class=panel-head><span class=panel-title>Starting &amp; Tactical Presets</span></div>"
        f"<div class=preset-grid>{''.join(preset_cards_html)}</div>"
        "</section></div>\n"
        f"<form id=mv>{''.join(pipe_open)}{arena_html}{pipe_close}</form>\n"
        "</main>\n</body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone pure HTML+CSS+SVG Chess index.html")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--weights", default=WEIGHTS_DEFAULT_PATH)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"),
    )
    args = parser.parse_args()
    out_html = generate_html(max_plies=args.max_plies, weights_path=args.weights)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_html)
    print(f"Generated {args.out} ({len(out_html) / 1024:.1f} KB, max_plies={args.max_plies})")


if __name__ == "__main__":
    main()
