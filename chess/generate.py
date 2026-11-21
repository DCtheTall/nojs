#!/usr/bin/env python3
"""
Static Pure HTML + Inline CSS Generator for 2-Player FIDE Chess (`chess/index.html`).

Generates a standalone HTML5 document with inline CSS, zero JavaScript, and zero SVG
that executes full 2-player FIDE chess inside the browser's CSS cascade engine:
  - 100-ply move-history pipeline (`.p > .m0 > .m1`) with O(1) direct-child `:has()`
  - 13-stage Radix-3 Kogge-Stone ray evaluator (`.e0a .. .e11`) computing exact
    X-ray King safety, pins, single & double checks, castling, en passant (including
    horizontal discovered-check pins), promotions/underpromotions, checkmate,
    stalemate, insufficient material, and the 50-move rule
  - Interactive click-to-select, free piece switching, click-outside/self to deselect,
    promotion selector bar (`Q / N / R / B`), LIFO Undo, New Game reset, Flip Board,
    and 1-click tactical showcase presets.
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


# Standard White-perspective visual order: Rank 8 (a8..h8 = 56..63) down to Rank 1 (a1..h1 = 0..7)
VISUAL_SQ_ORDER: List[int] = [8 * r + f for r in range(7, -1, -1) for f in range(8)]


def minify_ui_css(raw: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([{};:,>~+])\s*", r"\1", s)
    return s.replace(";}", "}").strip()


def generate_html(max_plies: int = DEFAULT_MAX_PLIES) -> str:
    spec = build_chess_css_spec()
    css: List[str] = []

    # -----------------------------------------------------------------------
    # A. @counter-style definitions & @property <integer> registrations
    # -----------------------------------------------------------------------
    # Append U+FE0E (VS15 text presentation selector) so macOS/iOS render monochrome
    # vector glyphs styled by CSS `color` instead of fixed-color 3D pawn emojis!
    piece_syms = '"" "♟\\fe0e" "♞\\fe0e" "♝\\fe0e" "♜\\fe0e" "♛\\fe0e" "♚\\fe0e" "" "" "♟\\fe0e" "♞\\fe0e" "♝\\fe0e" "♜\\fe0e" "♛\\fe0e" "♚\\fe0e"'
    css.append(f"@counter-style chess-sym{{system:fixed 0;symbols:{piece_syms}}}")

    css.append(
        '@counter-style turn-lbl{system:fixed 0;symbols:"White" "Black"}'
    )
    css.append(
        '@counter-style promo-lbl{system:fixed 0;symbols:"" "" "Knight (♞)" "Bishop (♝)" "Rook (♜)" "Queen (♛)"}'
    )
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

    for prop in spec.registered_props:
        css.append(f'@property {prop}{{syntax:"<integer>";initial-value:0;inherits:true}}')

    # -----------------------------------------------------------------------
    # B. UI Theme, Board & HUD Styling (Zero JS, Zero SVG)
    # -----------------------------------------------------------------------
    ui_css = """
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0b0f17;color:#e2e8f0;line-height:1.5;padding:1.75rem 1rem 4rem;min-height:100vh
}
.chess-app{max-width:1060px;margin:0 auto;display:flex;flex-direction:column;gap:1.35rem}

header.hero{
  border:1px solid #1e293b;background:linear-gradient(180deg,#111827,#0f172a);
  border-radius:14px;padding:1.4rem 1.65rem;box-shadow:0 12px 32px rgba(0,0,0,.35)
}
.back-row{margin-bottom:.65rem}
.back-link{
  display:inline-flex;align-items:center;gap:.35rem;font-size:.76rem;font-weight:600;
  color:#94a3b8;text-decoration:none;padding:.22rem .62rem;border-radius:999px;
  background:#0b1120;border:1px solid #1e293b;transition:color .14s,border-color .14s,background .14s
}
.back-link:hover{color:#38bdf8;border-color:#38bdf8;background:#0f1d32}
.badge-row{display:flex;flex-wrap:wrap;gap:.45rem;margin-bottom:.6rem}
.badge{
  font-size:.71rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  padding:.2rem .55rem;border-radius:999px;background:#1e293b;color:#38bdf8;border:1px solid #334155
}
.badge.green{color:#4ade80;border-color:#14532d;background:#052e16}
.badge.purple{color:#c084fc;border-color:#4c1d95;background:#1e1035}
.badge.amber{color:#fbbf24;border-color:#78350f;background:#271705}
header.hero h1{font-size:1.62rem;font-weight:700;color:#f8fafc;letter-spacing:-.02em}
header.hero p{margin-top:.4rem;color:#94a3b8;font-size:.92rem}
header.hero code{color:#e2e8f0;background:#1e293b;padding:.1rem .32rem;border-radius:4px;font-size:.85em}

.panel{
  border:1px solid #1e293b;background:#111827;border-radius:14px;
  padding:1.15rem 1.35rem;box-shadow:0 8px 24px rgba(0,0,0,.25)
}
.panel-head{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;margin-bottom:.8rem}
.panel-title{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8}
.panel-hint{font-size:.78rem;color:#64748b}

.preset-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.5rem}
@media (max-width:960px){.preset-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:600px){.preset-grid{grid-template-columns:1fr}}
.pr-lbl{
  display:flex;flex-direction:column;gap:.2rem;padding:.55rem .68rem;border-radius:9px;
  background:#090d16;border:1px solid #263349;cursor:pointer;user-select:none;transition:all .12s
}
.pr-lbl:hover{border-color:#475569;background:#0f172a}
.pr-top{display:flex;align-items:center;gap:.4rem;font-size:.77rem;font-weight:700;color:#e2e8f0}
.pr-top input{accent-color:#38bdf8;cursor:pointer}
.pr-sub{font-size:.7rem;color:#64748b;line-height:1.35}
.pr-lbl:has(input:checked){border-color:#38bdf8;background:#0c2136;box-shadow:0 0 0 1px rgba(56,189,248,.28)}
.pr-lbl:has(input:checked) .pr-sub{color:#7dd3fc}

.flip-pill{
  display:inline-flex;align-items:center;gap:.45rem;padding:.35rem .72rem;border-radius:8px;
  background:#090d16;border:1px solid #334155;font-size:.78rem;font-weight:600;color:#cbd5e1;
  cursor:pointer;user-select:none;transition:border-color .12s,background .12s
}
.flip-pill:hover{border-color:#38bdf8;background:#0f172a}
.flip-pill input{accent-color:#38bdf8;cursor:pointer}

#mv{display:block}
.st,.ui-ctrl{display:contents}
#mv input{display:none}

.arena{
  display:grid;grid-template-columns:minmax(300px,544px) minmax(270px,1fr);
  gap:1.35rem;align-items:start
}
@media (max-width:860px){.arena{grid-template-columns:1fr;justify-items:center}}

.board-shell{
  position:relative;width:100%;max-width:544px;padding:12px;border-radius:16px;
  background:linear-gradient(180deg,#162032,#0e1524);border:1px solid #28364d;
  box-shadow:0 16px 40px rgba(0,0,0,.45)
}
.board-frame{
  position:relative;width:100%;aspect-ratio:1/1;border-radius:8px;overflow:hidden;
  border:2px solid #334155;transition:transform .28s cubic-bezier(.22,1,.36,1)
}
.chess-app:has(#flip-bd:checked) .board-frame{transform:rotate(180deg)}

.board-grid{
  position:absolute;inset:0;display:grid;
  grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr)
}
.sq{
  position:relative;display:flex;align-items:center;justify-content:center;
  user-select:none;overflow:hidden
}
.sq.lgt{background:#dce4ee}
.sq.drk{background:#475a74}

/* Subtle amber last-move tint */
.sq .lm{
  position:absolute;inset:0;background:rgba(250,204,21,.36);pointer-events:none
}
/* Sky-blue active selection highlight */
.sq .sl{
  position:absolute;inset:0;background:rgba(56,189,248,.42);
  box-shadow:inset 0 0 0 3px #0284c7;pointer-events:none
}
/* Crimson radial glow when King is in Check */
.sq .ck{
  position:absolute;inset:0;
  background:radial-gradient(circle,rgba(239,68,68,.9) 18%,rgba(220,38,38,.55) 62%,transparent 100%);
  box-shadow:inset 0 0 0 3px #ef4444;pointer-events:none
}

.sq-in{
  position:relative;z-index:2;width:100%;height:100%;
  display:flex;align-items:center;justify-content:center;
  transition:transform .28s cubic-bezier(.22,1,.36,1);pointer-events:none
}
.chess-app:has(#flip-bd:checked) .sq-in{transform:rotate(180deg)}

.pc{
  font-style:normal;font-size:clamp(1.85rem,5.3vw,2.95rem);line-height:1;
  font-variant-emoji:text;contain:style;transform:translateY(-1px)
}
.pc::after{content:counter(p,chess-sym)}

/* Coordinate labels on Rank 1..8 (a-file) & File a..h (rank 1) */
.crd{
  position:absolute;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.64rem;font-weight:800;line-height:1;pointer-events:none;opacity:.82
}
.sq.lgt .crd{color:#3b4d66}
.sq.drk .crd{color:#cbd5e1}
.crd-r{top:3px;left:4px}
.crd-f{bottom:3px;right:4px}

/* Quiet legal-move center dot & legal-capture target ring */
.mv-dot{
  position:absolute;z-index:3;width:30%;height:30%;border-radius:50%;
  background:rgba(16,185,129,.88);box-shadow:0 1px 4px rgba(0,0,0,.35);
  pointer-events:none
}
.cp-ring{
  position:absolute;inset:5%;z-index:3;border-radius:50%;
  border:4px solid rgba(239,68,68,.88);box-shadow:0 0 8px rgba(239,68,68,.35);
  pointer-events:none
}

/* Hit-target overlays: only active ply .ov{k} has display:grid! */
.ov-stack{position:absolute;inset:0;z-index:10}
.ov{
  display:none;position:absolute;inset:0;
  grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr)
}
.ov-bg{position:absolute;inset:0;z-index:1;cursor:default}
.ov i{position:relative;z-index:2;display:block;pointer-events:none}
.ov label.lf,.ov label.lt{
  position:absolute;top:0;display:block;cursor:pointer;pointer-events:auto
}
.ov label.lf{z-index:2}
.ov label.lt{z-index:3}
.ov label.lf:hover{background:rgba(56,189,248,.16)}
.ov label.lt:hover{background:rgba(16,185,129,.22)}

/* Game-over board shield + floating banner */
.term-banner{
  position:absolute;inset:auto 6% 6% 6%;z-index:20;
  padding:.75rem 1rem;border-radius:11px;text-align:center;
  background:rgba(9,13,22,.93);border:1px solid #38bdf8;
  box-shadow:0 12px 32px rgba(0,0,0,.65);backdrop-filter:blur(6px);
  pointer-events:none
}
.term-banner strong{
  display:block;font-size:.98rem;color:#f8fafc;letter-spacing:.01em
}
.term-banner strong::after{content:counter(st,game-status)}
.term-banner span{font-size:.76rem;color:#94a3b8}

/* Right-hand HUD & Controls Column */
.hud-col{width:100%;display:flex;flex-direction:column;gap:1rem}

.status-card{
  border-radius:14px;padding:1.15rem 1.3rem;
  background:linear-gradient(180deg,#0f1c2e,#0b1320);
  border:1px solid #1e3a5f;box-shadow:0 8px 24px rgba(0,0,0,.3)
}
.status-kicker{
  font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#7dd3fc
}
.status-main{
  margin-top:.3rem;font-size:1.22rem;font-weight:800;color:#f8fafc;letter-spacing:-.01em
}
.status-main i{font-style:normal}
.status-main i::after{content:counter(st,game-status)}

.stat-grid{
  margin-top:.9rem;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem
}
.stat-box{
  background:#080c14;border:1px solid #1e293b;border-radius:9px;padding:.55rem .75rem
}
.stat-lbl{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b}
.stat-val{
  margin-top:.12rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.96rem;font-weight:700;color:#e2e8f0
}
.stat-val i{font-style:normal}
.sv-turn i::after{content:counter(tn,turn-lbl)}
.sv-ply i::after{content:counter(pl)}
.sv-hmc i::after{content:counter(hm) " / 100"}
.sv-sel i::after{content:counter(slc,sq-coord)}
.sv-last i::after{content:counter(lf,sq-coord) " → " counter(lt,sq-coord)}

/* Action buttons (LIFO Undo + Scoped Reset) */
.ctrl-card{
  border:1px solid #1e293b;background:#111827;border-radius:14px;padding:1.1rem 1.25rem;
  display:flex;flex-direction:column;gap:.85rem
}
.btn-row{display:flex;flex-wrap:wrap;gap:.6rem}
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:.4rem;
  padding:.58rem .95rem;border-radius:9px;font-size:.82rem;font-weight:700;
  cursor:pointer;user-select:none;border:1px solid #334155;background:#162032;color:#f8fafc;
  transition:background .13s,border-color .13s,transform .1s
}
.btn:hover{background:#1e293b;border-color:#38bdf8;color:#7dd3fc}
.btn:active{transform:translateY(1px)}
.btn.rst{border-color:#7f1d1d;background:#231115;color:#fca5a5}
.btn.rst:hover{border-color:#ef4444;background:#35151b;color:#fecaca}
.undo-btn{display:none;border-color:#0369a1;background:#0c2438;color:#7dd3fc}
.undo-disabled{
  display:none;align-items:center;gap:.4rem;padding:.58rem .95rem;border-radius:9px;
  font-size:.82rem;font-weight:600;border:1px solid #1e293b;background:#0b0f17;color:#475569
}
#t0n:checked~.arena .undo-disabled{display:inline-flex}

/* Promotion piece chooser bar */
.promo-box{
  padding:.8rem .9rem;border-radius:10px;background:#090d16;border:1px solid #1e293b;
  transition:border-color .15s,box-shadow .15s
}
.promo-head{display:flex;justify-content:space-between;align-items:center;gap:.4rem;margin-bottom:.5rem}
.promo-title{font-size:.73rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#94a3b8}
.promo-cur{font-size:.75rem;font-weight:700;color:#fbbf24}
.promo-cur i{font-style:normal}
.promo-cur i::after{content:counter(pr,promo-lbl)}
.pr-bar{display:none;grid-template-columns:repeat(4,1fr);gap:.4rem}
.pr-opt{
  display:flex;flex-direction:column;align-items:center;gap:.1rem;padding:.42rem .3rem;
  border-radius:7px;background:#111827;border:1px solid #28364d;cursor:pointer;
  color:#cbd5e1;font-size:.74rem;font-weight:600;user-select:none;transition:all .12s
}
.pr-opt:hover{border-color:#fbbf24;color:#fef08a}
.pr-opt b{font-size:1.2rem;line-height:1.1;font-variant-emoji:text}

.rules-note{
  font-size:.77rem;color:#64748b;line-height:1.5;border-top:1px solid #1e293b;padding-top:.75rem
}
.rules-note strong{color:#94a3b8}
"""
    css.append(minify_ui_css(ui_css))

    # -----------------------------------------------------------------------
    # C. Initial Position, Presets, & Scoped O(1) Ply Input Rules
    # -----------------------------------------------------------------------
    std_vars = state_to_css_vars(parse_fen(START_FEN))
    init_decls = ";".join(f"{k}:{v}" for k, v in sorted(std_vars.items()))
    css.append(f".init{{{init_decls}}}")

    for pid, _title, fen, _sub in PRESETS:
        if pid == "std":
            continue
        p_vars = state_to_css_vars(parse_fen(fen))
        p_decls = ";".join(f"{k}:{v}" for k, v in sorted(p_vars.items()) if p_vars[k] != std_vars.get(k))
        if p_decls:
            css.append(f".chess-app:has(#pr-{pid}:checked) .init{{{p_decls}}}")

    # Shared ply input decoding via pure forward sibling combinator `~` (ZERO `:has()`!)
    css.append(".p{--in_f:64;--in_t:64;--in_c:0;--in_pr:5}")
    for sq in range(64):
        css.append(f".f{sq}:checked~.p{{--in_f:{sq}}}")
        css.append(f".t{sq}:checked~.p{{--in_t:{sq};--in_c:1}}")
    css.append(".pn:checked~.p{--in_pr:2}")
    css.append(".pb:checked~.p{--in_pr:3}")
    css.append(".pr:checked~.p{--in_pr:4}")

    # Active ply selector rules using O(1) adjacent sibling `+` on `#t0n..#t99n` (ZERO `:has()`!)
    css.append("#t0n:checked~.arena .ov0,#t0n:checked~.arena .pb0{display:grid}")
    for k in range(1, max_plies):
        sel_k = f"#t{k - 1}n:not(:checked)+#t{k}n:checked~.arena"
        css.append(f"{sel_k} .ov{k},{sel_k} .pb{k}{{display:grid}}")
        css.append(f"{sel_k} .ub{k}{{display:inline-flex}}")
    css.append(f"#t{max_plies - 1}n:not(:checked)~.arena .ub{max_plies}{{display:inline-flex}}")

    # Highlight active promotion choice button via `--cur_pr`
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
    # D. Compiled Chess Pipeline Stages (`.m0`, `.m1`, `.e0a .. .e11`)
    # -----------------------------------------------------------------------
    for cls_name, rules in spec.stage_defs.items():
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".{cls_name}{{{body}}}")

    # -----------------------------------------------------------------------
    # E. Per-Square Visual Board & Hit-Target Rules (`0..63`)
    # -----------------------------------------------------------------------
    for sq in range(64):
        bu = f"var(--b{sq})"
        is_blk = f"clamp(0,{bu} - 8,1)"
        is_last = f"min(1,{css_eq('var(--last_f)', sq)} + {css_eq('var(--last_t)', sq)})"
        quiet_dot = f"calc(var(--leg_{sq})*(1 - var(--cap_{sq})))"

        # Piece glyph counter + ivory vs obsidian color & crisp shadow
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

        # Geometric hit-target gating inside active `.ov{k}`: off-screen + 0x0 when inactive!
        css.append(
            f".lf{sq}{{"
            f"left:calc(-9999px*(1 - var(--can_sel_{sq})));"
            f"width:calc(100%*var(--can_sel_{sq}));"
            f"height:calc(100%*var(--can_sel_{sq}))}}"
        )
        css.append(
            f".lt{sq}{{"
            f"left:calc(-9999px*(1 - var(--leg_{sq})));"
            f"width:calc(100%*var(--leg_{sq}));"
            f"height:calc(100%*var(--leg_{sq}))}}"
        )

    # HUD live counter bindings & terminal banner visibility
    css.append(
        ".arena{"
        "counter-reset:st var(--status) tn var(--turn) pl var(--ply) hm var(--hmc) "
        "slc var(--cur_from) lf var(--last_f) lt var(--last_t) pr var(--cur_pr)}"
    )
    css.append(
        ".term-banner{"
        "opacity:var(--is_terminal);"
        "transform:translateY(calc(12px*(1 - var(--is_terminal)))) scale(calc(.95 + .05*var(--is_terminal)))}"
    )
    css.append(
        ".status-card{"
        "border-color:rgb(calc(30 + 209*min(1,var(--in_check) + var(--checkmate))),calc(58 + 10*var(--is_terminal)),calc(95 - 27*var(--in_check)))}"
    )

    # -----------------------------------------------------------------------
    # F. HTML Markup Assembly (Presets, 100-Ply Pipeline, Evaluators, & Board)
    # -----------------------------------------------------------------------
    preset_cards_html: List[str] = []
    for idx, (pid, title, _fen, sub) in enumerate(PRESETS):
        chk = " checked" if idx == 0 else ""
        preset_cards_html.append(
            f'<label class=pr-lbl for=pr-{pid}>'
            f'<span class=pr-top><input type=radio name=preset id=pr-{pid}{chk}>'
            f'<span>{html.escape(title)}</span></span>'
            f'<span class=pr-sub>{html.escape(sub)}</span></label>'
        )

    # Build visual 8x8 board cells (64 squares total)
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

    # Build per-ply overlays `.ov0 .. .ov{T-1}`, promo bars `.pb0 .. .pb{T-1}`, and undo buttons
    overlays_html: List[str] = []
    promo_bars_html: List[str] = []
    undo_btns_html: List[str] = ['<span class=undo-disabled>&#x21B6; Undo Move</span>']

    for k in range(max_plies):
        cells_k = "".join(
            f'<i><label class="lf lf{sq}" for=f{k}_{sq}></label>'
            f'<label class="lt lt{sq}" for=t{k}_{sq}></label></i>'
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

    tn_sentinels_html = "".join(
        f"<input type=radio name=t{k} id=t{k}n checked>" for k in range(max_plies)
    )
    arena_html = (
        f'<div class=ui-ctrl>{tn_sentinels_html}'
        '<div class=arena>'
        '<div class=board-shell>'
        '<div class=board-frame>'
        f'<div class=board-grid>{"".join(board_cells_html)}</div>'
        f'<div class=ov-stack>{"".join(overlays_html)}</div>'
        '</div>'
        '<div class=term-banner><strong></strong><span>Use Undo Move or Reset Moves to continue playing</span></div>'
        '</div>'
        '<aside class=hud-col aria-label="Game Status and Controls">'
        '<div class=status-card>'
        '<div class=status-kicker>Live CSS Cascade Referee</div>'
        '<div class=status-main id=chess-status><i></i></div>'
        '<div class=stat-grid>'
        '<div class=stat-box><div class=stat-lbl>Side to Move</div><div class="stat-val sv-turn" id=chess-turn><i></i></div></div>'
        '<div class=stat-box><div class=stat-lbl>Committed Plies</div><div class="stat-val sv-ply" id=chess-ply><i></i></div></div>'
        '<div class=stat-box><div class=stat-lbl>Selected Piece</div><div class="stat-val sv-sel" id=chess-sel><i></i></div></div>'
        '<div class=stat-box><div class=stat-lbl>Last Move</div><div class="stat-val sv-last" id=chess-last><i></i></div></div>'
        '<div class=stat-box style="grid-column:span 2"><div class=stat-lbl>50-Move Halfmove Clock</div><div class="stat-val sv-hmc" id=chess-hmc><i></i></div></div>'
        '</div></div>'
        '<div class=ctrl-card>'
        '<div class=btn-row>'
        f'{"".join(undo_btns_html)}'
        '<button type=reset class="btn rst">&#x21BA; Reset Moves</button>'
        '</div>'
        '<div class=promo-box>'
        '<div class=promo-head>'
        '<span class=promo-title>Pawn Promotion Piece</span>'
        '<span class=promo-cur><i></i></span>'
        '</div>'
        f'{"".join(promo_bars_html)}'
        '</div>'
        '<p class=rules-note>'
        '<strong>100% Pure HTML + CSS:</strong> Click any friendly piece with legal moves to view exact '
        'ray-cast &amp; pin-filtered targets (emerald dots = quiet moves, red rings = captures/en passant). '
        'Switch pieces freely, click empty squares to deselect, or choose an underpromotion piece before '
        'advancing a 7th-rank pawn.'
        '</p></div></aside></div></div>'
    )

    # Open the nested `.init` -> 100 x (sibling inputs + `.p` > `.m0` > `.m1`) -> `.e0a .. .e11` pipeline!
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
        "<title>NoJS Chess — Pure HTML + Inline CSS 2-Player FIDE Chess</title>\n"
        f"<style>{''.join(css)}</style>\n</head>\n<body>\n<main class=chess-app>\n"
        "<header class=hero>"
        "<nav class=back-row aria-label='Site navigation'>"
        "<a class=back-link href='../'>&larr; All NoJS Demos</a>"
        "</nav>"
        "<div class=badge-row>"
        "<span class='badge green'>0 Bytes JavaScript</span>"
        "<span class=badge>0 SVG / 0 Images</span>"
        "<span class='badge purple'>Radix-3 Kogge-Stone Ray Legality</span>"
        "<span class='badge amber'>Pins · Castling · En Passant · Mate &amp; Stalemate</span>"
        "</div>"
        "<h1>Pure HTML + CSS 2-Player Chess</h1>"
        "<p>Full FIDE legal-move generation, absolute pin &amp; double-check filtering, King X-ray safety, "
        "castling, en passant (including horizontal discovered-check pins), pawn underpromotions, checkmate, "
        "stalemate, insufficient material, and the 50-move rule computed strictly inside browser CSS style "
        "recalculation via <code>@property &lt;integer&gt;</code> wires.</p></header>\n"
        '<section class=panel aria-label="Position Presets and Board Orientation">'
        "<div class=panel-head>"
        "<span class=panel-title>Position Presets &amp; Board View</span>"
        "<label class=flip-pill for=flip-bd><input type=checkbox id=flip-bd><span>&#x21C5; Flip Board (Black View)</span></label>"
        "</div>"
        f"<div class=preset-grid>{''.join(preset_cards_html)}</div>"
        "</section>\n"
        f"<form id=mv>{''.join(pipe_open)}{arena_html}{pipe_close}</form>\n"
        "</main>\n</body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone pure HTML+CSS Chess index.html")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"),
    )
    args = parser.parse_args()
    out_html = generate_html(max_plies=args.max_plies)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_html)
    print(f"Generated {args.out} ({len(out_html) / 1024:.1f} KB, max_plies={args.max_plies})")


if __name__ == "__main__":
    main()
