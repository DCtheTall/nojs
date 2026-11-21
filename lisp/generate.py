#!/usr/bin/env python3
"""
Static Pure HTML + Inline CSS Generator for the NoJS Lisp Interpreter (`lisp/index.html`).

Generates a standalone HTML5 document with inline CSS, zero JavaScript, and zero SVG
that lexes, parses, and evaluates live Lisp S-expressions inside the browser's CSS
style-recalculation engine.

Input ergonomics (100% HTML+CSS, 0 JS):
  - 40-Slot 8-Bit Quick Token Composer (defaults to all `;` comment chars so it starts blank):
    * Selectable space (`\u00a0`, `␣`, `.`), `;` comment-to-end, `(`, `)`
    * All integers `-10` through `100` (covering all `0-99`) + preset milestones (`144, 462, 1071`)
    * 1-keystroke typeahead (`def`='d', `if`='i', `let`='l', `set`='s', `mod`='m')
    * Selecting any Showcase Preset writes its tokens directly into `T00..T39` via CSS
      `--ptid0..39` + `@counter-style` overlays, while any cell can be edited live on top!
  - Popup-free `<label for="t0">` / `<label for="c0">` focus triggers + `<button type="reset">`
  - Full 64-Character Raw ASCII Editor (with selectable space, `0-9`, `a-z`, `;`)
"""

from __future__ import annotations

import argparse
import html
import os
import re
from typing import List, Sequence

from css_vm import NUM_FUNCS, NUM_VARS, NUM_VM_TICKS, build_lisp_css_spec
from lisp_ref import (
    CHAR_COMMENT,
    CODE_TO_CHAR,
    DEFAULT_PROGRAM,
    MAX_TOKENS,
    PRESETS,
    SELECT_CHARS,
    SOURCE_LEN,
    TOKEN_COMPOSER_LEN,
    TOKEN_PALETTE,
    TOKEN_SELECT_OPTIONS,
    encode_token_composer_slots,
    eval_lisp_ref,
    pad_source_codes,
    tokenize_to_palette_ids,
)


BIT6_LETTERS = "abcdef"    # 6-bit char code: bit 0 (1) -> 'a' .. bit 5 (32) -> 'f'
BIT8_LETTERS = "abcdefgh"  # 8-bit token ID:  bit 0 (1) -> 'a' .. bit 7 (128) -> 'h'


def minify_ui_css(raw: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([{};:,>])\s*", r"\1", s)
    return s.replace(";}", "}").strip()


def _css_escape_str(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def generate_html(
    default_program: str = DEFAULT_PROGRAM,
    default_tokens: Sequence[str] | None = None,
    start_in_char_mode: bool = False,
) -> str:
    spec = build_lisp_css_spec(num_ticks=NUM_VM_TICKS)
    css: List[str] = []

    # -----------------------------------------------------------------------
    # A. @counter-style definitions for symbolic CSS readout & token overlays
    # -----------------------------------------------------------------------
    css.append('@counter-style tok-kind{system:fixed 0;symbols:"·" "(" ")" "NUM" "SYM"}')
    css.append('@counter-style vm-mode{system:fixed 0;symbols:"EVAL" "CONT" "DONE" "ERR" "DIV"}')
    css.append(
        '@counter-style err-msg{system:fixed 0;symbols:'
        '"OK — Evaluated Cleanly" '
        '"Syntax Error: Unbalanced parentheses or >40 tokens" '
        '"Runtime Error: Unbound symbol" '
        '"Runtime Error: Division or modulo by zero" '
        '"Runtime Error: Invalid call, arity mismatch, or table overflow" '
        f'"Runtime Error: Exceeded {NUM_VM_TICKS}-tick CSS cascade step budget"}}'
    )

    # Token Composer live S-expression preview counter-style (0..155)
    tok_prev_syms = " ".join(
        _css_escape_str(prev if idx >= 2 else " ")
        for idx, (_, _, _, prev) in enumerate(TOKEN_PALETTE)
    )
    css.append(f"@counter-style tok-src{{system:fixed 0;symbols:{tok_prev_syms}}}")

    # Token Editor cell overlay counter-style (0..155) for rendering active preset / ';' default
    tok_lbl_syms = " ".join(
        _css_escape_str("\u00a0" if idx == 0 else lbl)
        for idx, (lbl, _, _, _) in enumerate(TOKEN_PALETTE)
    )
    css.append(f"@counter-style tok-lbl{{system:fixed 0;symbols:{tok_lbl_syms}}}")

    # 64-Char ASCII live code preview counter-style (0..55)
    char_prev_list = [
        _css_escape_str(CODE_TO_CHAR.get(c, " ") if c != CHAR_COMMENT else " ")
        for c in range(56)
    ]
    css.append(f"@counter-style chr-src{{system:fixed 0;symbols:{' '.join(char_prev_list)}}}")

    extra_ui_props = (
        [f"--ucm{i}" for i in range(TOKEN_COMPOSER_LEN - 1)]
        + [f"--dsemi{i}" for i in range(TOKEN_COMPOSER_LEN - 1)]
        + [f"--dsemig{g}" for g in range((TOKEN_COMPOSER_LEN - 1 + 7) // 8)]
    )
    for prop in list(spec.registered_props) + extra_ui_props:
        css.append(f'@property {prop}{{syntax:"<integer>";initial-value:0;inherits:true}}')

    # -----------------------------------------------------------------------
    # B. UI Theme Styling (Dark hardware/VM inspector aesthetic, zero SVG/JS)
    # -----------------------------------------------------------------------
    ui_css = """
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0b0f17;color:#e2e8f0;line-height:1.5;padding:2rem 1rem 4rem
}
.lisp-app{max-width:1060px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}
.grid-form{display:block}
header.hero{
  border:1px solid #1e293b;background:linear-gradient(180deg,#111827,#0f172a);
  border-radius:14px;padding:1.45rem 1.7rem;box-shadow:0 12px 32px rgba(0,0,0,.35)
}
.back-row{margin-bottom:.7rem}
.back-link{
  display:inline-flex;align-items:center;gap:.35rem;font-size:.76rem;font-weight:600;
  color:#94a3b8;text-decoration:none;padding:.22rem .62rem;border-radius:999px;
  background:#0b1120;border:1px solid #1e293b;transition:color .14s,border-color .14s,background .14s
}
.back-link:hover{color:#38bdf8;border-color:#38bdf8;background:#0f1d32}
.badge-row{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:.65rem}
.badge{
  font-size:.71rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  padding:.2rem .55rem;border-radius:999px;background:#1e293b;color:#38bdf8;border:1px solid #334155
}
.badge.green{color:#4ade80;border-color:#14532d;background:#052e16}
.badge.purple{color:#c084fc;border-color:#4c1d95;background:#1e1035}
header.hero h1{font-size:1.62rem;font-weight:700;color:#f8fafc;letter-spacing:-.02em}
header.hero p{margin-top:.42rem;color:#94a3b8;font-size:.92rem}
header.hero code{color:#e2e8f0;background:#1e293b;padding:.1rem .32rem;border-radius:4px;font-size:.85em}

.panel{
  border:1px solid #1e293b;background:#111827;border-radius:14px;
  padding:1.3rem 1.5rem;box-shadow:0 8px 24px rgba(0,0,0,.25)
}
.panel-head{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:.5rem;margin-bottom:.85rem}
.panel-title{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8}
.panel-hint{font-size:.79rem;color:#64748b}

.preset-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.5rem;margin-bottom:.85rem}
@media (max-width:920px){.preset-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:580px){.preset-grid{grid-template-columns:1fr}}
.pr-lbl{
  display:flex;flex-direction:column;gap:.16rem;padding:.52rem .68rem;border-radius:9px;
  background:#090d16;border:1px solid #263349;cursor:pointer;user-select:none;transition:all .12s
}
.pr-lbl:hover{border-color:#475569;background:#0f172a}
.pr-lbl.edit-mode{border-color:#1e3a5f;background:#0a1626}
.pr-top{display:flex;align-items:center;gap:.4rem;font-size:.77rem;font-weight:600;color:#e2e8f0}
.pr-top input{accent-color:#38bdf8;cursor:pointer}
.pr-code{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;color:#64748b;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}

.kbd-bar{
  display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.65rem;
  margin-bottom:.75rem;padding:.55rem .8rem;border-radius:9px;background:#090d16;border:1px solid #1e293b
}
.kbd-actions{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}
.kbd-btn{
  display:inline-flex;align-items:center;gap:.35rem;padding:.38rem .75rem;border-radius:7px;
  background:#0284c7;color:#f8fafc;font-size:.78rem;font-weight:700;cursor:pointer;user-select:none;
  border:1px solid #38bdf8;transition:background .12s
}
.kbd-btn:hover{background:#0369a1}
.rst-btn{
  padding:.38rem .7rem;border-radius:7px;background:#1e293b;color:#cbd5e1;
  font-size:.77rem;font-weight:600;border:1px solid #334155;cursor:pointer;transition:all .12s
}
.rst-btn:hover{background:#334155;color:#f8fafc}
.kbd-tips{font-size:.74rem;color:#94a3b8;display:flex;flex-wrap:wrap;gap:.35rem .7rem}
.kbd-tips kbd{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.71rem;font-weight:700;
  padding:.08rem .35rem;border-radius:4px;background:#1e293b;color:#7dd3fc;border:1px solid #334155
}

.preset-banner{
  display:none;margin-bottom:.75rem;padding:.55rem .85rem;border-radius:8px;
  background:#0c1e33;border:1px solid #0369a1;color:#bae6fd;font-size:.79rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace
}

.tok-editor-wrap{display:block}
.char-editor-wrap,.kbd-for-chr{display:none}
.lisp-app:has(#pr-edit:checked) .tok-editor-wrap,.lisp-app:has(#pr-edit:checked) .kbd-for-tok{display:none}
.lisp-app:has(#pr-edit:checked) .kbd-for-chr{display:inline-flex}
.lisp-app:has(#pr-edit:checked) .char-editor-wrap{display:block}

.tok-editor-grid{
  display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:5px;
  background:#090d16;padding:9px;border-radius:10px;border:1px solid #263349
}
@media (max-width:780px){.tok-editor-grid{grid-template-columns:repeat(5,minmax(0,1fr))}}
.char-grid{
  display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:4px;
  background:#090d16;padding:8px;border-radius:10px;border:1px solid #263349
}
@media (max-width:720px){.char-grid{grid-template-columns:repeat(8,minmax(0,1fr))}}
.cw{display:flex;flex-direction:column;align-items:center;gap:2px;transition:opacity .12s}
.char-grid .cw:has(:checked[cm]) ~ .cw{opacity:.24}
.ci{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.58rem;color:#475569}
.tsw{position:relative;width:100%}
select.cc{
  width:100%;height:2.2rem;appearance:none;-webkit-appearance:none;
  background:#162032;color:#f8fafc;border:1px solid #334155;border-radius:5px;
  text-align:center;text-align-last:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92rem;font-weight:600;
  cursor:pointer;transition:border-color .12s,background .12s,box-shadow .12s
}
select.cc option{color:#f8fafc;background:#0f172a}
select.cc.tc-sel{height:2.35rem;font-size:.85rem}
select.cc.tc-sel:has(:checked[def]){color:transparent}
select.cc.tc-sel:has(:checked:not([def])){border-color:#38bdf8;color:#86efac;background:#10243a}
select.cc:hover{border-color:#475569;background:#1e293b}
select.cc:focus{outline:none;border-color:#38bdf8;background:#0c2438;box-shadow:0 0 0 2px rgba(56,189,248,.4)}
.ptok{
  position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  pointer-events:none;color:#f8fafc;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85rem;font-weight:600
}
.ptok i{font-style:normal}
.ptok i::after{content:counter(ptv,tok-lbl)}
.tsw:has(:checked:not([def])) .ptok{display:none}

.ref-bar{
  margin-top:.75rem;display:flex;flex-wrap:wrap;gap:.45rem .85rem;font-size:.76rem;color:#94a3b8;
  background:#090d16;padding:.55rem .8rem;border-radius:8px;border:1px solid #1e293b
}
.ref-bar code{color:#7dd3fc;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

.cg{display:flex;flex-direction:column;gap:1.35rem}
.cg div.st{display:contents}

.output-panel{
  order:-3;border:1px solid #0284c7;background:linear-gradient(180deg,#0c192c,#0b1322);
  border-radius:14px;padding:1.4rem 1.6rem;box-shadow:0 0 36px rgba(14,165,233,.14)
}
.src-echo{
  margin-bottom:.9rem;padding:.55rem .85rem;border-radius:8px;
  background:#060a12;border:1px solid #1e3a5f;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88rem;color:#7dd3fc
}
.src-echo-lbl{color:#64748b;font-size:.69rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-right:.5rem}
.src-tok-view{display:inline}
.src-chr-view{display:none}
.lisp-app:has(#pr-edit:checked) .src-tok-view{display:none}
.lisp-app:has(#pr-edit:checked) .src-chr-view{display:inline}
.src-tok-view i,.src-chr-view i{font-style:normal}
.src-tok-view i::after{content:counter(tpv,tok-src)}
.src-chr-view i::after{content:counter(cpv,chr-src)}

.res-grid{display:grid;grid-template-columns:1fr auto auto;gap:.9rem;align-items:stretch}
@media (max-width:720px){.res-grid{grid-template-columns:1fr}}
.res-box{
  padding:.95rem 1.15rem;background:#060a12;border:1px solid #1e3a5f;border-radius:10px;
  display:flex;flex-direction:column;justify-content:center
}
.res-lbl{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#64748b;margin-bottom:.2rem}
.res-val{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.85rem;font-weight:800;
  color:#4ade80;letter-spacing:-.01em;line-height:1.2
}
.res-val i,.stat-num i,.tk-k,.tk-v,.tk-n,.ev-k,.ev-v,.tr-c{font-style:normal}
.res-val i::after{content:counter(ans)}
.stat-num{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.3rem;font-weight:700;color:#38bdf8
}
.stat-num.st-cnt i::after{content:counter(stc)}
.stat-num.nt-cnt i::after{content:counter(ntc)}

.status-line{
  margin-top:.75rem;padding:.55rem .85rem;border-radius:8px;
  background:#071a12;border:1px solid #14532d;color:#86efac;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem
}
.status-line i{font-style:normal}
.status-line i::after{content:counter(errc,err-msg)}
.err-alert{
  margin-top:.65rem;padding:0 .85rem;border-radius:8px;
  background:#2c0b0e;border:0 solid #dc2626;color:#fca5a5;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.83rem;font-weight:600;
  overflow:hidden;transition:opacity .12s
}
.err-alert i{font-style:normal;display:block;padding:.55rem 0}
.err-alert i::after{content:counter(errc,err-msg)}

.insp-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1.1rem}
@media (max-width:780px){.insp-grid{grid-template-columns:1fr}}
.sub-card{background:#060a12;border:1px solid #1e293b;border-radius:10px;padding:.85rem 1rem}
.sub-title{font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#94a3b8;margin-bottom:.55rem}
.slot-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}
.slot-chip{
  display:flex;justify-content:space-between;gap:.4rem;padding:.3rem .55rem;border-radius:6px;
  background:#0f172a;border:1px solid #1e293b;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.74rem;color:#cbd5e1
}
.ev-k::after{content:"sym#" counter(vk) " = "}
.ev-v::after{content:counter(vv)}
.fn-k::after{content:"fn#" counter(fk)}
.fn-b::after{content:"@tok " counter(fb)}

.tok-strip{
  margin-top:1rem;background:#060a12;border:1px solid #1e293b;border-radius:10px;padding:.85rem 1rem
}
.tok-grid{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:4px}
@media (max-width:820px){.tok-grid{grid-template-columns:repeat(5,minmax(0,1fr))}}
.tok-cell{
  background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:.28rem .35rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;
  display:flex;flex-direction:column;gap:1px
}
.tok-top{display:flex;justify-content:space-between;color:#64748b;font-size:.6rem}
.tk-k{font-weight:700;color:#38bdf8}
.tk-k::after{content:counter(tk,tok-kind)}
.tk-v{color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tk-v::after{content:counter(tv)}
.tk-n{color:#64748b;font-size:.6rem}
.tk-n::after{content:"next:" counter(tn)}

.trace-head{order:5;border:1px solid #1e293b;background:#111827;border-radius:14px;padding:1.1rem 1.4rem}
.tr{
  display:grid;grid-template-columns:5.2rem 5.8rem 6rem 1fr 5.5rem;gap:.65rem;align-items:center;
  padding:.42rem .8rem;background:#0c121e;border:1px solid #1e293b;border-radius:7px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem
}
@media (max-width:720px){.tr{grid-template-columns:repeat(3,1fr)}}
.trn{font-weight:700;color:#38bdf8}
.tr-m{color:#c084fc;font-weight:700}
.tr-m::after{content:counter(md,vm-mode)}
.tr-pc::after{content:"pc=" counter(pcv)}
.tr-ac{color:#4ade80}
.tr-ac::after{content:"acc=" counter(acv)}
.tr-sp{color:#94a3b8}
.tr-sp::after{content:"sp=" counter(spv)}
"""
    css.append(minify_ui_css(ui_css))

    # -----------------------------------------------------------------------
    # C. Input Decoding Rules (:root ';' blank default, Presets -> Token Editor, 8-bit :has())
    # -----------------------------------------------------------------------
    root_decls = ["--use_tok:1"]
    root_decls.extend(f"--ptid{i}:1" for i in range(TOKEN_COMPOSER_LEN))
    root_decls.extend(f"--ovr{i}:0" for i in range(TOKEN_COMPOSER_LEN))
    root_decls.extend(f"--ucm{i}:0" for i in range(TOKEN_COMPOSER_LEN - 1))
    root_decls.extend(f"--tb{i}_{b}:0" for i in range(TOKEN_COMPOSER_LEN) for b in range(8))
    root_decls.extend(f"--b{i}_{b}:0" for i in range(SOURCE_LEN) for b in range(6))
    css.append(f":root{{{';'.join(root_decls)}}}")

    css.append(".lisp-app:has(#pr-edit:checked){--use_tok:0}")

    all_preset_ids = ["tok", "edit"] + [pid for pid, *_ in PRESETS]
    pr_active_lbl = ",".join(f".lisp-app:has(#pr-{pid}:checked) label[for=pr-{pid}]" for pid in all_preset_ids)
    pr_active_code = ",".join(f".lisp-app:has(#pr-{pid}:checked) label[for=pr-{pid}] .pr-code" for pid in all_preset_ids)
    css.append(f"{pr_active_lbl}{{border-color:#38bdf8;background:#0c2136;box-shadow:0 0 0 1px rgba(56,189,248,.28)}}")
    css.append(f"{pr_active_code}{{color:#7dd3fc}}")

    for slot in range(TOKEN_COMPOSER_LEN):
        css.append(f".lisp-app:has(#t{slot} :checked:not([def])){{--ovr{slot}:1}}")
        if slot < TOKEN_COMPOSER_LEN - 1:
            css.append(f".lisp-app:has(#t{slot} :checked[cm]){{--ucm{slot}:1}}")
        css.append(f".ptok{slot}{{counter-reset:ptv var(--ptid{slot})}}")
        for bit_idx, bchar in enumerate(BIT8_LETTERS):
            css.append(f".lisp-app:has(#t{slot} :checked[b*={bchar}]){{--tb{slot}_{bit_idx}:1}}")

    for slot in range(SOURCE_LEN):
        for bit_idx, bchar in enumerate(BIT6_LETTERS):
            css.append(f".lisp-app:has(#c{slot} :checked[b*={bchar}]){{--b{slot}_{bit_idx}:1}}")

    for pid, _title, src, _exp_val in PRESETS:
        ptids = tokenize_to_palette_ids(src, TOKEN_COMPOSER_LEN)
        p_decls = [f"--ptid{slot}:{ptid}" for slot, ptid in enumerate(ptids)]
        css.append(f".lisp-app:has(#pr-{pid}:checked){{{';'.join(p_decls)}}}")
        css.append(f".lisp-app:has(#pr-{pid}:checked) #bn-{pid}{{display:block}}")

    # Deterministic custom-property comment-tail dimming for #tw1..39 (avoids WebKit multi-:has() ~ sibling bug)
    dsemi_decls = [
        f"--dsemi{i}:calc((1 - var(--ovr{i}))*(1 - min(1,max(var(--ptid{i}) - 1,1 - var(--ptid{i})))) + var(--ucm{i}))"
        for i in range(TOKEN_COMPOSER_LEN - 1)
    ]
    css.append(f".lisp-app .tok-editor-wrap{{{';'.join(dsemi_decls)}}}")
    dsemig_decls = []
    for g in range((TOKEN_COMPOSER_LEN - 1 + 7) // 8):
        hi = min(8 * g + 8, TOKEN_COMPOSER_LEN - 1)
        dsemig_decls.append(f"--dsemig{g}:calc({' + '.join(f'var(--dsemi{j})' for j in range(8 * g, hi))})")
    css.append(f".lisp-app .tok-editor-grid{{{';'.join(dsemig_decls)}}}")
    for slot in range(1, TOKEN_COMPOSER_LEN):
        g, r = divmod(slot, 8)
        dim_terms = [f"var(--dsemig{bg})" for bg in range(g)] + [f"var(--dsemi{8 * g + br})" for br in range(r)]
        css.append(f"#tw{slot}{{opacity:calc(1 - .76*min(1,{' + '.join(dim_terms)}))}}")

    # -----------------------------------------------------------------------
    # D. Compiled Lexer, Compactor, Parser, & Reusable 5-Stage VM CSS Rules
    # -----------------------------------------------------------------------
    for cls_name, rules in spec.stage_defs.items():
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".{cls_name}{{{body}}}")

    # -----------------------------------------------------------------------
    # E. Dynamic Output, Live Source Preview, Error, AST, Env, & Trace Rules
    # -----------------------------------------------------------------------
    for i in range(TOKEN_COMPOSER_LEN):
        css.append(
            f".tkp{i}{{counter-reset:tpv var(--tvis{i});"
            f"font-size:calc(1em*clamp(0,var(--tvis{i}) - 1,1))}}"
        )
    for i in range(SOURCE_LEN):
        css.append(f".chp{i}{{counter-reset:cpv var(--ch{i})}}")

    css.append(
        "#lisp-empty-hint{color:#64748b;"
        "opacity:calc(1 - min(1,var(--ntok)));"
        "font-size:calc(.84rem*(1 - min(1,var(--ntok))))}"
    )
    css.append(".res-val i{counter-reset:ans var(--acc)}")
    css.append("#lisp-result{opacity:calc(1 - .65*min(1,var(--f_err)))}")
    css.append(".st-cnt i{counter-reset:stc var(--steps)}")
    css.append(".nt-cnt i{counter-reset:ntc var(--ntok)}")
    css.append(
        "#lisp-status-ok{counter-reset:errc var(--f_err);"
        "max-height:calc(4rem*(1 - min(1,var(--f_err))));"
        "opacity:calc(1 - min(1,var(--f_err)));"
        "border-width:calc(1px*(1 - min(1,var(--f_err))));"
        "padding-top:calc(.55rem*(1 - min(1,var(--f_err))));"
        "padding-bottom:calc(.55rem*(1 - min(1,var(--f_err))));overflow:hidden}"
    )
    css.append(
        "#lisp-status-err{counter-reset:errc var(--f_err);"
        "max-height:calc(5rem*min(1,var(--f_err)));"
        "opacity:min(1,var(--f_err));"
        "border-width:calc(1px*min(1,var(--f_err)))}"
    )

    for i in range(MAX_TOKENS):
        css.append(
            f".tc{i}{{counter-reset:tk var(--tk{i}) tv var(--tv{i}) tn var(--tn{i});"
            f"opacity:calc(.2 + .8*min(1,var(--tk{i})))}}"
        )
    for k in range(NUM_VARS):
        css.append(
            f".vc{k}{{counter-reset:vk var(--vk{k}) vv var(--vv{k});"
            f"opacity:calc(.22 + .78*min(1,var(--vk{k})))}}"
        )
    for f in range(NUM_FUNCS):
        css.append(
            f".fc{f}{{counter-reset:fk var(--fk{f}) fb var(--fb{f});"
            f"opacity:calc(.22 + .78*min(1,max(var(--fk{f}),0 - var(--fk{f}))))}}"
        )

    css.append(".tr-row{counter-reset:md var(--mode) pcv var(--pc) acv var(--acc) spv var(--sp)}")
    sampled_ticks = list(range(1, 21)) + list(range(24, NUM_VM_TICKS + 1, 4))
    for t in sampled_ticks:
        css.append(f".tr-t{t}{{opacity:calc(.24 + .76*clamp(0,var(--steps) - {t - 1},1))}}")

    # -----------------------------------------------------------------------
    # F. HTML Controls, Pipeline DOM Nesting, & Inspector Cards
    # -----------------------------------------------------------------------
    padded_char_src = default_program if len(default_program) >= SOURCE_LEN - 1 else f"{default_program} ;"
    default_codes = pad_source_codes(padded_char_src[:SOURCE_LEN], SOURCE_LEN)

    explicit_tids: List[int] | None = (
        encode_token_composer_slots(default_tokens, TOKEN_COMPOSER_LEN)
        if default_tokens is not None
        else None
    )

    def build_token_select(slot: int) -> str:
        use_def = explicit_tids is None or slot >= len(default_tokens or ())
        def_sel = " selected" if use_def else ""
        opts: List[str] = [f"<option hidden def{def_sel}></option>"]
        selected_emitted = use_def
        target_tid = -1 if explicit_tids is None else explicit_tids[slot]
        for disp_lbl, tid, is_cm in TOKEN_SELECT_OPTIONS:
            b_str = "".join(BIT8_LETTERS[b] for b in range(8) if (tid >> b) & 1)
            b_attr = f" b={b_str}" if b_str else ""
            cm_attr = " cm" if is_cm else ""
            is_sel = (not selected_emitted) and (tid == target_tid)
            if is_sel:
                selected_emitted = True
            sel_attr = " selected" if is_sel else ""
            opts.append(f"<option{b_attr}{cm_attr}{sel_attr}>{html.escape(disp_lbl)}")
        return (
            f'<div class=cw id=tw{slot}><span class=ci>T{slot:02d}</span>'
            f'<div class=tsw>'
            f'<select id=t{slot} class="cc tc-sel" aria-label="Token slot {slot}">'
            f'{"".join(opts)}</select>'
            f'<span class="ptok ptok{slot}"><i></i></span>'
            f'</div></div>'
        )

    def build_char_select(slot: int, default_code: int) -> str:
        opts: List[str] = []
        selected_emitted = False
        for disp_lbl, code, is_cm in SELECT_CHARS:
            b_str = "".join(BIT6_LETTERS[b] for b in range(6) if (code >> b) & 1)
            b_attr = f" b={b_str}" if b_str else ""
            cm_attr = " cm" if is_cm else ""
            is_sel = (code == default_code) and not selected_emitted
            if is_sel:
                selected_emitted = True
            sel_attr = " selected" if is_sel else ""
            opts.append(f"<option{b_attr}{cm_attr}{sel_attr}>{html.escape(disp_lbl)}")
        return (
            f'<div class=cw><span class=ci>{slot:02d}</span>'
            f'<select id=c{slot} class=cc aria-label="Source character {slot}">'
            f'{"".join(opts)}</select></div>'
        )

    tok_selects_html = "".join(build_token_select(i) for i in range(TOKEN_COMPOSER_LEN))
    char_selects_html = "".join(build_char_select(i, default_codes[i]) for i in range(SOURCE_LEN))

    tok_checked = "" if start_in_char_mode else " checked"
    chr_checked = " checked" if start_in_char_mode else ""
    default_res = eval_lisp_ref(default_program)
    default_disp = str(default_res.value) if default_res.err == 0 else "ERR"

    preset_cards_html: List[str] = [
        '<label class="pr-lbl edit-mode" for=pr-tok>'
        f'<span class=pr-top><input type=radio name=preset id=pr-tok{tok_checked}>'
        '<span>Blank Token Editor (;)</span></span>'
        '<span class=pr-code>; ; ; ... (blank — type from T00)</span></label>',
        '<label class="pr-lbl edit-mode" for=pr-edit>'
        f'<span class=pr-top><input type=radio name=preset id=pr-edit{chr_checked}>'
        '<span>Raw 64-Char Editor</span></span>'
        f'<span class=pr-code>{html.escape(default_program)} &rarr; {default_disp}</span></label>',
    ]
    preset_banners_html: List[str] = []
    for pid, title, src, exp_val in PRESETS:
        preset_cards_html.append(
            f'<label class=pr-lbl for=pr-{pid}>'
            f'<span class=pr-top><input type=radio name=preset id=pr-{pid}>'
            f'<span>{html.escape(title)}</span></span>'
            f'<span class=pr-code>{html.escape(src)} &rarr; {exp_val}</span></label>'
        )
        preset_banners_html.append(
            f'<div class=preset-banner id=bn-{pid}>'
            f'Preset written into Token Editor below: <strong>{html.escape(src)}</strong> '
            f'&mdash; Default Result: <strong>{exp_val}</strong> '
            f'(Edit any <strong>T00..T39</strong> cell live, or click <strong>&#x21BA; Clear Cell Edits</strong> to restore)</div>'
        )

    tok_prev_spans = "".join(f"<i class=tkp{i}></i>" for i in range(TOKEN_COMPOSER_LEN))
    chr_prev_spans = "".join(f"<i class=chp{i}></i>" for i in range(SOURCE_LEN))

    tok_cells_html = "".join(
        f'<span class="tok-cell tc{i}">'
        f'<span class=tok-top><span>#{i:02d}</span><i class=tk-k></i></span>'
        f'<i class=tk-v></i><i class=tk-n></i></span>'
        for i in range(MAX_TOKENS)
    )
    var_chips_html = "".join(
        f'<span class="slot-chip vc{k}"><i class=ev-k></i><i class=ev-v></i></span>'
        for k in range(NUM_VARS)
    )
    fn_chips_html = "".join(
        f'<span class="slot-chip fc{f}"><i class=fn-k></i><i class=fn-b></i></span>'
        for f in range(NUM_FUNCS)
    )

    output_card_html = (
        '<section class=output-panel aria-label="Lisp Evaluation Result">'
        '<div class=panel-head>'
        '<span class=panel-title style="color:#7dd3fc">Live Lisp Evaluation Result</span>'
        '<span class=panel-hint style="color:#38bdf8">Lexed, Parsed &amp; Evaluated 100% by CSS @property &amp; calc()</span>'
        '</div>'
        '<div class=src-echo id=lisp-active-src>'
        '<span class=src-echo-lbl>Active S-Expr:</span>'
        '<span id=lisp-empty-hint>; (blank — type in T00 or select a preset)</span>'
        f'<span class=src-tok-view>{tok_prev_spans}</span>'
        f'<span class=src-chr-view>{chr_prev_spans}</span>'
        '</div>'
        '<div class=res-grid>'
        '<div class=res-box><span class=res-lbl>Return Value (--acc)</span>'
        '<p class=res-val id=lisp-result><i></i></p></div>'
        '<div class=res-box><span class=res-lbl>VM Ticks (--steps)</span>'
        '<p class="stat-num st-cnt" id=lisp-steps><i></i></p></div>'
        '<div class=res-box><span class=res-lbl>Parsed Tokens (--ntok)</span>'
        '<p class="stat-num nt-cnt" id=lisp-ntok><i></i></p></div>'
        '</div>'
        '<div class=status-line id=lisp-status-ok><i></i></div>'
        '<div class=err-alert id=lisp-status-err><i></i></div>'
        '<div class=insp-grid>'
        '<div class=sub-card><div class=sub-title>Shallow-Binding Variable Table (--vk0..7 / --vv0..7)</div>'
        f'<div class=slot-list>{var_chips_html}</div></div>'
        '<div class=sub-card><div class=sub-title>Function Table (--fk0..3 / Body Token Index)</div>'
        f'<div class=slot-list>{fn_chips_html}</div></div>'
        '</div>'
        '<div class=tok-strip>'
        '<div class=sub-title>Compacted Token &amp; S-Expression Span Bus (--tk / --tv / next_expr --tn0..39)</div>'
        f'<div class=tok-grid>{tok_cells_html}</div>'
        '</div>'
        '</section>'
    )

    sampled_set = set(sampled_ticks)
    pipeline_open: List[str] = []
    depth_count = 0
    tick_counter = 0

    pipeline_open.append(
        '<section class=trace-head>'
        '<div class=panel-title style="margin-bottom:.25rem">Live CEK Virtual Machine Cascade Trace</div>'
        '<p style="color:#64748b;font-size:.8rem">'
        'Each row sits directly inside the nested <code>display: contents</code> CSS stage tree and reads '
        'inherited <code>--mode</code>, <code>--pc</code>, <code>--acc</code>, and <code>--sp</code> wires at its tick boundary.'
        '</p></section>'
    )

    for cls_name in spec.pipeline_classes:
        pipeline_open.append(f'<div class="st {cls_name}">')
        depth_count += 1
        if cls_name == "v4":
            tick_counter += 1
            if tick_counter in sampled_set:
                pipeline_open.append(
                    f'<article class="tr tr-row tr-t{tick_counter}" style="order:{10 + tick_counter}">'
                    f'<span class=trn>Tick {tick_counter:02d}</span>'
                    f'<i class="tr-c tr-m"></i>'
                    f'<i class="tr-c tr-pc"></i>'
                    f'<i class="tr-c tr-ac"></i>'
                    f'<i class="tr-c tr-sp"></i>'
                    f'</article>'
                )

    return (
        "<!DOCTYPE html>\n<html lang=en>\n<head>\n<meta charset=utf-8>\n"
        "<meta name=viewport content='width=device-width,initial-scale=1'>\n"
        "<title>NoJS Lisp — Pure HTML + Inline CSS S-Expression Interpreter</title>\n"
        f"<style>{''.join(css)}</style>\n</head>\n<body>\n<main class=lisp-app>\n"
        "<header class=hero>"
        "<nav class=back-row aria-label='Site navigation'>"
        "<a class=back-link href='../'>&larr; All NoJS Demos</a>"
        "</nav>"
        "<div class=badge-row>"
        "<span class='badge green'>0 Bytes JavaScript</span>"
        "<span class=badge>0 SVG Filters / 0 SMIL</span>"
        "<span class='badge purple'>8-Bit Token Composer (0-99 + Presets)</span>"
        "<span class=badge>Shallow-Binding Recursion + Higher-Order Fns</span>"
        "</div>"
        "<h1>Pure HTML + CSS Lisp Interpreter</h1>"
        "<p>Live S-expression lexing, parenthesis matching, lexical &amp; mutable environments, and recursive "
        "function evaluation executed strictly inside browser CSS style recalculation via "
        "<code>@property &lt;integer&gt;</code> wires, <code>calc()/clamp()/mod()</code> ALU gates, and "
        "reusable <code>display: contents</code> VM stages. Starts on a blank <code>;</code> Token Editor &mdash; "
        "type directly from <code>T00</code> or click any preset to load &amp; edit its tokens live.</p></header>\n"
        '<section class=panel aria-label="Lisp Program Input">'
        "<div class=panel-head>"
        "<span class=panel-title>1. Input Mode, Presets &amp; Keyboard S-Expression Composer</span>"
        "<span class=panel-hint>Clicking any preset writes it into the Token Editor; click the blue focus button to type with keys + <code>Tab</code></span>"
        "</div>"
        f"<div class=preset-grid>{''.join(preset_cards_html)}</div>"
        f"{''.join(preset_banners_html)}"
        "<form class=grid-form>"
        "<div class=kbd-bar>"
        "<div class=kbd-actions>"
        "<label for=t0 class='kbd-btn kbd-for-tok'>&#x2328; Focus Token T00 &amp; Start Typing</label>"
        "<label for=c0 class='kbd-btn kbd-for-chr'>&#x2328; Focus Char 00 &amp; Start Typing</label>"
        "<button type=reset class=rst-btn>&#x21BA; Clear Cell Edits</button>"
        "</div>"
        "<div class=kbd-tips>"
        "<span><kbd>Tab</kbd> / <kbd>Shift+Tab</kbd> move cell</span>"
        "<span>Space / <kbd>&#x2423;</kbd> / <kbd>.</kbd> blank</span>"
        "<span><kbd>;</kbd> comment to end</span>"
        "<span>All <kbd>0</kbd>&ndash;<kbd>99</kbd> selectable</span>"
        "<span>1-key: <kbd>d</kbd>=def <kbd>i</kbd>=if <kbd>l</kbd>=let <kbd>s</kbd>=set <kbd>m</kbd>=mod</span>"
        "</div></div>"
        f"<div class=tok-editor-wrap><div class=tok-editor-grid>{tok_selects_html}</div></div>"
        f"<div class=char-editor-wrap><div class=char-grid>{char_selects_html}</div></div>"
        "</form>"
        "<div class=ref-bar>"
        "<span>Supported forms:</span>"
        "<code>(+ - * / mod % = &lt; &gt; &lt;= &gt;= not)</code>"
        "<code>(if c t e)</code>"
        "<code>(begin ...)</code>"
        "<code>(set x v)</code>"
        "<code>(let ((x a) (y b)) body)</code>"
        "<code>(def f (x y) body)</code>"
        "<code>(fn (x) body)</code>"
        "</div></section>\n"
        f"<section class=cg>{''.join(pipeline_open)}{output_card_html}{'</div>' * depth_count}</section>\n"
        "</main>\n</body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone zero-JS HTML+CSS Lisp interpreter.")
    parser.add_argument(
        "-p", "--program",
        default=None,
        help="Optional Lisp program string (<= 64 chars) to pre-load into the 64-char editor on startup.",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"),
        help="Output HTML file path (defaults to lisp/index.html).",
    )
    args = parser.parse_args()

    if args.program is not None:
        content = generate_html(default_program=args.program, start_in_char_mode=True)
    else:
        content = generate_html()

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {args.output} ({len(content) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
