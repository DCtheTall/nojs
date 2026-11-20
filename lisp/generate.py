#!/usr/bin/env python3
"""
Static Pure HTML + Inline CSS Generator for the NoJS Lisp Interpreter (`lisp/index.html`).

Generates a standalone HTML5 document with inline CSS, zero JavaScript, and zero SVG
that lexes, parses, and evaluates live Lisp S-expressions inside the browser's CSS
style-recalculation engine using:
  - 64-character `<select>` keyboard source strip + 8 one-click program presets
  - Parallel 64-char SIMD lexer (`lx0..lx4`) & packed LSB binary compactor (`cp0..cpu`)
  - Prefix-depth S-expression `next_expr` parser (`ps0..ps4`) with syntax checking
  - 92-tick reusable 5-stage shallow-binding CEK VM (`v0..v4` + `vfin`) supporting
    signed arithmetic (`+`, `-`, `*`, `/`, `mod`), comparisons, `if`, `begin`,
    mutable `set`/`setq`, lexical `let`, recursive `def`/`defun`, and `fn`/`lambda`.
"""

from __future__ import annotations

import html
import os
import re
from typing import List

from css_vm import NUM_FUNCS, NUM_VARS, NUM_VM_TICKS, build_lisp_css_spec
from lisp_ref import (
    CHAR_TO_CODE,
    DEFAULT_PROGRAM,
    MAX_TOKENS,
    PRESETS,
    SELECT_CHARS,
    SOURCE_LEN,
    eval_lisp_ref,
    pad_source_codes,
)


BIT6_LETTERS = "abcdef"  # bit 0 (1) -> 'a', ..., bit 5 (32) -> 'f'


def minify_ui_css(raw: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([{};:,>])\s*", r"\1", s)
    return s.replace(";}", "}").strip()


def generate_html() -> str:
    spec = build_lisp_css_spec(num_ticks=NUM_VM_TICKS)
    css: List[str] = []

    # -----------------------------------------------------------------------
    # A. @counter-style definitions for symbolic CSS readout (0 JS!)
    # -----------------------------------------------------------------------
    css.append(
        '@counter-style tok-kind{system:fixed 0;symbols:"·" "(" ")" "NUM" "SYM"}'
    )
    css.append(
        '@counter-style vm-mode{system:fixed 0;symbols:"EVAL" "CONT" "DONE" "ERR" "DIV"}'
    )
    css.append(
        '@counter-style err-msg{system:fixed 0;symbols:'
        '"OK — Evaluated Cleanly" '
        '"Syntax Error: Unbalanced parentheses or >40 tokens" '
        '"Runtime Error: Unbound symbol" '
        '"Runtime Error: Division or modulo by zero" '
        '"Runtime Error: Invalid call, arity mismatch, or table overflow" '
        f'"Runtime Error: Exceeded {NUM_VM_TICKS}-tick CSS cascade step budget"}}'
    )

    # Register all computed <integer> custom properties
    for prop in spec.registered_props:
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
.lisp-app{max-width:1040px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}
header.hero{
  border:1px solid #1e293b;background:linear-gradient(180deg,#111827,#0f172a);
  border-radius:14px;padding:1.45rem 1.7rem;box-shadow:0 12px 32px rgba(0,0,0,.35)
}
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

.preset-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.55rem;margin-bottom:1rem}
@media (max-width:780px){.preset-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){.preset-grid{grid-template-columns:1fr}}
.pr-lbl{
  display:flex;flex-direction:column;gap:.18rem;padding:.6rem .8rem;border-radius:9px;
  background:#090d16;border:1px solid #263349;cursor:pointer;user-select:none;transition:all .12s
}
.pr-lbl:hover{border-color:#475569;background:#0f172a}
.pr-top{display:flex;align-items:center;gap:.45rem;font-size:.82rem;font-weight:600;color:#e2e8f0}
.pr-top input{accent-color:#38bdf8;cursor:pointer}
.pr-code{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.71rem;color:#64748b;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}
.pr-lbl:has(input:checked){border-color:#38bdf8;background:#0c2136;box-shadow:0 0 0 1px rgba(56,189,248,.25)}
.pr-lbl:has(input:checked) .pr-code{color:#7dd3fc}

.preset-banner{
  display:none;margin-bottom:.85rem;padding:.65rem .9rem;border-radius:8px;
  background:#0c1e33;border:1px solid #0369a1;color:#bae6fd;font-size:.82rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace
}

.char-grid{
  display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:4px;
  background:#090d16;padding:8px;border-radius:10px;border:1px solid #263349
}
@media (max-width:720px){.char-grid{grid-template-columns:repeat(8,minmax(0,1fr))}}
.cw{display:flex;flex-direction:column;align-items:center;gap:2px}
.ci{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.58rem;color:#475569}
select.cc{
  width:100%;height:2.15rem;appearance:none;-webkit-appearance:none;
  background:#162032;color:#f8fafc;border:1px solid #334155;border-radius:5px;
  text-align:center;text-align-last:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.98rem;font-weight:600;
  cursor:pointer;transition:border-color .12s,background .12s
}
select.cc:hover{border-color:#475569;background:#1e293b}
select.cc:focus{outline:none;border-color:#38bdf8;background:#0f172a;box-shadow:0 0 0 2px rgba(56,189,248,.28)}

.ref-bar{
  margin-top:.75rem;display:flex;flex-wrap:wrap;gap:.45rem .9rem;font-size:.77rem;color:#94a3b8;
  background:#090d16;padding:.55rem .8rem;border-radius:8px;border:1px solid #1e293b
}
.ref-bar code{color:#7dd3fc;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

.cg{display:flex;flex-direction:column;gap:1.35rem}
.cg div.st{display:contents}

.output-panel{
  order:-3;border:1px solid #0284c7;background:linear-gradient(180deg,#0c192c,#0b1322);
  border-radius:14px;padding:1.4rem 1.6rem;box-shadow:0 0 36px rgba(14,165,233,.14)
}
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
.res-val i,.stat-num i,.tk-k,.tk-v,.tk-n,.ev-k,.ev-v,.tr-c{font-style:normal;contain:style}
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
    # C. Input Decoding Rules (:root defaults, 384 :has() rules, & 8 Presets)
    # -----------------------------------------------------------------------
    b_defaults = ";".join(f"--b{i}_{b}:0" for i in range(SOURCE_LEN) for b in range(6))
    css.append(f":root{{{b_defaults}}}")

    for slot in range(SOURCE_LEN):
        for bit_idx, bchar in enumerate(BIT6_LETTERS):
            css.append(f".lisp-app:has(#c{slot} :checked[b*={bchar}]){{--b{slot}_{bit_idx}:1}}")

    for pid, title, src, exp_val in PRESETS:
        codes = pad_source_codes(src, SOURCE_LEN)
        p_decls: List[str] = []
        for slot, code in enumerate(codes):
            for bit_idx in range(6):
                p_decls.append(f"--b{slot}_{bit_idx}:{(code >> bit_idx) & 1}")
        css.append(f".lisp-app.lisp-app:has(#pr-{pid}:checked){{{';'.join(p_decls)}}}")
        css.append(f".lisp-app:has(#pr-{pid}:checked) #bn-{pid}{{display:block}}")

    css.append(
        ".lisp-app:has(input[name=preset]:not(#pr-edit):checked) .char-grid"
        "{opacity:.42;pointer-events:none}"
    )

    # -----------------------------------------------------------------------
    # D. Compiled Lexer, Compactor, Parser, & Reusable 5-Stage VM CSS Rules
    # -----------------------------------------------------------------------
    for cls_name, rules in spec.stage_defs.items():
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".{cls_name}{{{body}}}")

    # -----------------------------------------------------------------------
    # E. Dynamic Output, Error, AST Token, Env, & Trace Counter Bindings
    # -----------------------------------------------------------------------
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
    # Trace ticks sampled cleanly: ticks 1..20 every tick, plus every 4th tick up to NUM_VM_TICKS
    sampled_ticks = list(range(1, 21)) + list(range(24, NUM_VM_TICKS + 1, 4))
    for t in sampled_ticks:
        css.append(f".tr-t{t}{{opacity:calc(.24 + .76*clamp(0,var(--steps) - {t - 1},1))}}")

    # -----------------------------------------------------------------------
    # F. HTML Controls, Pipeline DOM Nesting, & Inspector Cards
    # -----------------------------------------------------------------------
    default_codes = pad_source_codes(DEFAULT_PROGRAM, SOURCE_LEN)

    def build_char_select(slot: int, default_code: int) -> str:
        opts: List[str] = []
        for ch in SELECT_CHARS:
            code = CHAR_TO_CODE[ch]
            b_str = "".join(BIT6_LETTERS[b] for b in range(6) if (code >> b) & 1)
            b_attr = f" b={b_str}" if b_str else ""
            sel_attr = " selected" if code == default_code else ""
            disp = "␣" if ch == " " else html.escape(ch)
            opts.append(f"<option{b_attr}{sel_attr}>{disp}")
        return (
            f'<div class=cw><span class=ci>{slot:02d}</span>'
            f'<select id=c{slot} class=cc aria-label="Source character {slot}">'
            f'{"".join(opts)}</select></div>'
        )

    selects_html = "".join(build_char_select(i, default_codes[i]) for i in range(SOURCE_LEN))

    preset_cards_html: List[str] = [
        '<label class=pr-lbl for=pr-edit>'
        '<span class=pr-top><input type=radio name=preset id=pr-edit checked>'
        '<span>Custom 64-Char Editor</span></span>'
        f'<span class=pr-code>{html.escape(DEFAULT_PROGRAM)} &rarr; 84</span></label>'
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
            f'Preset Active: <strong>{html.escape(src)}</strong> &mdash; Expected Result: <strong>{exp_val}</strong> '
            f'(Select &ldquo;Custom 64-Char Editor&rdquo; above to edit individual characters)</div>'
        )

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

    # Build nested `display: contents` compute pipeline + interleaved VM trace rows
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

    default_val = eval_lisp_ref(DEFAULT_PROGRAM).value

    return (
        "<!DOCTYPE html>\n<html lang=en>\n<head>\n<meta charset=utf-8>\n"
        "<meta name=viewport content='width=device-width,initial-scale=1'>\n"
        "<title>NoJS Lisp — Pure HTML + Inline CSS S-Expression Interpreter</title>\n"
        f"<style>{''.join(css)}</style>\n</head>\n<body>\n<main class=lisp-app>\n"
        "<header class=hero>"
        "<div class=badge-row>"
        "<span class='badge green'>0 Bytes JavaScript</span>"
        "<span class=badge>0 SVG Filters / 0 SMIL</span>"
        "<span class='badge purple'>Parallel SIMD Lexer + S-Expr Parser</span>"
        "<span class=badge>Shallow-Binding Recursion + Higher-Order Fns</span>"
        "</div>"
        "<h1>Pure HTML + CSS Lisp Interpreter</h1>"
        "<p>Live S-expression lexing, parenthesis matching, lexical &amp; mutable environments, and recursive "
        "function evaluation executed strictly inside browser CSS style recalculation via "
        "<code>@property &lt;integer&gt;</code> wires, <code>calc()/clamp()/mod()</code> ALU gates, and "
        f"reusable <code>display: contents</code> VM stages. Default program (<code>{html.escape(DEFAULT_PROGRAM)}</code>): "
        f"<code>{default_val}</code>.</p></header>\n"
        '<section class=panel aria-label="Lisp Program Input">'
        "<div class=panel-head>"
        "<span class=panel-title>1. Program Presets &amp; 64-Character Live Source Buffer</span>"
        "<span class=panel-hint>Pick a preset or select Custom Editor &amp; type keys in cells (<code>Tab</code> advances; <code>&#x2423;</code> = space)</span>"
        "</div>"
        f"<div class=preset-grid>{''.join(preset_cards_html)}</div>"
        f"{''.join(preset_banners_html)}"
        f"<div class=char-grid>{selects_html}</div>"
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
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "index.html")
    content = generate_html()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {out_path} ({len(content) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
