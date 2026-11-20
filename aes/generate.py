#!/usr/bin/env python3
"""
Static Pure HTML + Inline CSS Generator for NIST AES-128 (`aes/index.html`).

Generates a single standalone HTML file with inline CSS, zero JavaScript, and
zero SVG that executes full 10-round AES-128 + KeyExpansion in the browser's
CSS cascade engine.
"""

from __future__ import annotations

import html
import os
from typing import List

from aes_cipher import (
    RCON,
    aes128_encrypt_ref,
    build_aes_final_ark,
    build_aes_mixcolumns_ark,
    build_aes_shared_subbytes_sr_keysched,
)


DEFAULT_PLAINTEXT = b"Hello, CSS AES!!"  # 16 bytes
NIST_PT_HEX = "3243f6a8885a308d313198a2e0370734"
NIST_KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
NIST_CT_HEX = "3925841d02dc09fbdc118597196a0b32"


def render_hex_spans(var_prefix: str, num_bytes: int = 16, group_bytes: bool = True) -> str:
    """Render 32 hex nibble spans reading `{var_prefix}0..{var_prefix}127`."""
    parts: List[str] = []
    for byte_idx in range(num_bytes):
        hi_idx = 2 * byte_idx
        lo_idx = 2 * byte_idx + 1
        cls_p = "pk" if var_prefix == "--k" else ("pp" if var_prefix == "--p" else "ps")
        b_html = (
            f'<span class="hb"><span class="hn {cls_p}{hi_idx}"></span>'
            f'<span class="hn {cls_p}{lo_idx}"></span></span>'
        )
        parts.append(b_html)
    sep = " " if group_bytes else ""
    return sep.join(parts)


def generate_html() -> str:
    # 1. Compile reusable AES circuit macros
    core_circ, core_out = build_aes_shared_subbytes_sr_keysched()
    core_la = core_circ.compile_to_css_layers(core_out, max_inline_depth=2, temp_prefix="--w")

    mc_circ, mc_out = build_aes_mixcolumns_ark()
    mc_la = mc_circ.compile_to_css_layers(mc_out, max_inline_depth=2, temp_prefix="--w")

    fin_circ, fin_out = build_aes_final_ark()
    fin_la = fin_circ.compile_to_css_layers(fin_out, max_inline_depth=2, temp_prefix="--w")

    max_temp_regs = max(core_la.num_temp_regs, mc_la.num_temp_regs, fin_la.num_temp_regs)

    css: List[str] = []

    # -----------------------------------------------------------------------
    # A. @counter-style & @property registrations
    # -----------------------------------------------------------------------
    css.append(
        '@counter-style hex-digit{system:fixed 0;'
        'symbols:"0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f";}'
    )

    def reg_prop(name: str) -> str:
        return f'@property {name}{{syntax:"<integer>";initial-value:0;inherits:true;}}'

    for i in range(128):
        css.append(reg_prop(f"--p{i}"))
        css.append(reg_prop(f"--k{i}"))
        css.append(reg_prop(f"--s{i}"))
    for i in range(8):
        css.append(reg_prop(f"--rc{i}"))
    for i in range(max_temp_regs):
        css.append(reg_prop(f"--w{i}"))

    # -----------------------------------------------------------------------
    # B. Page Styling (Modern Hardware / Cryptographic Lab UI, No SVG/JS)
    # -----------------------------------------------------------------------
    css.append(
        """
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0b0f17;
  color:#e2e8f0;
  line-height:1.5;
  padding:2rem 1rem 4rem;
}
.aes-app{
  max-width:980px;
  margin:0 auto;
  display:flex;
  flex-direction:column;
  gap:1.5rem;
}
header.hero{
  border:1px solid #1e293b;
  background:linear-gradient(180deg,#111827 0%,#0f172a 100%);
  border-radius:14px;
  padding:1.5rem 1.75rem;
  box-shadow:0 12px 32px rgba(0,0,0,.35);
}
.badge-row{
  display:flex;
  flex-wrap:wrap;
  gap:.5rem;
  margin-bottom:.65rem;
}
.badge{
  font-size:.72rem;
  font-weight:700;
  letter-spacing:.06em;
  text-transform:uppercase;
  padding:.2rem .55rem;
  border-radius:999px;
  background:#1e293b;
  color:#38bdf8;
  border:1px solid #334155;
}
.badge.green{color:#4ade80;border-color:#14532d;background:#052e16}
header.hero h1{
  font-size:1.65rem;
  font-weight:700;
  color:#f8fafc;
  letter-spacing:-.02em;
}
header.hero p{
  margin-top:.4rem;
  color:#94a3b8;
  font-size:.93rem;
}
.panel{
  border:1px solid #1e293b;
  background:#111827;
  border-radius:14px;
  padding:1.35rem 1.5rem;
  box-shadow:0 8px 24px rgba(0,0,0,.25);
}
.panel-head{
  display:flex;
  justify-content:space-between;
  align-items:baseline;
  flex-wrap:wrap;
  gap:.5rem;
  margin-bottom:.9rem;
}
.panel-title{
  font-size:.82rem;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:.08em;
  color:#94a3b8;
}
.panel-hint{
  font-size:.8rem;
  color:#64748b;
}
.char-strip{
  display:grid;
  grid-template-columns:repeat(16,minmax(0,1fr));
  gap:4px;
  background:#090d16;
  padding:8px;
  border-radius:10px;
  border:1px solid #263349;
}
@media (max-width:720px){
  .char-strip{grid-template-columns:repeat(8,minmax(0,1fr));}
}
.cell-wrap{
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:3px;
}
.cell-idx{
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:.63rem;
  color:#475569;
}
select.char-cell{
  width:100%;
  height:2.4rem;
  appearance:none;
  -webkit-appearance:none;
  background:#162032;
  color:#f8fafc;
  border:1px solid #334155;
  border-radius:6px;
  text-align:center;
  text-align-last:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:1.05rem;
  font-weight:600;
  cursor:pointer;
  transition:border-color .12s,background .12s,box-shadow .12s;
}
select.char-cell:hover{border-color:#475569;background:#1e293b}
select.char-cell:focus{
  outline:none;
  border-color:#38bdf8;
  background:#0f172a;
  box-shadow:0 0 0 2px rgba(56,189,248,.28);
}
.hex-echo{
  margin-top:.75rem;
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:.5rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:.83rem;
  color:#94a3b8;
  background:#090d16;
  padding:.55rem .8rem;
  border-radius:8px;
  border:1px solid #1e293b;
}
.hex-echo .lbl{
  color:#64748b;
  font-size:.73rem;
  text-transform:uppercase;
  letter-spacing:.06em;
  margin-right:.25rem;
}
.key-controls{
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  justify-content:space-between;
  gap:1rem;
}
.dip-bank{
  display:flex;
  gap:.45rem;
  flex-wrap:wrap;
}
.dip{
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:.3rem;
  cursor:pointer;
  user-select:none;
}
.dip input{
  position:absolute;
  opacity:0;
  width:1px;
  height:1px;
}
.dip-box{
  width:2.35rem;
  height:2.75rem;
  border-radius:7px;
  background:#090d16;
  border:1px solid #334155;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:space-between;
  padding:.3rem 0;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  transition:all .14s ease;
}
.dip-led{
  width:8px;
  height:8px;
  border-radius:50%;
  background:#1e293b;
}
.dip-bitval::after{
  content:"0";
  font-size:.85rem;
  font-weight:700;
  color:#64748b;
}
.dip input:checked + .dip-box{
  border-color:#38bdf8;
  background:#0c2438;
  box-shadow:0 0 12px rgba(56,189,248,.2);
}
.dip input:checked + .dip-box .dip-led{
  background:#38bdf8;
  box-shadow:0 0 8px #38bdf8;
}
.dip input:checked + .dip-box .dip-bitval::after{
  content:"1";
  color:#f8fafc;
}
.dip input:focus-visible + .dip-box{
  outline:2px solid #38bdf8;
  outline-offset:2px;
}
.dip-lbl{
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:.66rem;
  color:#64748b;
}
.nist-toggle{
  display:inline-flex;
  align-items:center;
  gap:.6rem;
  padding:.6rem .95rem;
  border-radius:9px;
  background:#090d16;
  border:1px solid #334155;
  cursor:pointer;
  user-select:none;
  font-size:.84rem;
  color:#cbd5e1;
  transition:border-color .15s,background .15s;
}
.nist-toggle:hover{border-color:#475569}
.nist-toggle input{accent-color:#f59e0b;width:1rem;height:1rem;cursor:pointer}
.nist-banner{
  display:none;
  margin-top:.75rem;
  padding:.65rem .9rem;
  border-radius:8px;
  background:#271c09;
  border:1px solid #92400e;
  color:#fde68a;
  font-size:.82rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
}
.aes-app:has(#nist-vec:checked) .nist-banner{display:block}
.aes-app:has(#nist-vec:checked) .char-strip,
.aes-app:has(#nist-vec:checked) .dip-bank{
  opacity:.4;
  pointer-events:none;
}

/* Circuit Pipeline Layout: compute layers use display:contents so zero layout boxes are created */
.circuit-grid{
  display:flex;
  flex-direction:column;
  gap:1.5rem;
}
.cl{display:contents}

.output-panel{
  order:-2;
  border:1px solid #0284c7;
  background:linear-gradient(180deg,#0c192c 0%,#0b1322 100%);
  border-radius:14px;
  padding:1.5rem 1.65rem;
  box-shadow:0 0 36px rgba(14,165,233,.14);
}
.ct-display{
  margin-top:.55rem;
  padding:1rem 1.15rem;
  background:#060a12;
  border:1px solid #1e3a5f;
  border-radius:10px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:1.25rem;
  font-weight:700;
  letter-spacing:.04em;
  color:#38bdf8;
  display:flex;
  flex-wrap:wrap;
  gap:.45rem .6rem;
}
.hb{display:inline-flex}
.hn{contain:style}
.hn::after{content:counter(h,hex-digit)}

.matrix-and-leds{
  margin-top:1.15rem;
  display:grid;
  grid-template-columns:auto 1fr;
  gap:1.25rem;
  align-items:start;
}
@media (max-width:680px){
  .matrix-and-leds{grid-template-columns:1fr;}
}
.sub-lbl{
  font-size:.72rem;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:.07em;
  color:#64748b;
  margin-bottom:.45rem;
}
.state-matrix{
  display:grid;
  grid-template-columns:repeat(4,3.1rem);
  gap:5px;
  background:#060a12;
  padding:8px;
  border-radius:9px;
  border:1px solid #1e293b;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:.92rem;
  font-weight:600;
  color:#e2e8f0;
}
.mat-cell{
  background:#111c2d;
  border:1px solid #24354d;
  border-radius:5px;
  padding:.35rem 0;
  text-align:center;
}
.led-grid{
  display:grid;
  grid-template-columns:repeat(16,minmax(0,1fr));
  gap:4px;
  background:#060a12;
  padding:10px;
  border-radius:9px;
  border:1px solid #1e293b;
}
.led{
  aspect-ratio:1;
  border-radius:3px;
  background:#38bdf8;
  border:1px solid rgba(255,255,255,.08);
}

.trace-section{
  order:1;
  border:1px solid #1e293b;
  background:#111827;
  border-radius:14px;
  padding:1.1rem 1.4rem;
}
.trace-title{
  font-size:.82rem;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:.08em;
  color:#94a3b8;
  margin-bottom:.75rem;
}
.trace-row{
  order:2;
  display:grid;
  grid-template-columns:5.5rem 1fr 1fr;
  gap:.75rem;
  align-items:center;
  padding:.5rem .85rem;
  background:#0c121e;
  border:1px solid #1e293b;
  border-radius:8px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  font-size:.78rem;
}
@media (max-width:820px){
  .trace-row{grid-template-columns:1fr;gap:.3rem}
}
.trace-rname{
  font-weight:700;
  color:#38bdf8;
}
.trace-kv{
  color:#cbd5e1;
  display:flex;
  flex-wrap:wrap;
  gap:.25rem;
}
.trace-kv .tag{
  color:#64748b;
  margin-right:.25rem;
}
"""
    )

    # -----------------------------------------------------------------------
    # C. Input Decoding Rules (:has() bit-class decoder + toy key + NIST preset)
    # -----------------------------------------------------------------------
    # Default plaintext & key bits are 0 via @property initial-value: 0.
    # 128 :has() rules decode all 16 <select> character slots!
    bit_classes = ["b7", "b6", "b5", "b4", "b3", "b2", "b1", "b0"]
    for slot in range(16):
        for bit_pos, bcls in enumerate(bit_classes):
            p_idx = 8 * slot + bit_pos
            css.append(f".aes-app:has(#c{slot} option.{bcls}:checked){{--p{p_idx}:1;}}")

    # 8-bit toy key switches (#kbit7 .. #kbit0 -> last byte of 128-bit key --k120..--k127)
    for bit_pos in range(8):
        bit_num = 7 - bit_pos  # kbit7 is MSB (128), kbit0 is LSB (1)
        k_idx = 120 + bit_pos
        css.append(f".aes-app:has(#kbit{bit_num}:checked){{--k{k_idx}:1;}}")

    # NIST FIPS-197 Appendix B override (#nist-vec:checked)
    nist_pt_bytes = bytes.fromhex(NIST_PT_HEX)
    nist_key_bytes = bytes.fromhex(NIST_KEY_HEX)
    nist_decls: List[str] = []
    for byte_idx in range(16):
        pb = nist_pt_bytes[byte_idx]
        kb = nist_key_bytes[byte_idx]
        for bit_pos in range(8):
            idx = 8 * byte_idx + bit_pos
            p_val = (pb >> (7 - bit_pos)) & 1
            k_val = (kb >> (7 - bit_pos)) & 1
            nist_decls.append(f"--p{idx}:{p_val}")
            nist_decls.append(f"--k{idx}:{k_val}")
    css.append(f".aes-app.aes-app.aes-app:has(#nist-vec:checked){{{';'.join(nist_decls)};}}")

    # -----------------------------------------------------------------------
    # D. Round 0 (Initial AddRoundKey) + Rcon[1..10] + Shared Round Circuit CSS
    # -----------------------------------------------------------------------
    r0_decls = [
        f"--s{i}:calc(var(--p{i}) + var(--k{i}) - 2 * var(--p{i}) * var(--k{i}))"
        for i in range(128)
    ]
    css.append(f".r0{{{';'.join(r0_decls)};}}")

    for r in range(1, 11):
        rc_val = RCON[r]
        rc_decls = [f"--rc{b}:{(rc_val >> (7 - b)) & 1}" for b in range(8)]
        css.append(f".rk{r}{{{';'.join(rc_decls)};}}")

    # Shared Core layers (.sc0 .. .sc{N-1}): KeySchedule + 16x SubBytes + ShiftRows
    for l_idx, rules in enumerate(core_la.layer_rules):
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".sc{l_idx}{{{body};}}")

    # Shared MixColumns + AddRoundKey layers (.mc0 .. .mc{M-1}) for Rounds 1..9
    for l_idx, rules in enumerate(mc_la.layer_rules):
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".mc{l_idx}{{{body};}}")

    # Final Round AddRoundKey layers (.fn0 .. .fn{F-1}) for Round 10
    for l_idx, rules in enumerate(fin_la.layer_rules):
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".fn{l_idx}{{{body};}}")

    # -----------------------------------------------------------------------
    # E. Hex Nibble Counter Rules & 128-Bit LED Opacity Rules
    # -----------------------------------------------------------------------
    for nib_idx in range(32):
        b0 = 4 * nib_idx
        for prefix, cls_p in (("--s", "ps"), ("--k", "pk"), ("--p", "pp")):
            expr = (
                f"calc(8 * var({prefix}{b0}) + 4 * var({prefix}{b0 + 1}) "
                f"+ 2 * var({prefix}{b0 + 2}) + var({prefix}{b0 + 3}))"
            )
            css.append(f".{cls_p}{nib_idx}{{counter-reset:h {expr};}}")

    for bit_idx in range(128):
        css.append(f".led{bit_idx}{{opacity:calc(0.12 + 0.88 * var(--s{bit_idx}));}}")

    # -----------------------------------------------------------------------
    # F. Build HTML Controls & Circuit DOM Tree
    # -----------------------------------------------------------------------
    # Precompute <option> list template per slot
    def build_select_for_slot(slot: int, default_byte: int) -> str:
        opts: List[str] = []
        # 0x00 Null byte option + printable ASCII 0x20..0x7E
        byte_choices = [0] + list(range(32, 127))
        for bval in byte_choices:
            classes = [f"b{bit}" for bit in range(7, -1, -1) if (bval >> bit) & 1]
            cls_attr = f' class="{" ".join(classes)}"' if classes else ""
            sel_attr = " selected" if bval == default_byte else ""
            if bval == 0:
                label = "·"
            elif bval == 32:
                label = "␣"
            else:
                label = html.escape(chr(bval))
            opts.append(f"<option{cls_attr}{sel_attr}>{label}</option>")
        return (
            f'<div class="cell-wrap">'
            f'<span class="cell-idx">{slot:02d}</span>'
            f'<select id="c{slot}" class="char-cell" aria-label="Plaintext byte {slot}">'
            f'{"".join(opts)}'
            f"</select></div>"
        )

    selects_html = "".join(
        build_select_for_slot(i, DEFAULT_PLAINTEXT[i]) for i in range(16)
    )

    dip_switches_html = "".join(
        f'<label class="dip" for="kbit{bit}">'
        f'<input type="checkbox" id="kbit{bit}">'
        f'<span class="dip-box"><span class="dip-led"></span><span class="dip-bitval"></span></span>'
        f'<span class="dip-lbl">2^{bit}</span>'
        f"</label>"
        for bit in range(7, -1, -1)
    )

    # Build 4x4 column-major AES state matrix cells (row r, col c -> byte 4*c + r)
    matrix_cells_html: List[str] = []
    for row in range(4):
        for col in range(4):
            b_idx = 4 * col + row
            hi = 2 * b_idx
            lo = 2 * b_idx + 1
            matrix_cells_html.append(
                f'<div class="mat-cell"><span class="hn ps{hi}"></span><span class="hn ps{lo}"></span></div>'
            )

    leds_html = "".join(f'<div class="led led{i}" title="Bit {i}"></div>' for i in range(128))

    def make_trace_row(r_num: int, label: str) -> str:
        return (
            f'<div class="trace-row" style="order:{10 + r_num}">'
            f'<span class="trace-rname">{label}</span>'
            f'<span class="trace-kv"><span class="tag">State:</span>{render_hex_spans("--s")}</span>'
            f'<span class="trace-kv"><span class="tag">RoundKey:</span>{render_hex_spans("--k")}</span>'
            f"</div>"
        )

    # Construct nested circuit pipeline inside .circuit-grid
    pipeline_open: List[str] = []
    pipeline_close: List[str] = []

    # Round 0
    pipeline_open.append('<div class="cl r0">')
    pipeline_open.append(
        '<div class="trace-section" style="order:5">'
        '<div class="trace-title">Live 10-Round Internal State &amp; Key-Schedule Trace (Pure CSS Cascade)</div>'
        '<div style="color:#64748b;font-size:.82rem">'
        "Every row below reads the inherited <code>--s0..127</code> and <code>--k0..127</code> integer wires directly at that DOM depth."
        "</div></div>"
    )
    pipeline_open.append(make_trace_row(0, "Round 0"))
    pipeline_close.append("</div>")

    # Rounds 1..10
    for r in range(1, 11):
        pipeline_open.append(f'<div class="cl rk{r}">')
        pipeline_close.append("</div>")
        for l_idx in range(core_la.num_layers):
            pipeline_open.append(f'<div class="cl sc{l_idx}">')
            pipeline_close.append("</div>")
        tail_la = mc_la if r < 10 else fin_la
        tail_cls = "mc" if r < 10 else "fn"
        for l_idx in range(tail_la.num_layers):
            pipeline_open.append(f'<div class="cl {tail_cls}{l_idx}">')
            pipeline_close.append("</div>")

        pipeline_open.append(make_trace_row(r, f"Round {r}"))

    # Innermost payload at the end of Round 10: Primary Ciphertext Output Card
    output_card_html = f"""
      <section class="output-panel" aria-label="AES-128 Ciphertext Output">
        <div class="panel-head">
          <span class="panel-title" style="color:#7dd3fc">AES-128 Ciphertext Output (Hex)</span>
          <span class="panel-hint" style="color:#38bdf8">Computed 100% by CSS @property &amp; calc()</span>
        </div>
        <div class="ct-display" id="ciphertext-hex">
          {render_hex_spans("--s")}
        </div>
        <div class="matrix-and-leds">
          <div>
            <div class="sub-lbl">4&times;4 State Matrix (Col-Major)</div>
            <div class="state-matrix">
              {"".join(matrix_cells_html)}
            </div>
          </div>
          <div>
            <div class="sub-lbl">128-Bit Ciphertext State Register (--s0 .. --s127)</div>
            <div class="led-grid">
              {leds_html}
            </div>
          </div>
        </div>
      </section>
    """

    default_ct_hex = aes128_encrypt_ref(DEFAULT_PLAINTEXT, bytes(16)).hex()

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NoJS AES-128 — Pure HTML + Inline CSS Block Cipher</title>
<style>
{"".join(css)}
</style>
</head>
<body>
<main class="aes-app">
  <header class="hero">
    <div class="badge-row">
      <span class="badge green">0 Bytes JavaScript</span>
      <span class="badge">0 SVG Filters</span>
      <span class="badge">NIST FIPS-197 AES-128</span>
      <span class="badge">Boyar–Peralta 113-Gate S-Box</span>
    </div>
    <h1>Pure HTML + CSS AES-128 Fixed-Circuit Cipher</h1>
    <p>
      Full 10-round 128-bit AES encryption &amp; key expansion executed strictly inside CSS style recalculation using
      <code>@property &lt;integer&gt;</code> wires, arithmetic boolean gates, <code>:has()</code> bit decoders, and
      <code>@counter-style</code> hexadecimal rendering. Default load cipher (<code>"Hello, CSS AES!!"</code>, key <code>0x00</code>):
      <code>{default_ct_hex}</code>.
    </p>
  </header>

  <section class="panel" aria-label="Plaintext Input">
    <div class="panel-head">
      <span class="panel-title">1. Plaintext Block (16 ASCII Bytes / 128 Bits)</span>
      <span class="panel-hint">Click a cell &amp; type on your keyboard (Tab advances; <code>&middot;</code> = 0x00 null, <code>&#x2423;</code> = space)</span>
    </div>
    <div class="char-strip">
      {selects_html}
    </div>
    <div class="hex-echo" id="plaintext-hex">
      <span class="lbl">Plaintext Hex:</span>
      {render_hex_spans("--p")}
    </div>
  </section>

  <section class="panel" aria-label="Toy Key Space">
    <div class="panel-head">
      <span class="panel-title">2. Toy Key Space (8-Bit Switch Bank: 256 Keys + NIST Vector)</span>
      <span class="panel-hint">Toggles bits <code>k[120..127]</code> (<code>00..00</code>–<code>00..ff</code>) fed into live CSS AES KeyExpansion</span>
    </div>
    <div class="key-controls">
      <div class="dip-bank">
        {dip_switches_html}
      </div>
      <label class="nist-toggle" for="nist-vec">
        <input type="checkbox" id="nist-vec">
        <span>Load NIST FIPS-197 Appendix B Vector</span>
      </label>
    </div>
    <div class="nist-banner">
      NIST FIPS-197 Appendix B Active &mdash; Expected Ciphertext: <strong>{NIST_CT_HEX}</strong>
    </div>
    <div class="hex-echo" id="key-hex">
      <span class="lbl">128-Bit Key Hex:</span>
      {render_hex_spans("--k")}
    </div>
  </section>

  <div class="circuit-grid">
    {"".join(pipeline_open)}
    {output_card_html}
    {"".join(reversed(pipeline_close))}
  </div>
</main>
</body>
</html>
"""
    return full_html


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "index.html")
    content = generate_html()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {out_path} ({len(content) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
