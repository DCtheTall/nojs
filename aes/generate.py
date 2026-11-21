#!/usr/bin/env python3
"""
Static Pure HTML + Inline CSS Generator for NIST AES-128 (`aes/index.html`).

Generates a compact standalone HTML file with inline CSS, zero JavaScript, and
zero SVG that executes full 10-round AES-128 + KeyExpansion in the browser's
CSS cascade engine using:
  - A reusable 4-byte `SubWord` unit (`w0..w3`) shared 5x per round
  - GF(2) Algebraic Normal Form + CSS `mod(..., 2)` flat modulo sums
  - `@property <integer>` wire canonicalization & `display: contents` compute nodes
  - HTML5 compact `<option b=...>` bit-mask attributes decoded by 128 `:has()` rules
"""

from __future__ import annotations

import html
import os
import re
from typing import List

from aes_cipher import (
    RCON,
    aes128_encrypt_ref,
    build_compact_aes_css_spec,
    get_round_stage_sequence,
)
from circuit import format_mod2_sum


DEFAULT_PLAINTEXT = b"Hello, CSS AES!!"  # 16 bytes
NIST_PT_HEX = "3243f6a8885a308d313198a2e0370734"
NIST_KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
NIST_CT_HEX = "3925841d02dc09fbdc118597196a0b32"
BIT_LETTERS = "hgfedcba"  # bit 7 (MSB) -> 'h', bit 0 (LSB) -> 'a'


def render_hex_strip(cls_prefix: str, num_nibbles: int = 32) -> str:
    """Render 32 compact `<i>` hex nibble elements (`s0..s31`, `k0..k31`, `p0..p31`)."""
    return "".join(f"<i class={cls_prefix}{n}></i>" for n in range(num_nibbles))


def minify_css_block(raw: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([{};:,>])\s*", r"\1", s)
    return s.replace(";}", "}").strip()


def generate_html() -> str:
    spec = build_compact_aes_css_spec()
    css: List[str] = []

    # -----------------------------------------------------------------------
    # A. @counter-style & @property registrations (only for computed wires!)
    # -----------------------------------------------------------------------
    css.append(
        '@counter-style hex-digit{system:fixed 0;'
        'symbols:"0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f"}'
    )

    def reg_prop(name: str) -> str:
        return f'@property {name}{{syntax:"<integer>";initial-value:0;inherits:true}}'

    for i in range(128):
        css.append(reg_prop(f"--s{i}"))
        css.append(reg_prop(f"--k{i}"))
    for i in range(68):
        css.append(reg_prop(f"--y{i}"))
    for i in range(16):
        css.append(reg_prop(f"--d{i}"))
    for i in range(36):
        css.append(reg_prop(f"--e{i}"))
    for i in range(32):
        css.append(reg_prop(f"--o{i}"))

    # -----------------------------------------------------------------------
    # B. UI Theme Styling (Minified inline CSS, zero SVG, zero JS)
    # -----------------------------------------------------------------------
    ui_css = """
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0b0f17;color:#e2e8f0;line-height:1.5;padding:2rem 1rem 4rem
}
.aes-app{max-width:980px;margin:0 auto;display:flex;flex-direction:column;gap:1.5rem}
header.hero{
  border:1px solid #1e293b;background:linear-gradient(180deg,#111827,#0f172a);
  border-radius:14px;padding:1.5rem 1.75rem;box-shadow:0 12px 32px rgba(0,0,0,.35)
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
  font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  padding:.2rem .55rem;border-radius:999px;background:#1e293b;color:#38bdf8;border:1px solid #334155
}
.badge.green{color:#4ade80;border-color:#14532d;background:#052e16}
header.hero h1{font-size:1.65rem;font-weight:700;color:#f8fafc;letter-spacing:-.02em}
header.hero p{margin-top:.4rem;color:#94a3b8;font-size:.93rem}
.panel{
  border:1px solid #1e293b;background:#111827;border-radius:14px;
  padding:1.35rem 1.5rem;box-shadow:0 8px 24px rgba(0,0,0,.25)
}
.panel-head{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:.5rem;margin-bottom:.9rem}
.panel-title{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8}
.panel-hint{font-size:.8rem;color:#64748b}
.char-strip{
  display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:4px;
  background:#090d16;padding:8px;border-radius:10px;border:1px solid #263349
}
@media (max-width:720px){.char-strip{grid-template-columns:repeat(8,minmax(0,1fr))}}
.cw{display:flex;flex-direction:column;align-items:center;gap:3px}
.ci{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.63rem;color:#475569}
select.cc{
  width:100%;height:2.4rem;appearance:none;-webkit-appearance:none;
  background:#162032;color:#f8fafc;border:1px solid #334155;border-radius:6px;
  text-align:center;text-align-last:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.05rem;font-weight:600;
  cursor:pointer;transition:border-color .12s,background .12s,box-shadow .12s
}
select.cc:hover{border-color:#475569;background:#1e293b}
select.cc:focus{outline:none;border-color:#38bdf8;background:#0f172a;box-shadow:0 0 0 2px rgba(56,189,248,.28)}
.hex-echo{
  margin-top:.75rem;display:flex;align-items:center;flex-wrap:wrap;gap:.5rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.83rem;color:#94a3b8;
  background:#090d16;padding:.55rem .8rem;border-radius:8px;border:1px solid #1e293b
}
.hex-echo .lbl{color:#64748b;font-size:.73rem;text-transform:uppercase;letter-spacing:.06em;margin-right:.25rem}
.key-controls{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:1rem}
.dip-bank{display:flex;gap:.45rem;flex-wrap:wrap}
.dip{display:flex;flex-direction:column;align-items:center;gap:.3rem;cursor:pointer;user-select:none}
.dip input{position:absolute;opacity:0;width:1px;height:1px}
.dip-box{
  width:2.35rem;height:2.75rem;border-radius:7px;background:#090d16;border:1px solid #334155;
  display:flex;flex-direction:column;align-items:center;justify-content:space-between;
  padding:.3rem 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;transition:all .14s ease
}
.dip-led{width:8px;height:8px;border-radius:50%;background:#1e293b}
.dip-bv::after{content:"0";font-size:.85rem;font-weight:700;color:#64748b}
.dip input:checked+.dip-box{border-color:#38bdf8;background:#0c2438;box-shadow:0 0 12px rgba(56,189,248,.2)}
.dip input:checked+.dip-box .dip-led{background:#38bdf8;box-shadow:0 0 8px #38bdf8}
.dip input:checked+.dip-box .dip-bv::after{content:"1";color:#f8fafc}
.dip input:focus-visible+.dip-box{outline:2px solid #38bdf8;outline-offset:2px}
.dip-lbl{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;color:#64748b}
.nist-toggle{
  display:inline-flex;align-items:center;gap:.6rem;padding:.6rem .95rem;border-radius:9px;
  background:#090d16;border:1px solid #334155;cursor:pointer;user-select:none;font-size:.84rem;color:#cbd5e1
}
.nist-toggle:hover{border-color:#475569}
.nist-toggle input{accent-color:#f59e0b;width:1rem;height:1rem;cursor:pointer}
.nist-banner{
  display:none;margin-top:.75rem;padding:.65rem .9rem;border-radius:8px;
  background:#271c09;border:1px solid #92400e;color:#fde68a;font-size:.82rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace
}
.aes-app:has(#nist-vec:checked) .nist-banner{display:block}
.aes-app:has(#nist-vec:checked) .char-strip,.aes-app:has(#nist-vec:checked) .dip-bank{opacity:.4;pointer-events:none}

.cg{display:flex;flex-direction:column;gap:1.5rem}
.cg div{display:contents}

.output-panel{
  order:-2;border:1px solid #0284c7;background:linear-gradient(180deg,#0c192c,#0b1322);
  border-radius:14px;padding:1.5rem 1.65rem;box-shadow:0 0 36px rgba(14,165,233,.14)
}
.ct-display{
  margin-top:.55rem;padding:1rem 1.15rem;background:#060a12;border:1px solid #1e3a5f;
  border-radius:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:1.25rem;font-weight:700;letter-spacing:.04em;color:#38bdf8;line-height:1.7
}
.hx i{font-style:normal;display:inline-block}
.hx i::after{content:counter(h,hex-digit)}
.hx i:nth-child(2n):not(:last-child){margin-right:.55ch}

.matrix-and-leds{margin-top:1.15rem;display:grid;grid-template-columns:auto 1fr;gap:1.25rem;align-items:start}
@media (max-width:680px){.matrix-and-leds{grid-template-columns:1fr}}
.sub-lbl{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#64748b;margin-bottom:.45rem}
.state-matrix{
  display:grid;grid-template-columns:repeat(4,3.1rem);gap:5px;background:#060a12;
  padding:8px;border-radius:9px;border:1px solid #1e293b;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92rem;font-weight:600;color:#e2e8f0
}
.mc-cell{display:block;background:#111c2d;border:1px solid #24354d;border-radius:5px;padding:.35rem 0;text-align:center}
.mc-cell i{font-style:normal}
.mc-cell i::after{content:counter(h,hex-digit)}
.led-grid{
  display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:4px;
  background:#060a12;padding:10px;border-radius:9px;border:1px solid #1e293b
}
.led{display:block;aspect-ratio:1;border-radius:3px;background:#38bdf8;border:1px solid rgba(255,255,255,.08)}
.trace-head{order:5;border:1px solid #1e293b;background:#111827;border-radius:14px;padding:1.1rem 1.4rem}
.trace-title{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;margin-bottom:.35rem}
.tr{
  display:grid;grid-template-columns:5.5rem 1fr 1fr;gap:.75rem;align-items:center;
  padding:.5rem .85rem;background:#0c121e;border:1px solid #1e293b;border-radius:8px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem
}
@media (max-width:820px){.tr{grid-template-columns:1fr;gap:.3rem}}
.trn{font-weight:700;color:#38bdf8}
.tkv{color:#cbd5e1}
.tkv .tag{color:#64748b;margin-right:.35rem}
"""
    css.append(minify_css_block(ui_css))

    # -----------------------------------------------------------------------
    # C. Input Decoding Rules (Default --p0..127 = 0, 128 :has() rules, toy key, NIST)
    # -----------------------------------------------------------------------
    p_defaults = ";".join(f"--p{i}:0" for i in range(128))
    css.append(f":root{{{p_defaults}}}")

    for slot in range(16):
        for bit_pos, bchar in enumerate(BIT_LETTERS):
            p_idx = 8 * slot + bit_pos
            css.append(f".aes-app:has(#c{slot} :checked[b*={bchar}]){{--p{p_idx}:1}}")

    for bit_pos in range(8):
        bit_num = 7 - bit_pos
        css.append(f".aes-app:has(#kbit{bit_num}:checked){{--k{120 + bit_pos}:1}}")

    nist_pt_bytes = bytes.fromhex(NIST_PT_HEX)
    nist_key_bytes = bytes.fromhex(NIST_KEY_HEX)
    nist_decls: List[str] = []
    for byte_idx in range(16):
        pb = nist_pt_bytes[byte_idx]
        kb = nist_key_bytes[byte_idx]
        for bit_pos in range(8):
            idx = 8 * byte_idx + bit_pos
            nist_decls.append(f"--p{idx}:{(pb >> (7 - bit_pos)) & 1}")
            nist_decls.append(f"--k{idx}:{(kb >> (7 - bit_pos)) & 1}")
    css.append(f".aes-app.aes-app.aes-app:has(#nist-vec:checked){{{';'.join(nist_decls)}}}")

    # -----------------------------------------------------------------------
    # D. Round 0 + Rcon[1..10] + Compact 4-Byte `SubWord` Round Stages
    # -----------------------------------------------------------------------
    r0_decls = [f"--s{i}:{format_mod2_sum([f'var(--p{i})', f'var(--k{i})'])}" for i in range(128)]
    css.append(f".r0{{{';'.join(r0_decls)}}}")

    for r in range(1, 11):
        rc_val = RCON[r]
        rc_decls = [f"--rc{b}:{(rc_val >> (7 - b)) & 1}" for b in range(8)]
        css.append(f".rk{r}{{{';'.join(rc_decls)}}}")

    for cls_name, rules in spec.items():
        body = ";".join(f"{lhs}:{rhs}" for lhs, rhs in rules)
        css.append(f".{cls_name}{{{body}}}")

    # -----------------------------------------------------------------------
    # E. Hex Nibble Counter Rules & 128-Bit LED Opacity Rules
    # -----------------------------------------------------------------------
    for nib_idx in range(32):
        b0 = 4 * nib_idx
        for prefix, cls_p in (("--s", "s"), ("--k", "k"), ("--p", "t")):
            expr = (
                f"calc(8*var({prefix}{b0}) + 4*var({prefix}{b0 + 1}) "
                f"+ 2*var({prefix}{b0 + 2}) + var({prefix}{b0 + 3}))"
            )
            css.append(f".{cls_p}{nib_idx}{{counter-reset:h {expr}}}")

    for bit_idx in range(128):
        css.append(f".l{bit_idx}{{opacity:calc(.12 + .88*var(--s{bit_idx}))}}")

    # -----------------------------------------------------------------------
    # F. Compact HTML5 Controls & Circuit DOM Tree
    # -----------------------------------------------------------------------
    def build_select_for_slot(slot: int, default_byte: int) -> str:
        opts: List[str] = []
        for bval in [0] + list(range(32, 127)):
            b_letters = "".join(BIT_LETTERS[7 - bit] for bit in range(7, -1, -1) if (bval >> bit) & 1)
            b_attr = f" b={b_letters}" if b_letters else ""
            sel_attr = " selected" if bval == default_byte else ""
            if bval == 0:
                label = "·"
            elif bval == 32:
                label = "␣"
            else:
                label = html.escape(chr(bval))
            # HTML5 §13.1.2.4: </option> end tag is optional before <option> or </select>
            opts.append(f"<option{b_attr}{sel_attr}>{label}")
        return (
            f'<div class=cw><span class=ci>{slot:02d}</span>'
            f'<select id=c{slot} class=cc aria-label="Plaintext byte {slot}">'
            f'{"".join(opts)}</select></div>'
        )

    selects_html = "".join(build_select_for_slot(i, DEFAULT_PLAINTEXT[i]) for i in range(16))

    dip_switches_html = "".join(
        f'<label class=dip for=kbit{bit}><input type=checkbox id=kbit{bit}>'
        f'<span class=dip-box><span class=dip-led></span><span class=dip-bv></span></span>'
        f'<span class=dip-lbl>2^{bit}</span></label>'
        for bit in range(7, -1, -1)
    )

    matrix_cells_html: List[str] = []
    for row in range(4):
        for col in range(4):
            b_idx = 4 * col + row
            matrix_cells_html.append(
                f"<span class=mc-cell><i class=s{2 * b_idx}></i><i class=s{2 * b_idx + 1}></i></span>"
            )

    leds_html = "".join(f"<span class='led l{i}'></span>" for i in range(128))

    def make_trace_row(r_num: int, label: str) -> str:
        return (
            f'<article class=tr style="order:{10 + r_num}">'
            f"<span class=trn>{label}</span>"
            f'<span class="tkv hx"><span class=tag>State:</span>{render_hex_strip("s")}</span>'
            f'<span class="tkv hx"><span class=tag>RoundKey:</span>{render_hex_strip("k")}</span>'
            f"</article>"
        )

    pipeline_open: List[str] = []
    depth_count = 0

    pipeline_open.append("<div class=r0>")
    depth_count += 1
    pipeline_open.append(
        '<section class=trace-head>'
        '<h2 class=trace-title>Live 10-Round Internal State &amp; Key-Schedule Trace (Pure CSS Cascade)</h2>'
        '<p style="color:#64748b;font-size:.82rem">'
        "Every row reads inherited <code>--s0..127</code> and <code>--k0..127</code> wires directly at its round boundary."
        "</p></section>"
    )
    pipeline_open.append(make_trace_row(0, "Round 0"))

    for r in range(1, 11):
        seq = get_round_stage_sequence(r)
        for idx, st_cls in enumerate(seq):
            if idx == 0:
                pipeline_open.append(f'<div class="{st_cls} rk{r}">')
            else:
                pipeline_open.append(f"<div class={st_cls}>")
            depth_count += 1
        pipeline_open.append(make_trace_row(r, f"Round {r}"))

    output_card_html = (
        '<section class=output-panel aria-label="AES-128 Ciphertext Output">'
        '<header class=panel-head>'
        '<span class=panel-title style="color:#7dd3fc">AES-128 Ciphertext Output (Hex)</span>'
        '<span class=panel-hint style="color:#38bdf8">Computed 100% by CSS @property &amp; mod()/calc()</span>'
        "</header>"
        f'<p class="ct-display hx" id=ciphertext-hex>{render_hex_strip("s")}</p>'
        "<section class=matrix-and-leds>"
        "<span><span class=sub-lbl style=display:block>4&times;4 State Matrix (Col-Major)</span>"
        f'<span class=state-matrix>{"".join(matrix_cells_html)}</span></span>'
        "<span><span class=sub-lbl style=display:block>128-Bit Ciphertext State Register (--s0 .. --s127)</span>"
        f"<span class=led-grid>{leds_html}</span></span>"
        "</section></section>"
    )

    default_ct_hex = aes128_encrypt_ref(DEFAULT_PLAINTEXT, bytes(16)).hex()

    return (
        "<!DOCTYPE html>\n<html lang=en>\n<head>\n<meta charset=utf-8>\n"
        "<meta name=viewport content='width=device-width,initial-scale=1'>\n"
        "<title>NoJS AES-128 — Pure HTML + Inline CSS Block Cipher</title>\n"
        f"<style>{''.join(css)}</style>\n</head>\n<body>\n<main class=aes-app>\n"
        "<header class=hero>"
        "<nav class=back-row aria-label='Site navigation'>"
        "<a class=back-link href='../'>&larr; All NoJS Demos</a>"
        "</nav>"
        "<div class=badge-row>"
        "<span class='badge green'>0 Bytes JavaScript</span>"
        "<span class=badge>0 SVG Filters</span>"
        "<span class=badge>NIST FIPS-197 AES-128</span>"
        "<span class=badge>4-Byte SubWord + ANF Circuit</span>"
        "</div>"
        "<h1>Pure HTML + CSS AES-128 Fixed-Circuit Cipher</h1>"
        "<p>Full 10-round 128-bit AES encryption &amp; key expansion executed strictly inside CSS style "
        "recalculation using a reusable 32-bit <code>SubWord</code> unit, GF(2) algebraic normal form "
        "<code>mod(..., 2)</code> gates, <code>@property &lt;integer&gt;</code> wires, and "
        f"<code>@counter-style</code> hex rendering. Default cipher (<code>\"Hello, CSS AES!!\"</code>, key <code>0x00</code>): "
        f"<code>{default_ct_hex}</code>.</p></header>\n"
        '<section class=panel aria-label="Plaintext Input">'
        "<div class=panel-head>"
        "<span class=panel-title>1. Plaintext Block (16 ASCII Bytes / 128 Bits)</span>"
        "<span class=panel-hint>Click a cell &amp; type on your keyboard (Tab advances; <code>&middot;</code> = 0x00 null, <code>&#x2423;</code> = space)</span>"
        "</div>"
        f"<div class=char-strip>{selects_html}</div>"
        f'<div class="hex-echo hx" id=plaintext-hex><span class=lbl>Plaintext Hex:</span>{render_hex_strip("t")}</div>'
        "</section>\n"
        '<section class=panel aria-label="Toy Key Space">'
        "<div class=panel-head>"
        "<span class=panel-title>2. Toy Key Space (8-Bit Switch Bank: 256 Keys + NIST Vector)</span>"
        "<span class=panel-hint>Toggles bits <code>k[120..127]</code> (<code>00..00</code>–<code>00..ff</code>) fed into live CSS AES KeyExpansion</span>"
        "</div>"
        "<div class=key-controls>"
        f"<div class=dip-bank>{dip_switches_html}</div>"
        "<label class=nist-toggle for=nist-vec><input type=checkbox id=nist-vec>"
        "<span>Load NIST FIPS-197 Appendix B Vector</span></label>"
        "</div>"
        f"<div class=nist-banner>NIST FIPS-197 Appendix B Active &mdash; Expected Ciphertext: <strong>{NIST_CT_HEX}</strong></div>"
        f'<div class="hex-echo hx" id=key-hex><span class=lbl>128-Bit Key Hex:</span>{render_hex_strip("k")}</div>'
        "</section>\n"
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
