#!/usr/bin/env python3
"""
Automated Verification Suite for Pure HTML + Inline CSS AES-128 (`aes/`).

Tests:
  1. Exhaustive 256-entry Boyar-Peralta S-Box circuit equivalence to FIPS-197
  2. NIST FIPS-197 standard vectors (Appendix B, Appendix C.1, all-zero)
  3. Full 10-round CSS layer-inheritance simulation (all round states + keys)
  4. Cross-check against system `openssl enc -aes-128-ecb -nopad` across toy keys
  5. Static audit of `aes/index.html` (zero JS, zero SVG, DOM depth, CSS cycles)
"""

from __future__ import annotations

from html.parser import HTMLParser
import os
import random
import re
import shutil
import subprocess
import unittest
from typing import Dict, List

from aes_cipher import (
    RCON,
    SBOX,
    aes128_encrypt_ref,
    aes128_encrypt_with_trace,
    bits_to_bytes,
    build_aes_final_ark,
    build_aes_mixcolumns_ark,
    build_aes_shared_subbytes_sr_keysched,
    build_sbox_boyar_peralta,
    bytes_to_bits,
)
from circuit import Circuit, simulate_css_layers
from generate import DEFAULT_PLAINTEXT, generate_html


class _DOMAuditor(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.max_depth = 0
        self.tags_seen: set[str] = set()
        self.event_attrs: List[str] = []
        self.js_uris: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        t = tag.lower()
        self.tags_seen.add(t)
        for k, v in attrs:
            kl = k.lower()
            if kl.startswith("on"):
                self.event_attrs.append(f"{t}[{kl}]")
            if v and "javascript:" in v.lower():
                self.js_uris.append(f"{t}[{kl}={v}]")
        if t not in self.VOID_TAGS:
            self.depth += 1
            if self.depth > self.max_depth:
                self.max_depth = self.depth

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t not in self.VOID_TAGS:
            self.depth -= 1


class TestNoJSAES(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        c_core, o_core = build_aes_shared_subbytes_sr_keysched()
        cls.core_la = c_core.compile_to_css_layers(o_core, max_inline_depth=2, temp_prefix="--w")

        c_mc, o_mc = build_aes_mixcolumns_ark()
        cls.mc_la = c_mc.compile_to_css_layers(o_mc, max_inline_depth=2, temp_prefix="--w")

        c_fin, o_fin = build_aes_final_ark()
        cls.fin_la = c_fin.compile_to_css_layers(o_fin, max_inline_depth=2, temp_prefix="--w")

    def _run_css_pipeline(self, pt16: bytes, key16: bytes) -> tuple[bytes, List[bytes], List[bytes]]:
        env: Dict[str, int] = {}
        pt_bits = bytes_to_bits(pt16, "--p")
        k_bits = bytes_to_bits(key16, "--k")
        env.update(pt_bits)
        env.update(k_bits)

        # Round 0 AddRoundKey
        for i in range(128):
            env[f"--s{i}"] = env[f"--p{i}"] ^ env[f"--k{i}"]

        states = [bits_to_bytes(env, "--s", 16)]
        rkeys = [bits_to_bytes(env, "--k", 16)]

        for r in range(1, 11):
            env.update(bytes_to_bits(bytes([RCON[r]]), "--rc"))
            env = simulate_css_layers(self.core_la, env)
            env = simulate_css_layers(self.mc_la if r < 10 else self.fin_la, env)
            states.append(bits_to_bytes(env, "--s", 16))
            rkeys.append(bits_to_bytes(env, "--k", 16))

        return states[-1], states, rkeys

    def test_sbox_exhaustive_256(self) -> None:
        c = Circuit()
        u = [c.input(f"u{i}") for i in range(8)]
        s = build_sbox_boyar_peralta(c, u)
        outs = {f"s{i}": s[i] for i in range(8)}
        self.assertEqual(SBOX[0x00], 0x63)
        self.assertEqual(SBOX[0x53], 0xED)
        self.assertEqual(SBOX[0xFF], 0x16)

        for x in range(256):
            inp = {f"u{i}": (x >> (7 - i)) & 1 for i in range(8)}
            res = c.evaluate(inp, outs)
            got = sum(res[f"s{i}"] << (7 - i) for i in range(8))
            self.assertEqual(got, SBOX[x], f"S-Box mismatch at {x:#04x}")

    def test_nist_fips197_standard_vectors(self) -> None:
        vectors = [
            # NIST FIPS-197 Appendix B
            (
                "3243f6a8885a308d313198a2e0370734",
                "2b7e151628aed2a6abf7158809cf4f3c",
                "3925841d02dc09fbdc118597196a0b32",
            ),
            # NIST FIPS-197 Appendix C.1 (AES-128)
            (
                "00112233445566778899aabbccddeeff",
                "000102030405060708090a0b0c0d0e0f",
                "69c4e0d86a7b0430d8cdb78070b4c55a",
            ),
            # All-zero block & key
            (
                "00000000000000000000000000000000",
                "00000000000000000000000000000000",
                "66e94bd4ef8a2c3b884cfa59ca342b2e",
            ),
        ]
        for pt_h, key_h, exp_ct_h in vectors:
            pt = bytes.fromhex(pt_h)
            key = bytes.fromhex(key_h)
            exp_ct = bytes.fromhex(exp_ct_h)

            ref_ct, ref_states, ref_keys = aes128_encrypt_with_trace(pt, key)
            self.assertEqual(ref_ct, exp_ct)

            css_ct, css_states, css_keys = self._run_css_pipeline(pt, key)
            self.assertEqual(css_ct, exp_ct)
            self.assertEqual(css_states, ref_states)
            self.assertEqual(css_keys, ref_keys)

    def test_toy_key_space_against_openssl(self) -> None:
        openssl_bin = shutil.which("openssl")
        rng = random.Random(20261119)

        test_cases = [
            (DEFAULT_PLAINTEXT, bytes(16)),
            (DEFAULT_PLAINTEXT, bytes(15) + b"\x2a"),
            (DEFAULT_PLAINTEXT, bytes(15) + b"\xff"),
        ]
        for _ in range(15):
            ascii_pt = bytes(rng.choice([0] + list(range(32, 127))) for _ in range(16))
            toy_key_byte = rng.randrange(256)
            key16 = bytes(15) + bytes([toy_key_byte])
            test_cases.append((ascii_pt, key16))

        for pt, key in test_cases:
            css_ct, _, _ = self._run_css_pipeline(pt, key)
            ref_ct = aes128_encrypt_ref(pt, key)
            self.assertEqual(css_ct, ref_ct)

            if openssl_bin:
                proc = subprocess.run(
                    [openssl_bin, "enc", "-aes-128-ecb", "-K", key.hex(), "-nopad"],
                    input=pt,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(css_ct, proc.stdout)

    def test_generated_html_static_audit(self) -> None:
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        self.assertTrue(os.path.exists(html_path), "index.html should exist")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Regenerate check: file on disk is up to date with generate_html()
        self.assertEqual(content, generate_html())

        auditor = _DOMAuditor()
        auditor.feed(content)

        # Strictly zero JS and zero SVG
        self.assertNotIn("script", auditor.tags_seen)
        self.assertNotIn("svg", auditor.tags_seen)
        self.assertEqual(auditor.event_attrs, [])
        self.assertEqual(auditor.js_uris, [])
        self.assertEqual(auditor.depth, 0, "All non-void HTML tags must be balanced")
        self.assertLess(auditor.max_depth, 250, "DOM depth must stay well below HTML5 512 limit")

        # Audit CSS blocks for intra-rule custom property self-cycles and token bounds
        style_match = re.search(r"<style>(.*)</style>", content, flags=re.DOTALL)
        self.assertIsNotNone(style_match)
        assert style_match is not None
        css_text = style_match.group(1)

        for rule_match in re.finditer(r"([^{}]+)\{([^{}]+)\}", css_text):
            sel = rule_match.group(1).strip()
            if sel.startswith("@"):
                continue
            body = rule_match.group(2)
            decls = [d.strip() for d in body.split(";") if d.strip()]
            lhs_vars = set()
            parsed = []
            for d in decls:
                if ":" not in d:
                    continue
                k, v = d.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k.startswith("--"):
                    lhs_vars.add(k)
                parsed.append((k, v))
            for k, v in parsed:
                self.assertLess(len(v), 600, f"CSS expression too long on {sel} {k}")
                for lv in lhs_vars:
                    self.assertNotIn(f"var({lv})", v, f"Self/intra-rule cycle in {sel}: {k} reads {lv}")


if __name__ == "__main__":
    unittest.main()
