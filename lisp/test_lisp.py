#!/usr/bin/env python3
"""
Automated Verification Suite for the Pure HTML + Inline CSS Lisp Interpreter (`lisp/`).

Tests:
  1. Parallel CSS Lexer, Binary Compactor, & S-Expr `next_expr` Parser equivalence
  2. Arithmetic, variadic ops, comparisons, & 20-bit signed integer division/modulo
  3. Mutable variables (`set`), lexical shadowing (`let`), & multi-expr sequencing
  4. User functions (`def`), lambdas (`fn`), recursion (`fact`, `fib`, `gcd`), & HOFs
  5. Syntax & runtime error handling (unbalanced parens, unbound sym, div/0, step limit)
  6. Static HTML5 & CSS audit of `lisp/index.html` (0 JS, 0 SVG, DOM depth < 500, 0 cycles)
"""

from __future__ import annotations

from html.parser import HTMLParser
import os
import re
import unittest
from typing import List

from css_vm import (
    MODE_DONE,
    MODE_ERR,
    LispCSSSpec,
    build_lisp_css_spec,
    simulate_css_lisp,
    simulate_css_token_composer,
)
from generate import generate_html
from lisp_ref import (
    DEFAULT_PROGRAM,
    DEFAULT_TOKEN_LIST,
    ERR_CALL,
    ERR_DIV_ZERO,
    ERR_NONE,
    ERR_STEP_LIMIT,
    ERR_SYNTAX,
    ERR_UNBOUND,
    MAX_TOKENS,
    PRESETS,
    SELECT_CHARS,
    TOKEN_SELECT_OPTIONS,
    eval_lisp_ref,
    lex_and_parse_ref,
)


class _DOMAuditor(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: List[str] = []
        self.max_depth = 0
        self.tags_seen: set[str] = set()
        self.event_attrs: List[str] = []
        self.js_uris: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "option" and self.stack and self.stack[-1] == "option":
            self.stack.pop()
        self.tags_seen.add(t)
        for k, v in attrs:
            kl = k.lower()
            if kl.startswith("on"):
                self.event_attrs.append(f"{t}[{kl}]")
            if v and "javascript:" in v.lower():
                self.js_uris.append(f"{t}[{kl}={v}]")
        if t not in self.VOID_TAGS:
            self.stack.append(t)
            if len(self.stack) > self.max_depth:
                self.max_depth = len(self.stack)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.VOID_TAGS:
            return
        if t == "select" and self.stack and self.stack[-1] == "option":
            self.stack.pop()
        assert self.stack and self.stack[-1] == t, f"Mismatched closing </{t}>, tail={self.stack[-3:]}"
        self.stack.pop()


class TestNoJSLisp(unittest.TestCase):
    spec: LispCSSSpec

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = build_lisp_css_spec()

    def _assert_eval_matches(self, src: str, expected_val: int | None = None) -> None:
        ref = eval_lisp_ref(src)
        if expected_val is not None:
            self.assertEqual(ref.err, ERR_NONE, f"Ref unexpectedly errored on {src!r}")
            self.assertEqual(ref.value, expected_val)

        env = simulate_css_lisp(src, self.spec)
        if ref.err == ERR_NONE:
            self.assertEqual(env["--f_mode"], MODE_DONE, f"CSS did not halt DONE on {src!r}: err={env['--f_err']}")
            self.assertEqual(env["--f_err"], ERR_NONE)
            self.assertEqual(env["--acc"], ref.value, f"Value mismatch on {src!r}")
        else:
            self.assertEqual(env["--f_mode"], MODE_ERR, f"Expected ERR on {src!r}")
            self.assertEqual(env["--f_err"], ref.err, f"Error code mismatch on {src!r}")

    def test_lexer_compactor_and_parser_bus(self) -> None:
        samples = [
            DEFAULT_PROGRAM,
            "   (+(* -12 3)(- 100 5))  ",
            "(<= -45 0)",
            "(>= 999999 100000)",
            "",
        ]
        for src in samples:
            ref_p = lex_and_parse_ref(src)
            env = simulate_css_lisp(src, self.spec)
            self.assertEqual(env["--ntok"], ref_p.ntok, f"ntok mismatch on {src!r}")
            self.assertEqual(env["--syn_err"], ref_p.syn_err, f"syn_err mismatch on {src!r}")
            for i in range(MAX_TOKENS):
                self.assertEqual(env[f"--tk{i}"], ref_p.kinds[i], f"tk[{i}] on {src!r}")
                self.assertEqual(env[f"--tv{i}"], ref_p.vals[i], f"tv[{i}] on {src!r}")
                if i < ref_p.ntok:
                    self.assertEqual(env[f"--tn{i}"], ref_p.next_expr[i], f"tn[{i}] on {src!r}")

    def test_arithmetic_comparisons_and_signed_division(self) -> None:
        cases = [
            ("42", 42),
            ("-305", -305),
            ("(+)", 0),
            ("(*)", 1),
            ("(- 19)", -19),
            ("(+ 1 2 3 4 5)", 15),
            ("(* 2 3 4 5)", 120),
            ("(- 50 10 7 3)", 30),
            ("(/ 144 12)", 12),
            ("(/ -100 7)", -14),
            ("(/ 100 -7)", -14),
            ("(/ -100 -7)", 14),
            ("(/ 40320 120)", 336),
            ("(mod 29 5)", 4),
            ("(% 1071 462)", 147),
            ("(= 7 7)", 1),
            ("(= 7 8)", 0),
            ("(<= 5 5)", 1),
            ("(>= 4 5)", 0),
            ("(not 0)", 1),
            ("(not 9)", 0),
            ("(if t 11 22)", 11),
            ("(if nil 11 22)", 22),
            ("(if 0 99)", 0),
        ]
        for src, exp in cases:
            self._assert_eval_matches(src, exp)

    def test_variables_let_shadowing_and_begin(self) -> None:
        cases = [
            ("(begin (set a 5) (set a (+ a 9)) (* a 3))", 42),
            ("(set x 100) (+ (let ((x 5)) (* x x)) x)", 125),
            ("(let ((a 7) (b (* a 6))) (+ a b))", 49),
        ]
        for src, exp in cases:
            self._assert_eval_matches(src, exp)

    def test_default_and_all_showcase_presets(self) -> None:
        self._assert_eval_matches(DEFAULT_PROGRAM, 84)
        for pid, _title, src, exp in PRESETS:
            with self.subTest(preset=pid):
                self._assert_eval_matches(src, exp)

    def test_zero_arg_fn_and_first_class_lambda(self) -> None:
        self._assert_eval_matches("(def k () 37) (+ (k) 5)", 42)
        self._assert_eval_matches("(let ((sq (fn (x) (* x x)))) (+ (sq 5) (sq 12)))", 169)

    def test_ergonomic_token_composer_comments_and_cli_program(self) -> None:
        # 1. Selectable space ('\u00a0', '␣', '.') in both 64-char and Token Editor dropdowns
        char_disp_map = {lbl: code for lbl, code, _ in SELECT_CHARS}
        tok_disp_map = {lbl: tid for lbl, tid, _ in TOKEN_SELECT_OPTIONS}
        for sp_lbl in ("\u00a0", "␣", "."):
            self.assertEqual(char_disp_map.get(sp_lbl), 0)
            self.assertEqual(tok_disp_map.get(sp_lbl), 0)

        self._assert_eval_matches("(+\u00a0(* 6␣7).10); )))) leftover garbage", 52)

        # 2. All numbers 0-99 are selectable options and evaluate accurately in the CSS Token Composer
        for n in range(100):
            s_n = str(n)
            self.assertIn(s_n, tok_disp_map, f"Number {n} missing from TOKEN_SELECT_OPTIONS")
            res_n = simulate_css_token_composer([s_n], self.spec)
            self.assertEqual(res_n["--f_mode"], MODE_DONE)
            self.assertEqual(res_n["--acc"], n)

        two_digit_arith = simulate_css_token_composer(["(", "*", "19", "23", ")"], self.spec)
        self.assertEqual(two_digit_arith["--f_mode"], MODE_DONE)
        self.assertEqual(two_digit_arith["--acc"], 437)

        # 3. Default Blank Token Editor (all ';' comment characters) starts blank -> 0 tokens, acc = 0
        blank_env = simulate_css_token_composer((), self.spec)
        self.assertEqual(blank_env["--ntok"], 0)
        self.assertEqual(blank_env["--f_mode"], MODE_DONE)
        self.assertEqual(blank_env["--f_err"], ERR_NONE)
        self.assertEqual(blank_env["--acc"], 0)

        # 4. Every Showcase Preset writes into the Token Editor (`--ptid0..39`) and evaluates cleanly
        for pid, _title, src, exp in PRESETS:
            with self.subTest(token_preset=pid):
                p_env = simulate_css_token_composer((), self.spec, preset_src=src)
                self.assertEqual(p_env["--f_mode"], MODE_DONE)
                self.assertEqual(p_env["--f_err"], ERR_NONE)
                self.assertEqual(p_env["--acc"], exp)

        # 5. Live per-cell user overrides on top of a preset written into the Token Editor
        fact_src = dict((pid, src) for pid, _, src, _ in PRESETS)["rec-fact"]
        # Token 30 in `(def fact (n) (if (< n 2) 1 (* n (fact (- n 1))))) (fact 8)` is `"8"`
        fact_6 = simulate_css_token_composer((), self.spec, preset_src=fact_src, overrides={30: "6"})
        self.assertEqual(fact_6["--f_mode"], MODE_DONE)
        self.assertEqual(fact_6["--acc"], 720)

        # 6. Token Composer with interior space/blank gaps ('\u00a0', '␣', '.') and ';' comment
        tok_default = simulate_css_token_composer(DEFAULT_TOKEN_LIST, self.spec)
        self.assertEqual(tok_default["--f_mode"], MODE_DONE)
        self.assertEqual(tok_default["--acc"], 84)

        fact_tokens = [
            "(", "def", "fact", "(", "n", ")",
            "(", "if", "(", "<", "n", "2", ")", "1",
            "(", "*", "n", "\u00a0", "(", "fact", "␣", "(", "-", "n", "1", ")", ")", ")",
            ")", ")",
            "(", "fact", "7", ")", ";", ")", "(",
        ]
        tok_fact = simulate_css_token_composer(fact_tokens, self.spec)
        self.assertEqual(tok_fact["--f_mode"], MODE_DONE)
        self.assertEqual(tok_fact["--acc"], 5040)

        # 7. Custom CLI `--program` HTML generation sanity check
        custom_html = generate_html(default_program="(def sq (x) (* x x)) (sq 19)", start_in_char_mode=True)
        self.assertIn("id=pr-edit checked", custom_html)
        self.assertIn("&rarr; 361", custom_html)

    def test_syntax_and_runtime_errors(self) -> None:
        err_cases = [
            ("(+ 1 2", ERR_SYNTAX),
            ("(+ 1 2))", ERR_SYNTAX),
            (")(", ERR_SYNTAX),
            ("(+ nope 1)", ERR_UNBOUND),
            ("(/ 15 0)", ERR_DIV_ZERO),
            ("(mod 15 0)", ERR_DIV_ZERO),
            ("(99 1)", ERR_CALL),
            ("(def f (x) x) (f 1 2)", ERR_CALL),
        ]
        for src, exp_err in err_cases:
            with self.subTest(src=src):
                env = simulate_css_lisp(src, self.spec)
                self.assertEqual(env["--f_mode"], MODE_ERR)
                self.assertEqual(env["--f_err"], exp_err)

        # Non-tail infinite recursion catches continuation stack overflow (ERR_CALL)
        ovf_env = simulate_css_lisp("(def inf (x) (+ 1 (inf x))) (inf 0)", self.spec, fast_halt=True)
        self.assertEqual(ovf_env["--f_mode"], MODE_ERR)
        self.assertEqual(ovf_env["--f_err"], ERR_CALL)

        # Tail-recursive infinite loop runs in O(1) stack space and halts at step budget (ERR_STEP_LIMIT)
        inf_env = simulate_css_lisp("(def inf () (inf)) (inf)", self.spec, fast_halt=False)
        self.assertEqual(inf_env["--f_mode"], MODE_ERR)
        self.assertEqual(inf_env["--f_err"], ERR_STEP_LIMIT)

    def test_generated_html_static_audit(self) -> None:
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        self.assertTrue(os.path.exists(html_path), "lisp/index.html should exist")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertEqual(content, generate_html())

        auditor = _DOMAuditor()
        auditor.feed(content)

        self.assertNotIn("script", auditor.tags_seen)
        self.assertNotIn("svg", auditor.tags_seen)
        self.assertEqual(auditor.event_attrs, [])
        self.assertEqual(auditor.js_uris, [])
        self.assertEqual(auditor.stack, [], "All non-void HTML5 tags must close cleanly")
        self.assertLess(auditor.max_depth, 500, f"DOM depth {auditor.max_depth} must stay below HTML5 512 limit")

        style_match = re.search(r"<style>(.*)</style>", content, flags=re.DOTALL)
        self.assertIsNotNone(style_match)
        assert style_match is not None
        css_text = style_match.group(1)
        top_level_css = re.sub(r"@media[^{]+\{(?:[^{}]+\{[^{}]*\})*\}", "", css_text)

        var_re = re.compile(r"var\((--[a-zA-Z0-9_]+)\)")
        seen_class_selectors: set[str] = set()
        for rule_match in re.finditer(r"([^{}]+)\{([^{}]+)\}", top_level_css):
            sel = rule_match.group(1).strip()
            if sel.startswith("@"):
                continue
            if re.fullmatch(r"\.[a-z0-9_-]+", sel):
                self.assertNotIn(sel, seen_class_selectors, f"Duplicate class rule collision on {sel}")
                seen_class_selectors.add(sel)
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
                refs = set(var_re.findall(v))
                overlap = lhs_vars & refs
                self.assertEqual(overlap, set(), f"Intra-rule CSS cycle in {sel}: {k} reads {overlap}")


if __name__ == "__main__":
    unittest.main()
