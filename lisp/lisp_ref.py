"""
Reference Lisp Interpreter, Symbol Encoder, and Preset Suite (`lisp/lisp_ref.py`).

Defines the canonical character encoding, 4-letter base-27 symbol IDs,
S-expression parser, and lexical/shallow-binding reference evaluator used to
verify the pure HTML+CSS Lisp interpreter (`lisp/index.html`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Canonical 6-Bit Character Set & Symbol Encoding
# ---------------------------------------------------------------------------

SOURCE_LEN = 64
MAX_TOKENS = 40

# Character codes (0..55, fitting in 6 bits: a=1, b=2, c=4, d=8, e=16, f=32)
CHAR_TO_CODE: Dict[str, int] = {
    " ": 0,
    "(": 1,
    ")": 2,
    "+": 20,
    "-": 21,
    "*": 22,
    "/": 23,
    "%": 24,
    "=": 25,
    "<": 26,
    ">": 27,
}
for _d in range(10):
    CHAR_TO_CODE[str(_d)] = 10 + _d
for _i in range(26):
    CHAR_TO_CODE[chr(ord("a") + _i)] = 30 + _i

CODE_TO_CHAR: Dict[int, str] = {v: k for k, v in CHAR_TO_CODE.items()}

# Ordered characters for <select> UI controls
SELECT_CHARS: Tuple[str, ...] = (
    (" ", "(", ")", "+", "-", "*", "/", "%", "=", "<", ">")
    + tuple(str(d) for d in range(10))
    + tuple(chr(ord("a") + i) for i in range(26))
)

# Token kinds
TOK_EOF = 0
TOK_LP = 1
TOK_RP = 2
TOK_NUM = 3
TOK_SYM = 4

# Operator symbol IDs (120..129, disjoint from alphabetic symbol IDs >= 1000)
SYM_ADD = 120
SYM_SUB = 121
SYM_MUL = 122
SYM_DIV = 123
SYM_PCT = 124
SYM_EQ = 125
SYM_LT = 126
SYM_GT = 127
SYM_LE = 128
SYM_GE = 129


def encode_symbol(name: str) -> int:
    """
    Encode an operator or lowercase alphabetic identifier into its canonical
    32-bit signed <integer> symbol ID.
    """
    op_map = {
        "+": SYM_ADD,
        "-": SYM_SUB,
        "*": SYM_MUL,
        "/": SYM_DIV,
        "%": SYM_PCT,
        "=": SYM_EQ,
        "<": SYM_LT,
        ">": SYM_GT,
        "<=": SYM_LE,
        ">=": SYM_GE,
    }
    if name in op_map:
        return op_map[name]
    if not name or not all("a" <= ch <= "z" for ch in name):
        raise ValueError(f"Unsupported Lisp identifier: {name!r}")
    letters = [ord(ch) - ord("a") + 1 for ch in name[:4]]
    while len(letters) < 4:
        letters.append(0)
    return 1000 + letters[0] * 19683 + letters[1] * 729 + letters[2] * 27 + letters[3]


def decode_symbol(sym_id: int) -> str:
    """Decode a canonical symbol ID back to its display name (up to 4-char prefix)."""
    rev_op = {
        SYM_ADD: "+",
        SYM_SUB: "-",
        SYM_MUL: "*",
        SYM_DIV: "/",
        SYM_PCT: "%",
        SYM_EQ: "=",
        SYM_LT: "<",
        SYM_GT: ">",
        SYM_LE: "<=",
        SYM_GE: ">=",
    }
    if sym_id in rev_op:
        return rev_op[sym_id]
    if sym_id < 1000:
        return f"#{sym_id}"
    rem = sym_id - 1000
    a0, rem = divmod(rem, 19683)
    a1, rem = divmod(rem, 729)
    a2, a3 = divmod(rem, 27)
    chars = []
    for a in (a0, a1, a2, a3):
        if 1 <= a <= 26:
            chars.append(chr(ord("a") + a - 1))
    return "".join(chars)


# Reserved special-form & built-in symbol IDs
SYM_MOD = encode_symbol("mod")
SYM_NOT = encode_symbol("not")
SYM_IF = encode_symbol("if")
SYM_BEGIN = encode_symbol("begin")
SYM_PROGN = encode_symbol("progn")
SYM_SET = encode_symbol("set")
SYM_SETQ = encode_symbol("setq")
SYM_DEFINE = encode_symbol("define")
SYM_LET = encode_symbol("let")
SYM_DEF = encode_symbol("def")
SYM_DEFUN = encode_symbol("defun")
SYM_FN = encode_symbol("fn")
SYM_LAMBDA = encode_symbol("lambda")
SYM_T = encode_symbol("t")
SYM_TRUE = encode_symbol("true")
SYM_NIL = encode_symbol("nil")
SYM_FALSE = encode_symbol("false")

FN_HANDLE_BASE = 900000

# Error codes
ERR_NONE = 0
ERR_SYNTAX = 1
ERR_UNBOUND = 2
ERR_DIV_ZERO = 3
ERR_CALL = 4
ERR_STEP_LIMIT = 5


# ---------------------------------------------------------------------------
# 2. Showcase Presets & Default Editable Program
# ---------------------------------------------------------------------------

DEFAULT_PROGRAM = "(+ (* 6 7) (- 100 58))"  # = 84

PRESETS: Tuple[Tuple[str, str, str, int], ...] = (
    (
        "div-mod",
        "Integer / & mod",
        "(+ (/ 144 12) (* (mod 29 5) (- 15 8)))",
        40,
    ),
    (
        "vars-set",
        "Mutable Variables (set)",
        "(set x 12) (set y (+ (* x x) 6)) (- y x)",
        138,
    ),
    (
        "let-scope",
        "Lexical Scope (let)",
        "(let ((x 6) (y (* x 7))) (+ x y))",
        48,
    ),
    (
        "defun-hyp",
        "User Functions (def)",
        "(def sq (x) (* x x)) (def hyp (a b) (+ (sq a) (sq b))) (hyp 3 4)",
        25,
    ),
    (
        "rec-fact",
        "Recursive Factorial",
        "(def fact (n) (if (< n 2) 1 (* n (fact (- n 1))))) (fact 8)",
        40320,
    ),
    (
        "rec-fib",
        "Recursive Fibonacci",
        "(def f (n) (if (< n 2) n (+ (f (- n 1)) (f (- n 2))))) (f 6)",
        8,
    ),
    (
        "rec-gcd",
        "Euclidean GCD",
        "(def gcd (a b) (if (= b 0) a (gcd b (mod a b)))) (gcd 1071 462)",
        21,
    ),
    (
        "hof-twice",
        "Higher-Order Function",
        "(def inc (x) (+ x 1)) (def twice (f x) (f (f x))) (twice inc 40)",
        42,
    ),
)


def pad_source_codes(src: str, length: int = SOURCE_LEN) -> List[int]:
    """Convert an ASCII Lisp string (len <= 64) into 64 canonical character codes."""
    norm = " ".join(src.lower().replace("\n", " ").replace("\t", " ").split(" "))
    # Preserve exact characters if already <= length, only lowercasing whitespace-normalized newlines
    raw = src.lower().replace("\n", " ").replace("\t", " ")
    if len(raw) > length:
        raise ValueError(f"Source length {len(raw)} exceeds {length}: {raw!r}")
    codes: List[int] = []
    for ch in raw:
        if ch not in CHAR_TO_CODE:
            raise ValueError(f"Unsupported character {ch!r} in {src!r}")
        codes.append(CHAR_TO_CODE[ch])
    codes.extend([0] * (length - len(codes)))
    return codes


# ---------------------------------------------------------------------------
# 3. Reference Lexer & Linearized S-Expression Parser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedProgram:
    kinds: List[int]       # length MAX_TOKENS
    vals: List[int]        # length MAX_TOKENS
    next_expr: List[int]   # length MAX_TOKENS
    ntok: int
    syn_err: int


def lex_and_parse_ref(src: str) -> ParsedProgram:
    """Reference lexer + S-expression next_expr parser matching the CSS pipeline."""
    codes = pad_source_codes(src, SOURCE_LEN)
    raw_tokens: List[Tuple[int, int]] = []
    i = 0
    n = len(codes)
    while i < n:
        c = codes[i]
        if c == 0:
            i += 1
        elif c == 1:
            raw_tokens.append((TOK_LP, 0))
            i += 1
        elif c == 2:
            raw_tokens.append((TOK_RP, 0))
            i += 1
        elif c == 21 and (i + 1 < n and 10 <= codes[i + 1] <= 19) and (
            i == 0 or not (10 <= codes[i - 1] <= 19 or codes[i - 1] >= 30)
        ):
            # Negative integer literal
            i += 1
            val = 0
            digits = 0
            while i < n and 10 <= codes[i] <= 19:
                if digits < 6:
                    val = val * 10 + (codes[i] - 10)
                digits += 1
                i += 1
            raw_tokens.append((TOK_NUM, -val))
        elif 10 <= c <= 19:
            val = 0
            digits = 0
            while i < n and 10 <= codes[i] <= 19:
                if digits < 6:
                    val = val * 10 + (codes[i] - 10)
                digits += 1
                i += 1
            raw_tokens.append((TOK_NUM, val))
        elif c >= 30:
            chars: List[str] = []
            while i < n and codes[i] >= 30:
                chars.append(CODE_TO_CHAR[codes[i]])
                i += 1
            raw_tokens.append((TOK_SYM, encode_symbol("".join(chars))))
        elif c in (26, 27) and (i + 1 < n and codes[i + 1] == 25):
            raw_tokens.append((TOK_SYM, SYM_LE if c == 26 else SYM_GE))
            i += 2
        elif 20 <= c <= 27:
            raw_tokens.append((TOK_SYM, 100 + c))
            i += 1
        else:
            i += 1

    ntok = len(raw_tokens)
    syn_err = 1 if ntok > MAX_TOKENS else 0

    kinds = [0] * MAX_TOKENS
    vals = [0] * MAX_TOKENS
    for idx, (tk, tv) in enumerate(raw_tokens[:MAX_TOKENS]):
        kinds[idx] = tk
        vals[idx] = tv

    # Compute parenthesis depths & syntax check
    depth = 0
    for idx in range(min(ntok, MAX_TOKENS)):
        if kinds[idx] == TOK_LP:
            # Disallow empty () at expression level
            if idx + 1 < MAX_TOKENS and kinds[idx + 1] == TOK_RP:
                # Wait: empty parameter list `()` in `(def seven () 7)` IS valid!
                # Only `()` as head-less call is caught at eval time, so `()` is syntactically balanced.
                pass
            depth += 1
        elif kinds[idx] == TOK_RP:
            depth -= 1
            if depth < 0:
                syn_err = 1
    if depth != 0:
        syn_err = 1

    next_expr = [idx + 1 for idx in range(MAX_TOKENS)]
    if not syn_err:
        stack: List[int] = []
        for idx in range(min(ntok, MAX_TOKENS)):
            if kinds[idx] == TOK_LP:
                stack.append(idx)
            elif kinds[idx] == TOK_RP:
                open_idx = stack.pop()
                next_expr[open_idx] = idx + 1

    return ParsedProgram(kinds=kinds, vals=vals, next_expr=next_expr, ntok=ntok, syn_err=syn_err)


# ---------------------------------------------------------------------------
# 4. Pure-Python Reference Lisp Evaluator (Oracle)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    value: int
    err: int


class _LispRuntimeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"Lisp error {code}")
        self.code = code


def eval_lisp_ref(src: str) -> EvalResult:
    """Evaluate a Lisp program string using standard recursive tree evaluation."""
    parsed = lex_and_parse_ref(src)
    if parsed.syn_err:
        return EvalResult(0, ERR_SYNTAX)
    if parsed.ntok == 0:
        return EvalResult(0, ERR_NONE)

    kinds = parsed.kinds
    vals = parsed.vals
    nx = parsed.next_expr

    env: Dict[int, int] = {}
    # Function table: handle_id -> (param_syms, body_pc)
    funcs: Dict[int, Tuple[List[int], int]] = {}
    fn_by_name: Dict[int, int] = {}

    def trunc_div(a: int, b: int) -> int:
        if b == 0:
            raise _LispRuntimeError(ERR_DIV_ZERO)
        sign = -1 if (a < 0) ^ (b < 0) else 1
        return sign * (abs(a) // abs(b))

    def css_mod(a: int, b: int) -> int:
        if b == 0:
            raise _LispRuntimeError(ERR_DIV_ZERO)
        # CSS `mod(a, abs(b))` with positive divisor matches Python floored `a % abs(b)`
        # Wait: let's match Python / Lisp floored mod `a % b`!
        return a % b

    def eval_at(pc: int, depth: int = 0) -> int:
        if depth > 60:
            raise _LispRuntimeError(ERR_STEP_LIMIT)
        tk = kinds[pc]
        tv = vals[pc]
        if tk == TOK_NUM:
            return tv
        if tk == TOK_SYM:
            if tv in (SYM_T, SYM_TRUE):
                return 1
            if tv in (SYM_NIL, SYM_FALSE):
                return 0
            if tv in env:
                return env[tv]
            if tv in fn_by_name:
                return fn_by_name[tv]
            raise _LispRuntimeError(ERR_UNBOUND)
        if tk != TOK_LP:
            raise _LispRuntimeError(ERR_SYNTAX)

        if kinds[pc + 1] == TOK_RP:
            raise _LispRuntimeError(ERR_CALL)

        head_tk = kinds[pc + 1]
        head_tv = vals[pc + 1]

        # Special forms when head is a symbol
        if head_tk == TOK_SYM:
            if head_tv == SYM_IF:
                c_pc = pc + 2
                if kinds[c_pc] == TOK_RP:
                    raise _LispRuntimeError(ERR_CALL)
                th_pc = nx[c_pc]
                if kinds[th_pc] == TOK_RP:
                    raise _LispRuntimeError(ERR_CALL)
                el_pc = nx[th_pc]
                cond_val = eval_at(c_pc, depth + 1)
                if cond_val != 0:
                    return eval_at(th_pc, depth + 1)
                if kinds[el_pc] == TOK_RP:
                    return 0
                return eval_at(el_pc, depth + 1)

            if head_tv in (SYM_BEGIN, SYM_PROGN):
                cur = pc + 2
                res = 0
                while kinds[cur] != TOK_RP:
                    res = eval_at(cur, depth + 1)
                    cur = nx[cur]
                return res

            if head_tv in (SYM_SET, SYM_SETQ, SYM_DEFINE):
                sym_pc = pc + 2
                val_pc = pc + 3
                if kinds[sym_pc] != TOK_SYM or kinds[val_pc] == TOK_RP:
                    raise _LispRuntimeError(ERR_CALL)
                v = eval_at(val_pc, depth + 1)
                env[vals[sym_pc]] = v
                return v

            if head_tv == SYM_LET:
                bind_lp = pc + 2
                if kinds[bind_lp] != TOK_LP:
                    raise _LispRuntimeError(ERR_CALL)
                body_pc = nx[bind_lp]
                if kinds[body_pc] == TOK_RP:
                    raise _LispRuntimeError(ERR_CALL)
                saved: List[Tuple[int, Optional[int]]] = []
                b_cur = bind_lp + 1
                while kinds[b_cur] != TOK_RP:
                    if kinds[b_cur] != TOK_LP or kinds[b_cur + 1] != TOK_SYM:
                        raise _LispRuntimeError(ERR_CALL)
                    bsym = vals[b_cur + 1]
                    bval = eval_at(b_cur + 2, depth + 1)
                    saved.append((bsym, env.get(bsym)))
                    env[bsym] = bval
                    b_cur = nx[b_cur]
                try:
                    return eval_at(body_pc, depth + 1)
                finally:
                    for bsym, prev in reversed(saved):
                        if prev is None:
                            env.pop(bsym, None)
                        else:
                            env[bsym] = prev

            if head_tv in (SYM_DEF, SYM_DEFUN):
                name_pc = pc + 2
                params_lp = pc + 3
                if kinds[name_pc] != TOK_SYM or kinds[params_lp] != TOK_LP:
                    raise _LispRuntimeError(ERR_CALL)
                fn_sym = vals[name_pc]
                params: List[int] = []
                p_cur = params_lp + 1
                while kinds[p_cur] != TOK_RP:
                    if kinds[p_cur] != TOK_SYM:
                        raise _LispRuntimeError(ERR_CALL)
                    params.append(vals[p_cur])
                    p_cur += 1
                body_pc = nx[params_lp]
                handle = fn_by_name.get(fn_sym, FN_HANDLE_BASE + len(funcs))
                funcs[handle] = (params, body_pc)
                fn_by_name[fn_sym] = handle
                return handle

            if head_tv in (SYM_FN, SYM_LAMBDA):
                params_lp = pc + 2
                if kinds[params_lp] != TOK_LP:
                    raise _LispRuntimeError(ERR_CALL)
                params = []
                p_cur = params_lp + 1
                while kinds[p_cur] != TOK_RP:
                    if kinds[p_cur] != TOK_SYM:
                        raise _LispRuntimeError(ERR_CALL)
                    params.append(vals[p_cur])
                    p_cur += 1
                body_pc = nx[params_lp]
                handle = FN_HANDLE_BASE + len(funcs)
                funcs[handle] = (params, body_pc)
                return handle

            # Built-in arithmetic & comparison operators
            if head_tv in (
                SYM_ADD, SYM_SUB, SYM_MUL, SYM_DIV, SYM_PCT, SYM_MOD,
                SYM_EQ, SYM_LT, SYM_GT, SYM_LE, SYM_GE, SYM_NOT,
            ):
                args: List[int] = []
                a_cur = pc + 2
                while kinds[a_cur] != TOK_RP:
                    args.append(eval_at(a_cur, depth + 1))
                    a_cur = nx[a_cur]
                if head_tv == SYM_ADD:
                    return sum(args)
                if head_tv == SYM_MUL:
                    prod = 1
                    for x in args:
                        prod *= x
                    return prod
                if head_tv == SYM_SUB:
                    if not args:
                        raise _LispRuntimeError(ERR_CALL)
                    if len(args) == 1:
                        return -args[0]
                    res = args[0]
                    for x in args[1:]:
                        res -= x
                    return res
                if head_tv == SYM_NOT:
                    if len(args) != 1:
                        raise _LispRuntimeError(ERR_CALL)
                    return 1 if args[0] == 0 else 0
                if len(args) != 2:
                    raise _LispRuntimeError(ERR_CALL)
                lhs, rhs = args
                if head_tv == SYM_DIV:
                    return trunc_div(lhs, rhs)
                if head_tv in (SYM_PCT, SYM_MOD):
                    return css_mod(lhs, rhs)
                if head_tv == SYM_EQ:
                    return 1 if lhs == rhs else 0
                if head_tv == SYM_LT:
                    return 1 if lhs < rhs else 0
                if head_tv == SYM_GT:
                    return 1 if lhs > rhs else 0
                if head_tv == SYM_LE:
                    return 1 if lhs <= rhs else 0
                if head_tv == SYM_GE:
                    return 1 if lhs >= rhs else 0

        # Otherwise: user-defined function or first-class callable expression
        callee_val = eval_at(pc + 1, depth + 1)
        if callee_val not in funcs:
            raise _LispRuntimeError(ERR_CALL)
        params, body_pc = funcs[callee_val]
        call_args: List[int] = []
        a_cur = pc + 2
        while kinds[a_cur] != TOK_RP:
            call_args.append(eval_at(a_cur, depth + 1))
            a_cur = nx[a_cur]
        if len(call_args) != len(params):
            raise _LispRuntimeError(ERR_CALL)

        saved_call: List[Tuple[int, Optional[int]]] = []
        for psym, aval in zip(params, call_args):
            saved_call.append((psym, env.get(psym)))
        for psym, aval in zip(params, call_args):
            env[psym] = aval
        try:
            return eval_at(body_pc, depth + 1)
        finally:
            for psym, prev in reversed(saved_call):
                if prev is None:
                    env.pop(psym, None)
                else:
                    env[psym] = prev

    try:
        top = 0
        last_val = 0
        while top < parsed.ntok:
            last_val = eval_at(top, 0)
            top = nx[top]
        return EvalResult(last_val, ERR_NONE)
    except _LispRuntimeError as exc:
        return EvalResult(0, exc.code)
