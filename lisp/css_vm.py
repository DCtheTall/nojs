"""
CSS Custom-Property Pipeline Compiler & Cascade Simulator (`lisp/css_vm.py`).

Compiles a zero-JavaScript Lisp interpreter into nested `display: contents`
CSS stages using strictly registered `@property <integer>` wires:
  1. Parallel SIMD Lexer (`lx0`..`lx4`): 64-char ASCII -> signed ints, operators, symbols
  2. Packed Monotone LSB-First Binary Compactor (`cp0`..`cp5`): 64 slots -> 40 dense tokens
     (packs `64 * kind + rank` into a single integer since 2^{s+1} | 64 for s = 0..5!)
  3. Prefix-Depth S-Expression Parser (`ps0`..`ps4`): `next_expr[0..39]` + syntax check
  4. Reusable 5-Stage Shallow-Binding VM (`v0`..`v4` x 92 ticks + `vfin`)
  5. Linear-scan liveness register allocator: colors all temporary wires into `--t0..N`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Set, Tuple

from lisp_ref import (
    ERR_CALL,
    ERR_DIV_ZERO,
    ERR_NONE,
    ERR_STEP_LIMIT,
    ERR_SYNTAX,
    ERR_UNBOUND,
    FN_HANDLE_BASE,
    MAX_TOKENS,
    SOURCE_LEN,
    SYM_ADD,
    SYM_BEGIN,
    SYM_DEF,
    SYM_DEFINE,
    SYM_DEFUN,
    SYM_DIV,
    SYM_EQ,
    SYM_FALSE,
    SYM_FN,
    SYM_GE,
    SYM_GT,
    SYM_IF,
    SYM_LAMBDA,
    SYM_LE,
    SYM_LET,
    SYM_LT,
    SYM_MOD,
    SYM_MUL,
    SYM_NIL,
    SYM_NOT,
    SYM_PCT,
    SYM_PROGN,
    SYM_SET,
    SYM_SETQ,
    SYM_SUB,
    SYM_T,
    SYM_TRUE,
    TOK_LP,
    TOK_NUM,
    TOK_RP,
    TOK_SYM,
    TOKEN_COMPOSER_LEN,
    TOKEN_EXTRA_TIDS,
    TOKEN_LABEL_TO_ID,
    TOKEN_NUM_MIN,
    TOKEN_PALETTE,
    TOKEN_TID_CONSEC_END,
    TOKEN_TID_KW_START,
    TOKEN_TID_NUM_ALL_END,
    TOKEN_TID_NUM_START,
    TOKEN_TID_OP_END,
    TOKEN_TID_OP_START,
    encode_token_composer_slots,
    pad_source_codes,
    tokenize_to_palette_ids,
)


NUM_VARS = 8
NUM_FUNCS = 4
STACK_SIZE = 18
NUM_VM_TICKS = 92

# VM execution modes
MODE_EVAL = 0
MODE_CONT = 1
MODE_DONE = 2
MODE_ERR = 3
MODE_DIV = 4

# Built-in ALU opcodes (1..11)
OP_ADD = 1
OP_SUB = 2
OP_MUL = 3
OP_DIV = 4
OP_MOD = 5
OP_EQ = 6
OP_LT = 7
OP_GT = 8
OP_LE = 9
OP_GE = 10
OP_NOT = 11

# Continuation kinds on stack
CONT_TOP_SEQ = 1
CONT_BEGIN_SEQ = 2
CONT_IF = 3
CONT_SET = 4
CONT_OP_FIRST = 6
CONT_OP_NEXT = 7
CONT_RESTORE = 8
CONT_CALL1 = 9
CONT_CALL2_A = 10
CONT_CALL2_B = 11
CONT_BIND_P1 = 12
CONT_LET1 = 13
CONT_LET2_A = 14


# ---------------------------------------------------------------------------
# 1. Strict <integer> CSS Expression Helpers
# ---------------------------------------------------------------------------

def css_eq(var_expr: str, const_k: int) -> str:
    if const_k == 0:
        return f"(1 - min(1,max({var_expr},0 - {var_expr})))"
    return f"(1 - min(1,max({var_expr} - {const_k},{const_k} - {var_expr})))"


def css_eq_vars(a: str, b: str) -> str:
    return f"(1 - min(1,max({a} - ({b}),({b}) - ({a}))))"


def css_gt(a: str, b: str) -> str:
    if b == "0":
        return f"clamp(0,{a},1)"
    return f"clamp(0,{a} - ({b}),1)"


def css_ge_const(var_expr: str, const_k: int) -> str:
    return f"clamp(0,{var_expr} - {const_k - 1},1)"


def css_le_const(var_expr: str, const_k: int) -> str:
    return f"clamp(0,{const_k + 1} - {var_expr},1)"


def css_in_set(var_expr: str, vals: Sequence[int]) -> str:
    terms = [css_eq(var_expr, v) for v in vals]
    return f"({' + '.join(terms)})" if len(terms) > 1 else terms[0]


def wrap_calc(expr: str) -> str:
    s = expr.strip()
    if s.lstrip("-").isdigit() or (
        s.startswith("var(") and s.endswith(")") and " " not in s and "+" not in s and "*" not in s
    ):
        return s
    if s.startswith(("clamp(", "min(", "max(", "mod(")):
        depth = 0
        top_call = True
        p0 = s.index("(")
        for idx in range(p0, len(s)):
            if s[idx] == "(":
                depth += 1
            elif s[idx] == ")":
                depth -= 1
                if depth == 0 and idx < len(s) - 1:
                    top_call = False
                    break
        if top_call:
            return s
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        matched = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(s) - 1:
                    matched = False
                    break
        if matched:
            s = s[1:-1]
    return f"calc({s})"


# ---------------------------------------------------------------------------
# 2. Liveness-Based CSS Temporary Register Allocator
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"var\((--[a-zA-Z0-9_]+)\)")


def _extract_vars(rhs: str) -> Set[str]:
    return set(_VAR_RE.findall(rhs))


def _allocate_temp_registers(
    stage_defs: Dict[str, List[Tuple[str, str]]],
    one_shot_order: Sequence[str],
    vm_loop_order: Sequence[str],
    keep_names: Set[str],
) -> Dict[str, List[Tuple[str, str]]]:
    def_stage: Dict[str, int] = {}
    last_use: Dict[str, int] = {}
    for st_idx, cls in enumerate(one_shot_order):
        for lhs, rhs in stage_defs[cls]:
            if lhs not in keep_names and not lhs.startswith(("--b", "--tb", "--ovr", "--ptid", "--use_tok")):
                assert lhs not in def_stage, f"Non-SSA temp {lhs} redefined in {cls}"
                def_stage[lhs] = st_idx
                last_use[lhs] = st_idx
            for ref in _extract_vars(rhs):
                if ref in last_use:
                    last_use[ref] = max(last_use[ref], st_idx)

    busy_until: List[int] = []
    rename_map: Dict[str, str] = {}
    for var_name in sorted(def_stage.keys(), key=lambda v: (def_stage[v], v)):
        ds = def_stage[var_name]
        lu = last_use[var_name]
        chosen = -1
        for r_idx, end_st in enumerate(busy_until):
            if end_st < ds:
                chosen = r_idx
                busy_until[r_idx] = lu
                break
        if chosen < 0:
            chosen = len(busy_until)
            busy_until.append(lu)
        rename_map[var_name] = f"--t{chosen}"

    vm_def: Dict[str, int] = {}
    vm_last: Dict[str, int] = {}
    for st_idx, cls in enumerate(vm_loop_order):
        for lhs, rhs in stage_defs[cls]:
            if lhs not in keep_names:
                assert lhs not in vm_def, f"Non-SSA VM temp {lhs} redefined in {cls}"
                vm_def[lhs] = st_idx
                vm_last[lhs] = st_idx
            for ref in _extract_vars(rhs):
                if ref in vm_last:
                    assert st_idx > vm_def[ref], f"VM scratch {ref} read at {st_idx} <= def {vm_def[ref]}"
                    vm_last[ref] = max(vm_last[ref], st_idx)

    vm_busy: List[int] = []
    for var_name in sorted(vm_def.keys(), key=lambda v: (vm_def[v], v)):
        ds = vm_def[var_name]
        lu = vm_last[var_name]
        chosen = -1
        for r_idx, end_st in enumerate(vm_busy):
            if end_st < ds:
                chosen = r_idx
                vm_busy[r_idx] = lu
                break
        if chosen < 0:
            chosen = len(vm_busy)
            vm_busy.append(lu)
        rename_map[var_name] = f"--t{chosen}"

    new_defs: Dict[str, List[Tuple[str, str]]] = {}
    for cls, rules in stage_defs.items():
        new_rules: List[Tuple[str, str]] = []
        for lhs, rhs in rules:
            nlhs = rename_map.get(lhs, lhs)
            nrhs = _VAR_RE.sub(lambda m: f"var({rename_map.get(m.group(1), m.group(1))})", rhs)
            new_rules.append((nlhs, nrhs))
        new_defs[cls] = new_rules
    return new_defs


# ---------------------------------------------------------------------------
# 3. Full CSS Lisp Pipeline Specification Builder
# ---------------------------------------------------------------------------

@dataclass
class LispCSSSpec:
    stage_defs: Dict[str, List[Tuple[str, str]]]
    pipeline_classes: List[str]
    registered_props: List[str]


def build_lisp_css_spec(num_ticks: int = NUM_VM_TICKS) -> LispCSSSpec:
    stage_defs: Dict[str, List[Tuple[str, str]]] = {}

    # -----------------------------------------------------------------------
    # Phase 1: Compact Parallel 64-Char Lexer + 40-Slot 8-Bit Token Composer (lx0 .. lx4)
    # -----------------------------------------------------------------------
    lx0: List[Tuple[str, str]] = []
    char_sc_exprs: List[str] = []
    for i in range(SOURCE_LEN):
        terms = [f"{1 << b}*var(--b{i}_{b})" if b > 0 else f"var(--b{i}_0)" for b in range(6)]
        lx0.append((f"--rch{i}", wrap_calc(" + ".join(terms))))
        # CHAR_COMMENT == 3 == 0b000011
        sc_e = (
            f"var(--b{i}_0)*var(--b{i}_1)*(1 - var(--b{i}_2))"
            f"*(1 - var(--b{i}_3))*(1 - var(--b{i}_4))*(1 - var(--b{i}_5))"
        )
        char_sc_exprs.append(sc_e)
        lx0.append((f"--sc{i}", wrap_calc(sc_e)))
    for g in range(8):
        lx0.append((f"--scb{g}", wrap_calc(" + ".join(char_sc_exprs[8 * g : 8 * g + 8]))))

    # Each token cell inherits `--ptid{i}` from the active preset (or 1 = `;` on Blank),
    # and switches to the user's 8-bit `--tb{i}_{0..7}` selection whenever `--ovr{i} == 1`.
    for i in range(TOKEN_COMPOSER_LEN):
        utid_terms = [f"{1 << b}*var(--tb{i}_{b})" if b > 0 else f"var(--tb{i}_0)" for b in range(8)]
        utid_s = " + ".join(utid_terms)
        lx0.append((
            f"--rtid{i}",
            wrap_calc(f"(1 - var(--ovr{i}))*var(--ptid{i}) + var(--ovr{i})*({utid_s})"),
        ))
    stage_defs["lx0"] = lx0

    # lx0b: Apply ';' masking -> --ch0..63 (live char preview) & compute token ';' flags (--tsc, --tscb)
    lx0b: List[Tuple[str, str]] = []
    for i in range(SOURCE_LEN):
        g, r = divmod(i, 8)
        c_terms = [f"var(--scb{bg})" for bg in range(g)] + [f"var(--sc{8 * g + br})" for br in range(r + 1)]
        not_cm = f"(1 - min(1,{' + '.join(c_terms)}))"
        lx0b.append((f"--ch{i}", wrap_calc(f"var(--rch{i})*{not_cm}")))
    for i in range(TOKEN_COMPOSER_LEN):
        lx0b.append((f"--tsc{i}", wrap_calc(css_eq(f"var(--rtid{i})", 1))))
    for g in range(TOKEN_COMPOSER_LEN // 8):
        lx0b.append((
            f"--tscb{g}",
            wrap_calc(" + ".join(css_eq(f"var(--rtid{8 * g + r})", 1) for r in range(8))),
        ))
    stage_defs["lx0b"] = lx0b

    # lx0c: Digit (--dg) & Alpha (--al) flags + masked visible token IDs (--tvis0..39)
    lx0c: List[Tuple[str, str]] = []
    for i in range(SOURCE_LEN):
        ch = f"var(--ch{i})"
        lx0c.append((f"--dg{i}", wrap_calc(f"{css_ge_const(ch, 10)}*{css_le_const(ch, 19)}")))
        lx0c.append((f"--al{i}", wrap_calc(css_ge_const(ch, 30))))
    for i in range(TOKEN_COMPOSER_LEN):
        g, r = divmod(i, 8)
        tc_terms = [f"var(--tscb{bg})" for bg in range(g)] + [f"var(--tsc{8 * g + br})" for br in range(r + 1)]
        not_tcm = f"(1 - min(1,{' + '.join(tc_terms)}))"
        lx0c.append((f"--tvis{i}", wrap_calc(f"var(--rtid{i})*{not_tcm}")))
    stage_defs["lx0c"] = lx0c

    # lx1: Negative sign (--ng), 6-digit number (--nmag), 4-char symbol (--symv), & 8-bit Token Composer decode
    lx1: List[Tuple[str, str]] = []
    for i in range(SOURCE_LEN):
        ch = f"var(--ch{i})"
        if i + 1 < SOURCE_LEN:
            ch_nxt = f"var(--ch{i + 1})"
            prev_ok = "1" if i == 0 else f"(1 - var(--dg{i - 1}))*(1 - var(--al{i - 1}))"
            lx1.append((f"--ng{i}", wrap_calc(f"{css_eq(ch, 21)}*var(--dg{i + 1})*{prev_ok}")))
            lx1.append((f"--le{i}", wrap_calc(f"{css_eq(ch, 26)}*{css_eq(ch_nxt, 25)}")))
            lx1.append((f"--ge{i}", wrap_calc(f"{css_eq(ch, 27)}*{css_eq(ch_nxt, 25)}")))
        else:
            lx1.append((f"--ng{i}", "0"))
            lx1.append((f"--le{i}", "0"))
            lx1.append((f"--ge{i}", "0"))

        # Unsigned up-to-6-digit value ending at i
        m_exprs = ["1"] + [
            "*".join(f"var(--dg{i - r})" for r in range(1, k + 1)) if i - k >= 0 else "0"
            for k in range(1, 6)
        ]
        num_terms = [f"(var(--ch{i}) - 10)"]
        for k, scale in ((1, 10), (2, 100), (3, 1000), (4, 10000), (5, 100000)):
            if i - k >= 0:
                num_terms.append(f"{scale}*{m_exprs[k]}*(var(--ch{i - k}) - 10)")
        lx1.append((f"--nmag{i}", wrap_calc(" + ".join(num_terms))))

        # 4-letter base-27 symbol ID starting at i
        sym_terms = ["1000", f"19683*(var(--ch{i}) - 29)"]
        if i + 1 < SOURCE_LEN:
            sym_terms.append(f"729*var(--al{i + 1})*(var(--ch{i + 1}) - 29)")
        if i + 2 < SOURCE_LEN:
            c2 = f"var(--al{i + 1})*var(--al{i + 2})"
            sym_terms.append(f"27*{c2}*(var(--ch{i + 2}) - 29)")
            if i + 3 < SOURCE_LEN:
                sym_terms.append(f"{c2}*var(--al{i + 3})*(var(--ch{i + 3}) - 29)")
        lx1.append((f"--symv{i}", wrap_calc(" + ".join(sym_terms))))

    consec_offset = TOKEN_TID_NUM_START - TOKEN_NUM_MIN  # 14
    op_offset = SYM_ADD - TOKEN_TID_OP_START             # 2
    for i in range(TOKEN_COMPOSER_LEN):
        tid = f"var(--tvis{i})"
        is_tlp = css_eq(tid, 2)
        is_trp = css_eq(tid, 3)
        is_tnum = f"({css_ge_const(tid, TOKEN_TID_NUM_START)}*{css_le_const(tid, TOKEN_TID_NUM_ALL_END)})"
        is_tsym = css_ge_const(tid, TOKEN_TID_OP_START)
        lx1.append((f"--tem{i}", wrap_calc(css_ge_const(tid, 2))))
        lx1.append((f"--tak{i}", wrap_calc(f"1*{is_tlp} + 2*{is_trp} + 3*{is_tnum} + 4*{is_tsym}")))

        consec_num = f"({css_ge_const(tid, TOKEN_TID_NUM_START)}*{css_le_const(tid, TOKEN_TID_CONSEC_END)})*({tid} - {consec_offset})"
        mile_terms = [f"{val}*{css_eq(tid, idx)}" for idx, val in TOKEN_EXTRA_TIDS]
        op_sym = f"({css_ge_const(tid, TOKEN_TID_OP_START)}*{css_le_const(tid, TOKEN_TID_OP_END)})*({tid} + {op_offset})"
        kw_terms = [f"{TOKEN_PALETTE[idx][2]}*{css_eq(tid, idx)}" for idx in range(TOKEN_TID_KW_START, len(TOKEN_PALETTE))]
        lx1.append((f"--tav{i}", wrap_calc(" + ".join([consec_num] + mile_terms + [op_sym] + kw_terms))))
    stage_defs["lx1"] = lx1

    # lx2: Emit bit (--em{i}), raw kind (--ak{i}), and signed/symbol value (--av{i})
    lx2: List[Tuple[str, str]] = []
    for i in range(SOURCE_LEN):
        ch = f"var(--ch{i})"
        is_lp = css_eq(ch, 1)
        is_rp = css_eq(ch, 2)
        nxt_not_dg = f"(1 - var(--dg{i + 1}))" if i + 1 < SOURCE_LEN else "1"
        is_num_end = f"(var(--dg{i})*{nxt_not_dg})"
        prv_not_al = f"(1 - var(--al{i - 1}))" if i > 0 else "1"
        is_sym_start = f"(var(--al{i})*{prv_not_al})"

        # Negative sign before start of digit run ending at i
        m_exprs = ["1"] + [
            "*".join(f"var(--dg{i - r})" for r in range(1, k + 1)) if i - k >= 0 else "0"
            for k in range(1, 6)
        ]
        nsg_terms: List[str] = []
        for k in range(1, 7):
            if i - k >= 0:
                cond = (
                    f"(1 - var(--dg{i - 1}))"
                    if k == 1
                    else (f"{m_exprs[k - 1]}*(1 - var(--dg{i - k}))" if k < 6 else m_exprs[5])
                )
                nsg_terms.append(f"{cond}*var(--ng{i - k})")
        nsg_s = f"({' + '.join(nsg_terms)})" if nsg_terms else "0"

        is_le_end = f"var(--le{i - 1})" if i > 0 else "0"
        is_ge_end = f"var(--ge{i - 1})" if i > 0 else "0"

        in_op_range = f"({css_ge_const(ch, 20)}*{css_le_const(ch, 27)})"
        not_compound = f"(1 - var(--ng{i}) - var(--le{i}) - var(--ge{i}) - {is_le_end} - {is_ge_end})"
        is_op1 = f"({in_op_range}*{not_compound})"

        kind_expr = f"1*{is_lp} + 2*{is_rp} + 3*{is_num_end} + 4*({is_sym_start} + {is_le_end} + {is_ge_end} + {is_op1})"
        emit_expr = f"{is_lp} + {is_rp} + {is_num_end} + {is_sym_start} + {is_le_end} + {is_ge_end} + {is_op1}"
        val_expr = (
            f"{is_num_end}*(1 - 2*{nsg_s})*var(--nmag{i}) + {is_sym_start}*var(--symv{i}) "
            f"+ {SYM_LE}*{is_le_end} + {SYM_GE}*{is_ge_end} + {is_op1}*(100 + {ch})"
        )
        if i < TOKEN_COMPOSER_LEN:
            lx2.append((f"--em{i}", wrap_calc(f"(1 - var(--use_tok))*({emit_expr}) + var(--use_tok)*var(--tem{i})")))
            lx2.append((f"--ak{i}", wrap_calc(f"(1 - var(--use_tok))*({kind_expr}) + var(--use_tok)*var(--tak{i})")))
            lx2.append((f"--av{i}", wrap_calc(f"(1 - var(--use_tok))*({val_expr}) + var(--use_tok)*var(--tav{i})")))
        else:
            lx2.append((f"--em{i}", wrap_calc(f"(1 - var(--use_tok))*({emit_expr})")))
            lx2.append((f"--ak{i}", wrap_calc(f"(1 - var(--use_tok))*({kind_expr})")))
            lx2.append((f"--av{i}", wrap_calc(f"(1 - var(--use_tok))*({val_expr})")))
    stage_defs["lx2"] = lx2

    lx3: List[Tuple[str, str]] = [
        (f"--eb{g}", wrap_calc(" + ".join(f"var(--em{8 * g + r})" for r in range(8))))
        for g in range(8)
    ]
    stage_defs["lx3"] = lx3

    # lx4: Pack `64 * kind + rank` into `--c_init_kd{i}` (0 when empty!) + `--ntok`
    lx4: List[Tuple[str, str]] = [("--ntok", wrap_calc(" + ".join(f"var(--eb{g})" for g in range(8))))]
    for i in range(SOURCE_LEN):
        g, r = divmod(i, 8)
        terms = [f"var(--eb{bg})" for bg in range(g)] + [f"var(--em{8 * g + br})" for br in range(r)]
        rank_s = f"({' + '.join(terms)})" if terms else "0"
        lx4.append((f"--ckd_init{i}", wrap_calc(f"64*var(--ak{i}) + var(--em{i})*{rank_s}")))
    stage_defs["lx4"] = lx4

    # -----------------------------------------------------------------------
    # Phase 2: Packed 2-Wire LSB-First Binary Compactor (cp0 .. cp5)
    # -----------------------------------------------------------------------
    for s in range(6):
        d = 1 << s
        mod_m = 1 << (s + 1)  # 2, 4, 8, 16, 32, 64 — all divide 64!
        kd_src = "ckd_init" if s == 0 else f"cp{s - 1}_kd"
        v_src = "av" if s == 0 else f"cp{s - 1}_v"
        kd_dst = f"cp{s}_kd"
        v_dst = f"cp{s}_v"
        cp_rules: List[Tuple[str, str]] = []
        max_slot = MAX_TOKENS if s == 5 else SOURCE_LEN
        for p in range(max_slot):
            act_p = f"min(1,var(--{kd_src}{p}))"
            mov_p = f"clamp(0,mod({p} - var(--{kd_src}{p}),{mod_m}),1)"
            stay = f"({act_p}*(1 - {mov_p}))"
            if p + d < SOURCE_LEN:
                q = p + d
                act_q = f"min(1,var(--{kd_src}{q}))"
                mov_q = f"clamp(0,mod({q} - var(--{kd_src}{q}),{mod_m}),1)"
                from_r = f"({act_q}*{mov_q})"
                kd_next = f"({stay}*var(--{kd_src}{p}) + {from_r}*var(--{kd_src}{q}))"
                v_next = f"{stay}*var(--{v_src}{p}) + {from_r}*var(--{v_src}{q})"
            else:
                kd_next = f"({stay}*var(--{kd_src}{p}))"
                v_next = f"{stay}*var(--{v_src}{p})"

            if s == 5:
                cp_rules.append((f"--cp5_kd{p}", wrap_calc(kd_next)))
                cp_rules.append((f"--tv{p}", wrap_calc(v_next)))
            else:
                cp_rules.append((f"--{kd_dst}{p}", wrap_calc(kd_next)))
                cp_rules.append((f"--{v_dst}{p}", wrap_calc(v_next)))
        stage_defs[f"cp{s}"] = cp_rules

    # Unpack `--tk0..39` from `--cp5_kd0..39` (where ckd == 64*kind + p for active tokens, 0 for empty!)
    cp_unpack: List[Tuple[str, str]] = []
    for p in range(MAX_TOKENS):
        kd = f"var(--cp5_kd{p})"
        cp_unpack.append((
            f"--tk{p}",
            wrap_calc(f"clamp(0,{kd} - 63,1) + clamp(0,{kd} - 127,1) + clamp(0,{kd} - 191,1) + clamp(0,{kd} - 255,1)"),
        ))
    stage_defs["cpu"] = cp_unpack

    # -----------------------------------------------------------------------
    # Phase 3: Prefix-Depth S-Expression Parser + VM Initializer (ps0 .. ps4)
    # -----------------------------------------------------------------------
    ps0: List[Tuple[str, str]] = [
        (f"--dl{i}", wrap_calc(f"{css_eq(f'var(--tk{i})', TOK_LP)} - {css_eq(f'var(--tk{i})', TOK_RP)}"))
        for i in range(MAX_TOKENS)
    ]
    stage_defs["ps0"] = ps0

    ps0b: List[Tuple[str, str]] = [
        (f"--db{g}", wrap_calc(" + ".join(f"var(--dl{8 * g + r})" for r in range(8))))
        for g in range(5)
    ]
    stage_defs["ps0b"] = ps0b

    ps1: List[Tuple[str, str]] = []
    for i in range(MAX_TOKENS):
        g, r = divmod(i, 8)
        before = [f"var(--db{bg})" for bg in range(g)] + [f"var(--dl{8 * g + br})" for br in range(r)]
        ps1.append((f"--dp{i}", wrap_calc(" + ".join(before)) if before else "0"))
        ps1.append((f"--adp{i}", wrap_calc(" + ".join(before + [f"var(--dl{i})"]))))
    stage_defs["ps1"] = ps1

    ps2: List[Tuple[str, str]] = []
    for g in range(5):
        for r in range(8):
            idx = 8 * g + r
            ps2.append((
                f"--pm{idx}",
                f"var(--adp{idx})" if r == 0 else f"min({','.join(f'var(--adp{8 * g + k})' for k in range(r + 1))})",
            ))
    for i in range(MAX_TOKENS):
        g0 = i // 8
        for B in range(g0 + 1, 5):
            args = [f"var(--adp{k})" for k in range(i, min(8 * B, MAX_TOKENS))]
            ps2.append((f"--rm{i}_{B}", args[0] if len(args) == 1 else f"min({','.join(args)})"))
    stage_defs["ps2"] = ps2

    ps3: List[Tuple[str, str]] = []
    min_all = "min(var(--pm7),var(--pm15),var(--pm23),var(--pm31),var(--pm39))"
    neg_depth = f"clamp(0,0 - {min_all},1)"
    unclosed = "min(1,max(var(--adp39),0 - var(--adp39)))"
    tok_ovf = f"clamp(0,var(--ntok) - {MAX_TOKENS},1)"
    ps3.append(("--syn_err", wrap_calc(f"min(1,{neg_depth} + {unclosed} + {tok_ovf})")))

    for i in range(MAX_TOKENS):
        d = f"var(--dp{i})"
        g0 = i // 8
        open_terms: List[str] = []
        for j in range(i + 1, 8 * g0 + 8):
            m_ij = f"var(--adp{i})" if j == i + 1 else f"min({','.join(f'var(--adp{k})' for k in range(i, j))})"
            open_terms.append(f"clamp(0,{m_ij} - {d},1)")
        for B in range(g0 + 1, 5):
            prior_min = f"var(--rm{i}_{B})"
            for r in range(8):
                j = 8 * B + r
                if j >= MAX_TOKENS:
                    break
                m_ij = prior_min if r == 0 else f"min({prior_min},var(--pm{8 * B + r - 1}))"
                open_terms.append(f"clamp(0,{m_ij} - {d},1)")
        if not open_terms:
            ps3.append((f"--tn{i}", str(i + 1)))
        else:
            is_lp = css_eq(f"var(--tk{i})", TOK_LP)
            ps3.append((f"--tn{i}", wrap_calc(f"{i + 1} + {is_lp}*({' + '.join(open_terms)})")))
    stage_defs["ps3"] = ps3

    ps4: List[Tuple[str, str]] = []
    has_err = "var(--syn_err)"
    is_empty = css_eq("var(--ntok)", 0)
    ps4.append(("--pc", "0"))
    ps4.append(("--mode", wrap_calc(f"3*{has_err} + 2*(1 - {has_err})*{is_empty}")))
    ps4.append(("--err", "var(--syn_err)"))
    ps4.append(("--acc", "0"))
    ps4.append(("--steps", "0"))
    has_more_top = f"(1 - {has_err})*(1 - {is_empty})*clamp(0,var(--ntok) - var(--tn0),1)"
    ps4.append(("--sp", wrap_calc(has_more_top)))
    ps4.append(("--sk0", wrap_calc(has_more_top)))
    ps4.append(("--sa0", wrap_calc(f"{has_more_top}*var(--tn0)")))
    ps4.append(("--sb0", "0"))
    for s in range(1, STACK_SIZE):
        ps4.append((f"--sk{s}", "0"))
        ps4.append((f"--sa{s}", "0"))
        ps4.append((f"--sb{s}", "0"))
    for k in range(NUM_VARS):
        ps4.append((f"--vk{k}", "0"))
        ps4.append((f"--vv{k}", "0"))
    for f in range(NUM_FUNCS):
        ps4.append((f"--fk{f}", "0"))
        ps4.append((f"--fp0_{f}", "0"))
        ps4.append((f"--fp1_{f}", "0"))
        ps4.append((f"--fb{f}", "0"))
    ps4.append(("--drem", "0"))
    ps4.append(("--dden", "1"))
    ps4.append(("--dq", "0"))
    ps4.append(("--dstep", "0"))
    ps4.append(("--dsgn", "1"))
    stage_defs["ps4"] = ps4

    # -----------------------------------------------------------------------
    # Phase 4: Streamlined 5-Stage Shallow-Binding VM (.v0 .. .v4)
    # -----------------------------------------------------------------------
    def build_var_atom_lookup(tk_v: str, tv_v: str) -> Tuple[str, str]:
        is_num = css_eq(tk_v, TOK_NUM)
        is_sym = css_eq(tk_v, TOK_SYM)
        is_true = css_in_set(tv_v, (SYM_T, SYM_TRUE))
        is_false = css_in_set(tv_v, (SYM_NIL, SYM_FALSE))
        v_match = " + ".join(f"{css_eq_vars(f'var(--vk{k})', tv_v)}*var(--vv{k})" for k in range(NUM_VARS))
        v_found = f"({' + '.join(css_eq_vars(f'var(--vk{k})', tv_v) for k in range(NUM_VARS))})"
        f_match = " + ".join(f"{css_eq_vars(f'var(--fk{f})', tv_v)}*{FN_HANDLE_BASE + f}" for f in range(NUM_FUNCS))
        f_found = f"({' + '.join(css_eq_vars(f'var(--fk{f})', tv_v) for f in range(NUM_FUNCS))})"
        builtin_f = f"({is_true} + {is_false})"
        ok_e = f"{is_num} + {is_sym}*min(1,{builtin_f} + {v_found} + {f_found})"
        val_e = f"{is_num}*{tv_v} + {is_sym}*({is_true} + (1 - {builtin_f})*({v_match} + (1 - {v_found})*({f_match})))"
        return wrap_calc(ok_e), wrap_calc(val_e)

    # .v0: One-hot PC (--ep0..39), early head fetch (--q_tk1, --q_tv1), Stack Top/Next, Div D, & lambda free slot!
    v0: List[Tuple[str, str]] = []
    for i in range(MAX_TOKENS):
        v0.append((f"--ep{i}", wrap_calc(css_eq("var(--pc)", i))))
    for off in (0, 1, 3):
        v0.append((f"--q_tk{off}", wrap_calc(" + ".join(f"{css_eq('var(--pc)', j - off)}*var(--tk{j})" for j in range(off, MAX_TOKENS)))))
        v0.append((f"--q_tv{off}", wrap_calc(" + ".join(f"{css_eq('var(--pc)', j - off)}*var(--tv{j})" for j in range(off, MAX_TOKENS)))))

    for prefix, offset in (("top", 1), ("nxt", 2)):
        for field in ("k", "a", "b"):
            terms = [f"{css_eq('var(--sp)', s + offset)}*var(--s{field}{s})" for s in range(STACK_SIZE)]
            v0.append((f"--{prefix}_{field}", wrap_calc(" + ".join(terms))))
    d_terms = [
        f"{css_eq('var(--dstep)', st)}*min(var(--dden),{cap})*{sc}"
        for st, (sc, cap) in enumerate(((1, 1048576), (16, 65536), (256, 4096), (4096, 256), (65536, 16)))
    ]
    v0.append(("--div_D", wrap_calc(" + ".join(d_terms))))

    # Lambda free slot & var free slot only depend on canonical parent fk*/vk* -> compute right in v0!
    lam_emp_any: List[str] = []
    lam_emp_idx: List[str] = []
    for f in range(NUM_FUNCS):
        prev_nz = "*".join(f"min(1,max(var(--fk{p}),0 - var(--fk{p})))" for p in range(f))
        cond = css_eq(f"var(--fk{f})", 0) if f == 0 else f"{prev_nz}*{css_eq(f'var(--fk{f})', 0)}"
        lam_emp_any.append(cond)
        if f > 0:
            lam_emp_idx.append(f"{f}*{cond}")
    v0.append(("--lam_ok", wrap_calc(" + ".join(lam_emp_any))))
    v0.append(("--lam_slot", wrap_calc(" + ".join(lam_emp_idx)) if lam_emp_idx else "0"))

    v_emp_any: List[str] = []
    v_emp_idx: List[str] = []
    for k in range(NUM_VARS):
        prev_nz = "*".join(f"min(1,var(--vk{p}))" for p in range(k))
        cond = css_eq(f"var(--vk{k})", 0) if k == 0 else f"{prev_nz}*{css_eq(f'var(--vk{k})', 0)}"
        v_emp_any.append(cond)
        if k > 0:
            v_emp_idx.append(f"{k}*{cond}")
    v0.append(("--v_has_free", wrap_calc(" + ".join(v_emp_any))))
    v0.append(("--v_free_idx", wrap_calc(" + ".join(v_emp_idx)) if v_emp_idx else "0"))
    stage_defs["v0"] = v0

    # .v1: Remaining token window, head callee lookup (--aok1, --aval1), op1, CONT_RESTORE fold, stack-fn metadata, div_dig
    v1: List[Tuple[str, str]] = []

    def tok_win(bus: str, offset: int, default_oob: int = 0) -> str:
        terms = [f"var(--ep{j - offset})*var(--{bus}{j})" for j in range(offset, MAX_TOKENS)]
        if default_oob != 0:
            for i in range(MAX_TOKENS - offset, MAX_TOKENS):
                terms.append(f"var(--ep{i})*{default_oob}")
        return wrap_calc(" + ".join(terms))

    for off in (2, 4, 5, 6):
        v1.append((f"--q_tk{off}", tok_win("tk", off, 0)))
    for off in (2, 4, 5):
        v1.append((f"--q_tv{off}", tok_win("tv", off, 0)))
    for off in (0, 2, 3, 7):
        v1.append((f"--q_tn{off}", tok_win("tn", off, MAX_TOKENS)))

    for off in (0, 1):
        aok_e, aval_e = build_var_atom_lookup(f"var(--q_tk{off})", f"var(--q_tv{off})")
        v1.append((f"--aok{off}", aok_e))
        v1.append((f"--aval{off}", aval_e))

    op_pairs = [
        (SYM_ADD, OP_ADD), (SYM_SUB, OP_SUB), (SYM_MUL, OP_MUL), (SYM_DIV, OP_DIV),
        (SYM_PCT, OP_MOD), (SYM_MOD, OP_MOD), (SYM_EQ, OP_EQ), (SYM_LT, OP_LT),
        (SYM_GT, OP_GT), (SYM_LE, OP_LE), (SYM_GE, OP_GE), (SYM_NOT, OP_NOT),
    ]
    for off in (1, 3):
        v1.append((
            f"--op{off}",
            wrap_calc(f"{css_eq(f'var(--q_tk{off})', TOK_SYM)}*({' + '.join(f'{c}*{css_eq(f'var(--q_tv{off})', s)}' for s, c in op_pairs)})"),
        ))

    sp_ge1 = css_ge_const("var(--sp)", 1)
    top_is_rest = css_eq("var(--top_k)", CONT_RESTORE)
    sp_eq1 = css_eq("var(--sp)", 1)
    nxt_foldable = css_in_set("var(--nxt_k)", (CONT_TOP_SEQ, CONT_BEGIN_SEQ, CONT_IF, CONT_OP_FIRST, CONT_OP_NEXT))
    fold_expr = f"({sp_ge1}*{top_is_rest}*min(1,{sp_eq1} + {nxt_foldable}))"
    v1.append(("--fold_r", wrap_calc(fold_expr)))
    for field in ("k", "a", "b"):
        v1.append((
            f"--e{field}",
            wrap_calc(f"var(--top_{field}) + {fold_expr}*(var(--nxt_{field}) - var(--top_{field}))"),
        ))

    for f_prop, arr_p in (("p0", "fp0_"), ("p1", "fp1_"), ("fb", "fb")):
        v1.append((
            f"--tf_{f_prop}",
            wrap_calc(" + ".join(f"{css_eq('var(--top_a)', f)}*var(--{arr_p}{f})" for f in range(NUM_FUNCS))),
        ))
    top_hi = "(var(--top_a) - mod(var(--top_a),64))"
    v1.append(("--top_fidx", wrap_calc(" + ".join(f"{f}*{css_eq(top_hi, 64 * f)}" for f in range(NUM_FUNCS)))))
    v1.append(("--div_dig", wrap_calc(" + ".join(f"clamp(0,var(--drem) - {k}*var(--div_D) + 1,1)" for k in range(1, 16)))))
    stage_defs["v1"] = v1

    # .v2: Atom lookups (0, 2..5), op3, dynamic AST tokens, callee metadata (--cf_*), def_slot, AND form flags!
    v2: List[Tuple[str, str]] = []
    eidx_expr = "mod(var(--ea),64)"
    for out_name, bus, shift, oob in (
        ("--etk", "tk", 0, 0),
        ("--etv1", "tv", 1, 0),
        ("--etn", "tn", 0, MAX_TOKENS),
    ):
        terms = []
        for i in range(MAX_TOKENS):
            val_ref = f"var(--{bus}{i + shift})" if (i + shift < MAX_TOKENS) else str(oob)
            terms.append(f"{css_eq(eidx_expr, i)}*{val_ref}")
        v2.append((out_name, wrap_calc(" + ".join(terms))))

    ea_hi = f"(var(--ea) - {eidx_expr})"
    v2.append(("--c_op", wrap_calc(" + ".join(f"{code}*{css_eq(ea_hi, 64 * code)}" for code in range(1, 12)))))
    v2.append(("--eidx", eidx_expr))

    for off in (2, 3, 4, 5):
        ok_e, val_e = build_var_atom_lookup(f"var(--q_tk{off})", f"var(--q_tv{off})")
        v2.append((f"--aok{off}", ok_e))
        v2.append((f"--aval{off}", val_e))

    # Materialize callee metadata right in v2 (since --aval1 and --aok1 were computed in v1!)
    is_fn_h = f"({' + '.join(css_eq('var(--aval1)', FN_HANDLE_BASE + f) for f in range(NUM_FUNCS))})"
    v2.append(("--is_fn_h", wrap_calc(is_fn_h)))
    v2.append(("--c_fidx", wrap_calc(" + ".join(f"{f}*{css_eq('var(--aval1)', FN_HANDLE_BASE + f)}" for f in range(NUM_FUNCS)))))
    for f_prop, arr_p in (("p0", "fp0_"), ("p1", "fp1_"), ("fb", "fb")):
        v2.append((
            f"--cf_{f_prop}",
            wrap_calc(" + ".join(f"{css_eq('var(--aval1)', FN_HANDLE_BASE + f)}*var(--{arr_p}{f})" for f in range(NUM_FUNCS))),
        ))

    # Def slot (matches q_tv2 in fk* or uses lam_ok/lam_slot from v0!)
    has_m = f"({' + '.join(css_eq_vars(f'var(--fk{f})', 'var(--q_tv2)') for f in range(NUM_FUNCS))})"
    m_slot = " + ".join(f"{f}*{css_eq_vars(f'var(--fk{f})', 'var(--q_tv2)')}" for f in range(NUM_FUNCS))
    def_ok_s = f"min(1,{has_m} + var(--lam_ok))"
    def_slot_s = f"(({m_slot}) + (1 - {has_m})*var(--lam_slot))"
    v2.append(("--def_ok", wrap_calc(def_ok_s)))
    v2.append(("--def_slot", wrap_calc(def_slot_s)))

    # Materialize special-form head flags AND fast23 / fast45 / rc_ek / cont_in right in v2!
    m_ev_v2 = css_eq("var(--mode)", MODE_EVAL)
    m_ct_v2 = css_eq("var(--mode)", MODE_CONT)
    lp0_v2 = css_eq("var(--q_tk0)", TOK_LP)
    at0_v2 = css_in_set("var(--q_tk0)", (TOK_NUM, TOK_SYM))
    sym1_v2 = css_eq("var(--q_tk1)", TOK_SYM)
    rp1_v2 = css_eq("var(--q_tk1)", TOK_RP)
    rp2_v2 = css_eq("var(--q_tk2)", TOK_RP)
    lp2_v2 = css_eq("var(--q_tk2)", TOK_LP)
    sym2_v2 = css_eq("var(--q_tk2)", TOK_SYM)
    at2_v2 = css_in_set("var(--q_tk2)", (TOK_NUM, TOK_SYM))
    at3_v2 = css_in_set("var(--q_tk3)", (TOK_NUM, TOK_SYM))
    lp3_v2 = css_eq("var(--q_tk3)", TOK_LP)
    rp3_v2 = css_eq("var(--q_tk3)", TOK_RP)
    sym3_v2 = css_eq("var(--q_tk3)", TOK_SYM)
    rp4_v2 = css_eq("var(--q_tk4)", TOK_RP)
    sym4_v2 = css_eq("var(--q_tk4)", TOK_SYM)
    at4_v2 = css_in_set("var(--q_tk4)", (TOK_NUM, TOK_SYM))
    rp5_v2 = css_eq("var(--q_tk5)", TOK_RP)
    sym5_v2 = css_eq("var(--q_tk5)", TOK_SYM)
    at5_v2 = css_in_set("var(--q_tk5)", (TOK_NUM, TOK_SYM))
    rp6_v2 = css_eq("var(--q_tk6)", TOK_RP)

    has_op1_v2 = "clamp(0,var(--op1),1)"
    head_kw_any = css_in_set(
        "var(--q_tv1)",
        (SYM_IF, SYM_BEGIN, SYM_PROGN, SYM_SET, SYM_SETQ, SYM_DEFINE, SYM_LET, SYM_DEF, SYM_DEFUN, SYM_FN, SYM_LAMBDA),
    )
    f_def_s = f"({m_ev_v2}*{lp0_v2}*{sym1_v2}*{css_in_set('var(--q_tv1)', (SYM_DEF, SYM_DEFUN))})"
    f_lam_s = f"({m_ev_v2}*{lp0_v2}*{sym1_v2}*{css_in_set('var(--q_tv1)', (SYM_FN, SYM_LAMBDA))})"
    f_bop_s = f"({m_ev_v2}*{lp0_v2}*{has_op1_v2})"

    v2.append(("--f_if", wrap_calc(f"{m_ev_v2}*{lp0_v2}*{sym1_v2}*{css_eq('var(--q_tv1)', SYM_IF)}")))
    v2.append(("--f_beg", wrap_calc(f"{m_ev_v2}*{lp0_v2}*{sym1_v2}*{css_in_set('var(--q_tv1)', (SYM_BEGIN, SYM_PROGN))}")))
    v2.append(("--f_set", wrap_calc(f"{m_ev_v2}*{lp0_v2}*{sym1_v2}*{css_in_set('var(--q_tv1)', (SYM_SET, SYM_SETQ, SYM_DEFINE))}")))
    v2.append(("--f_let", wrap_calc(f"{m_ev_v2}*{lp0_v2}*{sym1_v2}*{css_eq('var(--q_tv1)', SYM_LET)}")))
    v2.append(("--f_def", wrap_calc(f_def_s)))
    v2.append(("--f_lam", wrap_calc(f_lam_s)))
    v2.append(("--f_bop", wrap_calc(f_bop_s)))
    v2.append(("--f_call", wrap_calc(f"{m_ev_v2}*{lp0_v2}*(1 - {rp1_v2})*(1 - {has_op1_v2} - {sym1_v2}*{head_kw_any})")))

    op1_fast_v2 = css_in_set("var(--op1)", (OP_ADD, OP_SUB, OP_MUL, OP_MOD, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE))
    op3_fast_v2 = css_in_set("var(--op3)", (OP_ADD, OP_SUB, OP_MUL, OP_MOD, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE))
    v2.append(("--fast23", wrap_calc(f"{f_bop_s}*{op1_fast_v2}*{at2_v2}*{at3_v2}*{rp4_v2}")))
    v2.append(("--fast45", wrap_calc(f"{lp2_v2}*{op3_fast_v2}*{at4_v2}*{at5_v2}*{rp6_v2}")))

    def_v_v2 = f"({sym2_v2}*{lp3_v2}*({def_ok_s})*({rp4_v2} + {sym4_v2}*{rp5_v2} + {sym4_v2}*{sym5_v2}*{rp6_v2}))"
    lam_v_v2 = f"({lp2_v2}*var(--lam_ok)*({rp3_v2} + {sym3_v2}*{rp4_v2} + {sym3_v2}*{sym4_v2}*{rp5_v2}))"
    null_am_v2 = f"({f_bop_s}*{css_in_set('var(--op1)', (OP_ADD, OP_MUL))}*{rp2_v2})"
    imm_at_v2 = f"({m_ev_v2}*{at0_v2}*var(--aok0))"
    imm_df_v2 = f"({f_def_s}*{def_v_v2})"
    imm_lm_v2 = f"({f_lam_s}*{lam_v_v2})"
    rc_v2 = f"({m_ct_v2} + {imm_at_v2} + {null_am_v2} + {imm_df_v2} + {imm_lm_v2})"
    sp_af_v2 = "(var(--sp) - var(--fold_r))"
    v2.append(("--rc_s", wrap_calc(rc_v2)))
    v2.append(("--a_done", wrap_calc(f"{rc_v2}*{css_eq(sp_af_v2, 0)}")))
    v2.append(("--rc_ek", wrap_calc(f"{rc_v2}*clamp(0,{sp_af_v2},1)")))
    v2.append((
        "--cont_in",
        wrap_calc(
            f"{m_ct_v2}*var(--acc) + {imm_at_v2}*var(--aval0) + {null_am_v2}*{css_eq('var(--op1)', OP_MUL)} "
            f"+ {imm_df_v2}*({FN_HANDLE_BASE} + {def_slot_s}) + {imm_lm_v2}*({FN_HANDLE_BASE} + var(--lam_slot))"
        ),
    ))
    stage_defs["v2"] = v2

    # -----------------------------------------------------------------------
    # .v2b -> Wait: can we avoid adding a stage by keeping v3 concise using --cf_p0/p1 from v2?!
    # YES!! Look why v3 hit 52 KB earlier: ONLY because `cf_p0`, `cf_p1`, `is_fn_h`, `c_fidx` were inline strings,
    # AND `ek_any_s` summed 12 `ek_*` strings!
    # Now in v3:
    #   - `cf_p0` is `var(--cf_p0)` (11 chars instead of 280 chars!)
    #   - `cf_p1` is `var(--cf_p1)` (11 chars instead of 280 chars!)
    #   - `is_fn_h` is `var(--is_fn_h)` (13 chars instead of 240 chars!)
    #   - `ek_any_s` is `rc_has_ek * (1 - (1 - var(--fold_r)) * [top_k == 8])` (1 copy of `rc_has_ek` instead of 12 copies!!)
    # Let's check exact max rule size of v3 with those two changes: < 2,800 chars, ~28 KB total!!
    # -----------------------------------------------------------------------
    v3: List[Tuple[str, str]] = []
    for name in ("pc", "mode", "acc", "err", "steps", "sp", "drem", "dden", "dq", "dstep", "dsgn"):
        v3.append((f"--l_{name}", f"var(--{name})"))
    for k in range(NUM_VARS):
        v3.append((f"--l_vk{k}", f"var(--vk{k})"))
        v3.append((f"--l_vv{k}", f"var(--vv{k})"))
    for f in range(NUM_FUNCS):
        v3.append((f"--l_fk{f}", f"var(--fk{f})"))
        v3.append((f"--l_fp0_{f}", f"var(--fp0_{f})"))
        v3.append((f"--l_fp1_{f}", f"var(--fp1_{f})"))
        v3.append((f"--l_fb{f}", f"var(--fb{f})"))
    for s in range(STACK_SIZE):
        v3.append((f"--l_sk{s}", f"var(--sk{s})"))
        v3.append((f"--l_sa{s}", f"var(--sa{s})"))
        v3.append((f"--l_sb{s}", f"var(--sb{s})"))

    def alu_expr(op_v: str, lhs_v: str, rhs_v: str) -> Tuple[str, str]:
        rz = css_eq(rhs_v, 0)
        g_rhs = f"({rhs_v} + {rz})"
        terms = [
            f"{css_eq(op_v, OP_ADD)}*({lhs_v} + {rhs_v})",
            f"{css_eq(op_v, OP_SUB)}*({lhs_v} - {rhs_v})",
            f"{css_eq(op_v, OP_MUL)}*({lhs_v}*{rhs_v})",
            f"{css_eq(op_v, OP_MOD)}*mod({lhs_v},{g_rhs})",
            f"{css_eq(op_v, OP_EQ)}*{css_eq_vars(lhs_v, rhs_v)}",
            f"{css_eq(op_v, OP_LT)}*{css_gt(rhs_v, lhs_v)}",
            f"{css_eq(op_v, OP_GT)}*{css_gt(lhs_v, rhs_v)}",
            f"{css_eq(op_v, OP_LE)}*(1 - {css_gt(lhs_v, rhs_v)})",
            f"{css_eq(op_v, OP_GE)}*(1 - {css_gt(rhs_v, lhs_v)})",
        ]
        dz = f"{css_in_set(op_v, (OP_DIV, OP_MOD))}*{rz}"
        return f"({' + '.join(terms)})", f"({dz})"

    alu23_s, dz23_s = alu_expr("var(--op1)", "var(--aval2)", "var(--aval3)")
    alu45_s, dz45_s = alu_expr("var(--op3)", "var(--aval4)", "var(--aval5)")
    v3.append(("--alu23", wrap_calc(alu23_s)))
    v3.append(("--dz23", wrap_calc(dz23_s)))
    v3.append(("--alu45", wrap_calc(alu45_s)))
    v3.append(("--dz45", wrap_calc(dz45_s)))

    m_eval = css_eq("var(--mode)", MODE_EVAL)
    m_cont = css_eq("var(--mode)", MODE_CONT)
    k0_atom = css_in_set("var(--q_tk0)", (TOK_NUM, TOK_SYM))
    k0_lp = css_eq("var(--q_tk0)", TOK_LP)
    k1_atom = css_in_set("var(--q_tk1)", (TOK_NUM, TOK_SYM))
    k2_rp = css_eq("var(--q_tk2)", TOK_RP)
    k2_lp = css_eq("var(--q_tk2)", TOK_LP)
    k2_sym = css_eq("var(--q_tk2)", TOK_SYM)
    k2_atom = css_in_set("var(--q_tk2)", (TOK_NUM, TOK_SYM))
    k3_atom = css_in_set("var(--q_tk3)", (TOK_NUM, TOK_SYM))
    k3_lp = css_eq("var(--q_tk3)", TOK_LP)
    k3_rp = css_eq("var(--q_tk3)", TOK_RP)
    k3_sym = css_eq("var(--q_tk3)", TOK_SYM)
    k4_rp = css_eq("var(--q_tk4)", TOK_RP)
    k4_sym = css_eq("var(--q_tk4)", TOK_SYM)
    k4_atom = css_in_set("var(--q_tk4)", (TOK_NUM, TOK_SYM))
    k5_rp = css_eq("var(--q_tk5)", TOK_RP)
    k5_sym = css_eq("var(--q_tk5)", TOK_SYM)
    k5_atom = css_in_set("var(--q_tk5)", (TOK_NUM, TOK_SYM))
    k6_rp = css_eq("var(--q_tk6)", TOK_RP)

    fast23 = "var(--fast23)"
    fast45 = "var(--fast45)"

    def lookup_existing_var(sym_expr: str) -> Tuple[str, str]:
        nz = f"clamp(0,{sym_expr},1)"
        had = f"{nz}*({' + '.join(css_eq_vars(f'var(--vk{k})', sym_expr) for k in range(NUM_VARS))})"
        oldv = f"{nz}*({' + '.join(f'{css_eq_vars(f'var(--vk{k})', sym_expr)}*var(--vv{k})' for k in range(NUM_VARS))})"
        return f"({had})", f"({oldv})"

    cf0_had, cf0_old = lookup_existing_var("var(--cf_p0)")
    tf0_had, tf0_old = lookup_existing_var("var(--tf_p0)")
    tf1_had, tf1_old = lookup_existing_var("var(--tf_p1)")
    let_had, let_old = lookup_existing_var("var(--top_b)")

    def_p0_ok = f"{k4_rp}"
    def_p1_ok = f"({k4_sym}*{k5_rp})"
    def_p2_ok = f"({k4_sym}*{k5_sym}*{k6_rp})"
    def_valid = f"({k2_sym}*{k3_lp}*var(--def_ok)*({def_p0_ok} + {def_p1_ok} + {def_p2_ok}))"

    lam_p0_ok = f"{k3_rp}"
    lam_p1_ok = f"({k3_sym}*{k4_rp})"
    lam_p2_ok = f"({k3_sym}*{k4_sym}*{k5_rp})"
    lam_valid = f"({k2_lp}*var(--lam_ok)*({lam_p0_ok} + {lam_p1_ok} + {lam_p2_ok}))"

    is_nullary_am = f"(var(--f_bop)*{css_in_set('var(--op1)', (OP_ADD, OP_MUL))}*{k2_rp})"
    imm_atom_err = f"({m_eval}*{k0_atom}*(1 - var(--aok0)))"
    imm_def_ok = f"(var(--f_def)*{def_valid})"
    imm_def_err = f"(var(--f_def)*(1 - {def_valid}))"
    imm_lam_ok = f"(var(--f_lam)*{lam_valid})"
    imm_lam_err = f"(var(--f_lam)*(1 - {lam_valid}))"

    rc_s = "var(--rc_s)"
    cin_s = "var(--cont_in)"
    calu_s, cdz_s = alu_expr("var(--c_op)", "var(--eb)", "var(--cont_in)")
    v3.append(("--calu", wrap_calc(calu_s)))
    v3.append(("--cdz", wrap_calc(cdz_s)))

    # Function write signals
    v3.append(("--fn_we", wrap_calc(f"{imm_def_ok} + {imm_lam_ok}")))
    v3.append(("--fn_idx", wrap_calc(f"{imm_def_ok}*var(--def_slot) + {imm_lam_ok}*var(--lam_slot)")))
    v3.append(("--fn_k", wrap_calc(f"{imm_def_ok}*var(--q_tv2) + {imm_lam_ok}*({FN_HANDLE_BASE} + var(--lam_slot))")))
    v3.append(("--fn_p0", wrap_calc(f"{imm_def_ok}*{k4_sym}*var(--q_tv4) + {imm_lam_ok}*{k3_sym}*var(--q_tv3)")))
    v3.append(("--fn_p1", wrap_calc(f"{imm_def_ok}*{def_p2_ok}*var(--q_tv5) + {imm_lam_ok}*{lam_p2_ok}*var(--q_tv4)")))
    v3.append(("--fn_fb", wrap_calc(f"{imm_def_ok}*var(--q_tn3) + {imm_lam_ok}*var(--q_tn2)")))

    rc_done = "var(--a_done)"
    rc_has_ek = "var(--rc_ek)"
    is_unfolded_rest = f"((1 - var(--fold_r))*{css_eq('var(--top_k)', CONT_RESTORE)})"

    ek_is = {
        c: css_eq("var(--ek)", c)
        for c in (
            CONT_TOP_SEQ, CONT_BEGIN_SEQ, CONT_IF, CONT_SET, CONT_OP_FIRST, CONT_OP_NEXT,
            CONT_CALL1, CONT_CALL2_A, CONT_CALL2_B, CONT_BIND_P1, CONT_LET1, CONT_LET2_A,
        )
    }

    # Non-immediate MODE_EVAL branches (now using short `var(--cf_p0)`, `var(--cf_p1)`, `var(--is_fn_h)` from v2!)
    f23_unbound = f"({fast23}*(1 - var(--aok2)*var(--aok3)))"
    f23_ok = f"({fast23}*var(--aok2)*var(--aok3))"

    tn2_not_end = f"(1 - {css_eq_vars('var(--q_tn2)', 'var(--q_tn0) - 1')})"
    if_ok_head = f"(var(--f_if)*(1 - {k2_rp})*{tn2_not_end})"
    if_err = f"(var(--f_if)*(1 - (1 - {k2_rp})*{tn2_not_end}))"
    if_f45_unb = f"({if_ok_head}*{fast45}*(1 - var(--aok4)*var(--aok5)))"
    if_f45_ok = f"({if_ok_head}*{fast45}*var(--aok4)*var(--aok5))"
    if_norm = f"({if_ok_head}*(1 - {fast45}))"

    beg_empty = f"(var(--f_beg)*{k2_rp})"
    beg_single = f"(var(--f_beg)*(1 - {k2_rp})*(1 - {tn2_not_end}))"
    beg_multi = f"(var(--f_beg)*(1 - {k2_rp})*{tn2_not_end})"

    set_ok_cond = f"({k2_sym}*(1 - {k3_rp})*{css_eq_vars('var(--q_tn3)', 'var(--q_tn0) - 1')})"
    set_valid = f"(var(--f_set)*{set_ok_cond})"
    set_err = f"(var(--f_set)*(1 - {set_ok_cond}))"

    let_0 = f"(var(--f_let)*{k2_lp}*{k3_rp}*{tn2_not_end})"
    let_1 = f"(var(--f_let)*{k2_lp}*{k3_lp}*{k4_sym}*(1 - {k5_rp})*{css_eq_vars('var(--q_tn3)', 'var(--q_tn2) - 1')}*{tn2_not_end})"
    let_2 = f"(var(--f_let)*{k2_lp}*{k3_lp}*{k4_sym}*(1 - {k5_rp})*clamp(0,var(--q_tn2) - 1 - var(--q_tn3),1)*{tn2_not_end})"
    let_err = f"(var(--f_let)*(1 - {k2_lp}*{tn2_not_end}*({k3_rp} + {k3_lp}*{k4_sym}*(1 - {k5_rp}))))"

    bop_rem = f"(var(--f_bop)*(1 - {is_nullary_am})*(1 - {fast23}))"
    bop_empty_err = f"({bop_rem}*{k2_rp})"
    bop_fast_a1 = f"({bop_rem}*(1 - {k2_rp})*(1 - {css_eq('var(--op1)', OP_NOT)})*{k2_atom}*{tn2_not_end})"
    bop_fa1_unb = f"({bop_fast_a1}*(1 - var(--aok2)))"
    bop_fa1_ok = f"({bop_fast_a1}*var(--aok2))"
    bop_gen = f"({bop_rem}*(1 - {k2_rp})*(1 - {bop_fast_a1}))"

    call_unbound = f"(var(--f_call)*{k1_atom}*(1 - var(--aok1)))"
    call_not_fn = f"(var(--f_call)*((1 - {k1_atom}) + {k1_atom}*var(--aok1)*(1 - var(--is_fn_h))))"
    call_fn_ok = f"(var(--f_call)*{k1_atom}*var(--aok1)*var(--is_fn_h))"
    fn_arity0 = css_eq("var(--cf_p0)", 0)
    fn_arity1 = f"(clamp(0,var(--cf_p0),1)*{css_eq('var(--cf_p1)', 0)})"
    fn_arity2 = "(clamp(0,var(--cf_p0),1)*clamp(0,var(--cf_p1),1))"

    call0_ok = f"({call_fn_ok}*{fn_arity0}*{k2_rp})"
    call0_err = f"({call_fn_ok}*{fn_arity0}*(1 - {k2_rp}))"

    arg1_exact = f"((1 - {k2_rp})*(1 - {tn2_not_end}))"
    call1_err = f"({call_fn_ok}*{fn_arity1}*(1 - {arg1_exact}))"
    call1_atom_unb = f"({call_fn_ok}*{fn_arity1}*{arg1_exact}*{k2_atom}*(1 - var(--aok2)))"
    call1_atom_ok = f"({call_fn_ok}*{fn_arity1}*{arg1_exact}*{k2_atom}*var(--aok2))"
    call1_f45_unb = f"({call_fn_ok}*{fn_arity1}*{arg1_exact}*{fast45}*(1 - var(--aok4)*var(--aok5)))"
    call1_f45_ok = f"({call_fn_ok}*{fn_arity1}*{arg1_exact}*{fast45}*var(--aok4)*var(--aok5))"
    call1_gen = f"({call_fn_ok}*{fn_arity1}*{arg1_exact}*(1 - {k2_atom})*(1 - {fast45}))"

    arg2_has2 = f"((1 - {k2_rp})*clamp(0,var(--q_tn0) - 1 - var(--q_tn2),1))"
    call2_ok = f"({call_fn_ok}*{fn_arity2}*{arg2_has2})"
    call2_err = f"({call_fn_ok}*{fn_arity2}*(1 - {arg2_has2}))"

    empty_list_err = f"({m_eval}*{k0_lp}*{css_eq('var(--q_tk1)', TOK_RP)})"
    unexpected_rp_err = f"({m_eval}*{css_eq('var(--q_tk0)', TOK_RP)})"

    # Emit compact action wires for v4 (factoring `rc_has_ek` cleanly!)
    v3.append(("--a_bemp", wrap_calc(beg_empty)))
    v3.append(("--a_ek_if", wrap_calc(f"{rc_has_ek}*{ek_is[CONT_IF]}")))
    v3.append(("--a_ek_set", wrap_calc(f"{rc_has_ek}*{ek_is[CONT_SET]}")))
    v3.append(("--a_ek_op1", wrap_calc(f"{rc_has_ek}*{ek_is[CONT_OP_FIRST]}")))
    v3.append(("--a_ek_opn", wrap_calc(f"{rc_has_ek}*{ek_is[CONT_OP_NEXT]}")))
    v3.append(("--a_ek_rest", wrap_calc(f"{rc_has_ek}*{is_unfolded_rest}")))
    v3.append(("--a_ek_c2b", wrap_calc(f"{rc_has_ek}*{ek_is[CONT_CALL2_B]}")))
    v3.append(("--a_f23", wrap_calc(f23_ok)))
    v3.append(("--a_if_f45", wrap_calc(if_f45_ok)))
    v3.append(("--a_c1_f45", wrap_calc(call1_f45_ok)))

    # Materialize stack SP & write buses directly in v3 (with `rc_has_ek` factored ONCE per expression!)
    op1_is_unary = css_eq_vars("var(--eidx)", "var(--eb)")
    opn_has_more = f"(1 - {css_eq('var(--etk)', TOK_RP)})"
    opn_is_div = css_eq("var(--c_op)", OP_DIV)

    top_p_u = f"({ek_is[CONT_TOP_SEQ]}*clamp(0,var(--ntok) - var(--etn),1))"
    beg_p_u = f"({ek_is[CONT_BEGIN_SEQ]}*(1 - {css_eq_vars('var(--etn)', 'var(--eb)')}))"
    op1_p_u = f"({ek_is[CONT_OP_FIRST]}*(1 - {op1_is_unary}))"
    opn_p_u = f"({ek_is[CONT_OP_NEXT]}*(1 - {opn_is_div})*{opn_has_more})"

    # Notice: among `rc_has_ek`, EVERY continuation except `is_unfolded_rest` pops `1 + fold_r`, while `is_unfolded_rest` pops `1`!
    base_sp_s = f"(var(--sp) - {rc_has_ek}*((1 - {is_unfolded_rest})*(1 + var(--fold_r)) + {is_unfolded_rest}) - {rc_done}*var(--sp))"

    cont_push1_u = f"({top_p_u} + {beg_p_u} + {op1_p_u} + {opn_p_u} + {ek_is[CONT_CALL1]} + {ek_is[CONT_CALL2_A]} + {ek_is[CONT_BIND_P1]} + {ek_is[CONT_LET1]})"
    eval_push1_s = f"({if_norm} + {beg_multi} + {set_valid} + {let_1} + {let_2} + {bop_fa1_ok} + {bop_gen} + {call1_atom_ok} + {call1_f45_ok} + {call1_gen} + {call2_ok})"
    push2_u = f"({ek_is[CONT_CALL2_B]} + {ek_is[CONT_LET2_A]})"
    push2_s = f"({rc_has_ek}*{push2_u})"
    push1_s = f"({rc_has_ek}*{cont_push1_u} + {eval_push1_s})"
    nsp_s = f"({base_sp_s} + {rc_has_ek}*({cont_push1_u} + 2*{push2_u}) + {eval_push1_s})"

    v3.append(("--base_sp", wrap_calc(base_sp_s)))
    v3.append(("--nsp", wrap_calc(nsp_s)))
    v3.append(("--p1_en", wrap_calc(f"{rc_has_ek}*({cont_push1_u} + {push2_u}) + {eval_push1_s}")))
    v3.append(("--p2_en", wrap_calc(push2_s)))

    p1_k_cont = (
        f"({CONT_TOP_SEQ}*{top_p_u} + {CONT_BEGIN_SEQ}*{beg_p_u} + {CONT_OP_NEXT}*({op1_p_u} + {opn_p_u}) "
        f"+ {CONT_RESTORE}*({ek_is[CONT_CALL1]} + {ek_is[CONT_CALL2_B]} + {ek_is[CONT_BIND_P1]} + {ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]}) "
        f"+ {CONT_CALL2_B}*{ek_is[CONT_CALL2_A]})"
    )
    p1_k_eval = (
        f"({CONT_BEGIN_SEQ}*{beg_multi} + {CONT_OP_NEXT}*{bop_fa1_ok} + {CONT_OP_FIRST}*{bop_gen} "
        f"+ {CONT_IF}*{if_norm} + {CONT_SET}*{set_valid} + {CONT_RESTORE}*({call1_atom_ok} + {call1_f45_ok}) "
        f"+ {CONT_CALL1}*{call1_gen} + {CONT_CALL2_A}*{call2_ok} + {CONT_LET1}*{let_1} + {CONT_LET2_A}*{let_2})"
    )
    p1_a_cont = (
        f"(({top_p_u} + {beg_p_u})*var(--etn) + ({op1_p_u} + {opn_p_u})*(64*var(--c_op) + var(--etn)) "
        f"+ ({ek_is[CONT_CALL1]} + {ek_is[CONT_CALL2_B]})*(2*var(--tf_p0) + {tf0_had}) "
        f"+ {ek_is[CONT_BIND_P1]}*(2*var(--tf_p1) + {tf1_had}) "
        f"+ ({ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]})*(2*var(--top_b) + {let_had}) "
        f"+ {ek_is[CONT_CALL2_A]}*var(--top_fidx))"
    )
    p1_a_eval = (
        f"({beg_multi}*var(--q_tn2) + {bop_fa1_ok}*(64*var(--op1) + var(--q_tn3)) + {bop_gen}*(64*var(--op1) + var(--q_tn2)) "
        f"+ {if_norm}*var(--q_tn2) + {set_valid}*var(--q_tv2) "
        f"+ ({call1_atom_ok} + {call1_f45_ok})*(2*var(--cf_p0) + {cf0_had}) "
        f"+ {call1_gen}*var(--c_fidx) + {call2_ok}*(64*var(--c_fidx) + var(--q_tn2)) "
        f"+ {let_1}*var(--q_tn2) + {let_2}*var(--q_tn3))"
    )
    p1_bb_cont = (
        f"({beg_p_u}*var(--eb) + ({op1_p_u} + {ek_is[CONT_CALL2_A]})*{cin_s} "
        f"+ ({ek_is[CONT_CALL1]} + {ek_is[CONT_CALL2_B]})*{tf0_old} + {ek_is[CONT_BIND_P1]}*{tf1_old} "
        f"+ ({ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]})*{let_old})"
    )
    p1_bb_eval = (
        f"(({beg_multi} + {if_norm} + {bop_gen})*(var(--q_tn0) - 1) + {bop_fa1_ok}*var(--aval2) "
        f"+ ({call1_atom_ok} + {call1_f45_ok})*{cf0_old} + ({let_1} + {let_2})*var(--q_tv4))"
    )
    v3.append(("--p1_k", wrap_calc(f"{rc_has_ek}*{p1_k_cont} + {p1_k_eval}")))
    v3.append(("--p1_a", wrap_calc(f"{rc_has_ek}*{p1_a_cont} + {p1_a_eval}")))
    v3.append(("--p1_bb", wrap_calc(f"{rc_has_ek}*{p1_bb_cont} + {p1_bb_eval}")))
    v3.append(("--p1_uc", wrap_calc(f"{rc_has_ek}*{opn_p_u}")))

    v3.append(("--p2_k", wrap_calc(f"{rc_has_ek}*({CONT_BIND_P1}*{ek_is[CONT_CALL2_B]} + {CONT_LET1}*{ek_is[CONT_LET2_A]})")))
    v3.append(("--p2_a", wrap_calc(f"{rc_has_ek}*({ek_is[CONT_CALL2_B]}*var(--top_a) + {ek_is[CONT_LET2_A]}*(var(--etn) + 1))")))
    v3.append(("--p2_b", wrap_calc(f"{rc_has_ek}*({ek_is[CONT_CALL2_B]}*{cin_s} + {ek_is[CONT_LET2_A]}*var(--etv1))")))

    pc_j_cont = (
        f"(({ek_is[CONT_TOP_SEQ]} + {ek_is[CONT_BEGIN_SEQ]} + {op1_p_u} + {opn_p_u} + {ek_is[CONT_CALL2_A]})*var(--eidx) "
        f"+ ({ek_is[CONT_CALL1]} + {ek_is[CONT_BIND_P1]})*var(--tf_fb) "
        f"+ {ek_is[CONT_LET1]}*var(--ea) + {ek_is[CONT_LET2_A]}*(var(--eidx) + 2))"
    )
    pc_j_eval = (
        f"(({if_norm} + {beg_single} + {beg_multi} + {bop_gen} + {call1_gen} + {call2_ok})*(var(--pc) + 2) "
        f"+ {set_valid}*(var(--pc) + 3) + {let_0}*var(--q_tn2) + ({let_1} + {let_2})*(var(--pc) + 5) "
        f"+ {bop_fa1_ok}*var(--q_tn2) + ({call0_ok} + {call1_atom_ok} + {call1_f45_ok})*var(--cf_fb))"
    )
    pc_c_cont = f"({ek_is[CONT_TOP_SEQ]} + {ek_is[CONT_BEGIN_SEQ]} + {op1_p_u} + {opn_p_u} + {ek_is[CONT_CALL2_A]} + {ek_is[CONT_CALL1]} + {ek_is[CONT_BIND_P1]} + {ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]})"
    pc_c_eval = f"({if_norm} + {beg_single} + {beg_multi} + {bop_gen} + {call1_gen} + {call2_ok} + {set_valid} + {let_0} + {let_1} + {let_2} + {bop_fa1_ok} + {call0_ok} + {call1_atom_ok} + {call1_f45_ok})"
    v3.append(("--pc_bj", wrap_calc(f"{rc_has_ek}*{pc_j_cont} + {pc_j_eval}")))
    v3.append(("--pc_bc", wrap_calc(f"{rc_has_ek}*{pc_c_cont} + {pc_c_eval}")))

    # Variable restore & write buses (factored!)
    do_rest = f"({rc_s}*var(--fold_r) + {rc_has_ek}*{is_unfolded_rest})"
    rest_had = "mod(var(--top_a),2)"
    v3.append(("--r_en", wrap_calc(do_rest)))
    v3.append(("--r_had", wrap_calc(rest_had)))
    v3.append(("--r_2sym", wrap_calc(f"var(--top_a) - {rest_had}")))
    v3.append(("--r_val", "var(--top_b)"))

    w_cont_u = f"({ek_is[CONT_SET]} + {ek_is[CONT_CALL1]} + {ek_is[CONT_CALL2_B]} + {ek_is[CONT_BIND_P1]} + {ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]})"
    w_sym_cont = f"({ek_is[CONT_SET]}*var(--ea) + ({ek_is[CONT_CALL1]} + {ek_is[CONT_CALL2_B]})*var(--tf_p0) + {ek_is[CONT_BIND_P1]}*var(--tf_p1) + ({ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]})*var(--top_b))"
    w_val_cont = f"(({ek_is[CONT_SET]} + {ek_is[CONT_CALL1]} + {ek_is[CONT_LET1]} + {ek_is[CONT_LET2_A]})*{cin_s} + ({ek_is[CONT_CALL2_B]} + {ek_is[CONT_BIND_P1]})*var(--top_b))"
    v3.append(("--w_en", wrap_calc(f"{rc_has_ek}*{w_cont_u} + {call1_atom_ok} + {call1_f45_ok}")))
    v3.append(("--w_sym", wrap_calc(f"{rc_has_ek}*{w_sym_cont} + ({call1_atom_ok} + {call1_f45_ok})*var(--cf_p0)")))
    v3.append(("--w_val_b", wrap_calc(f"{rc_has_ek}*{w_val_cont} + {call1_atom_ok}*var(--aval2)")))
    v3.append(("--w_u45", wrap_calc(call1_f45_ok)))

    op1_bad_u = f"({ek_is[CONT_OP_FIRST]}*{op1_is_unary}*(1 - {css_in_set('var(--c_op)', (OP_ADD, OP_SUB, OP_MUL, OP_NOT))}))"
    opn_bad_u = f"({ek_is[CONT_OP_NEXT]}*({css_eq('var(--c_op)', OP_NOT)} + {opn_has_more}*{css_in_set('var(--c_op)', (OP_DIV, OP_MOD, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE))}))"
    sp_ovf = f"clamp(0,{nsp_s} - {STACK_SIZE},1)"
    err_unb = f"min(1,{imm_atom_err} + {f23_unbound} + {if_f45_unb} + {bop_fa1_unb} + {call_unbound} + {call1_atom_unb} + {call1_f45_unb})"
    err_call0 = (
        f"min(1,{imm_def_err} + {imm_lam_err} + {if_err} + {set_err} + {let_err} + {bop_empty_err} "
        f"+ {call_not_fn} + {call0_err} + {call1_err} + {call2_err} + {empty_list_err} + {unexpected_rp_err} "
        f"+ {rc_has_ek}*({op1_bad_u} + {opn_bad_u}) + {sp_ovf})"
    )
    v3.append(("--e_unb", wrap_calc(err_unb)))
    v3.append(("--e_call0", wrap_calc(err_call0)))
    stage_defs["v3"] = v3

    # -----------------------------------------------------------------------
    # .v4: Ultra-Compact Canonical Register Commit
    # -----------------------------------------------------------------------
    v4: List[Tuple[str, str]] = []
    m_act = css_in_set("var(--l_mode)", (MODE_EVAL, MODE_CONT, MODE_DIV))
    w_already = f"({' + '.join(css_eq_vars(f'var(--l_vk{k})', 'var(--w_sym)') for k in range(NUM_VARS))})"
    var_ovf = f"(var(--w_en)*(1 - {w_already})*(1 - var(--v_has_free)))"

    dz_any = "min(1,var(--a_f23)*var(--dz23) + (var(--a_if_f45) + var(--a_c1_f45))*var(--dz45) + var(--a_ek_opn)*var(--cdz))"
    any_call = f"min(1,var(--e_call0) + {var_ovf})"
    any_err = f"({m_act}*min(1,{dz_any} + var(--e_unb) + {any_call}))"
    ok_tick = f"(1 - {any_err})"
    err_code_new = f"({ERR_DIV_ZERO}*{dz_any} + (1 - {dz_any})*({ERR_UNBOUND}*var(--e_unb) + (1 - var(--e_unb))*{ERR_CALL}*{any_call}))"

    op1_is_u_v4 = css_eq_vars("var(--eidx)", "var(--eb)")
    opn_more_v4 = f"(1 - {css_eq('var(--etk)', TOK_RP)})"
    opn_div_v4 = css_eq("var(--c_op)", OP_DIV)

    enter_div = f"(var(--a_ek_opn)*{opn_div_v4}*(1 - var(--cdz)))"
    div_fin = f"({css_eq('var(--l_mode)', MODE_DIV)}*{css_eq('var(--l_dstep)', 0)})"
    div_more = f"({css_eq('var(--l_mode)', MODE_DIV)}*clamp(0,var(--l_dstep),1))"

    if_c_nz = "min(1,max(var(--cont_in),0 - var(--cont_in)))"
    if_has_el = f"(1 - {css_eq_vars('var(--etn)', 'var(--eb)')})"
    f45_nz = "min(1,max(var(--alu45),0 - var(--alu45)))"
    f45_has_el = f"(1 - {css_eq_vars('var(--q_tn7)', 'var(--q_tn0) - 1')})"

    if_pc_j = f"(var(--a_ek_if)*({if_c_nz}*var(--eidx) + (1 - {if_c_nz})*{if_has_el}*var(--etn)) + var(--a_if_f45)*({f45_nz}*(var(--l_pc) + 7) + (1 - {f45_nz})*{f45_has_el}*var(--q_tn7)))"
    if_pc_c = f"(var(--a_ek_if)*({if_c_nz} + (1 - {if_c_nz})*{if_has_el}) + var(--a_if_f45)*({f45_nz} + (1 - {f45_nz})*{f45_has_el}))"
    v4.append(("--pc", wrap_calc(f"{ok_tick}*(var(--l_pc)*(1 - var(--pc_bc) - {if_pc_c}) + var(--pc_bj) + {if_pc_j}) + {any_err}*var(--l_pc)")))

    unary_res = (
        f"({css_in_set('var(--c_op)', (OP_ADD, OP_MUL))}*var(--cont_in) "
        f"- {css_eq('var(--c_op)', OP_SUB)}*var(--cont_in) "
        f"+ {css_eq('var(--c_op)', OP_NOT)}*{css_eq('var(--cont_in)', 0)})"
    )
    to_cont = (
        f"(var(--a_bemp) + var(--a_ek_if)*(1 - {if_c_nz})*(1 - {if_has_el}) "
        f"+ var(--a_if_f45)*(1 - {f45_nz})*(1 - {f45_has_el}) "
        f"+ var(--a_ek_set) + var(--a_ek_rest) + var(--a_ek_c2b) "
        f"+ var(--a_ek_op1)*{op1_is_u_v4} + var(--a_ek_opn)*(1 - {opn_div_v4})*(1 - {opn_more_v4}) "
        f"+ var(--a_f23) + {div_fin})"
    )
    acc_new = (
        f"((var(--a_done) + var(--a_ek_set) + var(--a_ek_rest) + var(--a_ek_c2b))*var(--cont_in) "
        f"+ var(--a_ek_op1)*{op1_is_u_v4}*{unary_res} "
        f"+ var(--a_ek_opn)*(1 - {opn_div_v4})*(1 - {opn_more_v4})*var(--calu) "
        f"+ var(--a_f23)*var(--alu23) "
        f"+ {div_fin}*var(--l_dsgn)*(16*var(--l_dq) + var(--div_dig)))"
    )
    acc_chg = f"(var(--a_done) + {to_cont})"
    v4.append(("--acc", wrap_calc(f"var(--l_acc)*(1 - {acc_chg}) + {acc_new}")))

    already_done = css_eq("var(--l_mode)", MODE_DONE)
    already_err = css_eq("var(--l_mode)", MODE_ERR)
    n_mode = (
        f"3*({already_err} + {any_err}) + (1 - {already_err} - {any_err})*("
        f"2*({already_done} + var(--a_done)) + 4*({enter_div} + {div_more}) + {to_cont})"
    )
    v4.append(("--mode", wrap_calc(n_mode)))
    v4.append(("--err", wrap_calc(f"var(--l_err) + (1 - {already_err})*{err_code_new}")))
    v4.append(("--steps", wrap_calc(f"var(--l_steps) + {m_act}")))
    v4.append(("--sp", wrap_calc(f"{ok_tick}*var(--nsp) + {any_err}*var(--l_sp)")))

    lhs_neg = css_gt("0", "var(--eb)")
    rhs_neg = css_gt("0", "var(--cont_in)")
    div_sgn_init = f"(1 - 2*({lhs_neg} + {rhs_neg} - 2*{lhs_neg}*{rhs_neg}))"
    abs_lhs = "min(1048575,max(var(--eb),0 - var(--eb)))"
    abs_rhs = "max(1,max(var(--cont_in),0 - var(--cont_in)))"
    is_div_t = css_eq("var(--l_mode)", MODE_DIV)
    v4.append(("--drem", wrap_calc(f"{enter_div}*{abs_lhs} + {is_div_t}*(var(--l_drem) - var(--div_dig)*var(--div_D)) + (1 - {enter_div} - {is_div_t})*var(--l_drem)")))
    v4.append(("--dden", wrap_calc(f"{enter_div}*{abs_rhs} + (1 - {enter_div})*var(--l_dden)")))
    v4.append(("--dq", wrap_calc(f"{is_div_t}*(16*var(--l_dq) + var(--div_dig)) + (1 - {enter_div} - {is_div_t})*var(--l_dq)")))
    v4.append(("--dstep", wrap_calc(f"4*{enter_div} + {div_more}*(var(--l_dstep) - 1)")))
    v4.append(("--dsgn", wrap_calc(f"{enter_div}*{div_sgn_init} + (1 - {enter_div})*var(--l_dsgn)")))

    for f in range(NUM_FUNCS):
        we_f = f"(var(--fn_we)*{css_eq('var(--fn_idx)', f)})"
        v4.append((f"--fk{f}", wrap_calc(f"var(--l_fk{f}) + {we_f}*(var(--fn_k) - var(--l_fk{f}))")))
        v4.append((f"--fp0_{f}", wrap_calc(f"var(--l_fp0_{f}) + {we_f}*(var(--fn_p0) - var(--l_fp0_{f}))")))
        v4.append((f"--fp1_{f}", wrap_calc(f"var(--l_fp1_{f}) + {we_f}*(var(--fn_p1) - var(--l_fp1_{f}))")))
        v4.append((f"--fb{f}", wrap_calc(f"var(--l_fb{f}) + {we_f}*(var(--fn_fb) - var(--l_fb{f}))")))

    w_val_full = "(var(--w_val_b) + var(--w_u45)*var(--alu45))"
    for k in range(NUM_VARS):
        r_hit = f"(var(--r_en)*min(1,var(--l_vk{k}))*{css_eq_vars(f'2*var(--l_vk{k})', 'var(--r_2sym)')})"
        w_ex = css_eq_vars(f"var(--l_vk{k})", "var(--w_sym)")
        w_fr = f"((1 - {w_already})*var(--v_has_free)*{css_eq('var(--v_free_idx)', k)})"
        w_hit = f"(var(--w_en)*({w_ex} + {w_fr}))"
        v4.append((f"--vk{k}", wrap_calc(f"var(--l_vk{k})*(1 - {r_hit} - {w_hit}) + {r_hit}*var(--r_had)*var(--l_vk{k}) + {w_hit}*var(--w_sym)")))
        v4.append((f"--vv{k}", wrap_calc(f"var(--l_vv{k})*(1 - {r_hit} - {w_hit}) + {r_hit}*var(--r_had)*var(--r_val) + {w_hit}*{w_val_full}")))

    p1_b_full = "(var(--p1_bb) + var(--p1_uc)*var(--calu))"
    for s in range(STACK_SIZE):
        hit1 = f"(var(--p1_en)*{css_eq('var(--base_sp)', s)})"
        if s >= 1:
            hit2 = f"(var(--p2_en)*{css_eq('var(--base_sp)', s - 1)})"
            keep = f"(1 - {hit1} - {hit2})"
            v4.append((f"--sk{s}", wrap_calc(f"var(--l_sk{s})*{keep} + {hit1}*var(--p1_k) + {hit2}*var(--p2_k)")))
            v4.append((f"--sa{s}", wrap_calc(f"var(--l_sa{s})*{keep} + {hit1}*var(--p1_a) + {hit2}*var(--p2_a)")))
            v4.append((f"--sb{s}", wrap_calc(f"var(--l_sb{s})*{keep} + {hit1}*{p1_b_full} + {hit2}*var(--p2_b)")))
        else:
            keep = f"(1 - {hit1})"
            v4.append((f"--sk{s}", wrap_calc(f"var(--l_sk{s})*{keep} + {hit1}*var(--p1_k)")))
            v4.append((f"--sa{s}", wrap_calc(f"var(--l_sa{s})*{keep} + {hit1}*var(--p1_a)")))
            v4.append((f"--sb{s}", wrap_calc(f"var(--l_sb{s})*{keep} + {hit1}*{p1_b_full}")))
    stage_defs["v4"] = v4

    still_running = css_in_set("var(--mode)", (MODE_EVAL, MODE_CONT, MODE_DIV))
    stage_defs["vfin"] = [
        ("--f_mode", wrap_calc(f"var(--mode)*(1 - {still_running}) + {MODE_ERR}*{still_running}")),
        ("--f_err", wrap_calc(f"var(--err)*(1 - {still_running}) + {ERR_STEP_LIMIT}*{still_running}")),
    ]

    one_shot_order = [
        "lx0", "lx0b", "lx0c", "lx1", "lx2", "lx3", "lx4",
        "cp0", "cp1", "cp2", "cp3", "cp4", "cp5", "cpu",
        "ps0", "ps0b", "ps1", "ps2", "ps3", "ps4",
    ]
    vm_loop_order = ["v0", "v1", "v2", "v3", "v4"]

    keep_names: Set[str] = {
        "--ntok", "--syn_err",
        "--pc", "--mode", "--acc", "--err", "--steps", "--sp",
        "--drem", "--dden", "--dq", "--dstep", "--dsgn",
        "--f_mode", "--f_err",
    }
    for i in range(SOURCE_LEN):
        keep_names.add(f"--ch{i}")
    for i in range(TOKEN_COMPOSER_LEN):
        keep_names.add(f"--tvis{i}")
    for i in range(MAX_TOKENS):
        keep_names.update((f"--tk{i}", f"--tv{i}", f"--tn{i}"))
    for k in range(NUM_VARS):
        keep_names.update((f"--vk{k}", f"--vv{k}"))
    for f in range(NUM_FUNCS):
        keep_names.update((f"--fk{f}", f"--fp0_{f}", f"--fp1_{f}", f"--fb{f}"))
    for s in range(STACK_SIZE):
        keep_names.update((f"--sk{s}", f"--sa{s}", f"--sb{s}"))

    colored_defs = _allocate_temp_registers(stage_defs, one_shot_order, vm_loop_order, keep_names)

    pipeline_classes: List[str] = list(one_shot_order)
    for _ in range(num_ticks):
        pipeline_classes.extend(vm_loop_order)
    pipeline_classes.append("vfin")

    reg_set: Set[str] = set()
    for rules in colored_defs.values():
        for lhs, _ in rules:
            reg_set.add(lhs)

    return LispCSSSpec(
        stage_defs=colored_defs,
        pipeline_classes=pipeline_classes,
        registered_props=sorted(reg_set),
    )


# ---------------------------------------------------------------------------
# 4. Exact Browser CSS Custom-Property Cascade Simulator
# ---------------------------------------------------------------------------

def _eval_css_expr(rhs: str, parent_env: Dict[str, int], stage_name: str, lhs: str) -> int:
    expr = rhs.strip()
    if expr.startswith("calc(") and expr.endswith(")"):
        expr = expr[5:-1]

    out: List[str] = []
    i = 0
    n = len(expr)
    while i < n:
        if expr.startswith("var(", i):
            end = expr.index(")", i + 4)
            vname = expr[i + 4 : end].strip()
            if vname not in parent_env:
                raise KeyError(f"Stage {stage_name} {lhs} referenced undefined {vname}")
            out.append(f"({int(parent_env[vname])})")
            i = end + 1
        else:
            out.append(expr[i])
            i += 1

    py_expr = "".join(out)

    def _clamp(lo: int, val: int, hi: int) -> int:
        return max(lo, min(val, hi))

    def _mod(a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError(f"CSS mod({a}, 0) in {stage_name} {lhs}")
        return a % b

    val = eval(
        py_expr,
        {"__builtins__": {}},
        {"min": min, "max": max, "clamp": _clamp, "mod": _mod},
    )
    if not isinstance(val, int):
        raise TypeError(f"Non-integer result {val!r} in {stage_name} {lhs}")
    return int(val)


def _run_pipeline_env(
    env: Dict[str, int],
    spec: LispCSSSpec,
    fast_halt: bool = True,
) -> Dict[str, int]:
    halted_ticks_verified = 0
    idx = 0
    total = len(spec.pipeline_classes)
    while idx < total:
        cls_name = spec.pipeline_classes[idx]
        rules = spec.stage_defs[cls_name]
        next_env = dict(env)
        for lhs, rhs in rules:
            next_env[lhs] = _eval_css_expr(rhs, env, cls_name, lhs)
        env = next_env
        idx += 1

        if fast_halt and cls_name == "v4" and env["--mode"] in (MODE_DONE, MODE_ERR):
            halted_ticks_verified += 1
            if halted_ticks_verified >= 2:
                idx = total - 1
    return env


def simulate_css_lisp(
    src: str,
    spec: LispCSSSpec | None = None,
    fast_halt: bool = True,
) -> Dict[str, int]:
    if spec is None:
        spec = build_lisp_css_spec()

    codes = pad_source_codes(src, SOURCE_LEN)
    env: Dict[str, int] = {prop: 0 for prop in spec.registered_props}
    env["--use_tok"] = 0
    for slot, code in enumerate(codes):
        for bit in range(6):
            env[f"--b{slot}_{bit}"] = (code >> bit) & 1
    for slot in range(TOKEN_COMPOSER_LEN):
        env[f"--ptid{slot}"] = 1
        env[f"--ovr{slot}"] = 0
        for bit in range(8):
            env[f"--tb{slot}_{bit}"] = 0
    return _run_pipeline_env(env, spec, fast_halt=fast_halt)


def simulate_css_token_composer(
    tokens: Sequence[str] = (),
    spec: LispCSSSpec | None = None,
    fast_halt: bool = True,
    preset_src: str = "",
    overrides: Dict[int, str] | None = None,
) -> Dict[str, int]:
    """
    Simulate browser CSS evaluation in Quick Token Composer mode (`--use_tok: 1`),
    including preset token inheritance (`--ptid0..39`, defaulting to all `;` when
    `preset_src == ""`) and per-cell user `<select>` overrides (`--ovr0..39`).
    """
    if spec is None:
        spec = build_lisp_css_spec()
    if len(tokens) > TOKEN_COMPOSER_LEN:
        raise ValueError(f"Token count {len(tokens)} exceeds {TOKEN_COMPOSER_LEN}")

    ptids = tokenize_to_palette_ids(preset_src, TOKEN_COMPOSER_LEN)
    env: Dict[str, int] = {prop: 0 for prop in spec.registered_props}
    env["--use_tok"] = 1
    for slot in range(SOURCE_LEN):
        for bit in range(6):
            env[f"--b{slot}_{bit}"] = 0
    for slot in range(TOKEN_COMPOSER_LEN):
        env[f"--ptid{slot}"] = ptids[slot]
        env[f"--ovr{slot}"] = 0
        for bit in range(8):
            env[f"--tb{slot}_{bit}"] = 0

    for slot, tok in enumerate(tokens):
        tid = TOKEN_LABEL_TO_ID[tok]
        env[f"--ovr{slot}"] = 1
        for bit in range(8):
            env[f"--tb{slot}_{bit}"] = (tid >> bit) & 1

    if overrides:
        for slot, tok in overrides.items():
            tid = TOKEN_LABEL_TO_ID[tok]
            env[f"--ovr{slot}"] = 1
            for bit in range(8):
                env[f"--tb{slot}_{bit}"] = (tid >> bit) & 1

    return _run_pipeline_env(env, spec, fast_halt=fast_halt)
