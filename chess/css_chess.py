"""
CSS Custom-Property Pipeline Compiler & Cascade Simulator (`chess/css_chess.py`).

Compiles full 2-player FIDE chess rules into nested `display: contents` CSS stages
using strictly registered `@property <integer>` custom properties:
  1. `.m0` + `.m1` (2-stage reusable per-ply move applicator: handles normal moves,
     captures, double-pawn pushes, en passant, castling, promotions, castling-rights
     revocation, halfmove clock, and active-ply `--cur_from` latching)
  2. `.e0 .. .e10` (11-stage combinational evaluator using Radix-3 Kogge-Stone
     8-directional ray propagation for X-ray King attacks, pins, single/double
     checks, en-passant horizontal/diagonal pins, castling transit safety,
     per-square legal-move existence, destination masks, checkmate, stalemate,
     insufficient material, and the 50-move rule)
  3. Linear-scan liveness temporary register allocator (`--t0..N`) + cycle verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from chess_ref import (
    BR,
    ChessState,
    Move,
    PT_BISHOP,
    PT_KING,
    PT_KNIGHT,
    PT_PAWN,
    PT_QUEEN,
    PT_ROOK,
    SQ_NONE,
    START_FEN,
    STATUS_BLACK_CHECK,
    STATUS_BLACK_TURN,
    STATUS_BLACK_WINS_MATE,
    STATUS_DRAW_50,
    STATUS_DRAW_INSUF,
    STATUS_STALEMATE,
    STATUS_WHITE_CHECK,
    STATUS_WHITE_TURN,
    STATUS_WHITE_WINS_MATE,
    TURN_WHITE,
    WR,
    parse_fen,
)


DEFAULT_MAX_PLIES = 80

# 8 ray directions (d = 0..7), paired so (d ^ 1) is the exact opposite ray:
#   0: E (0,+1) ax=1   1: W (0,-1) ax=1
#   2: N (+1,0) ax=2   3: S (-1,0) ax=2
#   4: NE(+1,+1) ax=3  5: SW(-1,-1) ax=3
#   6: NW(+1,-1) ax=4  7: SE(-1,+1) ax=4
RAY_DIRS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 1),   # 0: E
    (0, -1, 1),  # 1: W
    (1, 0, 2),   # 2: N
    (-1, 0, 2),  # 3: S
    (1, 1, 3),   # 4: NE (/)
    (-1, -1, 3), # 5: SW (/)
    (1, -1, 4),  # 6: NW (\)
    (-1, 1, 4),  # 7: SE (\)
)

KNIGHT_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
KING_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _step_sq(u: int, d: int, dist: int = 1) -> Optional[int]:
    r, f = divmod(u, 8)
    dr, df, _ = RAY_DIRS[d]
    nr, nf = r + dist * dr, f + dist * df
    if 0 <= nr < 8 and 0 <= nf < 8:
        return 8 * nr + nf
    return None


def _delta_sq(u: int, dr: int, df: int) -> Optional[int]:
    r, f = divmod(u, 8)
    nr, nf = r + dr, f + df
    if 0 <= nr < 8 and 0 <= nf < 8:
        return 8 * nr + nf
    return None


# ---------------------------------------------------------------------------
# 1. Strict <integer> CSS Expression Helpers
# ---------------------------------------------------------------------------

def css_eq(var_expr: str, const_k: int) -> str:
    if const_k == 0:
        return f"(1 - min(1,max({var_expr},0 - ({var_expr}))))"
    return f"(1 - min(1,max({var_expr} - {const_k},{const_k} - ({var_expr}))))"


def css_eq_vars(a: str, b: str) -> str:
    return f"(1 - min(1,max({a} - ({b}),({b}) - ({a}))))"


def css_ge_const(var_expr: str, const_k: int) -> str:
    if const_k <= 1:
        return f"clamp(0,{var_expr} - {const_k - 1},1)" if const_k != 1 else f"clamp(0,{var_expr},1)"
    return f"clamp(0,{var_expr} - {const_k - 1},1)"


def css_or_terms(terms: Sequence[str]) -> str:
    nz = [t for t in terms if t != "0"]
    if not nz:
        return "0"
    if len(nz) == 1:
        return nz[0]
    return f"min(1,{' + '.join(nz)})"


def css_sum_terms(terms: Sequence[str]) -> str:
    nz = [t for t in terms if t != "0"]
    if not nz:
        return "0"
    if len(nz) == 1:
        return nz[0]
    return f"({' + '.join(nz)})"


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
    ply_loop_order: Sequence[str],
    eval_order: Sequence[str],
    keep_names: Set[str],
) -> Tuple[Dict[str, List[Tuple[str, str]]], int]:
    rename_map: Dict[str, str] = {}

    # 1. Allocate temps inside the reusable ply macro (.m0 -> .m1)
    ply_def: Dict[str, int] = {}
    ply_last: Dict[str, int] = {}
    for st_idx, cls in enumerate(ply_loop_order):
        for lhs, rhs in stage_defs[cls]:
            if lhs not in keep_names:
                assert lhs not in ply_def, f"Duplicate ply temp {lhs}"
                ply_def[lhs] = st_idx
                ply_last[lhs] = st_idx
            for ref in _extract_vars(rhs):
                if ref in ply_last:
                    assert st_idx > ply_def[ref], f"Intra-stage read of {ref} in {cls}"
                    ply_last[ref] = max(ply_last[ref], st_idx)

    ply_busy: List[int] = []
    for vname in sorted(ply_def.keys(), key=lambda v: (ply_def[v], v)):
        ds, lu = ply_def[vname], ply_last[vname]
        chosen = -1
        for r_idx, end_st in enumerate(ply_busy):
            if end_st < ds:
                chosen = r_idx
                ply_busy[r_idx] = lu
                break
        if chosen < 0:
            chosen = len(ply_busy)
            ply_busy.append(lu)
        rename_map[vname] = f"--t{chosen}"

    # 2. Allocate temps across the 11-stage evaluator (.e0 .. .e10), reusing --t0..N!
    ev_def: Dict[str, int] = {}
    ev_last: Dict[str, int] = {}
    for st_idx, cls in enumerate(eval_order):
        for lhs, rhs in stage_defs[cls]:
            if lhs not in keep_names:
                assert lhs not in ev_def, f"Duplicate eval temp {lhs} in {cls}"
                ev_def[lhs] = st_idx
                ev_last[lhs] = st_idx
            for ref in _extract_vars(rhs):
                if ref in ev_last:
                    assert st_idx > ev_def[ref], f"Intra-stage read of {ref} in {cls}"
                    ev_last[ref] = max(ev_last[ref], st_idx)

    ev_busy: List[int] = []
    for vname in sorted(ev_def.keys(), key=lambda v: (ev_def[v], v)):
        ds, lu = ev_def[vname], ev_last[vname]
        chosen = -1
        for r_idx, end_st in enumerate(ev_busy):
            if end_st < ds:
                chosen = r_idx
                ev_busy[r_idx] = lu
                break
        if chosen < 0:
            chosen = len(ev_busy)
            ev_busy.append(lu)
        rename_map[vname] = f"--t{chosen}"

    num_temps = max(len(ply_busy), len(ev_busy))

    new_defs: Dict[str, List[Tuple[str, str]]] = {}
    for cls, rules in stage_defs.items():
        compiled_rules: List[Tuple[str, str]] = []
        lhs_in_stage: Set[str] = set()
        for lhs, rhs in rules:
            nlhs = rename_map.get(lhs, lhs)
            nrhs = _VAR_RE.sub(lambda m: f"var({rename_map.get(m.group(1), m.group(1))})", rhs)
            assert nlhs not in lhs_in_stage, f"Duplicate LHS {nlhs} on stage {cls}"
            lhs_in_stage.add(nlhs)
            compiled_rules.append((nlhs, nrhs))
        # Static intra-stage cycle assertion
        for nlhs, nrhs in compiled_rules:
            refs = _extract_vars(nrhs)
            overlap = refs & lhs_in_stage
            assert not overlap, f"Intra-stage CSS var dependency on {cls}: {nlhs} reads {overlap}"
        new_defs[cls] = compiled_rules

    return new_defs, num_temps


# ---------------------------------------------------------------------------
# 3. Full CSS Chess Pipeline Builder
# ---------------------------------------------------------------------------

@dataclass
class ChessCSSSpec:
    stage_defs: Dict[str, List[Tuple[str, str]]]
    ply_classes: Tuple[str, ...]
    eval_classes: Tuple[str, ...]
    registered_props: List[str]
    num_temp_regs: int


def state_to_css_vars(state: ChessState) -> Dict[str, int]:
    env: Dict[str, int] = {f"--b{sq}": state.board[sq] for sq in range(64)}
    env["--turn"] = state.turn
    env["--cast_wk"] = state.cast_wk
    env["--cast_wq"] = state.cast_wq
    env["--cast_bk"] = state.cast_bk
    env["--cast_bq"] = state.cast_bq
    env["--ep_sq"] = state.ep_sq
    env["--hmc"] = state.hmc
    env["--ply"] = state.ply
    env["--halted"] = 0
    env["--last_f"] = state.last_f
    env["--last_t"] = state.last_t
    env["--cur_from"] = SQ_NONE
    env["--cur_pr"] = PT_QUEEN
    return env


def build_chess_css_spec() -> ChessCSSSpec:
    stage_defs: Dict[str, List[Tuple[str, str]]] = {}

    # -----------------------------------------------------------------------
    # Part A: Reusable 2-Stage Per-Ply Move Applicator (.m0 -> .m1)
    # -----------------------------------------------------------------------
    # IMPORTANT: In CSS custom properties, `--b0: calc(var(--b0) + ...)` on the
    # same element is a self-cycle! So `.m0` latches all persistent registers
    # that `.m1` updates into temp wires (`--mb*`, `--m_*`), and `.m1` reads ONLY
    # `.m0` latches + `.p` inputs (`--in_*`), guaranteeing zero intra-element cycles!
    m0: List[Tuple[str, str]] = []
    for sq in range(64):
        m0.append((f"--mb{sq}", f"var(--b{sq})"))
    for kreg, treg in (
        ("--turn", "--m_trn"),
        ("--cast_wk", "--m_owk"),
        ("--cast_wq", "--m_owq"),
        ("--cast_bk", "--m_obk"),
        ("--cast_bq", "--m_obq"),
        ("--ep_sq", "--m_oep"),
        ("--hmc", "--m_ohmc"),
        ("--ply", "--m_oply"),
        ("--halted", "--m_ohlt"),
        ("--last_f", "--m_olf"),
        ("--last_t", "--m_olt"),
        ("--cur_from", "--m_ocf"),
        ("--cur_pr", "--m_ocpr"),
    ):
        m0.append((treg, f"var({kreg})"))

    m0.append(("--m_do", wrap_calc("(1 - var(--halted))*var(--in_c)")))
    m0.append(("--m_lat", wrap_calc("(1 - var(--halted))*(1 - var(--in_c))")))
    m0.append((
        "--m_mp",
        wrap_calc(" + ".join(f"{css_eq('var(--in_f)', sq)}*var(--b{sq})" for sq in range(64))),
    ))
    m0.append((
        "--m_docc",
        wrap_calc(f"clamp(0,{' + '.join(f'{css_eq('var(--in_t)', sq)}*var(--b{sq})' for sq in range(64))},1)"),
    ))
    m0.append(("--m_sfile", wrap_calc(css_eq_vars("mod(var(--in_f),8)", "mod(var(--in_t),8)"))))
    m0.append((
        "--m_dbl",
        wrap_calc(f"{css_eq('var(--in_t) - var(--in_f)', 16)} + {css_eq('var(--in_f) - var(--in_t)', 16)}"),
    ))
    m0.append(("--m_cwk", wrap_calc(f"{css_eq('var(--in_f)', 4)}*{css_eq('var(--in_t)', 6)}")))
    m0.append(("--m_cwq", wrap_calc(f"{css_eq('var(--in_f)', 4)}*{css_eq('var(--in_t)', 2)}")))
    m0.append(("--m_cbk", wrap_calc(f"{css_eq('var(--in_f)', 60)}*{css_eq('var(--in_t)', 62)}")))
    m0.append(("--m_cbq", wrap_calc(f"{css_eq('var(--in_f)', 60)}*{css_eq('var(--in_t)', 58)}")))
    stage_defs["m0"] = m0

    m1: List[Tuple[str, str]] = []
    is_mp_pawn = css_eq("mod(var(--m_mp),8)", PT_PAWN)
    is_mp_king = css_eq("mod(var(--m_mp),8)", PT_KING)

    for sq in range(64):
        r = sq // 8
        eq_t = css_eq("var(--in_t)", sq)
        eq_f = css_eq("var(--in_f)", sq)

        # Piece value placed if `sq == --in_t`
        if r == 0 or r == 7:
            placed = f"({is_mp_pawn}*(8*var(--m_trn) + var(--in_pr)) + (1 - {is_mp_pawn})*var(--m_mp))"
        else:
            placed = "var(--m_mp)"

        delta_terms: List[str] = [
            f"{eq_t}*({placed} - var(--mb{sq}))",
            f"(0 - {eq_f}*var(--mb{sq}))",
        ]

        # Castling rook updates on corners & transit squares
        if sq == 7:
            delta_terms.append(f"(0 - {is_mp_king}*var(--m_cwk)*var(--mb7))")
        elif sq == 5:
            delta_terms.append(f"{is_mp_king}*var(--m_cwk)*({WR} - var(--mb5))")
        elif sq == 0:
            delta_terms.append(f"(0 - {is_mp_king}*var(--m_cwq)*var(--mb0))")
        elif sq == 3:
            delta_terms.append(f"{is_mp_king}*var(--m_cwq)*({WR} - var(--mb3))")
        elif sq == 63:
            delta_terms.append(f"(0 - {is_mp_king}*var(--m_cbk)*var(--mb63))")
        elif sq == 61:
            delta_terms.append(f"{is_mp_king}*var(--m_cbk)*({BR} - var(--mb61))")
        elif sq == 56:
            delta_terms.append(f"(0 - {is_mp_king}*var(--m_cbq)*var(--mb56))")
        elif sq == 59:
            delta_terms.append(f"{is_mp_king}*var(--m_cbq)*({BR} - var(--mb59))")

        # En-passant victim clearing on rank 3 (White pawn captured by Black) & rank 4 (Black pawn captured by White)
        if r == 3:
            ep_t_sq = sq - 8  # rank 2
            ep_clr = (
                f"var(--m_trn)*{is_mp_pawn}*(1 - var(--m_sfile))"
                f"*(1 - var(--m_docc))*{css_eq('var(--in_t)', ep_t_sq)}*{css_eq('var(--m_oep)', ep_t_sq)}"
            )
            delta_terms.append(f"(0 - {ep_clr}*var(--mb{sq}))")
        elif r == 4:
            ep_t_sq = sq + 8  # rank 5
            ep_clr = (
                f"(1 - var(--m_trn))*{is_mp_pawn}*(1 - var(--m_sfile))"
                f"*(1 - var(--m_docc))*{css_eq('var(--in_t)', ep_t_sq)}*{css_eq('var(--m_oep)', ep_t_sq)}"
            )
            delta_terms.append(f"(0 - {ep_clr}*var(--mb{sq}))")

        m1.append((f"--b{sq}", wrap_calc(f"var(--mb{sq}) + var(--m_do)*({' + '.join(delta_terms)})")))

    # Metadata state transitions
    m1.append(("--ply", "calc(var(--m_oply) + var(--m_do))"))
    m1.append(("--turn", "calc(var(--m_trn) + var(--m_do)*(1 - 2*var(--m_trn)))"))
    m1.append(("--halted", "calc(var(--m_ohlt) + var(--m_lat))"))
    m1.append(("--cur_from", "calc(var(--m_ocf) + var(--m_lat)*(var(--in_f) - var(--m_ocf)))"))
    m1.append(("--cur_pr", "calc(var(--m_ocpr) + var(--m_lat)*(var(--in_pr) - var(--m_ocpr)))"))
    m1.append(("--last_f", "calc(var(--m_olf) + var(--m_do)*(var(--in_f) - var(--m_olf)))"))
    m1.append(("--last_t", "calc(var(--m_olt) + var(--m_do)*(var(--in_t) - var(--m_olt)))"))

    is_dbl_push = f"({is_mp_pawn}*var(--m_dbl))"
    next_ep_val = f"({is_dbl_push}*(var(--in_f) + 8 - 16*var(--m_trn)) + (1 - {is_dbl_push})*{SQ_NONE})"
    m1.append(("--ep_sq", wrap_calc(f"var(--m_oep) + var(--m_do)*({next_ep_val} - var(--m_oep))")))

    rst_hmc = f"min(1,{is_mp_pawn} + var(--m_docc))"
    next_hmc_val = f"((1 - {rst_hmc})*(var(--m_ohmc) + 1))"
    m1.append(("--hmc", wrap_calc(f"var(--m_ohmc) + var(--m_do)*({next_hmc_val} - var(--m_ohmc))")))

    for cname, lreg, s_king, s_rook in (
        ("--cast_wk", "--m_owk", 4, 7),
        ("--cast_wq", "--m_owq", 4, 0),
        ("--cast_bk", "--m_obk", 60, 63),
        ("--cast_bq", "--m_obq", 60, 56),
    ):
        touch = css_or_terms([
            css_eq("var(--in_f)", s_king),
            css_eq("var(--in_f)", s_rook),
            css_eq("var(--in_t)", s_king),
            css_eq("var(--in_t)", s_rook),
        ])
        m1.append((cname, wrap_calc(f"var({lreg})*(1 - var(--m_do)*{touch})")))

    stage_defs["m1"] = m1

    # -----------------------------------------------------------------------
    # Part B: 11-Stage Combinational Legality & Terminal Evaluator (e0 .. e10)
    # -----------------------------------------------------------------------

    # .e0a & .e0b: 2-step square decode (defers non-ray piece flags to .e2 to cut peak register liveness!)
    e0a: List[Tuple[str, str]] = []
    e0b: List[Tuple[str, str]] = []
    e2_roles: List[Tuple[str, str]] = []
    e4_roles: List[Tuple[str, str]] = []
    for u in range(64):
        bu = f"var(--b{u})"
        occ_e = f"clamp(0,{bu},1)"
        blk_e = f"clamp(0,{bu} - 8,1)"
        wht_e = f"({occ_e} - {blk_e})"
        e0a.append((f"--occ{u}", wrap_calc(occ_e)))
        e0a.append((f"--own{u}", wrap_calc(f"(1 - var(--turn))*{wht_e} + var(--turn)*{blk_e}")))
        e0a.append((f"--opp{u}", wrap_calc(f"var(--turn)*{wht_e} + (1 - var(--turn))*{blk_e}")))
        e0a.append((f"--pt{u}", f"mod({bu},8)"))
        e0a.append((f"--isel{u}", wrap_calc(css_eq("var(--cur_from)", u))))

        pt_e = f"mod({bu},8)"
        is_b0 = css_eq(pt_e, PT_BISHOP)
        is_r0 = css_eq(pt_e, PT_ROOK)
        is_q0 = css_eq(pt_e, PT_QUEEN)
        is_k0 = css_eq(pt_e, PT_KING)
        own_e = f"((1 - var(--turn))*{wht_e} + var(--turn)*{blk_e})"
        opp_e = f"(var(--turn)*{wht_e} + (1 - var(--turn))*{blk_e})"
        e0a.append((f"--own_k{u}", wrap_calc(f"{own_e}*{is_k0}")))
        e0a.append((f"--opp_dg{u}", wrap_calc(f"{opp_e}*({is_b0} + {is_q0})")))
        e0a.append((f"--opp_or{u}", wrap_calc(f"{opp_e}*({is_r0} + {is_q0})")))
        e0a.append((f"--onk{u}", wrap_calc(f"{occ_e} - {own_e}*{is_k0}")))

        pt_v = f"var(--pt{u})"
        is_p = css_eq(pt_v, PT_PAWN)
        is_n = css_eq(pt_v, PT_KNIGHT)
        is_b = css_eq(pt_v, PT_BISHOP)
        is_r = css_eq(pt_v, PT_ROOK)
        is_q = css_eq(pt_v, PT_QUEEN)
        is_k = css_eq(pt_v, PT_KING)

        e2_roles.append((f"--opp_p{u}", wrap_calc(f"var(--opp{u})*{is_p}")))
        e2_roles.append((f"--opp_n{u}", wrap_calc(f"var(--opp{u})*{is_n}")))
        e2_roles.append((f"--opp_k{u}", wrap_calc(f"var(--opp{u})*{is_k}")))

        e4_roles.append((f"--own_p{u}", wrap_calc(f"var(--own{u})*{is_p}")))
        e4_roles.append((f"--own_n{u}", wrap_calc(f"var(--own{u})*{is_n}")))
        e4_roles.append((f"--own_b{u}", wrap_calc(f"var(--own{u})*{is_b}")))
        e4_roles.append((f"--own_r{u}", wrap_calc(f"var(--own{u})*{is_r}")))
        e4_roles.append((f"--own_q{u}", wrap_calc(f"var(--own{u})*{is_q}")))
        e4_roles.append((f"--own_dg{u}", wrap_calc(f"var(--own{u})*({is_b} + {is_q})")))
        e4_roles.append((f"--own_or{u}", wrap_calc(f"var(--own{u})*({is_r} + {is_q})")))
    stage_defs["e0a"] = e0a

    # Helper to emit a 2-stage Radix-3 Kogge-Stone `first_hit` ray scan (`3 -> 7` steps),
    # packing the 3-step empty flag `e3` directly into `h3` (`h3 + emp_scale * e3`)!
    def add_radix3_ray_scan(
        s_blk3_list: List[Tuple[str, str]],
        s_full7_list: List[Tuple[str, str]],
        prefix: str,
        prop_expr_fn,
        occ_var_fn,
        allow_empty_prop: bool = False,
        emp_scale: int = 2,
        pack_emp: bool = True,
        ext_emp_prefix: Optional[str] = None,
        ext_emp_scale: int = 2,
        collapse_or_by_sq: bool = False,
    ) -> None:
        for d in range(8):
            for u in range(64):
                u1 = _step_sq(u, d, 1)
                if u1 is None:
                    continue
                u2 = _step_sq(u, d, 2)
                u3 = _step_sq(u, d, 3)
                e_u1 = f"(1 - {occ_var_fn(u1)})"
                h_terms = [prop_expr_fn(u1, d)]
                if u2 is not None:
                    h_terms.append(f"{e_u1}*{prop_expr_fn(u2, d)}")
                if u3 is not None and u2 is not None:
                    e_u2 = f"(1 - {occ_var_fn(u2)})"
                    h_terms.append(f"{e_u1}*{e_u2}*{prop_expr_fn(u3, d)}")
                expr_h3 = css_or_terms(h_terms) if allow_empty_prop else " + ".join(h_terms)

                if pack_emp and u3 is not None and u2 is not None:
                    e_u2 = f"(1 - {occ_var_fn(u2)})"
                    e_u3 = f"(1 - {occ_var_fn(u3)})"
                    s_blk3_list.append((f"--{prefix}3_{d}_{u}", wrap_calc(f"({expr_h3}) + {emp_scale}*{e_u1}*{e_u2}*{e_u3}")))
                else:
                    s_blk3_list.append((f"--{prefix}3_{d}_{u}", wrap_calc(expr_h3)))

        def _h3_val(d: int, sq: int) -> str:
            if pack_emp and _step_sq(sq, d, 3) is not None:
                return f"mod(var(--{prefix}3_{d}_{sq}),{emp_scale})"
            return f"var(--{prefix}3_{d}_{sq})"

        def _e3_val(d: int, sq: int) -> str:
            src_p = ext_emp_prefix or prefix
            sc = ext_emp_scale if ext_emp_prefix else emp_scale
            return f"clamp(0,var(--{src_p}3_{d}_{sq}) - {sc - 1},1)"

        if collapse_or_by_sq:
            for u in range(64):
                sq_terms: List[str] = []
                for d in range(8):
                    if _step_sq(u, d, 1) is None:
                        continue
                    u3 = _step_sq(u, d, 3)
                    u4 = _step_sq(u, d, 4)
                    u6 = _step_sq(u, d, 6)
                    u7 = _step_sq(u, d, 7)
                    sq_terms.append(_h3_val(d, u))
                    if u4 is not None and u3 is not None:
                        sq_terms.append(f"{_e3_val(d, u)}*{_h3_val(d, u3)}")
                    if u7 is not None and u6 is not None and u3 is not None:
                        sq_terms.append(f"{_e3_val(d, u)}*{_e3_val(d, u3)}*{_h3_val(d, u6)}")
                s_full7_list.append((f"--{prefix}_any_{u}", wrap_calc(css_or_terms(sq_terms))))
        else:
            for d in range(8):
                for u in range(64):
                    if _step_sq(u, d, 1) is None:
                        continue
                    u3 = _step_sq(u, d, 3)
                    u4 = _step_sq(u, d, 4)
                    u6 = _step_sq(u, d, 6)
                    u7 = _step_sq(u, d, 7)
                    h_terms = [_h3_val(d, u)]
                    if u4 is not None and u3 is not None:
                        h_terms.append(f"{_e3_val(d, u)}*{_h3_val(d, u3)}")
                    if u7 is not None and u6 is not None and u3 is not None:
                        h_terms.append(f"{_e3_val(d, u)}*{_e3_val(d, u3)}*{_h3_val(d, u6)}")
                    expr_h7 = css_or_terms(h_terms) if allow_empty_prop else " + ".join(h_terms)
                    s_full7_list.append((f"--{prefix}7_{d}_{u}", wrap_calc(expr_h7)))

    # .e0b, .e1, .e2: Staggered `ka` (`.e0b -> .e1`) & `hks` (`.e1 -> .e2`) Radix-3 Ray Scans!
    e1: List[Tuple[str, str]] = []
    e2: List[Tuple[str, str]] = list(e2_roles)

    def _opp_slider_var(sq: int, d: int) -> str:
        return f"var(--opp_or{sq})" if d < 4 else f"var(--opp_dg{sq})"

    def _packed_hks_expr(sq: int, d: int) -> str:
        return f"(var(--own_k{sq}) + 2*{_opp_slider_var(sq, d)})"

    add_radix3_ray_scan(
        e0b, e1,
        prefix="ka",
        prop_expr_fn=_opp_slider_var,
        occ_var_fn=lambda sq: f"var(--onk{sq})",
        allow_empty_prop=False,
        emp_scale=2,
        pack_emp=True,
        collapse_or_by_sq=True,
    )
    add_radix3_ray_scan(
        e1, e2,
        prefix="hks",
        prop_expr_fn=_packed_hks_expr,
        occ_var_fn=lambda sq: f"var(--occ{sq})",
        allow_empty_prop=False,
        emp_scale=4,
        pack_emp=True,
        collapse_or_by_sq=False,
    )

    stage_defs["e0b"] = e0b
    stage_defs["e1"] = e1
    stage_defs["e2"] = e2

    # .e3: `kingless_att[0..63]`, `is_checker[0..63]`, `pin_dir[0..63]`,
    #      `between_k_slider[0..63]`, AND horizontal EP rank pins!
    e3: List[Tuple[str, str]] = []

    def _hk7_expr(d: int, sq: int) -> str:
        return f"mod(var(--hks7_{d}_{sq}),2)"

    def _hs7_expr(d: int, sq: int) -> str:
        return f"clamp(0,var(--hks7_{d}_{sq}) - 1,1)"

    for u in range(64):
        # 1. `kingless_att[u]`: attacked by enemy slider (on `onk`), pawn, knight, or king?
        att_terms: List[str] = [f"var(--ka_any_{u})"]
        bp_att: List[str] = []
        wp_att: List[str] = []
        for df in (-1, 1):
            sq_above = _delta_sq(u, 1, df)
            if sq_above is not None:
                bp_att.append(f"var(--opp_p{sq_above})")
            sq_below = _delta_sq(u, -1, df)
            if sq_below is not None:
                wp_att.append(f"var(--opp_p{sq_below})")
        if bp_att:
            att_terms.append(f"(1 - var(--turn))*({' + '.join(bp_att)})")
        if wp_att:
            att_terms.append(f"var(--turn)*({' + '.join(wp_att)})")

        kn_att = [
            f"var(--opp_n{w})"
            for dr, df in KNIGHT_DELTAS
            if (w := _delta_sq(u, dr, df)) is not None
        ]
        if kn_att:
            att_terms.append(" + ".join(kn_att))

        kg_att = [
            f"var(--opp_k{w})"
            for dr, df in KING_DELTAS
            if (w := _delta_sq(u, dr, df)) is not None
        ]
        if kg_att:
            att_terms.append(" + ".join(kg_att))

        e3.append((f"--katt{u}", wrap_calc(css_or_terms(att_terms))))

        # 2. `is_checker[u]`: does an enemy piece AT `u` directly check our friendly King?
        chk_parts: List[str] = []
        orth_hit_k = [
            _hk7_expr(d, u) for d in range(4) if _step_sq(u, d, 1) is not None
        ]
        diag_hit_k = [
            _hk7_expr(d, u) for d in range(4, 8) if _step_sq(u, d, 1) is not None
        ]
        if orth_hit_k:
            chk_parts.append(f"var(--opp_or{u})*({' + '.join(orth_hit_k)})")
        if diag_hit_k:
            chk_parts.append(f"var(--opp_dg{u})*({' + '.join(diag_hit_k)})")

        kn_k = [
            f"var(--own_k{w})"
            for dr, df in KNIGHT_DELTAS
            if (w := _delta_sq(u, dr, df)) is not None
        ]
        if kn_k:
            chk_parts.append(f"var(--opp_n{u})*({' + '.join(kn_k)})")

        bp_chk_k = [f"var(--own_k{w})" for df in (-1, 1) if (w := _delta_sq(u, -1, df)) is not None]
        wp_chk_k = [f"var(--own_k{w})" for df in (-1, 1) if (w := _delta_sq(u, 1, df)) is not None]
        p_chk_sub: List[str] = []
        if bp_chk_k:
            p_chk_sub.append(f"(1 - var(--turn))*({' + '.join(bp_chk_k)})")
        if wp_chk_k:
            p_chk_sub.append(f"var(--turn)*({' + '.join(wp_chk_k)})")
        if p_chk_sub:
            chk_parts.append(f"var(--opp_p{u})*({' + '.join(p_chk_sub)})")

        e3.append((f"--ichk{u}", wrap_calc(css_sum_terms(chk_parts))))

        # 3. `pin_dir[u]` (0 = none, 1 = H, 2 = V, 3 = /, 4 = \) & `between_k_slider[u]` (0/1)
        pin_axis_terms: List[str] = []
        btw_terms: List[str] = []
        for d in range(8):
            opp_d = d ^ 1
            if _step_sq(u, d, 1) is not None and _step_sq(u, opp_d, 1) is not None:
                _, _, axis_id = RAY_DIRS[d]
                seg = f"{_hk7_expr(opp_d, u)}*{_hs7_expr(d, u)}"
                pin_axis_terms.append(f"{axis_id}*{seg}")
                btw_terms.append(seg)
        e3.append((f"--pdir{u}", wrap_calc(css_sum_terms(pin_axis_terms))))
        e3.append((f"--btwk{u}", wrap_calc(css_sum_terms(btw_terms))))

    # Horizontal EP rank discovered-check flags for adjacent file pairs c in 0..6
    for prefix_ep, r_ep in (("ep_hpin_w", 4), ("ep_hpin_b", 3)):
        base = 8 * r_ep
        for c in range(7):
            lk_terms: List[str] = []
            lrq_terms: List[str] = []
            for i in range(c):
                clear_mid = [f"(1 - var(--occ{base + j}))" for j in range(i + 1, c)]
                prod_mid = ("*" + "*".join(clear_mid)) if clear_mid else ""
                lk_terms.append(f"var(--own_k{base + i}){prod_mid}")
                lrq_terms.append(f"var(--opp_or{base + i}){prod_mid}")

            rk_terms: List[str] = []
            rrq_terms: List[str] = []
            for i in range(c + 2, 8):
                clear_mid = [f"(1 - var(--occ{base + j}))" for j in range(c + 2, i)]
                prod_mid = ("*" + "*".join(clear_mid)) if clear_mid else ""
                rk_terms.append(f"var(--own_k{base + i}){prod_mid}")
                rrq_terms.append(f"var(--opp_or{base + i}){prod_mid}")

            if lk_terms and rrq_terms:
                t1 = f"{css_sum_terms(lk_terms)}*{css_sum_terms(rrq_terms)}"
                t2 = f"{css_sum_terms(lrq_terms)}*{css_sum_terms(rk_terms)}"
                e3.append((f"--{prefix_ep}{c}", wrap_calc(f"min(1,{t1} + {t2})")))
            else:
                e3.append((f"--{prefix_ep}{c}", "0"))

    stage_defs["e3"] = e3

    # .e4: `--num_checkers` & deferred own-piece role flags
    e4: List[Tuple[str, str]] = list(e4_roles)
    e4.append(("--num_checkers", wrap_calc(" + ".join(f"var(--ichk{u})" for u in range(64)))))
    stage_defs["e4"] = e4

    # .e5: `evade[0..63]`, `vtgt[0..63]`, & selected-piece metadata
    e5: List[Tuple[str, str]] = []
    no_chk = css_eq("var(--num_checkers)", 0)
    one_chk = css_eq("var(--num_checkers)", 1)

    for v in range(64):
        ev_single = f"min(1,var(--ichk{v}) + (1 - var(--occ{v}))*var(--btwk{v}))"
        ev_expr = f"({no_chk} + {one_chk}*{ev_single})"
        e5.append((f"--evd{v}", wrap_calc(ev_expr)))
        e5.append((f"--vtgt{v}", wrap_calc(f"(1 - var(--own{v}))*{ev_expr}")))

    for role_out, src_prefix in (
        ("--sel_p", "own_p"),
        ("--sel_n", "own_n"),
        ("--sel_dg", "own_dg"),
        ("--sel_or", "own_or"),
        ("--sel_k", "own_k"),
        ("--sel_pdir", "pdir"),
    ):
        e5.append((
            role_out,
            wrap_calc(" + ".join(f"var(--isel{u})*var(--{src_prefix}{u})" for u in range(64))),
        ))

    w_pr_terms = [f"var(--isel{u})*var(--own_p{u})" for u in range(48, 56)]
    b_pr_terms = [f"var(--isel{u})*var(--own_p{u})" for u in range(8, 16)]
    e5.append((
        "--sel_can_promo",
        wrap_calc(f"(1 - var(--turn))*({' + '.join(w_pr_terms)}) + var(--turn)*({' + '.join(b_pr_terms)})"),
    ))
    stage_defs["e5"] = e5

    # .e6 & .e7: En Passant, Castling, Pin Axes, & `rv` 2-Stage Radix-3 Ray Scan (`rv3` in e6, `rv7` in e7)
    e6: List[Tuple[str, str]] = []
    e7: List[Tuple[str, str]] = []

    for ef in range(8):
        t_sq = 40 + ef
        vic_sq = 32 + ef
        for df, cap_ax in ((-1, 3), (1, 4)):
            uf = ef + df
            if not (0 <= uf < 8):
                continue
            u_sq = 32 + uf
            c_pair = min(uf, ef)
            pin_u_ok = f"({css_eq(f'var(--pdir{u_sq})', 0)} + {css_eq(f'var(--pdir{u_sq})', cap_ax)})"
            pin_vic_ok = f"(1 - {css_eq(f'var(--pdir{vic_sq})', 3)} - {css_eq(f'var(--pdir{vic_sq})', 4)})"
            chk_ep_ok = f"({no_chk} + {one_chk}*min(1,var(--evd{t_sq}) + var(--ichk{vic_sq})))"
            ep_expr = (
                f"(1 - var(--turn))*var(--own_p{u_sq})*{css_eq('var(--ep_sq)', t_sq)}"
                f"*(1 - var(--occ{t_sq}))*var(--opp_p{vic_sq})"
                f"*{pin_u_ok}*{pin_vic_ok}*{chk_ep_ok}*(1 - var(--ep_hpin_w{c_pair}))"
            )
            e6.append((f"--epw_{uf}_{ef}", wrap_calc(ep_expr)))

    for ef in range(8):
        t_sq = 16 + ef
        vic_sq = 24 + ef
        for df, cap_ax in ((-1, 4), (1, 3)):
            uf = ef + df
            if not (0 <= uf < 8):
                continue
            u_sq = 24 + uf
            c_pair = min(uf, ef)
            pin_u_ok = f"({css_eq(f'var(--pdir{u_sq})', 0)} + {css_eq(f'var(--pdir{u_sq})', cap_ax)})"
            pin_vic_ok = f"(1 - {css_eq(f'var(--pdir{vic_sq})', 3)} - {css_eq(f'var(--pdir{vic_sq})', 4)})"
            chk_ep_ok = f"({no_chk} + {one_chk}*min(1,var(--evd{t_sq}) + var(--ichk{vic_sq})))"
            ep_expr = (
                f"var(--turn)*var(--own_p{u_sq})*{css_eq('var(--ep_sq)', t_sq)}"
                f"*(1 - var(--occ{t_sq}))*var(--opp_p{vic_sq})"
                f"*{pin_u_ok}*{pin_vic_ok}*{chk_ep_ok}*(1 - var(--ep_hpin_b{c_pair}))"
            )
            e6.append((f"--epb_{uf}_{ef}", wrap_calc(ep_expr)))

    c_wk_leg = (
        f"(1 - var(--turn))*var(--cast_wk)*{css_eq('var(--b4)', 6)}*{css_eq('var(--b7)', 4)}"
        f"*(1 - var(--occ5))*(1 - var(--occ6))"
        f"*(1 - var(--katt4))*(1 - var(--katt5))*(1 - var(--katt6))"
    )
    c_wq_leg = (
        f"(1 - var(--turn))*var(--cast_wq)*{css_eq('var(--b4)', 6)}*{css_eq('var(--b0)', 4)}"
        f"*(1 - var(--occ1))*(1 - var(--occ2))*(1 - var(--occ3))"
        f"*(1 - var(--katt4))*(1 - var(--katt3))*(1 - var(--katt2))"
    )
    c_bk_leg = (
        f"var(--turn)*var(--cast_bk)*{css_eq('var(--b60)', 14)}*{css_eq('var(--b63)', 12)}"
        f"*(1 - var(--occ61))*(1 - var(--occ62))"
        f"*(1 - var(--katt60))*(1 - var(--katt61))*(1 - var(--katt62))"
    )
    c_bq_leg = (
        f"var(--turn)*var(--cast_bq)*{css_eq('var(--b60)', 14)}*{css_eq('var(--b56)', 12)}"
        f"*(1 - var(--occ57))*(1 - var(--occ58))*(1 - var(--occ59))"
        f"*(1 - var(--katt60))*(1 - var(--katt59))*(1 - var(--katt58))"
    )
    e6.append(("--cleg_wk", wrap_calc(c_wk_leg)))
    e6.append(("--cleg_wq", wrap_calc(c_wq_leg)))
    e6.append(("--cleg_bk", wrap_calc(c_bk_leg)))
    e6.append(("--cleg_bq", wrap_calc(c_bq_leg)))

    sp0_e = "(1 - min(1,var(--sel_pdir)))"
    e6.append(("--sp0", wrap_calc(sp0_e)))
    for ax in (1, 2, 3, 4):
        e6.append((f"--sp_ax{ax}", wrap_calc(f"{sp0_e} + {css_eq('var(--sel_pdir)', ax)}")))
    for u in range(64):
        p0_u = f"(1 - min(1,var(--pdir{u})))"
        for ax in (1, 2, 3, 4):
            e6.append((f"--pax{u}_{ax}", wrap_calc(f"{p0_u} + {css_eq(f'var(--pdir{u})', ax)}")))

    add_radix3_ray_scan(
        e6, e7,
        prefix="rv",
        prop_expr_fn=lambda sq, _d: f"var(--vtgt{sq})",
        occ_var_fn=lambda sq: f"var(--occ{sq})",
        allow_empty_prop=True,
        emp_scale=2,
        pack_emp=True,
    )
    stage_defs["e6"] = e6
    stage_defs["e7"] = e7

    # .e8 & .e9: `--has_leg_0..63`, material counts, `msl` Ray Scan (`msl3` in e8, `msl7` in e9,
    #            reusing `rv`'s temporary registers & replacing the redundant 840-rule `sr` scan!),
    #            and terminal game flags in `.e9`.
    e8: List[Tuple[str, str]] = []
    e8b: List[Tuple[str, str]] = []
    e9: List[Tuple[str, str]] = []

    def _msl_prop_expr(sq: int, d_scan: int) -> str:
        ax = RAY_DIRS[d_scan][2]
        if d_scan < 4:
            return f"((64 + {sq})*var(--own_r{sq}) + (128 + {sq})*var(--own_q{sq}))*var(--pax{sq}_{ax})"
        return f"((64 + {sq})*var(--own_b{sq}) + (128 + {sq})*var(--own_q{sq}))*var(--pax{sq}_{ax})"

    add_radix3_ray_scan(
        e8b, e9,
        prefix="msl",
        prop_expr_fn=_msl_prop_expr,
        occ_var_fn=lambda sq: f"var(--occ{sq})",
        allow_empty_prop=False,
        emp_scale=256,
        pack_emp=True,
    )

    for u in range(64):
        ur, uf = divmod(u, 8)
        p0 = f"(1 - min(1,var(--pdir{u})))"
        ax_ok = {ax: f"var(--pax{u}_{ax})" for ax in (1, 2, 3, 4)}

        hl_parts: List[str] = []

        p_sub: List[str] = []
        if ur < 7:
            wp_mvs: List[str] = [f"(1 - var(--occ{u + 8}))*var(--evd{u + 8})*{ax_ok[2]}"]
            if ur == 1:
                wp_mvs.append(f"(1 - var(--occ{u + 8}))*(1 - var(--occ{u + 16}))*var(--evd{u + 16})*{ax_ok[2]}")
            if uf < 7:
                wp_mvs.append(f"var(--opp{u + 9})*var(--evd{u + 9})*{ax_ok[3]}")
            if uf > 0:
                wp_mvs.append(f"var(--opp{u + 7})*var(--evd{u + 7})*{ax_ok[4]}")
            if ur == 4:
                if uf > 0:
                    wp_mvs.append(f"var(--epw_{uf}_{uf - 1})")
                if uf < 7:
                    wp_mvs.append(f"var(--epw_{uf}_{uf + 1})")
            p_sub.append(f"(1 - var(--turn))*min(1,{' + '.join(wp_mvs)})")

        if ur > 0:
            bp_mvs: List[str] = [f"(1 - var(--occ{u - 8}))*var(--evd{u - 8})*{ax_ok[2]}"]
            if ur == 6:
                bp_mvs.append(f"(1 - var(--occ{u - 8}))*(1 - var(--occ{u - 16}))*var(--evd{u - 16})*{ax_ok[2]}")
            if uf > 0:
                bp_mvs.append(f"var(--opp{u - 9})*var(--evd{u - 9})*{ax_ok[3]}")
            if uf < 7:
                bp_mvs.append(f"var(--opp{u - 7})*var(--evd{u - 7})*{ax_ok[4]}")
            if ur == 3:
                if uf > 0:
                    bp_mvs.append(f"var(--epb_{uf}_{uf - 1})")
                if uf < 7:
                    bp_mvs.append(f"var(--epb_{uf}_{uf + 1})")
            p_sub.append(f"var(--turn)*min(1,{' + '.join(bp_mvs)})")

        if p_sub:
            hl_parts.append(f"var(--own_p{u})*({' + '.join(p_sub)})")

        kn_tgts = [
            f"var(--vtgt{w})"
            for dr, df in KNIGHT_DELTAS
            if (w := _delta_sq(u, dr, df)) is not None
        ]
        if kn_tgts:
            hl_parts.append(f"var(--own_n{u})*{p0}*{css_or_terms(kn_tgts)}")

        or_rays = [
            f"{ax_ok[RAY_DIRS[d][2]]}*var(--rv7_{d}_{u})"
            for d in range(4)
            if _step_sq(u, d, 1) is not None
        ]
        if or_rays:
            hl_parts.append(f"var(--own_or{u})*{css_or_terms(or_rays)}")

        dg_rays = [
            f"{ax_ok[RAY_DIRS[d][2]]}*var(--rv7_{d}_{u})"
            for d in range(4, 8)
            if _step_sq(u, d, 1) is not None
        ]
        if dg_rays:
            hl_parts.append(f"var(--own_dg{u})*{css_or_terms(dg_rays)}")

        kg_tgts = [
            f"(1 - var(--own{w}))*(1 - var(--katt{w}))"
            for dr, df in KING_DELTAS
            if (w := _delta_sq(u, dr, df)) is not None
        ]
        if u == 4:
            kg_tgts.extend(["var(--cleg_wk)", "var(--cleg_wq)"])
        elif u == 60:
            kg_tgts.extend(["var(--cleg_bk)", "var(--cleg_bq)"])
        hl_parts.append(f"var(--own_k{u})*{css_or_terms(kg_tgts)}")

        e8.append((f"--has_leg_{u}", wrap_calc(css_or_terms(hl_parts))))

    prq_terms: List[str] = []
    n_terms: List[str] = []
    bl_terms: List[str] = []
    bd_terms: List[str] = []
    for u in range(64):
        ur, uf = divmod(u, 8)
        pt_u = f"var(--pt{u})"
        prq_terms.append(f"({css_eq(pt_u, PT_PAWN)} + {css_eq(pt_u, PT_ROOK)} + {css_eq(pt_u, PT_QUEEN)})")
        n_terms.append(css_eq(pt_u, PT_KNIGHT))
        if (ur + uf) % 2 == 1:
            bl_terms.append(css_eq(pt_u, PT_BISHOP))
        else:
            bd_terms.append(css_eq(pt_u, PT_BISHOP))

    e8.append(("--mat_prq", wrap_calc(" + ".join(prq_terms))))
    e8.append(("--mat_n", wrap_calc(" + ".join(n_terms))))
    e8.append(("--mat_bl", wrap_calc(" + ".join(bl_terms))))
    e8.append(("--mat_bd", wrap_calc(" + ".join(bd_terms))))
    stage_defs["e8"] = e8
    stage_defs["e8b"] = e8b

    any_leg_s = f"min(1,{' + '.join(f'var(--has_leg_{u})' for u in range(64))})"
    in_chk_s = "min(1,var(--num_checkers))"
    mate_s = f"({in_chk_s}*(1 - {any_leg_s}))"
    stale_s = f"((1 - {in_chk_s})*(1 - {any_leg_s}))"

    no_prq = "(1 - min(1,var(--mat_prq)))"
    n_eq0 = "(1 - min(1,var(--mat_n)))"
    n_eq1 = css_eq("var(--mat_n)", 1)
    bl_eq0 = "(1 - min(1,var(--mat_bl)))"
    bd_eq0 = "(1 - min(1,var(--mat_bd)))"
    insuf_raw = f"({no_prq}*min(1,{n_eq1}*{bl_eq0}*{bd_eq0} + {n_eq0}*min(1,{bl_eq0} + {bd_eq0})))"
    insuf_s = f"({any_leg_s}*{insuf_raw})"
    d50_s = f"({any_leg_s}*(1 - {insuf_raw})*clamp(0,var(--hmc) - 99,1))"
    term_s = f"min(1,(1 - {any_leg_s}) + {insuf_raw} + clamp(0,var(--hmc) - 99,1))"

    e9.append(("--any_leg", wrap_calc(any_leg_s)))
    e9.append(("--in_check", wrap_calc(in_chk_s)))
    e9.append(("--checkmate", wrap_calc(mate_s)))
    e9.append(("--stalemate", wrap_calc(stale_s)))
    e9.append(("--draw_insuf", wrap_calc(insuf_s)))
    e9.append(("--draw_50", wrap_calc(d50_s)))
    e9.append(("--is_terminal", wrap_calc(term_s)))
    stage_defs["e9"] = e9

    # .e10: Selected destination `--rleg_0..63` & Ply-1 piece candidates `--mc_*_0..63`
    e10: List[Tuple[str, str]] = []
    sp0 = "var(--sp0)"
    sp_ax = {ax: f"var(--sp_ax{ax})" for ax in (1, 2, 3, 4)}

    for v in range(64):
        vr, vf = divmod(v, 8)
        dst_parts: List[str] = []

        vp_terms: List[str] = []
        if vr >= 2:
            vp_terms.append(f"(1 - var(--turn))*var(--isel{v - 8})*(1 - var(--occ{v}))*var(--evd{v})*{sp_ax[2]}")
            if vr == 3:
                vp_terms.append(
                    f"(1 - var(--turn))*var(--isel{v - 16})*(1 - var(--occ{v - 8}))*(1 - var(--occ{v}))*var(--evd{v})*{sp_ax[2]}"
                )
            if vf > 0:
                vp_terms.append(f"(1 - var(--turn))*var(--isel{v - 9})*var(--opp{v})*var(--evd{v})*{sp_ax[3]}")
            if vf < 7:
                vp_terms.append(f"(1 - var(--turn))*var(--isel{v - 7})*var(--opp{v})*var(--evd{v})*{sp_ax[4]}")
            if vr == 5:
                if vf > 0:
                    vp_terms.append(f"var(--isel{32 + vf - 1})*var(--epw_{vf - 1}_{vf})")
                if vf < 7:
                    vp_terms.append(f"var(--isel{32 + vf + 1})*var(--epw_{vf + 1}_{vf})")

        if vr <= 5:
            vp_terms.append(f"var(--turn)*var(--isel{v + 8})*(1 - var(--occ{v}))*var(--evd{v})*{sp_ax[2]}")
            if vr == 4:
                vp_terms.append(
                    f"var(--turn)*var(--isel{v + 16})*(1 - var(--occ{v + 8}))*(1 - var(--occ{v}))*var(--evd{v})*{sp_ax[2]}"
                )
            if vf < 7:
                vp_terms.append(f"var(--turn)*var(--isel{v + 9})*var(--opp{v})*var(--evd{v})*{sp_ax[3]}")
            if vf > 0:
                vp_terms.append(f"var(--turn)*var(--isel{v + 7})*var(--opp{v})*var(--evd{v})*{sp_ax[4]}")
            if vr == 2:
                if vf > 0:
                    vp_terms.append(f"var(--isel{24 + vf - 1})*var(--epb_{vf - 1}_{vf})")
                if vf < 7:
                    vp_terms.append(f"var(--isel{24 + vf + 1})*var(--epb_{vf + 1}_{vf})")

        if vp_terms:
            dst_parts.append(f"var(--sel_p)*{css_or_terms(vp_terms)}")

        kn_src = [
            f"var(--isel{u})"
            for dr, df in KNIGHT_DELTAS
            if (u := _delta_sq(v, dr, df)) is not None
        ]
        if kn_src:
            dst_parts.append(f"var(--sel_n)*{sp0}*var(--vtgt{v})*({' + '.join(kn_src)})")

        or_src = [
            f"clamp(0,var(--msl7_{d ^ 1}_{v}),1)*{css_eq_vars(f'mod(var(--msl7_{d ^ 1}_{v}),64)', 'var(--cur_from)')}"
            for d in range(4)
            if _step_sq(v, d ^ 1, 1) is not None
        ]
        if or_src:
            dst_parts.append(f"var(--sel_or)*var(--vtgt{v})*({' + '.join(or_src)})")

        dg_src = [
            f"clamp(0,var(--msl7_{d ^ 1}_{v}),1)*{css_eq_vars(f'mod(var(--msl7_{d ^ 1}_{v}),64)', 'var(--cur_from)')}"
            for d in range(4, 8)
            if _step_sq(v, d ^ 1, 1) is not None
        ]
        if dg_src:
            dst_parts.append(f"var(--sel_dg)*var(--vtgt{v})*({' + '.join(dg_src)})")

        kg_src = [
            f"var(--isel{u})*(1 - var(--own{v}))*(1 - var(--katt{v}))"
            for dr, df in KING_DELTAS
            if (u := _delta_sq(v, dr, df)) is not None
        ]
        if v == 6:
            kg_src.append("var(--isel4)*var(--cleg_wk)")
        elif v == 2:
            kg_src.append("var(--isel4)*var(--cleg_wq)")
        elif v == 62:
            kg_src.append("var(--isel60)*var(--cleg_bk)")
        elif v == 58:
            kg_src.append("var(--isel60)*var(--cleg_bq)")
        if kg_src:
            dst_parts.append(f"var(--sel_k)*{css_or_terms(kg_src)}")

        e10.append((f"--rleg_{v}", wrap_calc(css_or_terms(dst_parts))))

    def _chain_first(pairs: Sequence[Tuple[str, str]]) -> str:
        if not pairs:
            return "0"
        g_cur, val_cur = pairs[0]
        if len(pairs) == 1:
            return f"{g_cur}*{val_cur}"
        rest = _chain_first(pairs[1:])
        return f"({g_cur}*{val_cur} + (1 - {g_cur})*{rest})"

    for v in range(64):
        vr, vf = divmod(v, 8)
        # 1. Pawn candidates (`64 + u` normal/promo, `128 + u` en passant)
        p_pairs: List[Tuple[str, str]] = []
        if vr >= 2:
            u1 = v - 8
            p_pairs.append((f"((1 - var(--turn))*var(--own_p{u1})*(1 - var(--occ{v}))*var(--evd{v})*var(--pax{u1}_2))", str(64 + u1)))
            if vr == 3:
                u2 = v - 16
                p_pairs.append((f"((1 - var(--turn))*var(--own_p{u2})*(1 - var(--occ{v - 8}))*(1 - var(--occ{v}))*var(--evd{v})*var(--pax{u2}_2))", str(64 + u2)))
            if vf > 0:
                ul = v - 9
                p_pairs.append((f"((1 - var(--turn))*var(--own_p{ul})*var(--opp{v})*var(--evd{v})*var(--pax{ul}_3))", str(64 + ul)))
            if vf < 7:
                ur_s = v - 7
                p_pairs.append((f"((1 - var(--turn))*var(--own_p{ur_s})*var(--opp{v})*var(--evd{v})*var(--pax{ur_s}_4))", str(64 + ur_s)))
            if vr == 5:
                if vf > 0:
                    p_pairs.append((f"var(--epw_{vf - 1}_{vf})", str(128 + 32 + vf - 1)))
                if vf < 7:
                    p_pairs.append((f"var(--epw_{vf + 1}_{vf})", str(128 + 32 + vf + 1)))
        if vr <= 5:
            u1 = v + 8
            p_pairs.append((f"(var(--turn)*var(--own_p{u1})*(1 - var(--occ{v}))*var(--evd{v})*var(--pax{u1}_2))", str(64 + u1)))
            if vr == 4:
                u2 = v + 16
                p_pairs.append((f"(var(--turn)*var(--own_p{u2})*(1 - var(--occ{v + 8}))*(1 - var(--occ{v}))*var(--evd{v})*var(--pax{u2}_2))", str(64 + u2)))
            if vf < 7:
                ul = v + 9
                p_pairs.append((f"(var(--turn)*var(--own_p{ul})*var(--opp{v})*var(--evd{v})*var(--pax{ul}_3))", str(64 + ul)))
            if vf > 0:
                ur_s = v + 7
                p_pairs.append((f"(var(--turn)*var(--own_p{ur_s})*var(--opp{v})*var(--evd{v})*var(--pax{ur_s}_4))", str(64 + ur_s)))
            if vr == 2:
                if vf > 0:
                    p_pairs.append((f"var(--epb_{vf - 1}_{vf})", str(128 + 24 + vf - 1)))
                if vf < 7:
                    p_pairs.append((f"var(--epb_{vf + 1}_{vf})", str(128 + 24 + vf + 1)))
        e10.append((f"--mc_p_{v}", wrap_calc(_chain_first(p_pairs))))

        # 2. Knight candidates (`64 + u`)
        n_pairs = [
            (f"(var(--own_n{u})*(1 - min(1,var(--pdir{u}))))", str(64 + u))
            for dr, df in KNIGHT_DELTAS
            if (u := _delta_sq(v, dr, df)) is not None
        ]
        e10.append((f"--mc_n_{v}", wrap_calc(f"var(--vtgt{v})*{_chain_first(n_pairs)}" if n_pairs else "0")))

        # 3. Bishop candidates (`64 + u` from diagonal rays)
        b_pairs = [
            (
                f"(clamp(0,var(--msl7_{d ^ 1}_{v}) - 63,1)*(1 - clamp(0,var(--msl7_{d ^ 1}_{v}) - 127,1)))",
                f"var(--msl7_{d ^ 1}_{v})",
            )
            for d in range(4, 8)
            if _step_sq(v, d ^ 1, 1) is not None
        ]
        e10.append((f"--mc_b_{v}", wrap_calc(f"var(--vtgt{v})*{_chain_first(b_pairs)}" if b_pairs else "0")))

        # 4. Rook candidates (`64 + u` from orthogonal rays)
        r_pairs = [
            (
                f"(clamp(0,var(--msl7_{d ^ 1}_{v}) - 63,1)*(1 - clamp(0,var(--msl7_{d ^ 1}_{v}) - 127,1)))",
                f"var(--msl7_{d ^ 1}_{v})",
            )
            for d in range(0, 4)
            if _step_sq(v, d ^ 1, 1) is not None
        ]
        e10.append((f"--mc_r_{v}", wrap_calc(f"var(--vtgt{v})*{_chain_first(r_pairs)}" if r_pairs else "0")))

        # 5. Queen candidates (`64 + u` from all 8 rays where msl7 >= 128)
        q_pairs = [
            (
                f"clamp(0,var(--msl7_{d ^ 1}_{v}) - 127,1)",
                f"(var(--msl7_{d ^ 1}_{v}) - 64)",
            )
            for d in range(0, 8)
            if _step_sq(v, d ^ 1, 1) is not None
        ]
        e10.append((f"--mc_q_{v}", wrap_calc(f"var(--vtgt{v})*{_chain_first(q_pairs)}" if q_pairs else "0")))

        # 6. King candidates (`64 + u` normal, `128 + u` O-O, `192 + u` O-O-O; mutually exclusive!)
        k_terms = [
            f"var(--own_k{u})*(1 - var(--own{v}))*(1 - var(--katt{v}))*{64 + u}"
            for dr, df in KING_DELTAS
            if (u := _delta_sq(v, dr, df)) is not None
        ]
        if v == 6:
            k_terms.append(f"var(--cleg_wk)*{128 + 4}")
        elif v == 2:
            k_terms.append(f"var(--cleg_wq)*{192 + 4}")
        elif v == 62:
            k_terms.append(f"var(--cleg_bk)*{128 + 60}")
        elif v == 58:
            k_terms.append(f"var(--cleg_bq)*{192 + 60}")
        e10.append((f"--mc_k_{v}", wrap_calc(css_sum_terms(k_terms))))

        # 7. Precompute NBRQ chain (`--mc_nbrq_u_{v}`, `--mc_nbrq_c_{v}`) in e10? No — cannot read `--mc_n_{v}` in e10!
    stage_defs["e10"] = e10

    # .e11: Status readout, UI board square gates, & Ply-1 per-square `(ru, rc)` selection!
    e11: List[Tuple[str, str]] = []
    live_v = "(1 - var(--is_terminal))"
    status_s = (
        f"{STATUS_WHITE_TURN}*{live_v}*(1 - var(--turn))*(1 - var(--in_check))"
        f" + {STATUS_BLACK_TURN}*{live_v}*var(--turn)*(1 - var(--in_check))"
        f" + {STATUS_WHITE_CHECK}*{live_v}*(1 - var(--turn))*var(--in_check)"
        f" + {STATUS_BLACK_CHECK}*{live_v}*var(--turn)*var(--in_check)"
        f" + {STATUS_BLACK_WINS_MATE}*(1 - var(--turn))*var(--checkmate)"
        f" + {STATUS_WHITE_WINS_MATE}*var(--turn)*var(--checkmate)"
        f" + {STATUS_STALEMATE}*var(--stalemate)"
        f" + {STATUS_DRAW_INSUF}*var(--draw_insuf)"
        f" + {STATUS_DRAW_50}*var(--draw_50)"
    )
    e11.append(("--status", wrap_calc(status_s)))

    for u in range(64):
        e11.append((f"--can_sel_{u}", wrap_calc(f"{live_v}*var(--has_leg_{u})*(1 - var(--isel{u}))")))
        e11.append((f"--leg_{u}", wrap_calc(f"{live_v}*var(--rleg_{u})")))
        e11.append((f"--chk_k_{u}", wrap_calc(f"var(--own_k{u})*var(--in_check)")))
        is_ep_cap_ui = f"var(--sel_p)*(1 - var(--occ{u}))*{css_eq(f'var(--ep_sq)', u)}"
        e11.append((f"--cap_{u}", wrap_calc(f"{live_v}*var(--rleg_{u})*min(1,var(--occ{u}) + {is_ep_cap_ui})")))

        vr = u // 8
        gp = f"clamp(0,var(--mc_p_{u}),1)"
        gn = f"clamp(0,var(--mc_n_{u}),1)"
        gb = f"clamp(0,var(--mc_b_{u}),1)"
        gr = f"clamp(0,var(--mc_r_{u}),1)"
        gq = f"clamp(0,var(--mc_q_{u}),1)"
        gk = f"clamp(0,var(--mc_k_{u}),1)"
        is_ep = f"clamp(0,var(--mc_p_{u}) - 127,1)"
        up_e = f"(var(--mc_p_{u}) - 64 - 64*{is_ep})"
        cp_e = "5" if vr in (0, 7) else f"(1 + 8*{is_ep})"
        is_oo = f"(clamp(0,var(--mc_k_{u}) - 127,1)*(1 - clamp(0,var(--mc_k_{u}) - 191,1)))"
        is_ooo = f"clamp(0,var(--mc_k_{u}) - 191,1)"
        uk_e = f"(var(--mc_k_{u}) - 64 - 64*{is_oo} - 128*{is_ooo})"
        ck_e = f"(6 + {is_oo} + 2*{is_ooo})"

        is_cap_m = f"min(1,var(--occ{u}) + {is_ep})"
        gp_pre = f"({is_cap_m}*{gp})"
        gp_post = f"((1 - {is_cap_m})*{gp})"

        u_prio = _chain_first([
            (gp_pre, up_e),
            (gn, f"(var(--mc_n_{u}) - 64)"),
            (gb, f"(var(--mc_b_{u}) - 64)"),
            (gr, f"(var(--mc_r_{u}) - 64)"),
            (gq, f"(var(--mc_q_{u}) - 64)"),
            (gp_post, up_e),
            (gk, uk_e),
        ])
        c_prio = _chain_first([
            (gp_pre, cp_e),
            (gn, "2"),
            (gb, "3"),
            (gr, "4"),
            (gq, "5"),
            (gp_post, cp_e),
            (gk, ck_e),
        ])
        e11.append((f"--m1_ru_{u}", wrap_calc(u_prio)))
        e11.append((f"--m1_rc_{u}", wrap_calc(c_prio)))

    stage_defs["e11"] = e11

    # .e12: Orientation-normalized Ply-1 Legal Move Strip (`--m1_u_0..63`, `--m1_v_0..63`, `--m1_c_0..63`)
    #       for the SVG 2-Ply CNN Minimax Engine (`ar = r` when White to move, `7 - r` when Black to move)
    e12: List[Tuple[str, str]] = []
    for a in range(64):
        ar, af = divmod(a, 8)
        sq_w = 8 * ar + af
        sq_b = 8 * (7 - ar) + af
        c_raw = f"((1 - var(--turn))*var(--m1_rc_{sq_w}) + var(--turn)*var(--m1_rc_{sq_b}))"
        c_live = f"({live_v}*{c_raw})"
        has_m = f"clamp(0,{c_live},1)"
        uw = f"var(--m1_ru_{sq_w})"
        ub = f"var(--m1_ru_{sq_b})"
        ub_flip = f"(56 - {ub} + 2*mod({ub},8))"
        u_norm = f"((1 - var(--turn))*{uw} + var(--turn)*{ub_flip})"
        e12.append((f"--m1_c_{a}", wrap_calc(c_live)))
        e12.append((f"--m1_u_{a}", wrap_calc(f"{has_m}*{u_norm} + (1 - {has_m})*64")))
        e12.append((f"--m1_v_{a}", wrap_calc(f"{has_m}*{a} + (1 - {has_m})*64")))
    stage_defs["e12"] = e12

    ply_classes = ("m0", "m1")
    eval_classes = ("e0a", "e0b", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e8b", "e9", "e10", "e11", "e12")

    keep_names: Set[str] = {
        "--in_f", "--in_t", "--in_c", "--in_pr",
        "--turn", "--cast_wk", "--cast_wq", "--cast_bk", "--cast_bq",
        "--ep_sq", "--hmc", "--ply", "--halted",
        "--last_f", "--last_t", "--cur_from", "--cur_pr",
        "--num_checkers", "--sel_can_promo",
        "--any_leg", "--in_check", "--checkmate", "--stalemate",
        "--draw_insuf", "--draw_50", "--is_terminal", "--status",
    }
    for u in range(64):
        keep_names.update((
            f"--b{u}",
            f"--isel{u}",
            f"--can_sel_{u}",
            f"--leg_{u}",
            f"--chk_k_{u}",
            f"--cap_{u}",
            f"--m1_u_{u}",
            f"--m1_v_{u}",
            f"--m1_c_{u}",
        ))

    compiled_defs, num_temps = _allocate_temp_registers(
        stage_defs=stage_defs,
        ply_loop_order=ply_classes,
        eval_order=eval_classes,
        keep_names=keep_names,
    )

    registered_props = sorted(keep_names) + [f"--t{i}" for i in range(num_temps)]

    return ChessCSSSpec(
        stage_defs=compiled_defs,
        ply_classes=ply_classes,
        eval_classes=eval_classes,
        registered_props=registered_props,
        num_temp_regs=num_temps,
    )


# ---------------------------------------------------------------------------
# 4. Fast Bit-Accurate CSS Cascade Simulator
# ---------------------------------------------------------------------------

def _eval_css_stage(rules: Sequence[Tuple[str, str]], env: Dict[str, int]) -> Dict[str, int]:
    def _mod(a: int, b: int) -> int:
        # Match CSS Values 4 `mod(A, B)` (result has sign of B; for positive B, Python's `%` is exact!)
        return a % b

    def _clamp(lo: int, val: int, hi: int) -> int:
        return lo if val < lo else (hi if val > hi else val)

    ns = {"min": min, "max": max, "clamp": _clamp, "mod": _mod}
    nxt = dict(env)
    for lhs, rhs in rules:
        expr = rhs[5:-1] if (rhs.startswith("calc(") and rhs.endswith(")")) else rhs
        py_expr = _VAR_RE.sub(lambda m: f"({env[m.group(1)]})", expr)
        nxt[lhs] = int(eval(py_expr, {"__builtins__": {}}, ns))
    return nxt


def compile_fast_evaluator(spec: ChessCSSSpec):
    """Compiles stage rules into Python lambdas once so thousands of test positions run in < 1s."""
    def _mod(a: int, b: int) -> int:
        return a % b

    def _clamp(lo: int, val: int, hi: int) -> int:
        return lo if val < lo else (hi if val > hi else val)

    compiled_stages = {}
    for cls, rules in spec.stage_defs.items():
        lines = ["def _run_stage(env):"]
        lines.append("    nxt = dict(env)")
        for lhs, rhs in rules:
            expr = rhs[5:-1] if (rhs.startswith("calc(") and rhs.endswith(")")) else rhs
            py_expr = _VAR_RE.sub(lambda m: f"env[{m.group(1)!r}]", expr)
            lines.append(f"    nxt[{lhs!r}] = int({py_expr})")
        lines.append("    return nxt")
        loc: Dict[str, object] = {}
        exec(
            "\n".join(lines),
            {"__builtins__": {}, "dict": dict, "int": int, "min": min, "max": max, "clamp": _clamp, "mod": _mod},
            loc,
        )
        compiled_stages[cls] = loc["_run_stage"]
    return compiled_stages


def simulate_css_chess(
    spec: ChessCSSSpec,
    initial_fen: str = START_FEN,
    ply_inputs: Sequence[Tuple[int, int, int, int]] = (),
    cur_from_uncommitted: int = SQ_NONE,
    cur_promo_uncommitted: int = PT_QUEEN,
    fast_stages: Optional[Dict] = None,
) -> Dict[str, int]:
    """
    Simulates browser CSS custom-property cascade across `.init`, committed plies,
    an optional active uncommitted ply selection, and `.e0 .. .e12`.

    Each item in `ply_inputs` is `(in_f, in_t, in_c, in_pr)`.
    """
    init_state = parse_fen(initial_fen)
    env: Dict[str, int] = {prop: 0 for prop in spec.registered_props}
    env.update(state_to_css_vars(init_state))

    run_fn = (lambda cls, e: fast_stages[cls](e)) if fast_stages else (
        lambda cls, e: _eval_css_stage(spec.stage_defs[cls], e)
    )

    all_plies = list(ply_inputs)
    if not all_plies or all(c == 1 for _, _, c, _ in all_plies):
        all_plies.append((cur_from_uncommitted, SQ_NONE, 0, cur_promo_uncommitted))

    for in_f, in_t, in_c, in_pr in all_plies:
        env = dict(env)
        env["--in_f"] = in_f
        env["--in_t"] = in_t
        env["--in_c"] = in_c
        env["--in_pr"] = in_pr
        for cls in spec.ply_classes:
            env = run_fn(cls, env)

    for cls in spec.eval_classes:
        env = run_fn(cls, env)

    return env
