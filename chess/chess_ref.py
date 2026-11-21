"""
Self-Contained FIDE Reference Chess Rule Engine (`chess/chess_ref.py`).

Provides an exact, zero-dependency reference implementation of FIDE chess rules
using the same integer square/piece encoding as the CSS `@property <integer>`
cascade pipeline:
  - Squares `0..63` (`sq = 8 * rank + file`, `a1 = 0`, `h8 = 63`), `64` = NONE
  - Pieces: `0` = Empty, `1..6` = White `P,N,B,R,Q,K`, `9..14` = Black `P,N,B,R,Q,K`
  - Full legal move generation (pins, checks, double-check, King X-ray, castling,
    en passant, promotion/underpromotion)
  - Terminal evaluation: Checkmate, Stalemate, Insufficient Material, 50-Move Rule
  - Curated showcase presets & Perft node counter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# Piece type codes (1..6)
EMPTY = 0
PT_PAWN = 1
PT_KNIGHT = 2
PT_BISHOP = 3
PT_ROOK = 4
PT_QUEEN = 5
PT_KING = 6

WHITE_BIT = 0
BLACK_BIT = 8

WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 9, 10, 11, 12, 13, 14

TURN_WHITE = 0
TURN_BLACK = 1
SQ_NONE = 64

# Unified UI / CSS `@counter-style` status codes
STATUS_WHITE_TURN = 0
STATUS_BLACK_TURN = 1
STATUS_WHITE_CHECK = 2
STATUS_BLACK_CHECK = 3
STATUS_BLACK_WINS_MATE = 4
STATUS_WHITE_WINS_MATE = 5
STATUS_STALEMATE = 6
STATUS_DRAW_INSUF = 7
STATUS_DRAW_50 = 8

FEN_CHAR_TO_PIECE: Dict[str, int] = {
    "P": WP, "N": WN, "B": WB, "R": WR, "Q": WQ, "K": WK,
    "p": BP, "n": BN, "b": BB, "r": BR, "q": BQ, "k": BK,
}
PIECE_TO_FEN_CHAR: Dict[int, str] = {v: k for k, v in FEN_CHAR_TO_PIECE.items()}

KNIGHT_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
KING_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
ORTH_DIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
DIAG_DIRS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Curated showcase presets: (id, title, fen, subtitle)
PRESETS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "std",
        "Standard FIDE Game",
        START_FEN,
        "Full 2-player starting position",
    ),
    (
        "mate1",
        "Mate in 1 (Scholar's)",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "Play Qxf7# (h5 -> f7) for Checkmate",
    ),
    (
        "stale1",
        "Stalemate in 1",
        "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1",
        "Play Qg6 for Stalemate (or Qg7# for Mate)",
    ),
    (
        "special",
        "Castling & En Passant",
        "r3k2r/ppp1qppp/2n1bn2/3pP3/8/2N1BN2/PPP1QPPP/R3K2R w KQkq d6 0 8",
        "O-O, O-O-O, and exd6 e.p. all live on move 1",
    ),
    (
        "promo",
        "Underpromotion Mate",
        "6nb/5Ppk/6pp/4K3/8/8/8/8 w - - 0 1",
        "Select Knight + play f8=N# for smothered mate!",
    ),
)


def sq_name(sq: int) -> str:
    if sq < 0 or sq >= 64:
        return "-"
    r, f = divmod(sq, 8)
    return f"{chr(ord('a') + f)}{r + 1}"


def parse_sq(name: str) -> int:
    if name == "-" or len(name) != 2:
        return SQ_NONE
    f = ord(name[0]) - ord("a")
    r = ord(name[1]) - ord("1")
    if 0 <= f < 8 and 0 <= r < 8:
        return 8 * r + f
    return SQ_NONE


def piece_type(p: int) -> int:
    if p == 0:
        return 0
    return p - 8 if p >= 8 else p


def piece_color(p: int) -> int:
    """Returns TURN_WHITE (0) for White, TURN_BLACK (1) for Black, or -1 for Empty."""
    if p == 0:
        return -1
    return TURN_BLACK if p >= 8 else TURN_WHITE


@dataclass(frozen=True)
class Move:
    from_sq: int
    to_sq: int
    promo: int = PT_QUEEN  # 2=N, 3=B, 4=R, 5=Q (only used when a pawn reaches rank 0/7)

    def uci(self, is_promo: bool = False) -> str:
        base = f"{sq_name(self.from_sq)}{sq_name(self.to_sq)}"
        if is_promo:
            p_map = {PT_KNIGHT: "n", PT_BISHOP: "b", PT_ROOK: "r", PT_QUEEN: "q"}
            return base + p_map.get(self.promo, "q")
        return base


@dataclass(frozen=True)
class ChessState:
    board: Tuple[int, ...]  # length 64
    turn: int               # 0 = White, 1 = Black
    cast_wk: int            # 0 or 1
    cast_wq: int            # 0 or 1
    cast_bk: int            # 0 or 1
    cast_bq: int            # 0 or 1
    ep_sq: int              # 0..63 or 64 (SQ_NONE)
    hmc: int                # halfmove clock >= 0
    ply: int = 0
    last_f: int = SQ_NONE
    last_t: int = SQ_NONE


@dataclass(frozen=True)
class EvalSummary:
    in_check: int
    num_checkers: int
    any_legal: int
    checkmate: int
    stalemate: int
    draw_insuf: int
    draw_50: int
    is_terminal: int
    status: int
    has_leg: Tuple[int, ...]       # length 64 (0/1 per square)
    legal_moves: Tuple[Move, ...]


def parse_fen(fen: str) -> ChessState:
    parts = fen.strip().split()
    rows = parts[0].split("/")
    assert len(rows) == 8, f"Invalid FEN board rows: {fen}"
    board = [EMPTY] * 64
    for r_inv, row_str in enumerate(rows):
        r = 7 - r_inv
        f = 0
        for ch in row_str:
            if ch.isdigit():
                f += int(ch)
            else:
                board[8 * r + f] = FEN_CHAR_TO_PIECE[ch]
                f += 1
        assert f == 8, f"Invalid row width in FEN: {row_str}"

    turn = TURN_WHITE if (len(parts) < 2 or parts[1] == "w") else TURN_BLACK
    cast_str = parts[2] if len(parts) >= 3 else "-"
    ep_sq = parse_sq(parts[3]) if len(parts) >= 4 else SQ_NONE
    hmc = int(parts[4]) if len(parts) >= 5 else 0
    return ChessState(
        board=tuple(board),
        turn=turn,
        cast_wk=1 if "K" in cast_str else 0,
        cast_wq=1 if "Q" in cast_str else 0,
        cast_bk=1 if "k" in cast_str else 0,
        cast_bq=1 if "q" in cast_str else 0,
        ep_sq=ep_sq,
        hmc=hmc,
        ply=0,
        last_f=SQ_NONE,
        last_t=SQ_NONE,
    )


def to_fen(state: ChessState) -> str:
    rows: List[str] = []
    for r in range(7, -1, -1):
        empty_run = 0
        row_chars: List[str] = []
        for f in range(8):
            p = state.board[8 * r + f]
            if p == EMPTY:
                empty_run += 1
            else:
                if empty_run:
                    row_chars.append(str(empty_run))
                    empty_run = 0
                row_chars.append(PIECE_TO_FEN_CHAR[p])
        if empty_run:
            row_chars.append(str(empty_run))
        rows.append("".join(row_chars))

    cast = "".join(
        ch
        for ch, bit in (
            ("K", state.cast_wk),
            ("Q", state.cast_wq),
            ("k", state.cast_bk),
            ("q", state.cast_bq),
        )
        if bit
    ) or "-"
    turn_s = "w" if state.turn == TURN_WHITE else "b"
    ep_s = sq_name(state.ep_sq) if state.ep_sq != SQ_NONE else "-"
    fullmove = (state.ply // 2) + 1
    return f"{'/'.join(rows)} {turn_s} {cast} {ep_s} {state.hmc} {fullmove}"


def count_attackers_on_square(board: Sequence[int], target_sq: int, by_color: int) -> int:
    """Counts how many pieces of `by_color` directly attack `target_sq` on `board`."""
    tr, tf = divmod(target_sq, 8)
    opp_off = BLACK_BIT if by_color == TURN_BLACK else WHITE_BIT
    opp_p = opp_off + PT_PAWN
    opp_n = opp_off + PT_KNIGHT
    opp_b = opp_off + PT_BISHOP
    opp_r = opp_off + PT_ROOK
    opp_q = opp_off + PT_QUEEN
    opp_k = opp_off + PT_KING

    cnt = 0
    # Pawn attacks: a White pawn at (tr-1, tf±1) attacks (tr, tf); Black at (tr+1, tf±1)
    pr = tr - 1 if by_color == TURN_WHITE else tr + 1
    if 0 <= pr < 8:
        for df in (-1, 1):
            pf = tf + df
            if 0 <= pf < 8 and board[8 * pr + pf] == opp_p:
                cnt += 1

    for dr, df in KNIGHT_DELTAS:
        nr, nf = tr + dr, tf + df
        if 0 <= nr < 8 and 0 <= nf < 8 and board[8 * nr + nf] == opp_n:
            cnt += 1

    for dr, df in KING_DELTAS:
        kr, kf = tr + dr, tf + df
        if 0 <= kr < 8 and 0 <= kf < 8 and board[8 * kr + kf] == opp_k:
            cnt += 1

    for dr, df in ORTH_DIRS:
        cr, cf = tr + dr, tf + df
        while 0 <= cr < 8 and 0 <= cf < 8:
            p = board[8 * cr + cf]
            if p != EMPTY:
                if p == opp_r or p == opp_q:
                    cnt += 1
                break
            cr += dr
            cf += df

    for dr, df in DIAG_DIRS:
        cr, cf = tr + dr, tf + df
        while 0 <= cr < 8 and 0 <= cf < 8:
            p = board[8 * cr + cf]
            if p != EMPTY:
                if p == opp_b or p == opp_q:
                    cnt += 1
                break
            cr += dr
            cf += df

    return cnt


def is_square_attacked(board: Sequence[int], target_sq: int, by_color: int) -> bool:
    return count_attackers_on_square(board, target_sq, by_color) > 0


def find_king(board: Sequence[int], color: int) -> int:
    target = (BLACK_BIT if color == TURN_BLACK else WHITE_BIT) + PT_KING
    for sq, p in enumerate(board):
        if p == target:
            return sq
    return SQ_NONE


def is_promotion_move(state: ChessState, from_sq: int, to_sq: int) -> bool:
    p = state.board[from_sq]
    if piece_type(p) != PT_PAWN:
        return False
    tr = to_sq // 8
    return (state.turn == TURN_WHITE and tr == 7) or (state.turn == TURN_BLACK and tr == 0)


def apply_move(state: ChessState, mv: Move) -> ChessState:
    """Applies a move `(from_sq, to_sq, promo)` and returns the updated `ChessState`."""
    b = list(state.board)
    f_sq, t_sq = mv.from_sq, mv.to_sq
    mov_p = b[f_sq]
    assert mov_p != EMPTY, f"Cannot move from empty square {sq_name(f_sq)}"
    pt = piece_type(mov_p)
    col = state.turn
    own_off = BLACK_BIT if col == TURN_BLACK else WHITE_BIT
    dst_was_occ = 1 if b[t_sq] != EMPTY else 0

    fr, ff = divmod(f_sq, 8)
    tr, tf = divmod(t_sq, 8)

    b[f_sq] = EMPTY

    # En passant capture removal
    if pt == PT_PAWN and ff != tf and t_sq == state.ep_sq and dst_was_occ == 0:
        victim_sq = t_sq - 8 if col == TURN_WHITE else t_sq + 8
        b[victim_sq] = EMPTY

    # Castling rook movement
    if pt == PT_KING:
        if f_sq == 4 and t_sq == 6:
            b[7] = EMPTY
            b[5] = WR
        elif f_sq == 4 and t_sq == 2:
            b[0] = EMPTY
            b[3] = WR
        elif f_sq == 60 and t_sq == 62:
            b[63] = EMPTY
            b[61] = BR
        elif f_sq == 60 and t_sq == 58:
            b[56] = EMPTY
            b[59] = BR

    # Destination piece placement (with promotion if applicable)
    if pt == PT_PAWN and (tr == 7 or tr == 0):
        promo_pt = mv.promo if mv.promo in (PT_KNIGHT, PT_BISHOP, PT_ROOK, PT_QUEEN) else PT_QUEEN
        b[t_sq] = own_off + promo_pt
    else:
        b[t_sq] = mov_p

    # Next en-passant target square
    next_ep = SQ_NONE
    if pt == PT_PAWN and abs(tr - fr) == 2:
        next_ep = (f_sq + t_sq) // 2

    # Castling rights revocation (moving King/Rook OR capturing Rook on its home corner)
    c_wk = state.cast_wk * (0 if (f_sq in (4, 7) or t_sq in (4, 7)) else 1)
    c_wq = state.cast_wq * (0 if (f_sq in (4, 0) or t_sq in (4, 0)) else 1)
    c_bk = state.cast_bk * (0 if (f_sq in (60, 63) or t_sq in (60, 63)) else 1)
    c_bq = state.cast_bq * (0 if (f_sq in (60, 56) or t_sq in (60, 56)) else 1)

    next_hmc = 0 if (pt == PT_PAWN or dst_was_occ == 1) else (state.hmc + 1)

    return ChessState(
        board=tuple(b),
        turn=1 - col,
        cast_wk=c_wk,
        cast_wq=c_wq,
        cast_bk=c_bk,
        cast_bq=c_bq,
        ep_sq=next_ep,
        hmc=next_hmc,
        ply=state.ply + 1,
        last_f=f_sq,
        last_t=t_sq,
    )


def generate_pseudo_legal_moves(state: ChessState) -> List[Move]:
    b = state.board
    col = state.turn
    opp = 1 - col
    moves: List[Move] = []

    for u in range(64):
        p = b[u]
        if p == EMPTY or piece_color(p) != col:
            continue
        pt = piece_type(p)
        ur, uf = divmod(u, 8)

        if pt == PT_PAWN:
            step = 8 if col == TURN_WHITE else -8
            start_r = 1 if col == TURN_WHITE else 6
            promo_r = 7 if col == TURN_WHITE else 0
            v1 = u + step
            if 0 <= v1 < 64 and b[v1] == EMPTY:
                if (v1 // 8) == promo_r:
                    for pr in (PT_QUEEN, PT_ROOK, PT_BISHOP, PT_KNIGHT):
                        moves.append(Move(u, v1, pr))
                else:
                    moves.append(Move(u, v1))
                    if ur == start_r:
                        v2 = u + 2 * step
                        if b[v2] == EMPTY:
                            moves.append(Move(u, v2))
            for df in (-1, 1):
                vf = uf + df
                vr = ur + (1 if col == TURN_WHITE else -1)
                if 0 <= vf < 8 and 0 <= vr < 8:
                    vc = 8 * vr + vf
                    if (b[vc] != EMPTY and piece_color(b[vc]) == opp) or vc == state.ep_sq:
                        if vr == promo_r:
                            for pr in (PT_QUEEN, PT_ROOK, PT_BISHOP, PT_KNIGHT):
                                moves.append(Move(u, vc, pr))
                        else:
                            moves.append(Move(u, vc))

        elif pt == PT_KNIGHT:
            for dr, df in KNIGHT_DELTAS:
                vr, vf = ur + dr, uf + df
                if 0 <= vr < 8 and 0 <= vf < 8:
                    v = 8 * vr + vf
                    if b[v] == EMPTY or piece_color(b[v]) == opp:
                        moves.append(Move(u, v))

        elif pt in (PT_BISHOP, PT_ROOK, PT_QUEEN):
            dirs: Tuple[Tuple[int, int], ...] = ()
            if pt in (PT_ROOK, PT_QUEEN):
                dirs = dirs + ORTH_DIRS
            if pt in (PT_BISHOP, PT_QUEEN):
                dirs = dirs + DIAG_DIRS
            for dr, df in dirs:
                vr, vf = ur + dr, uf + df
                while 0 <= vr < 8 and 0 <= vf < 8:
                    v = 8 * vr + vf
                    if b[v] == EMPTY:
                        moves.append(Move(u, v))
                    else:
                        if piece_color(b[v]) == opp:
                            moves.append(Move(u, v))
                        break
                    vr += dr
                    vf += df

        elif pt == PT_KING:
            for dr, df in KING_DELTAS:
                vr, vf = ur + dr, uf + df
                if 0 <= vr < 8 and 0 <= vf < 8:
                    v = 8 * vr + vf
                    if b[v] == EMPTY or piece_color(b[v]) == opp:
                        moves.append(Move(u, v))
            # Castling (checks path emptiness + e/f/g or e/d/c unattacked + home rook presence)
            if col == TURN_WHITE and u == 4:
                if (
                    state.cast_wk
                    and b[7] == WR
                    and b[5] == EMPTY
                    and b[6] == EMPTY
                    and not is_square_attacked(b, 4, opp)
                    and not is_square_attacked(b, 5, opp)
                    and not is_square_attacked(b, 6, opp)
                ):
                    moves.append(Move(4, 6))
                if (
                    state.cast_wq
                    and b[0] == WR
                    and b[1] == EMPTY
                    and b[2] == EMPTY
                    and b[3] == EMPTY
                    and not is_square_attacked(b, 4, opp)
                    and not is_square_attacked(b, 3, opp)
                    and not is_square_attacked(b, 2, opp)
                ):
                    moves.append(Move(4, 2))
            elif col == TURN_BLACK and u == 60:
                if (
                    state.cast_bk
                    and b[63] == BR
                    and b[61] == EMPTY
                    and b[62] == EMPTY
                    and not is_square_attacked(b, 60, opp)
                    and not is_square_attacked(b, 61, opp)
                    and not is_square_attacked(b, 62, opp)
                ):
                    moves.append(Move(60, 62))
                if (
                    state.cast_bq
                    and b[56] == BR
                    and b[57] == EMPTY
                    and b[58] == EMPTY
                    and b[59] == EMPTY
                    and not is_square_attacked(b, 60, opp)
                    and not is_square_attacked(b, 59, opp)
                    and not is_square_attacked(b, 58, opp)
                ):
                    moves.append(Move(60, 58))

    return moves


def generate_legal_moves(state: ChessState) -> List[Move]:
    col = state.turn
    opp = 1 - col
    legal: List[Move] = []
    for mv in generate_pseudo_legal_moves(state):
        nxt = apply_move(state, mv)
        k_sq = find_king(nxt.board, col)
        if k_sq != SQ_NONE and not is_square_attacked(nxt.board, k_sq, opp):
            legal.append(mv)
    return legal


def is_insufficient_material_ref(board: Sequence[int]) -> bool:
    """Exact FIDE / python-chess insufficient material check."""
    heavy_or_pawn = 0
    knights = 0
    bishops_light = 0
    bishops_dark = 0
    for sq, p in enumerate(board):
        pt = piece_type(p)
        if pt in (PT_PAWN, PT_ROOK, PT_QUEEN):
            heavy_or_pawn += 1
        elif pt == PT_KNIGHT:
            knights += 1
        elif pt == PT_BISHOP:
            r, f = divmod(sq, 8)
            if (r + f) % 2 == 1:
                bishops_light += 1
            else:
                bishops_dark += 1

    if heavy_or_pawn > 0:
        return False
    if knights > 0:
        return knights == 1 and bishops_light == 0 and bishops_dark == 0
    return bishops_light == 0 or bishops_dark == 0


def evaluate_state(state: ChessState) -> EvalSummary:
    col = state.turn
    opp = 1 - col
    k_sq = find_king(state.board, col)
    num_chk = count_attackers_on_square(state.board, k_sq, opp) if k_sq != SQ_NONE else 0
    in_check = 1 if num_chk > 0 else 0

    leg = generate_legal_moves(state)
    has_leg = [0] * 64
    for mv in leg:
        has_leg[mv.from_sq] = 1

    any_leg = 1 if leg else 0
    checkmate = 1 if (in_check == 1 and any_leg == 0) else 0
    stalemate = 1 if (in_check == 0 and any_leg == 0) else 0
    draw_insuf = 1 if (checkmate == 0 and stalemate == 0 and is_insufficient_material_ref(state.board)) else 0
    draw_50 = 1 if (checkmate == 0 and stalemate == 0 and draw_insuf == 0 and state.hmc >= 100) else 0
    is_term = 1 if (checkmate or stalemate or draw_insuf or draw_50) else 0

    if checkmate:
        status = STATUS_BLACK_WINS_MATE if col == TURN_WHITE else STATUS_WHITE_WINS_MATE
    elif stalemate:
        status = STATUS_STALEMATE
    elif draw_insuf:
        status = STATUS_DRAW_INSUF
    elif draw_50:
        status = STATUS_DRAW_50
    elif in_check:
        status = STATUS_WHITE_CHECK if col == TURN_WHITE else STATUS_BLACK_CHECK
    else:
        status = STATUS_WHITE_TURN if col == TURN_WHITE else STATUS_BLACK_TURN

    return EvalSummary(
        in_check=in_check,
        num_checkers=num_chk,
        any_legal=any_leg,
        checkmate=checkmate,
        stalemate=stalemate,
        draw_insuf=draw_insuf,
        draw_50=draw_50,
        is_terminal=is_term,
        status=status,
        has_leg=tuple(has_leg),
        legal_moves=tuple(leg),
    )


def perft(state: ChessState, depth: int) -> int:
    if depth == 0:
        return 1
    moves = generate_legal_moves(state)
    if depth == 1:
        return len(moves)
    return sum(perft(apply_move(state, mv), depth - 1) for mv in moves)
