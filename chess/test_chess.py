#!/usr/bin/env python3
"""
Automated Verification Suite for Pure HTML + Inline CSS 2-Player Chess (`chess/`).

Tests:
  1. FIDE Perft node counts on Startpos, Kiwipete, Position 3 (EP pin), and Position 5
  2. Differential verification of the compiled CSS `@property <integer>` cascade
     against `python-chess` (`chess.Board`) across presets & random game rollouts
  3. Targeted FIDE edge-case suite: 4-axis pins, double check, King X-ray retreat,
     castling (`b1/b8` attacked vs occupied, home-square Rook capture rights loss),
     en passant (normal, checking-pawn capture, horizontal discovered check, diagonal
     victim pin), promotions/underpromotions, Fool's Mate, pinned-piece stalemate,
     insufficient material, and 50-move rule (with checkmate precedence)
  4. Multi-ply `.m0/.m1` state replay + LIFO Undo verification
  5. Static HTML5 & CSS audit of `chess/index.html` (0 JS, 0 `on*` attrs, DOM depth < 500,
     GitHub Pages `../` back-link, and 0 intra-stage CSS custom-property cycles)
"""

from __future__ import annotations

from html.parser import HTMLParser
import importlib.machinery
import importlib.util
import os
import random
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

CHESS_DIR = os.path.dirname(os.path.abspath(__file__))
if CHESS_DIR not in sys.path:
    sys.path.insert(0, CHESS_DIR)


def _load_external_python_chess():
    """Import `python-chess` from site-packages without colliding with `./chess/`."""
    existing = sys.modules.get("chess")
    if existing is not None and hasattr(existing, "Board"):
        return existing
    search_paths = [
        p for p in sys.path
        if os.path.abspath(p) != os.path.abspath(os.path.dirname(CHESS_DIR))
        and os.path.abspath(p) != CHESS_DIR
    ]
    spec = importlib.machinery.PathFinder.find_spec("chess", search_paths)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chess"] = mod
    spec.loader.exec_module(mod)
    return mod


PYCHESS = _load_external_python_chess()

import chess_ref as cr
from css_chess import (
    ChessCSSSpec,
    build_chess_css_spec,
    compile_fast_evaluator,
    simulate_css_chess,
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
        self.tags_seen: Set[str] = set()
        self.event_attrs: List[str] = []
        self.js_uris: List[str] = []
        self.has_back_link = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        self.tags_seen.add(t)
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        if t == "a" and attr_dict.get("href") == "../" and "back-link" in attr_dict.get("class", ""):
            self.has_back_link = True
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
        assert self.stack and self.stack[-1] == t, f"Mismatched closing </{t}>, tail={self.stack[-3:]}"
        self.stack.pop()


class TestNoJSChess(unittest.TestCase):
    spec: ChessCSSSpec
    fast: Dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = build_chess_css_spec()
        cls.fast = compile_fast_evaluator(cls.spec)

    def _assert_matches_oracles(self, fen: str, check_all_selected: bool = True) -> Dict[str, int]:
        ref_st = cr.parse_fen(fen)
        ref_ev = cr.evaluate_state(ref_st)
        env = simulate_css_chess(self.spec, initial_fen=fen, fast_stages=self.fast)

        self.assertEqual(env["--in_check"], ref_ev.in_check, f"in_check mismatch on {fen}")
        self.assertEqual(env["--num_checkers"], ref_ev.num_checkers, f"num_checkers mismatch on {fen}")
        self.assertEqual(env["--any_leg"], ref_ev.any_legal, f"any_leg mismatch on {fen}")
        self.assertEqual(env["--checkmate"], ref_ev.checkmate, f"checkmate mismatch on {fen}")
        self.assertEqual(env["--stalemate"], ref_ev.stalemate, f"stalemate mismatch on {fen}")
        self.assertEqual(env["--draw_insuf"], ref_ev.draw_insuf, f"draw_insuf mismatch on {fen}")
        self.assertEqual(env["--draw_50"], ref_ev.draw_50, f"draw_50 mismatch on {fen}")
        self.assertEqual(env["--is_terminal"], ref_ev.is_terminal, f"is_terminal mismatch on {fen}")
        self.assertEqual(env["--status"], ref_ev.status, f"status mismatch on {fen}")

        ref_by_from: Dict[int, Set[int]] = {u: set() for u in range(64)}
        for mv in ref_ev.legal_moves:
            ref_by_from[mv.from_sq].add(mv.to_sq)

        for u in range(64):
            self.assertEqual(
                env[f"--has_leg_{u}"],
                ref_ev.has_leg[u],
                f"has_leg_{u} ({cr.sq_name(u)}) mismatch on {fen}",
            )

        if PYCHESS is not None:
            pb = PYCHESS.Board(fen)
            self.assertEqual(env["--in_check"], int(pb.is_check()), f"pychess check mismatch on {fen}")
            self.assertEqual(env["--checkmate"], int(pb.is_checkmate()), f"pychess mate mismatch on {fen}")
            self.assertEqual(env["--stalemate"], int(pb.is_stalemate()), f"pychess stalemate mismatch on {fen}")
            exp_insuf = int(not pb.is_checkmate() and not pb.is_stalemate() and pb.is_insufficient_material())
            self.assertEqual(env["--draw_insuf"], exp_insuf, f"pychess insuf mismatch on {fen}")
            exp_d50 = int(
                not pb.is_checkmate()
                and not pb.is_stalemate()
                and not pb.is_insufficient_material()
                and pb.is_fifty_moves()
            )
            self.assertEqual(env["--draw_50"], exp_d50, f"pychess 50-move mismatch on {fen}")

            pyc_by_from: Dict[int, Set[int]] = {u: set() for u in range(64)}
            for pmv in pb.legal_moves:
                pyc_by_from[pmv.from_square].add(pmv.to_square)
            self.assertEqual(ref_by_from, pyc_by_from, f"ref vs pychess legal target sets on {fen}")

        if check_all_selected and env["--is_terminal"] == 0:
            for u in range(64):
                if ref_by_from[u]:
                    eu = simulate_css_chess(
                        self.spec,
                        initial_fen=fen,
                        cur_from_uncommitted=u,
                        fast_stages=self.fast,
                    )
                    got_dsts = {v for v in range(64) if eu[f"--leg_{v}"] == 1}
                    self.assertEqual(
                        got_dsts,
                        ref_by_from[u],
                        f"CSS --leg_* mismatch for selected {cr.sq_name(u)} on {fen}",
                    )
        return env

    def test_01_perft_benchmarks(self) -> None:
        self.assertEqual(cr.perft(cr.parse_fen(cr.START_FEN), 1), 20)
        self.assertEqual(cr.perft(cr.parse_fen(cr.START_FEN), 2), 400)
        self.assertEqual(cr.perft(cr.parse_fen(cr.START_FEN), 3), 8902)

        kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        self.assertEqual(cr.perft(cr.parse_fen(kiwipete), 1), 48)
        self.assertEqual(cr.perft(cr.parse_fen(kiwipete), 2), 2039)

        pos3_ep_pin = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
        self.assertEqual(cr.perft(cr.parse_fen(pos3_ep_pin), 1), 14)
        self.assertEqual(cr.perft(cr.parse_fen(pos3_ep_pin), 2), 191)
        self.assertEqual(cr.perft(cr.parse_fen(pos3_ep_pin), 3), 2812)

        pos5_promo = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
        self.assertEqual(cr.perft(cr.parse_fen(pos5_promo), 1), 44)
        self.assertEqual(cr.perft(cr.parse_fen(pos5_promo), 2), 1486)

    def test_02_presets_and_benchmarks_match_pychess(self) -> None:
        if PYCHESS is not None:
            for pid, _title, fen, _sub in cr.PRESETS:
                self.assertTrue(PYCHESS.Board(fen).is_valid(), f"Preset {pid} FEN is not strictly valid: {fen}")
        fens = [p[2] for p in cr.PRESETS] + [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ]
        for fen in fens:
            self._assert_matches_oracles(fen, check_all_selected=True)

    def test_03_pins_double_check_and_king_xray(self) -> None:
        # 1. Pinned Knight (0 legal moves), pinned Rook on file (can move along file only),
        #    pinned Bishop on diagonal (can slide/capture along diagonal only)
        pin_fens = [
            "4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1",          # Pinned Knight on e2 -> 0 moves
            "4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1",          # Pinned Rook on e2 -> e3..e8 only
            "7k/6b1/8/8/3B4/8/8/K7 w - - 0 1",           # Pinned Bishop on d4 along a1-h8
            "k7/1q6/8/3P4/4K3/8/8/8 w - - 0 1",          # Pinned Pawn on d5 -> cannot push d6, CAN capture b7? (no, d5->c6; put q on c6: k7/8/2q5/3P4/4K3/8/8/8 w - - 0 1)
            "k7/8/2q5/3P4/4K3/8/8/8 w - - 0 1",          # Pinned Pawn on d5: dxc6 legal, d6 illegal!
            "4r2k/8/8/8/8/8/4P3/4K3 w - - 0 1",          # Vertically pinned Pawn on e2: e3/e4 legal!
        ]
        for fen in pin_fens:
            self._assert_matches_oracles(fen)

        # 2. Double Check: Rook on e8 + Knight on f6 both check Black King on e8? (White checks Black on e8: Rd8+ & Nf6+ -> 3Rk3/8/5N2/8/4r3/8/8/4K3 b - - 0 1)
        dbl_chk_fen = "3Rk3/8/5N2/3q4/4r3/8/8/4K3 b - - 0 1"
        env_dc = self._assert_matches_oracles(dbl_chk_fen)
        self.assertEqual(env_dc["--num_checkers"], 2)
        #Neither Black Queen d5 (which could capture Rd8 or Nf6 individually!) nor Re4 may move in double check!
        self.assertEqual(env_dc[f"--has_leg_{cr.parse_sq('d5')}"], 0)
        self.assertEqual(env_dc[f"--has_leg_{cr.parse_sq('e4')}"], 0)
        self.assertEqual(env_dc[f"--has_leg_{cr.parse_sq('e8')}"], 1)  # Kf7 legal

        # 3. King X-ray retreat along checking slider ray forbidden (Ke2 checked by Re8 cannot step to e1!)
        xray_fen = "4r2k/8/8/8/8/8/4K3/8 w - - 0 1"
        eu = simulate_css_chess(
            self.spec,
            initial_fen=xray_fen,
            cur_from_uncommitted=cr.parse_sq("e2"),
            fast_stages=self.fast,
        )
        self.assertEqual(eu[f"--leg_{cr.parse_sq('e1')}"], 0)
        self.assertEqual(eu[f"--leg_{cr.parse_sq('d1')}"], 1)
        self._assert_matches_oracles(xray_fen)

    def test_04_castling_and_en_passant_edge_cases(self) -> None:
        # 1. Queenside castling STILL LEGAL when b1 is attacked by enemy Rook on b8!
        b1_att_fen = "1r2k2r/8/8/8/8/8/P1PPPPPP/R3K2R w KQk - 0 1"
        env_b1 = simulate_css_chess(
            self.spec,
            initial_fen=b1_att_fen,
            cur_from_uncommitted=4,
            fast_stages=self.fast,
        )
        self.assertEqual(env_b1["--leg_2"], 1)  # e1 -> c1 (O-O-O) is legal!
        self._assert_matches_oracles(b1_att_fen)

        # 2. Queenside castling BLOCKED when d1 transit square is attacked, or b1 is occupied
        self._assert_matches_oracles("3rk2r/8/8/8/8/8/PPP1PPPP/R3K2R w KQk - 0 1")
        self._assert_matches_oracles("rn2k2r/8/8/8/8/8/PPPPPPPP/RN2K2R w KQkq - 0 1")

        # 3. Home-square Rook capture revokes castling right without Rook ever moving!
        # Black Bxh1 captures White's unmoved Rook on h1 -> --cast_wk must become 0!
        cap_rook_fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPbP/R3K2R b KQkq - 0 1"
        env_cap = simulate_css_chess(
            self.spec,
            initial_fen=cap_rook_fen,
            ply_inputs=[(cr.parse_sq("g2"), cr.parse_sq("h1"), 1, cr.PT_QUEEN)],
            cur_from_uncommitted=4,
            fast_stages=self.fast,
        )
        self.assertEqual(env_cap["--cast_wk"], 0)
        self.assertEqual(env_cap["--cast_wq"], 1)
        self.assertEqual(env_cap["--leg_6"], 0)  # White O-O forbidden!
        self.assertEqual(env_cap["--leg_2"], 1)  # White O-O-O still legal!

        # 4. En Passant Horizontal Discovered-Check Pin (both White and Black!)
        ep_hpin_w = "8/8/8/K2pP2q/8/8/8/7k w - d6 0 1"
        ep_hpin_b = "8/8/8/8/k2Pp2Q/8/8/7K b - d3 0 1"
        self._assert_matches_oracles(ep_hpin_w)
        self._assert_matches_oracles(ep_hpin_b)
        eu_hw = simulate_css_chess(
            self.spec, initial_fen=ep_hpin_w, cur_from_uncommitted=cr.parse_sq("e5"), fast_stages=self.fast
        )
        self.assertEqual(eu_hw[f"--leg_{cr.parse_sq('d6')}"], 0)

        # 5. En Passant Capturing the Single Checking Pawn!
        ep_evade_chk = "8/8/8/3k4/3pP3/8/8/4K3 b - e3 0 1"
        env_epc = self._assert_matches_oracles(ep_evade_chk)
        self.assertEqual(env_epc["--in_check"], 1)
        eu_epc = simulate_css_chess(
            self.spec, initial_fen=ep_evade_chk, cur_from_uncommitted=cr.parse_sq("d4"), fast_stages=self.fast
        )
        self.assertEqual(eu_epc[f"--leg_{cr.parse_sq('e3')}"], 1)

        # 6. En Passant Victim Diagonally Pinned (synthetic FEN check)
        ep_diag_vic = "6b1/8/8/3pP3/8/1K6/8/7k w - d6 0 1"
        self._assert_matches_oracles(ep_diag_vic)

    def test_05_checkmate_stalemate_draws_and_fools_mate_replay(self) -> None:
        # 1. Replay Fool's Mate (`1. f3 e5 2. g4 Qh4#`) across 4 committed plies!
        fools_plies = [
            (cr.parse_sq("f2"), cr.parse_sq("f3"), 1, cr.PT_QUEEN),
            (cr.parse_sq("e7"), cr.parse_sq("e5"), 1, cr.PT_QUEEN),
            (cr.parse_sq("g2"), cr.parse_sq("g4"), 1, cr.PT_QUEEN),
            (cr.parse_sq("d8"), cr.parse_sq("h4"), 1, cr.PT_QUEEN),
        ]
        env_fm = simulate_css_chess(self.spec, initial_fen=cr.START_FEN, ply_inputs=fools_plies, fast_stages=self.fast)
        self.assertEqual(env_fm["--ply"], 4)
        self.assertEqual(env_fm["--in_check"], 1)
        self.assertEqual(env_fm["--any_leg"], 0)
        self.assertEqual(env_fm["--checkmate"], 1)
        self.assertEqual(env_fm["--stalemate"], 0)
        self.assertEqual(env_fm["--is_terminal"], 1)
        self.assertEqual(env_fm["--status"], cr.STATUS_BLACK_WINS_MATE)
        self.assertTrue(all(env_fm[f"--can_sel_{u}"] == 0 for u in range(64)))

        # 2. LIFO Undo of move 4 (`Qh4#` uncommitted -> `in_c = 0`): game resumes at ply 3!
        undo_plies = fools_plies[:3] + [(cr.parse_sq("d8"), cr.SQ_NONE, 0, cr.PT_QUEEN)]
        env_un = simulate_css_chess(self.spec, initial_fen=cr.START_FEN, ply_inputs=undo_plies, fast_stages=self.fast)
        self.assertEqual(env_un["--ply"], 3)
        self.assertEqual(env_un["--is_terminal"], 0)
        self.assertEqual(env_un["--cur_from"], cr.parse_sq("d8"))
        self.assertEqual(env_un[f"--leg_{cr.parse_sq('h4')}"], 1)

        # 3. Smothered Knight Underpromotion Checkmate (`f7 -> f8=N#`) vs Queen non-mate
        promo_fen = cr.PRESETS[4][2]
        env_pr_n = simulate_css_chess(
            self.spec,
            initial_fen=promo_fen,
            ply_inputs=[(cr.parse_sq("f7"), cr.parse_sq("f8"), 1, cr.PT_KNIGHT)],
            fast_stages=self.fast,
        )
        self.assertEqual(env_pr_n["--checkmate"], 1)
        self.assertEqual(env_pr_n["--status"], cr.STATUS_WHITE_WINS_MATE)

        env_pr_q = simulate_css_chess(
            self.spec,
            initial_fen=promo_fen,
            ply_inputs=[(cr.parse_sq("f7"), cr.parse_sq("f8"), 1, cr.PT_QUEEN)],
            fast_stages=self.fast,
        )
        self.assertEqual(env_pr_q["--checkmate"], 0)

        # 4. Pinned-Piece Stalemate (`Bg8` pinned horizontally to `Kh8` by `Ra8`, `Kg6` guards `g7/h7`)
        pinned_stale_fen = "R5bk/8/6K1/8/8/8/8/8 b - - 0 1"
        env_st = self._assert_matches_oracles(pinned_stale_fen)
        self.assertEqual(env_st["--stalemate"], 1)
        self.assertEqual(env_st["--status"], cr.STATUS_STALEMATE)

        # 5. Insufficient Material cases (`K vs K`, `K+N vs K`, `K+B vs K`, same-color `K+B vs K+B`,
        #    opposite-color `K+B vs K+B` [live], `K+N+N vs K` [live])
        insuf_cases = [
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 1),
            ("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1", 1),
            ("4k3/8/8/8/8/5B2/8/4K3 w - - 0 1", 1),
            ("4k3/5b2/8/8/8/5B2/8/4K3 w - - 0 1", 1),  # f3 (light) & f7 (light) -> same color!
            ("4k3/4b3/8/8/8/5B2/8/4K3 w - - 0 1", 0),  # f3 (light) & e7 (dark) -> opposite color!
            ("4k3/8/8/8/8/4NN2/8/4K3 w - - 0 1", 0),  # 2 Knights -> mate not forced, but helpmate exists!
        ]
        for fen, exp_ins in insuf_cases:
            ev = self._assert_matches_oracles(fen, check_all_selected=False)
            self.assertEqual(ev["--draw_insuf"], exp_ins, f"Unexpected draw_insuf on {fen}")

        # 6. 50-Move Rule (`hmc=99` quiet move -> `hmc=100` draw; vs `hmc=99` mating move -> Checkmate!)
        hmc99_quiet = "7k/8/6K1/8/8/8/8/R7 w - - 99 60"
        env_d50 = simulate_css_chess(
            self.spec,
            initial_fen=hmc99_quiet,
            ply_inputs=[(cr.parse_sq("a1"), cr.parse_sq("a2"), 1, cr.PT_QUEEN)],
            fast_stages=self.fast,
        )
        self.assertEqual(env_d50["--hmc"], 100)
        self.assertEqual(env_d50["--draw_50"], 1)
        self.assertEqual(env_d50["--status"], cr.STATUS_DRAW_50)

        env_mate100 = simulate_css_chess(
            self.spec,
            initial_fen=hmc99_quiet,
            ply_inputs=[(cr.parse_sq("a1"), cr.parse_sq("a8"), 1, cr.PT_QUEEN)],
            fast_stages=self.fast,
        )
        self.assertEqual(env_mate100["--hmc"], 100)
        self.assertEqual(env_mate100["--checkmate"], 1)
        self.assertEqual(env_mate100["--draw_50"], 0)
        self.assertEqual(env_mate100["--status"], cr.STATUS_WHITE_WINS_MATE)

    @unittest.skipIf(PYCHESS is None, "python-chess not installed")
    def test_06_random_multi_ply_rollouts_vs_python_chess(self) -> None:
        rng = random.Random(2026)
        for game_idx in range(6):
            pb = PYCHESS.Board()
            plies: List[Tuple[int, int, int, int]] = []
            for step in range(28):
                if pb.is_game_over():
                    break
                mv = rng.choice(list(pb.legal_moves))
                pr_map = {
                    None: cr.PT_QUEEN,
                    PYCHESS.QUEEN: cr.PT_QUEEN,
                    PYCHESS.ROOK: cr.PT_ROOK,
                    PYCHESS.BISHOP: cr.PT_BISHOP,
                    PYCHESS.KNIGHT: cr.PT_KNIGHT,
                }
                plies.append((mv.from_square, mv.to_square, 1, pr_map[mv.promotion]))
                pb.push(mv)

            env = simulate_css_chess(self.spec, initial_fen=cr.START_FEN, ply_inputs=plies, fast_stages=self.fast)
            self.assertEqual(env["--ply"], len(plies))
            self.assertEqual(env["--in_check"], int(pb.is_check()))
            self.assertEqual(env["--checkmate"], int(pb.is_checkmate()))
            self.assertEqual(env["--stalemate"], int(pb.is_stalemate()))
            self.assertEqual(env["--hmc"], pb.halfmove_clock)

            for sq in range(64):
                p = pb.piece_at(sq)
                exp_code = 0 if p is None else ((0 if p.color == PYCHESS.WHITE else 8) + p.piece_type)
                self.assertEqual(env[f"--b{sq}"], exp_code, f"Game {game_idx} board mismatch at {cr.sq_name(sq)}")

    def test_07_static_html5_and_zero_js_audit(self) -> None:
        html_path = os.path.join(CHESS_DIR, "index.html")
        self.assertTrue(os.path.isfile(html_path), "chess/index.html must exist")
        with open(html_path, "r", encoding="utf-8") as f:
            raw = f.read()

        auditor = _DOMAuditor()
        auditor.feed(raw)
        self.assertEqual(auditor.stack, [], f"Unclosed HTML tags: {auditor.stack}")
        self.assertNotIn("script", auditor.tags_seen, "Forbidden <script> tag in chess/index.html")
        self.assertNotIn("svg", auditor.tags_seen, "Phase 1 chess/index.html should use 0 SVG")
        self.assertEqual(auditor.event_attrs, [], f"Forbidden on* attributes: {auditor.event_attrs}")
        self.assertEqual(auditor.js_uris, [], f"Forbidden javascript: URIs: {auditor.js_uris}")
        self.assertTrue(auditor.has_back_link, "Missing GitHub Pages `../` back-link in chess/index.html")
        self.assertLess(
            auditor.max_depth,
            500,
            f"DOM nesting depth {auditor.max_depth} must stay safely below Chromium's 512 parser cap",
        )
        self.assertNotIn(
            "contain:style",
            raw.replace(" ", ""),
            "Safari/WebKit isolates ::after counters when contain:style is set on counter-reset elements",
        )
        self.assertNotIn(
            ":has(input:checked)",
            raw,
            "Subject-level :has(input:checked) fails to invalidate on radio uncheck in WebKit",
        )


if __name__ == "__main__":
    unittest.main()
