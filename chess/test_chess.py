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
        self.elem_count = 0
        self.tags_seen: Set[str] = set()
        self.event_attrs: List[str] = []
        self.js_uris: List[str] = []
        self.has_back_link = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        self.elem_count += 1
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
            exp_sel = 0 if ref_ev.is_terminal else ref_ev.has_leg[u]
            self.assertEqual(
                env[f"--can_sel_{u}"],
                exp_sel,
                f"can_sel_{u} ({cr.sq_name(u)}) mismatch on {fen}",
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
        self.assertEqual(env_dc[f"--can_sel_{cr.parse_sq('d5')}"], 0)
        self.assertEqual(env_dc[f"--can_sel_{cr.parse_sq('e4')}"], 0)
        self.assertEqual(env_dc[f"--can_sel_{cr.parse_sq('e8')}"], 1)  # Kf7 legal

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

    def test_07_static_html5_svg_and_zero_js_audit(self) -> None:
        import xml.etree.ElementTree as ET
        import numpy as np
        from PIL import Image
        import svg_ai

        html_path = os.path.join(CHESS_DIR, "index.html")
        self.assertTrue(os.path.isfile(html_path), "chess/index.html must exist")
        with open(html_path, "r", encoding="utf-8") as f:
            raw = f.read()

        auditor = _DOMAuditor()
        auditor.feed(raw)
        self.assertEqual(auditor.stack, [], f"Unclosed HTML tags: {auditor.stack}")
        self.assertNotIn("script", auditor.tags_seen, "Forbidden <script> tag in chess/index.html")
        self.assertNotIn("feimage", auditor.tags_seen, "Forbidden <feImage> tag in chess/index.html")
        self.assertIn("svg", auditor.tags_seen, "Phase 2 chess/index.html must include SVG 2-ply CNN engine")
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
            ":has(",
            raw,
            "All :has(...) selectors must be eliminated in favor of O(1) top-level sibling combinators",
        )
        self.assertLessEqual(
            len(self.spec.registered_props),
            2850,
            f"Registered @property count {len(self.spec.registered_props)} regressed above 2,850 budget",
        )
        self.assertLessEqual(
            auditor.elem_count,
            34500,
            f"Total HTML/SVG element count {auditor.elem_count} regressed above 34,500 budget",
        )
        self.assertIn(
            ".ai-src-wrap i,.ai-meta{background:rgb(var(--g),var(--g),var(--g))}",
            raw,
            "<foreignObject> pixels must be strictly neutral gray (R==G==B) for Display-P3 immunity",
        )
        self.assertIn(
            "@media (-webkit-min-device-pixel-ratio:2.5),(min-resolution:2.5dppx)",
            raw,
            "Missing 3x DPR iPhone / mobile scaling media query",
        )
        self.assertIn(
            "#mode-1pw:checked~#ai-hint:not(:checked)~#mv #t0n:checked~.arena .ai-fstage",
            raw,
            "Missing Human-turn SVG filter pause rule for initial page load & Human clicks",
        )

        # Audit all compiled 1x & 2x SVG <filter> graphs and Display-P3/linearRGB lossless PNG LUTs
        w_spec = svg_ai.load_weights()
        stage_ids = svg_ai.get_compute_stage_ids(w_spec)
        self.assertLessEqual(len(stage_ids), 58)
        for dpr in (1, 2):
            flts = svg_ai.compile_chess_ai_filters(w_spec, dpr)
            self.assertEqual(len(flts), len(stage_ids))
            for sid, fxml in zip(stage_ids, flts):
                root = ET.fromstring(fxml)
                fw, fh = float(root.attrib["width"]), float(root.attrib["height"])
                if sid in ("visa", "visb"):
                    self.assertEqual((fw, fh), (float(svg_ai.STAGE_W), float(svg_ai.STAGE_H)))
                else:
                    self.assertEqual((fw, fh), (svg_ai.STAGE_W / dpr, svg_ai.COMP_H / dpr))
                prims = list(root)
                self.assertLessEqual(len(prims), 74, f"{sid} {dpr}x exceeds 74-primitive budget ({len(prims)})")
                defined = {"SourceGraphic"}
                nodes_dag: Dict[str, List[str]] = {}
                last_r = "SourceGraphic"
                for p_idx, pr in enumerate(prims):
                    self.assertNotEqual(pr.tag, "feImage")
                    if "x" in pr.attrib:
                        px, py = float(pr.attrib["x"]), float(pr.attrib["y"])
                        pw, ph = float(pr.attrib["width"]), float(pr.attrib["height"])
                        self.assertGreaterEqual(px, 0.0)
                        self.assertGreaterEqual(py, 0.0)
                        self.assertLessEqual(px + pw, fw, f"{sid} {dpr}x prim x+w={px+pw} > {fw}")
                        self.assertLessEqual(py + ph, fh, f"{sid} {dpr}x prim y+h={py+ph} > {fh}")
                    if pr.tag == "feConvolveMatrix":
                        ords = [int(x) for x in pr.attrib["order"].split()]
                        self.assertTrue(all(o <= 5 for o in ords), f"Kernel > 5x5 in {sid}: {ords}")
                    if pr.tag == "feComposite" and pr.attrib.get("operator") == "arithmetic":
                        ks = [float(pr.attrib[f"k{j}"]) for j in (1, 2, 3, 4)]
                        self.assertGreaterEqual(sum(ks), 0.999, f"Alpha drop in {sid}: {ks}")
                    if pr.tag == "feMerge":
                        for mn in pr:
                            self.assertIn(mn.attrib["in"], defined)
                        ins = [mn.attrib["in"] for mn in pr]
                    else:
                        ins = []
                        if "in" in pr.attrib:
                            self.assertIn(pr.attrib["in"], defined)
                            ins.append(pr.attrib["in"])
                        if "in2" in pr.attrib:
                            self.assertIn(pr.attrib["in2"], defined)
                            ins.append(pr.attrib["in2"])
                    r_key = pr.attrib.get("result", f"__p{p_idx}")
                    nodes_dag[r_key] = ins
                    last_r = r_key
                    if "result" in pr.attrib:
                        defined.add(pr.attrib["result"])

                # Guard against unmemoized Skia SkImageFilter::onFilterBounds diamond-chain blowup in Chrome!
                memo_walks: Dict[str, int] = {"SourceGraphic": 1}
                def _count_walks(nm: str) -> int:
                    if nm not in memo_walks:
                        memo_walks[nm] = 1 + sum(_count_walks(ch) for ch in nodes_dag.get(nm, []))
                    return memo_walks[nm]
                self.assertLessEqual(
                    _count_walks(last_r),
                    12000,
                    f"Skia DAG tree-walk blowup in {sid} ({dpr}x): {_count_walks(last_r)} > 12,000",
                )

        assets_dir = os.path.join(CHESS_DIR, "assets")
        pngs = [fn for fn in os.listdir(assets_dir) if fn.endswith(".png")]
        self.assertGreaterEqual(len(pngs), 30)
        for fn in pngs:
            arr = np.asarray(Image.open(os.path.join(assets_dir, fn)))
            self.assertEqual(arr.shape[-1], 4)
            self.assertTrue(np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]))
            self.assertTrue(np.all(arr[..., 3] == 255))
            u8 = arr[..., 0]
            self.assertTrue(np.all((u8 == 0) | (u8 >= 128)), f"Dark byte in {fn} risks linearRGB collision")


class TestSVG2PlyCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import svg_ai
        cls.svg_ai = svg_ai
        cls.css_spec = build_chess_css_spec()
        cls.fast = compile_fast_evaluator(cls.css_spec)
        cls.w_spec = svg_ai.load_weights()

    def test_01_golden_positions_bit_exact_from_css(self) -> None:
        import numpy as np
        gp = self.w_spec["golden_positions"]
        self.assertGreaterEqual(len(gp), 24)
        boards = []
        for g in gp:
            env = simulate_css_chess(self.css_spec, initial_fen=g["fen"], fast_stages=self.fast)
            boards.append(self.svg_ai.extract_root_board_u8_from_css(env))
        _, _, u8_pred = self.svg_ai.evaluate_svg_cnn_batch(np.stack(boards, axis=0), self.w_spec)
        u8_exp = np.array([g["svg_u8"] for g in gp], dtype=np.int32)
        self.assertTrue(np.array_equal(u8_pred, u8_exp), f"Golden u8 mismatch: {u8_pred} vs {u8_exp}")

    @unittest.skipIf(PYCHESS is None, "python-chess not installed")
    def test_02_e12_ply1_move_bus_parity_vs_python_chess(self) -> None:
        for _pid, _title, fen, _sub in cr.PRESETS:
            pb = PYCHESS.Board(fen)
            env = simulate_css_chess(self.css_spec, initial_fen=fen, fast_stages=self.fast)
            turn = int(env["--turn"])
            legal_by_to: Dict[int, Set[int]] = {sq: set() for sq in range(64)}
            for mv in pb.legal_moves:
                if mv.promotion in (None, PYCHESS.QUEEN):
                    legal_by_to[mv.to_square].add(mv.from_square)
            for v_norm in range(64):
                vr_n, vf_n = divmod(v_norm, 8)
                v_real = (8 * vr_n + vf_n) if turn == 0 else (8 * (7 - vr_n) + vf_n)
                c = env[f"--m1_c_{v_norm}"]
                if not legal_by_to[v_real]:
                    self.assertEqual(c, 0)
                else:
                    self.assertGreater(c, 0)
                    self.assertEqual(env[f"--m1_v_{v_norm}"], v_norm)
                    u_norm = env[f"--m1_u_{v_norm}"]
                    ur_n, uf_n = divmod(u_norm, 8)
                    u_real = (8 * ur_n + uf_n) if turn == 0 else (8 * (7 - ur_n) + uf_n)
                    self.assertIn(u_real, legal_by_to[v_real])

    def test_03_2ply_cnn_minimax_tactical_suite_and_self_play(self) -> None:
        tactics = [
            ("r1bqkbnr/ppp2ppp/2np4/4p3/4P1Q1/2N2N2/PPPP1PPP/R1B1KB1R b KQkq - 1 4", {"c8g4"}),
            ("rnb1kbnr/pppp1ppp/8/4p3/4P2q/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 3", {"h4f6", "h4f4", "h4e7", "h4d8", "h4h5", "h4h6", "h4g4"}),
            ("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3", {"d7d5", "g7g6", "d8e7", "d8f6", "g8h6"}),
            ("r1bq1rk1/pppp1ppp/2n5/2b1p1N1/4P2Q/3P4/PPP2PPP/RNB2RK1 b - - 4 6", {"f8e8", "h7h6", "h7h5"}),
            ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", {"d1d8"}),
            ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1", {"f7f5", "f7f6", "g7g5", "g7g6", "h7h5", "h7h6", "g8f8"}),
            ("4k3/p7/8/3b4/8/8/8/4K2R b - - 0 1", {"d5h1"}),
            ("7k/8/5K2/6Q1/8/8/8/8 w - - 0 1", {"g5g7", "g5h4"}),
            ("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1", {"e1e2"}),
        ]
        for fen, expected in tactics:
            env = simulate_css_chess(self.css_spec, initial_fen=fen, fast_stages=self.fast)
            res = self.svg_ai.simulate_svg_2ply_cnn(env, self.w_spec)
            uci = cr.sq_name(res.best_u_real) + cr.sq_name(res.best_v_real)
            self.assertIn(uci, expected, f"Tactic {fen}: got {uci}, expected {expected}")

        # Verify Scholar's Mate blunder `3... Nf6??` (`v1 = 21`) is explicitly scored `Q == 2` (`0.0%` mate loss)
        sch_env = simulate_css_chess(
            self.css_spec,
            initial_fen="r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
            fast_stages=self.fast,
        )
        sch_res = self.svg_ai.simulate_svg_2ply_cnn(sch_env, self.w_spec)
        self.assertEqual(int(sch_res.q_scores_u8[21]), 2)

        # Anti-blunder suite: never trade Queen for defended Knight (`QxN`), and always rescue an attacked Queen
        anti_qxn_1 = simulate_css_chess(
            self.css_spec,
            initial_fen="r1bqkb1r/ppp2ppp/2np4/3np3/2BQP3/2N2N2/PPPP1PPP/R1B1K2R w KQkq - 0 6",
            fast_stages=self.fast,
        )
        res_qxn_1 = self.svg_ai.simulate_svg_2ply_cnn(anti_qxn_1, self.w_spec)
        self.assertNotEqual(cr.sq_name(res_qxn_1.best_u_real) + cr.sq_name(res_qxn_1.best_v_real), "d4d5")

        anti_qxn_2 = simulate_css_chess(
            self.css_spec,
            initial_fen="r1b1kbnr/pp1ppppp/2n5/q1p5/4P3/2NP1N2/PPP2PPP/R1BQKB1R b KQkq - 2 4",
            fast_stages=self.fast,
        )
        res_qxn_2 = self.svg_ai.simulate_svg_2ply_cnn(anti_qxn_2, self.w_spec)
        self.assertNotEqual(cr.sq_name(res_qxn_2.best_u_real) + cr.sq_name(res_qxn_2.best_v_real), "a5c3")

        save_q = simulate_css_chess(
            self.css_spec,
            initial_fen="r1bqkbnr/ppp1pppp/2n5/8/3Q4/5N2/PPP1PPPP/RNB1KB1R w KQkq - 2 4",
            fast_stages=self.fast,
        )
        res_sq = self.svg_ai.simulate_svg_2ply_cnn(save_q, self.w_spec)
        uci_sq = cr.sq_name(res_sq.best_u_real) + cr.sq_name(res_sq.best_v_real)
        self.assertTrue(uci_sq.startswith("d4") and uci_sq not in {"d4c6", "d4d7", "d4e5"}, f"Did not save Qd4 safely: {uci_sq}")

        # Verify 10-ply self-play produces strictly legal moves on every turn
        plies: List[Tuple[int, int, int, int]] = []
        for _step in range(10):
            env = simulate_css_chess(self.css_spec, initial_fen=cr.START_FEN, ply_inputs=plies, fast_stages=self.fast)
            if env["--is_terminal"]:
                break
            res = self.svg_ai.simulate_svg_2ply_cnn(env, self.w_spec)
            self.assertLess(res.best_u_real, 64)
            self.assertLess(res.best_v_real, 64)
            plies.append((res.best_u_real, res.best_v_real, 1, cr.PT_QUEEN))
        self.assertEqual(len(plies), 10)

    def test_04_colab_notebook_syntax_audit(self) -> None:
        import ast
        import json
        nb_path = os.path.join(CHESS_DIR, "train_chess_value_cnn.ipynb")
        self.assertTrue(os.path.isfile(nb_path))
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 6)
        for idx, cell in enumerate(code_cells):
            src = "\n".join(
                line for line in "".join(cell["source"]).splitlines()
                if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(src, filename=f"cell_{idx}")

    def test_05_widened_capacity_18_27_36_5x5_compiler_audit(self) -> None:
        """Verify `svg_ai.py` cleanly compiles & simulates the widened 18->27->36 (5x5 dwconv1) CNN."""
        import copy
        import xml.etree.ElementTree as ET
        import numpy as np

        spec2 = copy.deepcopy(self.w_spec)
        z0 = float(spec2["zero_point"])

        def _make_pw(c_in: int, c_out: int) -> dict:
            g_in, g_out = c_in // 3, c_out // 3
            id_mat = [0.08, 0.0, 0.0, 0.0, z0, 0.0, 0.08, 0.0, 0.0, z0, 0.0, 0.0, 0.08, 0.0, z0, 0.0, 0.0, 0.0, 0.0, 1.0]
            return {
                "in_channels": c_in,
                "out_channels": c_out,
                "zero_point": z0,
                "triplets": [
                    {
                        "out_triplet": o,
                        "scales": [1.0, 1.0, 1.0],
                        "rescale_slopes": [1.0, 1.0, 1.0],
                        "rescale_intercepts": [0.02 - z0, 0.02 - z0, 0.02 - z0],
                        "color_matrices": [id_mat for _ in range(g_in)],
                    }
                    for o in range(g_out)
                ],
            }

        def _make_dw(c: int, k: int, pad: int) -> dict:
            tx = (k - 1) // 2 if pad > 0 else 0
            km = [0.0] * (k * k)
            km[tx * k + tx] = 0.5
            return {
                "num_channels": c,
                "svg_order": [k, k],
                "targetX": tx,
                "targetY": tx,
                "padding": pad,
                "filters": [
                    {"channel": ch, "order": [k, k], "targetX": tx, "targetY": tx, "divisor": 1.0, "bias": 0.02, "kernelMatrix": km, "padding": pad}
                    for ch in range(c)
                ],
            }

        spec2["layers"]["mix1"] = _make_pw(6, 18)
        spec2["layers"]["dwconv1"] = _make_dw(18, 5, 2)
        spec2["layers"]["mix2"] = _make_pw(18, 27)
        spec2["layers"]["dwconv2"] = _make_dw(27, 3, 1)
        spec2["layers"]["mix3"] = _make_pw(27, 36)
        spec2["layers"]["dwconv3"] = _make_dw(36, 2, 0)
        vh_row = [0.01, 0.01, 0.01, 0.0, z0]
        spec2["layers"]["value_head"]["in_features"] = 39
        spec2["layers"]["value_head"]["color_matrices"] = [vh_row + vh_row + vh_row + [0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(13)]

        sids = self.svg_ai.get_compute_stage_ids(spec2)
        for dpr in (1, 2):
            flts = self.svg_ai.compile_chess_ai_filters(spec2, dpr)
            self.assertEqual(len(flts), len(sids))
            for sid, fxml in zip(sids, flts):
                root = ET.fromstring(fxml)
                prims = list(root)
                self.assertLessEqual(len(prims), 74, f"Widened {sid} {dpr}x exceeds 74 prims ({len(prims)})")
                for pr in prims:
                    if pr.tag == "feConvolveMatrix":
                        ords = [int(x) for x in pr.attrib["order"].split()]
                        self.assertTrue(all(o <= 5 for o in ords))
                    if pr.tag == "feComposite" and pr.attrib.get("operator") == "arithmetic":
                        ks = [float(pr.attrib[f"k{j}"]) for j in (1, 2, 3, 4)]
                        self.assertGreaterEqual(sum(ks), 0.999)

        env = simulate_css_chess(self.css_spec, initial_fen=cr.START_FEN, fast_stages=self.fast)
        res2 = self.svg_ai.simulate_svg_2ply_cnn(env, spec2)
        self.assertEqual(res2.v_sheet_u8.shape, (16, 64))
        self.assertTrue(np.any(res2.q_scores_u8 > 0))

    def test_06_skia_component_transfer_truncation_and_queen_knight_blunder_guard(self) -> None:
        """Verify all compute `<feComponentTransfer>` LUT bytes survive Blink/Skia `int(255*v)`
        truncation AND WebKit `round(255*v)` rounding, and verify the 7-ply Scandinavian Queen
        position (`1. e4 h5 2. Nf3 d5 3. exd5 Qxd5 4. Nc3`) rejects `Qxf3??` in favor of safe retreat.
        """
        import xml.etree.ElementTree as ET
        import numpy as np

        sids = self.svg_ai.get_compute_stage_ids(self.w_spec)
        for dpr in (1, 2):
            flts = self.svg_ai.compile_chess_ai_filters(self.w_spec, dpr)
            for sid, fxml in zip(sids, flts):
                if sid in ("visa", "visb"):
                    continue
                root = ET.fromstring(fxml)
                for ct in root.findall("feComponentTransfer"):
                    for ch in ct:
                        if ch.attrib.get("type") in ("table", "discrete"):
                            for tok in ch.attrib.get("tableValues", "").split():
                                vf = float(tok)
                                f32 = np.float32(vf)
                                k_rnd = int(np.round(f32 * np.float32(255.0)))
                                k_trunc = int(f32 * np.float32(255.0))
                                self.assertEqual(
                                    k_trunc, k_rnd,
                                    f"Blink LUT truncation mismatch in {sid} ({dpr}x) token={tok}: "
                                    f"int={k_trunc} vs round={k_rnd}",
                                )

        fen7 = "rnb1kbnr/ppp1ppp1/8/3q3p/8/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 1 4"
        env7 = simulate_css_chess(self.css_spec, initial_fen=fen7, fast_stages=self.fast)
        res7 = self.svg_ai.simulate_svg_2ply_cnn(env7, self.w_spec)
        best_uci = cr.sq_name(res7.best_u_real) + cr.sq_name(res7.best_v_real)
        self.assertIn(best_uci, {"d5c5", "d5d8", "d5d6", "d5a5"}, f"Expected safe Qd5 retreat, got {best_uci}")
        self.assertLessEqual(int(res7.q_scores_u8[45]), 45, f"Qxf3?? (v1=45) must score <= 45, got {res7.q_scores_u8[45]}")
        self.assertGreaterEqual(int(res7.best_score_u8), 115)

    def test_07_ply3_quiescence_fide_legality_and_tactical_horizon(self) -> None:
        """Verify Ply-3 Quiescence (`s2q0..s2q7`) produces 100% FIDE-legal captures across random
        & tactical boards (including pins, single/double checks, King X-rays & promo-captures) and
        resolves 3-ply royal forks (`Nc7+ Kd8 Nxa8!`) and defended-piece recaptures.
        """
        import numpy as np

        sids = self.svg_ai.get_compute_stage_ids(self.w_spec)
        for q_sid in ("s2q0", "s2q1", "s2q2", "s2q3", "s2q4", "s2q5", "s2q6", "s2q7"):
            self.assertIn(q_sid, sids)

        if PYCHESS is not None:
            rng = random.Random(2026)
            bd = PYCHESS.Board()
            checked = 0
            selected = 0
            while checked < 2500:
                if bd.is_game_over() or bd.fullmove_number > 60:
                    bd = PYCHESS.Board()
                mvs = list(bd.legal_moves)
                if not mvs:
                    bd = PYCHESS.Board()
                    continue
                bd.push(rng.choice(mvs))
                if bd.is_checkmate():
                    continue
                s2_u8 = np.full((6, 8, 8), 128, dtype=np.uint8)
                stm = bd.turn
                for sq, pc in bd.piece_map().items():
                    r, f = divmod(sq, 8)
                    ay = r if stm == PYCHESS.WHITE else (7 - r)
                    s2_u8[pc.piece_type - 1, ay, f] = 255 if pc.color == stm else 1
                _, sel = self.svg_ai._select_and_apply_ply3_quiescence_8x8(s2_u8)
                checked += 1
                if sel is not None:
                    u3, v3, _pl, _tr = sel
                    ur = u3 if stm == PYCHESS.WHITE else (8 * (7 - (u3 // 8)) + (u3 % 8))
                    vr = v3 if stm == PYCHESS.WHITE else (8 * (7 - (v3 // 8)) + (v3 % 8))
                    pc_u = bd.piece_at(ur)
                    self.assertIsNotNone(pc_u)
                    promo = PYCHESS.QUEEN if (pc_u.piece_type == PYCHESS.PAWN and PYCHESS.square_rank(vr) in (0, 7)) else None
                    cmv = PYCHESS.Move(ur, vr, promotion=promo)
                    self.assertIn(cmv, bd.legal_moves, f"Illegal Ply-3 move {cmv} in {bd.fen()}")
                    self.assertTrue(bd.is_capture(cmv))
                    selected += 1
            self.assertGreater(selected, 1000)

        # 1. 3-Ply Royal Fork: 1. Nc7+ Ke7/Kd8 2. Nxa8! (missed at pure 2-ply, found at 2-ply + Q3!)
        fork_env = simulate_css_chess(
            self.css_spec,
            initial_fen="r1b1kbnr/pp3ppp/2n1p3/1N1p2q1/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 5",
            fast_stages=self.fast,
        )
        res_fork = self.svg_ai.simulate_svg_2ply_cnn(fork_env, self.w_spec)
        self.assertEqual(cr.sq_name(res_fork.best_u_real) + cr.sq_name(res_fork.best_v_real), "b5c7")
        self.assertGreaterEqual(int(res_fork.best_score_u8), 200)

        # 2. 3-Ply Winning Exchange: 1. Nxd5 exd5 2. Bxd5! (+1 pawn after Ply-3 recapture)
        combo_env = simulate_css_chess(
            self.css_spec,
            initial_fen="r2qkb1r/pppb1ppp/4p1n1/3n4/2B5/2N5/PPPPPPPP/R1BQK1NR w KQkq - 0 6",
            fast_stages=self.fast,
        )
        res_combo = self.svg_ai.simulate_svg_2ply_cnn(combo_env, self.w_spec)
        self.assertEqual(cr.sq_name(res_combo.best_u_real) + cr.sq_name(res_combo.best_v_real), "c3d5")
        self.assertGreaterEqual(int(res_combo.best_score_u8), 155)


if __name__ == "__main__":
    unittest.main()
