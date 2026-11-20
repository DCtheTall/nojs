"""
Combinational Boolean Circuit IR & CSS Layer Compiler.

Compiles arbitrary boolean DAGs into bounded-depth nested <div> stages using
registered CSS @property <integer> custom properties, guaranteeing:
  - Strict <integer> arithmetic (+, -, *) compatible with all modern browsers
  - Zero intra-element CSS custom-property cycles
  - Bounded calc() token length per property (no exponential var() blowup)
  - Liveness-based register reuse across layers and across repeated rounds
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple


class Op(str, Enum):
    INPUT = "INPUT"
    CONST = "CONST"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    XNOR = "XNOR"


@dataclass(frozen=True)
class Node:
    id: int
    op: Op
    a: Optional[int] = None
    b: Optional[int] = None
    const_val: Optional[int] = None
    input_name: Optional[str] = None


@dataclass
class LayerAssignment:
    """Compiled multi-layer CSS schedule for a combinational circuit macro."""

    num_layers: int
    num_temp_regs: int
    # layer_rules[layer_idx] = list of (lhs_css_var, rhs_calc_expr)
    layer_rules: List[List[Tuple[str, str]]]
    max_tokens: int


class Circuit:
    """DAG of 1-bit boolean gates with constant folding and CSE."""

    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self._cse: Dict[Tuple[Op, Optional[int], Optional[int]], int] = {}
        self._zero_id: Optional[int] = None
        self._one_id: Optional[int] = None

    def const(self, val: int) -> int:
        val = 1 if val else 0
        if val == 0:
            if self._zero_id is None:
                self._zero_id = self._add_node(Node(len(self.nodes), Op.CONST, const_val=0))
            return self._zero_id
        else:
            if self._one_id is None:
                self._one_id = self._add_node(Node(len(self.nodes), Op.CONST, const_val=1))
            return self._one_id

    def input(self, name: str) -> int:
        return self._add_node(Node(len(self.nodes), Op.INPUT, input_name=name))

    def inv(self, a: int) -> int:
        na = self.nodes[a]
        if na.op == Op.CONST:
            return self.const(1 - (na.const_val or 0))
        if na.op == Op.NOT and na.a is not None:
            return na.a
        if na.op == Op.XOR and na.a is not None and na.b is not None:
            return self.xnor(na.a, na.b)
        if na.op == Op.XNOR and na.a is not None and na.b is not None:
            return self.xor(na.a, na.b)
        key = (Op.NOT, a, None)
        if key in self._cse:
            return self._cse[key]
        nid = self._add_node(Node(len(self.nodes), Op.NOT, a=a))
        self._cse[key] = nid
        return nid

    def and_(self, a: int, b: int) -> int:
        if a == b:
            return a
        na, nb = self.nodes[a], self.nodes[b]
        if na.op == Op.CONST:
            return b if na.const_val == 1 else self.const(0)
        if nb.op == Op.CONST:
            return a if nb.const_val == 1 else self.const(0)
        x, y = (a, b) if a < b else (b, a)
        key = (Op.AND, x, y)
        if key in self._cse:
            return self._cse[key]
        nid = self._add_node(Node(len(self.nodes), Op.AND, a=x, b=y))
        self._cse[key] = nid
        return nid

    def or_(self, a: int, b: int) -> int:
        if a == b:
            return a
        na, nb = self.nodes[a], self.nodes[b]
        if na.op == Op.CONST:
            return self.const(1) if na.const_val == 1 else b
        if nb.op == Op.CONST:
            return self.const(1) if nb.const_val == 1 else a
        x, y = (a, b) if a < b else (b, a)
        key = (Op.OR, x, y)
        if key in self._cse:
            return self._cse[key]
        nid = self._add_node(Node(len(self.nodes), Op.OR, a=x, b=y))
        self._cse[key] = nid
        return nid

    def xor(self, a: int, b: int) -> int:
        if a == b:
            return self.const(0)
        na, nb = self.nodes[a], self.nodes[b]
        if na.op == Op.CONST:
            return self.inv(b) if na.const_val == 1 else b
        if nb.op == Op.CONST:
            return self.inv(a) if nb.const_val == 1 else a
        x, y = (a, b) if a < b else (b, a)
        key = (Op.XOR, x, y)
        if key in self._cse:
            return self._cse[key]
        nid = self._add_node(Node(len(self.nodes), Op.XOR, a=x, b=y))
        self._cse[key] = nid
        return nid

    def xnor(self, a: int, b: int) -> int:
        if a == b:
            return self.const(1)
        na, nb = self.nodes[a], self.nodes[b]
        if na.op == Op.CONST:
            return b if na.const_val == 1 else self.inv(b)
        if nb.op == Op.CONST:
            return a if nb.const_val == 1 else self.inv(a)
        x, y = (a, b) if a < b else (b, a)
        key = (Op.XNOR, x, y)
        if key in self._cse:
            return self._cse[key]
        nid = self._add_node(Node(len(self.nodes), Op.XNOR, a=x, b=y))
        self._cse[key] = nid
        return nid

    def xor_many(self, wires: Sequence[int]) -> int:
        """Balanced binary XOR tree to minimize gate depth."""
        cur = list(wires)
        if not cur:
            return self.const(0)
        while len(cur) > 1:
            nxt: List[int] = []
            for i in range(0, len(cur), 2):
                if i + 1 < len(cur):
                    nxt.append(self.xor(cur[i], cur[i + 1]))
                else:
                    nxt.append(cur[i])
            cur = nxt
        return cur[0]

    def _add_node(self, node: Node) -> int:
        self.nodes.append(node)
        return node.id

    def evaluate(self, inputs: Dict[str, int], outputs: Dict[str, int]) -> Dict[str, int]:
        """Bit-accurate simulation of the boolean DAG."""
        vals: List[int] = [0] * len(self.nodes)
        for n in self.nodes:
            if n.op == Op.CONST:
                vals[n.id] = int(n.const_val or 0)
            elif n.op == Op.INPUT:
                assert n.input_name is not None
                vals[n.id] = inputs[n.input_name] & 1
            elif n.op == Op.NOT:
                assert n.a is not None
                vals[n.id] = 1 - vals[n.a]
            elif n.op == Op.AND:
                assert n.a is not None and n.b is not None
                vals[n.id] = vals[n.a] & vals[n.b]
            elif n.op == Op.OR:
                assert n.a is not None and n.b is not None
                vals[n.id] = vals[n.a] | vals[n.b]
            elif n.op == Op.XOR:
                assert n.a is not None and n.b is not None
                vals[n.id] = vals[n.a] ^ vals[n.b]
            elif n.op == Op.XNOR:
                assert n.a is not None and n.b is not None
                vals[n.id] = 1 - (vals[n.a] ^ vals[n.b])
        return {name: vals[nid] for name, nid in outputs.items()}

    def compile_to_css_layers(
        self,
        outputs: Dict[str, int],
        max_inline_depth: int = 2,
        temp_prefix: str = "--w",
    ) -> LayerAssignment:
        """
        Partitions the DAG into nested <div> stages so that:
        1. Within any stage, inlined sub-expressions have depth <= max_inline_depth,
           keeping CSS calc() expressions compact (< ~100 tokens).
        2. Every emitted CSS declaration on stage L ONLY references CSS custom
           properties materialized strictly before stage L (stage < L, or inputs
           at stage -1). Thus zero intra-element var() dependencies or cycles exist.
        3. Named output properties (e.g. --s0..127, --k0..127) that also appear as
           circuit inputs are written ONLY in the final stage, while any input
           needed in the final stage is first latched in stage 0 if its name collides
           with an output name, guaranteeing zero self-cycles on the final stage.
        """
        num_nodes = len(self.nodes)
        output_node_ids: Set[int] = set(outputs.values())
        output_var_names: Set[str] = set(outputs.keys())

        # Compute fanout of each node among reachable nodes
        reachable: Set[int] = set()
        stack = list(outputs.values())
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            n = self.nodes[nid]
            if n.a is not None:
                stack.append(n.a)
            if n.b is not None:
                stack.append(n.b)

        fanout: Dict[int, int] = {nid: 0 for nid in reachable}
        for nid in reachable:
            n = self.nodes[nid]
            if n.a is not None:
                fanout[n.a] += 1
            if n.b is not None:
                fanout[n.b] += 1
        for nid in outputs.values():
            fanout[nid] = fanout.get(nid, 0) + 2  # Force materialization capability

        # Pass 1: Choose which intermediate nodes to materialize so inline trees stay <= max_inline_depth.
        inline_depth: Dict[int, int] = {}
        materialized: Set[int] = set()

        for n in self.nodes:
            if n.id not in reachable:
                continue
            if n.op in (Op.CONST, Op.INPUT):
                inline_depth[n.id] = 0
                if n.op == Op.INPUT:
                    materialized.add(n.id)
                continue

            deps = [x for x in (n.a, n.b) if x is not None]
            cur_inl_depth = 1
            for d in deps:
                d_depth = 1 if d in materialized else (inline_depth[d] + 1)
                cur_inl_depth = max(cur_inl_depth, d_depth)

            if cur_inl_depth > max_inline_depth:
                cur_inl_depth = 1
                for d in deps:
                    if d not in materialized and self.nodes[d].op not in (Op.CONST, Op.INPUT):
                        if inline_depth[d] >= max_inline_depth:
                            materialized.add(d)
                    d_depth = 1 if d in materialized else (inline_depth[d] + 1)
                    cur_inl_depth = max(cur_inl_depth, d_depth)

            inline_depth[n.id] = cur_inl_depth
            if fanout[n.id] >= 3 and cur_inl_depth >= max_inline_depth:
                materialized.add(n.id)

        # Pass 2: With `materialized` fixed, compute exact `node_stage` so every consumer
        # of a materialized node is scheduled in a strictly later stage (> dep stage).
        node_stage: Dict[int, int] = {}
        for n in self.nodes:
            if n.id not in reachable:
                continue
            if n.op in (Op.CONST, Op.INPUT):
                node_stage[n.id] = -1
                continue
            deps = [x for x in (n.a, n.b) if x is not None]
            req_stage = 0
            for d in deps:
                if d in materialized:
                    req_stage = max(req_stage, node_stage[d] + 1)
                else:
                    req_stage = max(req_stage, node_stage[d])
            node_stage[n.id] = req_stage

        # Final stage writes all named outputs
        max_internal_stage = max(
            (node_stage[nid] for nid in materialized if node_stage[nid] >= 0),
            default=-1,
        )
        for nid in output_node_ids:
            if self.nodes[nid].op not in (Op.CONST, Op.INPUT):
                max_internal_stage = max(max_internal_stage, node_stage[nid])

        final_stage = max(max_internal_stage, 0)
        # Ensure any materialized node that was scheduled AT final_stage and is read by another
        # output in final_stage pushes final_stage to max_stage + 1 so final_stage only reads < final_stage.
        # Let's collect for each output expression which materialized nodes it directly references!
        def get_materialized_deps(root_id: int, consumer_is_output: bool = False) -> Set[int]:
            res: Set[int] = set()

            def walk(u: int, is_root: bool) -> None:
                nu = self.nodes[u]
                if nu.op == Op.CONST:
                    return
                if not is_root and u in materialized:
                    res.add(u)
                    return
                if nu.op == Op.INPUT:
                    res.add(u)
                    return
                if nu.a is not None:
                    walk(nu.a, False)
                if nu.b is not None:
                    walk(nu.b, False)

            walk(root_id, True)
            return res

        for out_nid in outputs.values():
            if out_nid in materialized and self.nodes[out_nid].op not in (Op.CONST, Op.INPUT):
                # Output directly copies a materialized node -> needs final_stage > node_stage[out_nid]
                final_stage = max(final_stage, node_stage[out_nid] + 1)
            else:
                for dep in get_materialized_deps(out_nid):
                    if node_stage[dep] >= 0:
                        final_stage = max(final_stage, node_stage[dep] + 1)

        # Check if final_stage directly reads any INPUT whose input_name is ALSO in output_var_names!
        # If so, reading var(--s0) while writing --s0 on final_stage would be a CSS self-cycle!
        # Fix: create a stage-0 buffer copy for any such input read at final_stage (ensuring final_stage >= 1).
        final_input_reads: Set[int] = set()
        for out_nid in outputs.values():
            if self.nodes[out_nid].op == Op.INPUT:
                final_input_reads.add(out_nid)
            elif out_nid not in materialized:
                for dep in get_materialized_deps(out_nid):
                    if self.nodes[dep].op == Op.INPUT:
                        final_input_reads.add(dep)

        colliding_inputs: Dict[int, str] = {}
        for inp_nid in final_input_reads:
            inp_name = self.nodes[inp_nid].input_name
            assert inp_name is not None
            if inp_name in output_var_names:
                colliding_inputs[inp_nid] = inp_name
        if colliding_inputs and final_stage == 0:
            final_stage = 1

        num_layers = final_stage + 1

        # Compute liveness intervals [def_stage, last_use_stage] for all intermediate materialized nodes
        temp_nodes: List[int] = sorted(
            [
                nid
                for nid in materialized
                if self.nodes[nid].op not in (Op.CONST, Op.INPUT)
            ],
            key=lambda x: (node_stage[x], x),
        )

        last_use: Dict[int, int] = {nid: node_stage[nid] for nid in temp_nodes}
        for consumer in temp_nodes:
            cs = node_stage[consumer]
            for dep in get_materialized_deps(consumer):
                if dep in last_use:
                    last_use[dep] = max(last_use[dep], cs)

        for out_nid in outputs.values():
            if out_nid in last_use:
                last_use[out_nid] = max(last_use[out_nid], final_stage)
            else:
                for dep in get_materialized_deps(out_nid):
                    if dep in last_use:
                        last_use[dep] = max(last_use[dep], final_stage)

        # Allocate temp register indices via linear-scan interval coloring:
        # Register R occupied on [def_stage, last_use_stage] inclusive so no consumer stage
        # both reads var(--wR) and overwrites --wR on the same element!
        reg_busy_until: List[int] = []  # last_use_stage for each allocated register
        temp_reg_name: Dict[int, str] = {}

        # Also allocate registers for colliding_inputs latched at stage 0, read at final_stage
        latch_reg_name: Dict[int, str] = {}
        for inp_nid in sorted(colliding_inputs.keys()):
            r_idx = len(reg_busy_until)
            reg_busy_until.append(final_stage)
            latch_reg_name[inp_nid] = f"{temp_prefix}{r_idx}"

        for nid in temp_nodes:
            ds = node_stage[nid]
            lu = last_use[nid]
            chosen: Optional[int] = None
            for r_idx, busy_end in enumerate(reg_busy_until):
                if busy_end < ds:
                    chosen = r_idx
                    reg_busy_until[r_idx] = lu
                    break
            if chosen is None:
                chosen = len(reg_busy_until)
                reg_busy_until.append(lu)
            temp_reg_name[nid] = f"{temp_prefix}{chosen}"

        # Helper to format arithmetic integer expression (without outer calc())
        def render_expr(u: int, is_root: bool, in_final_stage: bool) -> str:
            nu = self.nodes[u]
            if nu.op == Op.CONST:
                return str(int(nu.const_val or 0))
            if nu.op == Op.INPUT:
                if in_final_stage and u in latch_reg_name:
                    return f"var({latch_reg_name[u]})"
                return f"var({nu.input_name})"
            if not is_root and u in temp_reg_name:
                return f"var({temp_reg_name[u]})"

            assert nu.a is not None
            ea = render_expr(nu.a, False, in_final_stage)
            if nu.op == Op.NOT:
                return f"(1 - {ea})"
            assert nu.b is not None
            eb = render_expr(nu.b, False, in_final_stage)
            if nu.op == Op.AND:
                return f"({ea} * {eb})"
            if nu.op == Op.OR:
                return f"({ea} + {eb} - {ea} * {eb})"
            if nu.op == Op.XOR:
                return f"({ea} + {eb} - 2 * {ea} * {eb})"
            if nu.op == Op.XNOR:
                return f"(1 - {ea} - {eb} + 2 * {ea} * {eb})"
            raise ValueError(f"Unsupported op {nu.op}")

        def wrap_calc(expr: str) -> str:
            if expr in ("0", "1") or (expr.startswith("var(") and expr.endswith(")") and " " not in expr):
                return expr
            # Strip redundant outer parens inside calc(...)
            if expr.startswith("(") and expr.endswith(")"):
                # Verify outer parens match each other
                depth = 0
                matched_at_end = True
                for idx, ch in enumerate(expr):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0 and idx < len(expr) - 1:
                            matched_at_end = False
                            break
                if matched_at_end:
                    expr = expr[1:-1]
            return f"calc({expr})"

        layer_rules: List[List[Tuple[str, str]]] = [[] for _ in range(num_layers)]

        # Stage 0 latches for colliding inputs
        for inp_nid, lreg in latch_reg_name.items():
            inp_name = self.nodes[inp_nid].input_name
            layer_rules[0].append((lreg, f"var({inp_name})"))

        # Intermediate temp assignments
        for nid in temp_nodes:
            st = node_stage[nid]
            rhs = wrap_calc(render_expr(nid, True, False))
            layer_rules[st].append((temp_reg_name[nid], rhs))

        # Final stage named outputs
        for out_name, out_nid in outputs.items():
            if out_nid in temp_reg_name:
                rhs = f"var({temp_reg_name[out_nid]})"
            else:
                rhs = wrap_calc(render_expr(out_nid, True, True))
            layer_rules[final_stage].append((out_name, rhs))

        # Compute max CSS token estimate and assert zero intra-layer cycles
        max_tokens = 0
        for st_idx, rules in enumerate(layer_rules):
            lhs_set = {lhs for lhs, _ in rules}
            for lhs, rhs in rules:
                for candidate in lhs_set:
                    if f"var({candidate})" in rhs:
                        raise AssertionError(
                            f"Intra-layer CSS cycle detected on stage {st_idx}: {lhs} references {candidate} in {rhs}"
                        )
                tok_count = rhs.count(" ") + rhs.count("(") + rhs.count(")") + 1
                max_tokens = max(max_tokens, tok_count)

        return LayerAssignment(
            num_layers=num_layers,
            num_temp_regs=len(reg_busy_until),
            layer_rules=layer_rules,
            max_tokens=max_tokens,
        )


def format_mod2_sum(terms: Sequence[str], invert: bool = False) -> str:
    """
    Format a GF(2) sum of `{0, 1}` expressions (e.g. `var(--a)`, `var(--a)*var(--b)`)
    into a minimal CSS `<integer>` expression using `mod(..., 2)` when needed.
    """
    items = list(terms)
    if invert:
        if not items:
            return "1"
        if len(items) == 1 and "*" not in items[0]:
            return f"calc(1 - {items[0]})"
        return f"mod(1 + {' + '.join(items)},2)"
    if not items:
        return "0"
    if len(items) == 1:
        return f"calc({items[0]})" if "*" in items[0] else items[0]
    return f"mod({' + '.join(items)},2)"


def mobius_anf_4bit(truth_table_16: Sequence[int], var_names: Sequence[str], invert_out: bool = False) -> str:
    """
    Compute the exact Algebraic Normal Form (polynomial over GF(2)) of a 4-input
    boolean function `f(v0, v1, v2, v3)` where index = (v0<<3)|(v1<<2)|(v2<<1)|v3,
    returning a compact CSS `mod(..., 2)` expression.
    """
    assert len(truth_table_16) == 16 and len(var_names) == 4
    coef = list(truth_table_16)
    # Fast binary Mobius transform over GF(2)^4
    for bit in range(4):
        step = 1 << bit
        for mask in range(16):
            if mask & step:
                coef[mask] ^= coef[mask ^ step]

    # Note: bit 3 corresponds to var_names[0], bit 0 to var_names[3]
    const_one = bool(coef[0]) ^ invert_out
    terms: List[str] = []
    for mask in range(1, 16):
        if coef[mask]:
            factors = [var_names[3 - b] for b in range(3, -1, -1) if (mask >> b) & 1]
            terms.append("*".join(factors))
    return format_mod2_sum(terms, invert=const_one)


def simulate_css_layers(
    assignment: LayerAssignment,
    initial_vars: Dict[str, int],
) -> Dict[str, int]:
    """
    Simulates browser CSS custom-property inheritance across the nested <div> stages
    (supporting both `calc(...)` and `mod(..., 2)`) to prove the compiled CSS
    expressions match Circuit.evaluate() bit-for-bit.
    """
    def _mod(a: int, b: int) -> int:
        return a % b

    env = dict(initial_vars)
    for st_idx, rules in enumerate(assignment.layer_rules):
        next_env = dict(env)
        for lhs, rhs in rules:
            expr = rhs
            if expr.startswith("calc(") and expr.endswith(")"):
                expr = expr[5:-1]
            # Substitute var(--foo) using ONLY parent env (matching CSS inheritance + zero intra-element deps)
            out_chars: List[str] = []
            i = 0
            while i < len(expr):
                if expr.startswith("var(", i):
                    end = expr.index(")", i + 4)
                    vname = expr[i + 4 : end]
                    if vname not in env:
                        raise KeyError(f"Stage {st_idx} rule {lhs} read undefined {vname}")
                    out_chars.append(str(int(env[vname])))
                    i = end + 1
                else:
                    out_chars.append(expr[i])
                    i += 1
            py_expr = "".join(out_chars)
            val = int(eval(py_expr, {"__builtins__": {}}, {"mod": _mod}))
            next_env[lhs] = val
        env = next_env
    return env
