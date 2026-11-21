#!/usr/bin/env python3
"""
Automated Verification Suite for Pure HTML5 SVG (`feConvolveMatrix`) + CSS MNIST CNN (`cnn/`).

Tests:
  1. W3C `feConvolveMatrix` 180-degree kernel flip equivalence to 2D cross-correlation
  2. Signed `+0.5` zero-point `feColorMatrix` dynamic-range non-clipping bounds across all layers
  3. Exact 8-bit SVG simulator parity & accuracy across all 30 unseen golden MNIST test digits,
     30 inverted light-paper OCR variants, and all 20 generated `cnn/assets/*.png` sample cards
  4. Display-P3-immune neutral grayscale (`R = G = B`) invariant on all 1D displacement LUT PNGs
  5. Static HTML5, CSS & SVG DAG audit of `cnn/index.html` (zero JS, zero `<feImage>`, `sRGB`
     filters, `order <= 25` GPU fast path, relative `assets/` paths, O(N) stage path count)
  6. Colab notebook (`train_mnist_cnn.ipynb`) JSON & Python AST validation
"""

from __future__ import annotations

import ast
from html.parser import HTMLParser
import json
import os
import sys
import unittest
import xml.etree.ElementTree as ET
from typing import List, Set, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import generate_html
from svg_cnn import (
    ASSETS_DIR,
    CNN_DIR,
    LAYER_VIEWS,
    WEIGHTS_DEFAULT_PATH,
    compile_compute_stages,
    compile_pre_filters,
    compile_visual_stage,
    load_weights,
    q8,
    simulate_svg_cnn,
    svg_fe_convolve_2d,
)


class _DOMAuditor(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.tags_seen: Set[str] = set()
        self.event_attrs: List[str] = []
        self.js_uris: List[str] = []
        self.data_uris: List[str] = []
        self.img_srcs: List[str] = []
        self.svg_image_hrefs: List[str] = []
        self.has_contenteditable_true = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        t = tag.lower()
        self.tags_seen.add(t)
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if attr_map.get("contenteditable", "").lower() == "true":
            self.has_contenteditable_true = True
        if t == "img" and "src" in attr_map:
            self.img_srcs.append(attr_map["src"])
        if t == "image" and "href" in attr_map:
            self.svg_image_hrefs.append(attr_map["href"])

        for k, v in attrs:
            kl = k.lower()
            val = v or ""
            if kl.startswith("on"):
                self.event_attrs.append(f"{t}[{kl}]")
            if "javascript:" in val.lower():
                self.js_uris.append(f"{t}[{kl}]")
            if val.startswith("data:"):
                self.data_uris.append(f"{t}[{kl}]")


class TestNoJSCNN(unittest.TestCase):
    spec: dict
    html_doc: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_weights(WEIGHTS_DEFAULT_PATH)
        index_path = os.path.join(CNN_DIR, "index.html")
        if not os.path.exists(index_path):
            generate_html()
        with open(index_path, "r", encoding="utf-8") as f:
            cls.html_doc = f.read()

    def test_feconvolvematrix_180_flip_matches_crosscorrelation(self) -> None:
        rng = np.random.default_rng(7)
        inp = q8(rng.uniform(0.0, 1.0, size=(28, 28)))
        for f in self.spec["layers"]["conv1"]["filters"]:
            w_raw = np.asarray(f["weights_unflipped_3x3_or_5x5"], dtype=np.float64)
            bias = float(f["bias"])
            # Direct 5x5 valid cross-correlation at interior 2..25
            ref = np.zeros((24, 24), dtype=np.float64)
            for r in range(5):
                for c in range(5):
                    ref += w_raw[r, c] * inp[r : r + 24, c : c + 24]
            ref_q = q8(ref + bias)

            svg_out = svg_fe_convolve_2d(inp, f["order"], f["targetX"], f["targetY"], f["kernelMatrix"], f["divisor"], bias)
            np.testing.assert_allclose(svg_out[2:26, 2:26], ref_q, atol=1e-9)

    def test_signed_zero_point_bounds_never_clip(self) -> None:
        for stage_name in ("mix2", "mix3"):
            pw = self.spec["layers"][stage_name]
            for trip in pw["triplets"]:
                mats = [np.asarray(m, dtype=np.float64).reshape(4, 5)[:3, :] for m in trip["color_matrices"]]
                # Worst-case partial sum over any prefix of RGB input groups with inputs in [0, 1]
                pos_sum = np.zeros(3)
                neg_sum = np.zeros(3)
                for m in mats:
                    w3 = m[:, :3]
                    pos_sum += np.maximum(w3, 0.0).sum(axis=1)
                    neg_sum += np.minimum(w3, 0.0).sum(axis=1)
                    self.assertTrue(np.all(0.5 + pos_sum <= 0.99), f"{stage_name} positive partial sum clips 1.0")
                    self.assertTrue(np.all(0.5 + neg_sum >= 0.01), f"{stage_name} negative partial sum clips 0.0")

        fc = self.spec["layers"]["fc"]
        for trip in fc["triplets"]:
            mats = [np.asarray(m, dtype=np.float64).reshape(4, 5)[:3, :] for m in trip["color_matrices"]]
            base_off = mats[0][:, 4]  # includes 0.5 + b / (2M)
            pos_sum = base_off.copy()
            neg_sum = base_off.copy()
            for m in mats:
                w3 = m[:, :3]
                pos_sum += np.maximum(w3, 0.0).sum(axis=1)
                neg_sum += np.minimum(w3, 0.0).sum(axis=1)
                self.assertTrue(np.all(pos_sum <= 0.99), "FC positive partial sum clips 1.0")
                self.assertTrue(np.all(neg_sum >= 0.01), "FC negative partial sum clips 0.0")

    def test_golden_samples_and_auto_polarity_ocr(self) -> None:
        metrics = self.spec["metrics"]
        self.assertGreaterEqual(metrics["test_accuracy_svg_uint8"], 0.90)

        for gs in self.spec["golden_samples"]:
            img_u8 = np.asarray(gs["pixels_u8"], dtype=np.uint8)
            res = simulate_svg_cnn(img_u8, self.spec, auto_polarity=True)
            self.assertEqual(res.is_light_bg, 0)
            self.assertEqual(res.pred, gs["label"], f"Golden {gs['id']} misclassified")
            np.testing.assert_allclose(res.probs, np.asarray(gs["svg_probs"]), atol=1e-4)

            # Light-paper inverted OCR version must produce identical x0 and Softmax probabilities
            paper_rgb = np.stack([255 - img_u8] * 3, axis=-1).astype(np.uint8)
            res_paper = simulate_svg_cnn(paper_rgb, self.spec, auto_polarity=True)
            self.assertEqual(res_paper.is_light_bg, 1)
            self.assertEqual(res_paper.pred, gs["label"])
            np.testing.assert_allclose(res_paper.probs, res.probs, atol=1e-6)

        # Verify all 20 generated draggable card PNGs in cnn/assets/
        for prefix in ("mnist", "paper"):
            for d in range(10):
                p = os.path.join(ASSETS_DIR, f"{prefix}_{d}.png")
                self.assertTrue(os.path.exists(p), f"Missing sample asset {p}")
                im28 = np.asarray(Image.open(p).resize((28, 28), resample=Image.Resampling.NEAREST))
                sim = simulate_svg_cnn(im28, self.spec, auto_polarity=True)
                self.assertEqual(sim.pred, d, f"Asset {prefix}_{d}.png predicted {sim.pred} instead of {d}")
                if prefix == "paper":
                    self.assertEqual(sim.is_light_bg, 1)

    def test_neutral_gray_luts_are_display_p3_immune(self) -> None:
        lut_files = [fn for fn in os.listdir(ASSETS_DIR) if fn.startswith("lut_")]
        self.assertGreaterEqual(len(lut_files), 28)
        for fn in lut_files:
            arr = np.asarray(Image.open(os.path.join(ASSETS_DIR, fn)))
            self.assertEqual(arr.shape[2], 4)
            # R == G == B guarantees exact neutral D65 gray (zero sRGB->Display-P3 matrix cross-talk!)
            np.testing.assert_array_equal(arr[..., 0], arr[..., 1], err_msg=f"R != G in {fn}")
            np.testing.assert_array_equal(arr[..., 1], arr[..., 2], err_msg=f"G != B in {fn}")
            self.assertTrue(np.all(arr[..., 0] >= 128), f"Dark LUT byte <128 in {fn} quantizes in WebKit Display-P3/CoreImage")
            self.assertTrue(np.all(arr[..., 3] == 255), f"Alpha != 255 in {fn}")

    def test_static_html_zero_js_and_svg_dag_audit(self) -> None:
        auditor = _DOMAuditor()
        auditor.feed(self.html_doc)

        self.assertNotIn("script", auditor.tags_seen, "Forbidden <script> tag found")
        self.assertEqual(auditor.event_attrs, [], f"Forbidden inline event handlers: {auditor.event_attrs}")
        self.assertEqual(auditor.js_uris, [], f"Forbidden javascript: URIs: {auditor.js_uris}")
        self.assertEqual(auditor.data_uris, [], "Expected external relative assets, found inline data: URIs")
        self.assertTrue(auditor.has_contenteditable_true, 'Missing contenteditable="true" OCR input element')
        self.assertGreaterEqual(len(auditor.img_srcs), 20)
        self.assertGreaterEqual(len(auditor.svg_image_hrefs), 32)
        for src in auditor.img_srcs + auditor.svg_image_hrefs:
            self.assertTrue(src.startswith("assets/"), f"Expected relative assets/ path, got {src}")
            self.assertTrue(os.path.exists(os.path.join(CNN_DIR, src)), f"Missing referenced image {src}")

        self.assertNotIn(":has(input:checked)", self.html_doc, "Safari/WebKit :has() radio bug: use root-anchored ID selectors")
        self.assertIn("<foreignObject", self.html_doc, "Expected SVG <foreignObject> stage for Safari/WebKit filter chain")

        # Audit all compiled SVG <filter> stages (Safari/WebKit <= 80 prims/filter + Blink O(N) DAG guards)
        all_filters: List[str] = list(compile_pre_filters())
        for dpr in (1, 2):
            all_filters.extend(compile_compute_stages(self.spec, dpr))
            for k, *_ in LAYER_VIEWS:
                all_filters.extend(compile_visual_stage(k, dpr))

        for fx in all_filters:
            root = ET.fromstring(fx)
            fid = root.attrib["id"]
            self.assertEqual(root.attrib.get("color-interpolation-filters"), "sRGB")
            self.assertEqual(root.attrib.get("filterUnits"), "userSpaceOnUse", f"{fid} must use userSpaceOnUse")
            self.assertEqual(root.attrib.get("primitiveUnits"), "userSpaceOnUse", f"{fid} must use userSpaceOnUse")
            top_prims = list(root)
            total_fe_tags = sum(1 for el in root.iter() if el is not root)
            self.assertLessEqual(len(top_prims), 80, f"{fid} has {len(top_prims)} top-level prims (risks WebKit CoreImage hang)")
            self.assertLessEqual(total_fe_tags, 100, f"{fid} has {total_fe_tags} <fe*> tags (WebKit aborts >200)")

            defined = {"SourceGraphic", "SourceAlpha"}
            paths = {"SourceGraphic": 1, "SourceAlpha": 1}
            for el in top_prims:
                self.assertNotEqual(el.tag, "feImage", f"Forbidden <feImage> in {fid}")
                if el.tag == "feConvolveMatrix":
                    ox, oy = [int(v) for v in el.attrib["order"].split()]
                    self.assertLessEqual(ox * oy, 25, f"Kernel {ox}x{oy} exceeds 25-tap GPU budget in {fid}")
                if el.tag == "feMorphology":
                    radii = [float(v) for v in el.attrib["radius"].split()]
                    self.assertTrue(all(r > 0 for r in radii), f"Zero feMorphology radius in {fid} fails in WebKit: {radii}")
                cnt = 0
                for a in ("in", "in2"):
                    ref = el.attrib.get(a)
                    if ref:
                        self.assertIn(ref, defined, f"Undefined input wire {ref} in {fid}")
                        cnt += paths[ref]
                if el.tag == "feMerge":
                    for mn in el:
                        ref = mn.attrib["in"]
                        self.assertIn(ref, defined, f"Undefined feMergeNode {ref} in {fid}")
                        cnt += paths[ref]
                cnt = max(cnt, 1)
                if "result" in el.attrib:
                    defined.add(el.attrib["result"])
                    paths[el.attrib["result"]] = cnt
                self.assertLess(cnt, 1000, f"Excessive DAG path fan-in ({cnt}) in {fid}")

    def test_colab_notebook_valid(self) -> None:
        nb_path = os.path.join(CNN_DIR, "train_mnist_cnn.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 6)
        for i, cell in enumerate(code_cells):
            ast.parse("".join(cell["source"]), filename=f"train_mnist_cnn_cell_{i}")


if __name__ == "__main__":
    unittest.main()
