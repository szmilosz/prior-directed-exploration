from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from network_prep import DEFAULT_EMBEDDING_TENSOR, add_graph_output, add_output_to_onnx_file


def _tiny_model() -> onnx.ModelProto:
    """input -> Relu -> (embedding tensor) -> Neg -> output; the embedding tensor is internal."""
    x = helper.make_tensor_value_info("/input/planes", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("/output/policy", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Relu", ["/input/planes"], [DEFAULT_EMBEDDING_TENSOR]),
        helper.make_node("Neg", [DEFAULT_EMBEDDING_TENSOR], ["/output/policy"]),
    ]
    graph = helper.make_graph(nodes, "tiny", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


class NetworkPrepTest(unittest.TestCase):
    def test_add_graph_output_exposes_internal_tensor_once(self) -> None:
        model = _tiny_model()
        self.assertTrue(add_graph_output(model, DEFAULT_EMBEDDING_TENSOR))
        self.assertEqual([o.name for o in model.graph.output], ["/output/policy", DEFAULT_EMBEDDING_TENSOR])
        self.assertEqual(len(model.graph.node), 2)
        self.assertFalse(add_graph_output(model, DEFAULT_EMBEDDING_TENSOR))  # already an output
        self.assertEqual(len(model.graph.output), 2)

    def test_unknown_tensor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            add_graph_output(_tiny_model(), "/no/such/tensor")

    def test_add_output_to_onnx_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "plain.onnx", Path(tmp) / "with_embedding.onnx"
            onnx.save(_tiny_model(), str(src))
            self.assertEqual(add_output_to_onnx_file(src, dst), dst)
            loaded = onnx.load(str(dst))
        self.assertEqual([o.name for o in loaded.graph.output], ["/output/policy", DEFAULT_EMBEDDING_TENSOR])


if __name__ == "__main__":
    unittest.main()
