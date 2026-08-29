from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path

import onnx


DEFAULT_EMBEDDING_TENSOR = "/encoder14/ln2/betas"


def _model_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".pb"):
        name = name[:-3]
    return name


def resolve_output_paths(
    input_path: Path,
    output_path: Path | None = None,
    plain_output_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    stem = _model_stem(input_path)
    base_dir = output_dir or Path("model_assets")
    if plain_output_path is None:
        plain_output_path = base_dir / f"{stem}.onnx"
    if output_path is None:
        output_path = base_dir / f"{stem}_with_embedding.onnx"
    return plain_output_path, output_path


def _decompress_weights_if_needed(input_path: Path, work_dir: Path) -> Path:
    if input_path.suffix != ".gz":
        return input_path

    decompressed_path = work_dir / input_path.stem
    with gzip.open(input_path, "rb") as src, open(decompressed_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return decompressed_path


def run_leela2onnx(lc0_bin: Path, weights_path: Path, output_path: Path) -> None:
    cmd = [
        str(lc0_bin),
        "leela2onnx",
        f"--input={weights_path}",
        f"--output={output_path}",
        "--onnx2pytorch",
    ]
    subprocess.run(cmd, check=True)


def add_graph_output(model: onnx.ModelProto, tensor_name: str) -> bool:
    if any(output.name == tensor_name for output in model.graph.output):
        return False

    known_names = set()
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                known_names.add(output_name)
    for value in model.graph.value_info:
        known_names.add(value.name)
    for value in model.graph.output:
        known_names.add(value.name)
    for value in model.graph.input:
        known_names.add(value.name)
    for value in model.graph.initializer:
        known_names.add(value.name)

    if tensor_name not in known_names:
        raise ValueError(f"Tensor '{tensor_name}' was not found in the ONNX graph.")

    model.graph.output.extend(
        [
            onnx.helper.make_tensor_value_info(
                tensor_name,
                onnx.TensorProto.FLOAT,
                None,
            )
        ]
    )
    return True


def add_output_to_onnx_file(input_path: Path, output_path: Path, tensor_name: str = DEFAULT_EMBEDDING_TENSOR) -> Path:
    model = onnx.load(str(input_path))
    add_graph_output(model, tensor_name)
    onnx.save(model, str(output_path))
    return output_path


def prepare_network_artifact(
    input_path: Path,
    lc0_bin: Path,
    output_path: Path | None = None,
    plain_output_path: Path | None = None,
    output_dir: Path | None = None,
    embedding_tensor: str = DEFAULT_EMBEDDING_TENSOR,
) -> tuple[Path, Path]:
    plain_output_path, output_path = resolve_output_paths(
        input_path=input_path,
        output_path=output_path,
        plain_output_path=plain_output_path,
        output_dir=output_dir,
    )
    plain_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="paper-repro-lc0-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        weights_path = _decompress_weights_if_needed(input_path, temp_dir)
        run_leela2onnx(lc0_bin=lc0_bin, weights_path=weights_path, output_path=plain_output_path)

    add_output_to_onnx_file(
        input_path=plain_output_path,
        output_path=output_path,
        tensor_name=embedding_tensor,
    )
    return plain_output_path, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an Lc0 network file to the ONNX asset the trainer loads.")
    parser.add_argument("--input", required=True, help="Path to an LC0 .pb or .pb.gz file.")
    parser.add_argument("--lc0-bin", required=True, help="Path to an lc0 binary that supports leela2onnx.")
    parser.add_argument("--output", default=None, help="Optional path for the final _with_embedding.onnx file.")
    parser.add_argument("--plain-output", default=None, help="Optional path for the plain ONNX export.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory when --output paths are omitted.")
    parser.add_argument("--embedding-tensor", default=DEFAULT_EMBEDDING_TENSOR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plain_output_path, output_path = prepare_network_artifact(
        input_path=Path(args.input),
        lc0_bin=Path(args.lc0_bin),
        output_path=Path(args.output) if args.output else None,
        plain_output_path=Path(args.plain_output) if args.plain_output else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        embedding_tensor=str(args.embedding_tensor),
    )
    print(f"plain_onnx={plain_output_path}")
    print(f"with_embedding_onnx={output_path}")


if __name__ == "__main__":
    main()