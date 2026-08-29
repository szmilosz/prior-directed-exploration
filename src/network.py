import copy

import numpy as np
import torch
import torch.nn as nn


def _as_numpy_lc0_meta(value, dtype=np.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _board_to_lc0_planes_with_history_fill(board, lc0_meta=None):
    board_np = board.detach().float().cpu().numpy()
    batch_size = board_np.shape[0]
    planes = np.zeros((batch_size, 112, 8, 8), dtype=np.float32)

    current_piece_planes = board_np[:, :, ::-1, :]
    history_piece_planes = np.repeat(current_piece_planes[:, None, :, :, :], 8, axis=1)
    history_repetition = np.zeros((batch_size, 8), dtype=np.float32)
    castling = np.zeros((batch_size, 4), dtype=np.float32)
    side_to_move = np.zeros((batch_size,), dtype=np.float32)
    rule50_count = np.zeros((batch_size,), dtype=np.float32)

    if lc0_meta is not None:
        castling_np = _as_numpy_lc0_meta(lc0_meta.get("castling"), dtype=np.float32)
        if castling_np is not None:
            castling = castling_np.reshape(batch_size, 4)

        side_np = _as_numpy_lc0_meta(lc0_meta.get("side_to_move"), dtype=np.float32)
        if side_np is not None:
            side_to_move = side_np.reshape(batch_size)

        rule50_np = _as_numpy_lc0_meta(lc0_meta.get("rule50_count"), dtype=np.float32)
        if rule50_np is not None:
            rule50_count = rule50_np.reshape(batch_size)

        hist_rep_np = _as_numpy_lc0_meta(lc0_meta.get("history_repetition"), dtype=np.float32)
        if hist_rep_np is not None:
            history_repetition = hist_rep_np.reshape(batch_size, 8)

        hist_piece_np = _as_numpy_lc0_meta(lc0_meta.get("history_piece_planes"), dtype=np.float32)
        if hist_piece_np is not None:
            history_piece_planes = hist_piece_np.reshape(batch_size, 8, 12, 8, 8)

    for idx in range(8):
        start = idx * 13
        planes[:, start : start + 12] = history_piece_planes[:, idx]
        planes[:, start + 12] = history_repetition[:, idx][:, None, None]

    planes[:, 104] = castling[:, 0][:, None, None]
    planes[:, 105] = castling[:, 1][:, None, None]
    planes[:, 106] = castling[:, 2][:, None, None]
    planes[:, 107] = castling[:, 3][:, None, None]
    planes[:, 108] = side_to_move[:, None, None]
    planes[:, 109] = rule50_count[:, None, None]
    planes[:, 110] = 0.0
    planes[:, 111] = 1.0
    return planes


def _to_b3_tensor(candidate, batch_size):
    if isinstance(candidate, np.ndarray):
        tensor = torch.as_tensor(candidate, dtype=torch.float32)
    elif torch.is_tensor(candidate):
        tensor = candidate.to(dtype=torch.float32)
    else:
        return None

    if tensor.ndim == 3 and tensor.shape[0] == batch_size and tensor.shape[1] == 1 and tensor.shape[2] == 3:
        tensor = tensor[:, 0, :]
    elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == batch_size and tensor.shape[2] == 3:
        tensor = tensor[0, :, :]

    if tensor.ndim == 2 and tensor.shape[0] == batch_size and tensor.shape[1] == 3:
        return tensor
    if tensor.ndim == 1 and tensor.numel() == batch_size * 3:
        return tensor.reshape(batch_size, 3)
    return None


def _extract_lc0_wdl_tensor(outputs, batch_size, preferred_keys=()):
    if isinstance(outputs, dict):
        for key in preferred_keys:
            if key in outputs:
                tensor = _to_b3_tensor(outputs[key], batch_size=batch_size)
                if tensor is not None:
                    return tensor
        for value in outputs.values():
            tensor = _to_b3_tensor(value, batch_size=batch_size)
            if tensor is not None:
                return tensor

    tensor = _to_b3_tensor(outputs, batch_size=batch_size)
    if tensor is not None:
        return tensor

    if isinstance(outputs, (list, tuple)):
        for value in outputs:
            tensor = _to_b3_tensor(value, batch_size=batch_size)
            if tensor is not None:
                return tensor

    raise RuntimeError("Could not extract LC0 WDL tensor [B,3] from model outputs.")


def _to_lc0_wdl_probs_tensor(wdl):
    if wdl.ndim != 2 or wdl.shape[1] != 3:
        raise RuntimeError(f"Expected WDL shape [B,3], got {tuple(wdl.shape)}")

    sums = wdl.sum(dim=1, keepdim=True)
    if bool(torch.all(wdl >= 0.0).item()) and bool(torch.all(torch.isfinite(wdl)).item()):
        if torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=0.0):
            return wdl

    return torch.softmax(wdl, dim=-1)


class Lc0MoveHeadModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        dense1_w, dense1_b, q_w, q_b, k_w, k_b, scale, promotion_w = self._load_policy_tensors_from_onnx(onnx_path)

        self.lc0_emb_size = int(dense1_w.shape[0])
        self.dense = nn.Linear(self.lc0_emb_size, int(dense1_w.shape[1]))
        self.q = nn.Linear(int(q_w.shape[0]), int(q_w.shape[1]))
        self.k = nn.Linear(int(k_w.shape[0]), int(k_w.shape[1]))
        self.activation = nn.Mish()

        self._copy_linear_from_onnx(self.dense, dense1_w, dense1_b)
        self._copy_linear_from_onnx(self.q, q_w, q_b)
        self._copy_linear_from_onnx(self.k, k_w, k_b)

        self.register_buffer("policy_scale", torch.tensor(float(scale), dtype=torch.float32))
        self.promotion_matmul_w = nn.Parameter(torch.from_numpy(np.array(promotion_w, copy=True)).to(torch.float32))

    @staticmethod
    def _reorder_b64d_square_order(x, from_order, to_order):
        if from_order == to_order:
            return x
        if {from_order, to_order} != {"model", "lc0"}:
            raise RuntimeError(f"Unsupported square order conversion: {from_order} -> {to_order}")
        return x.reshape(x.shape[0], 8, 8, x.shape[2])[:, torch.arange(7, -1, -1, device=x.device), :, :].reshape(
            x.shape[0], 64, x.shape[2]
        )

    @staticmethod
    def _reorder_b6464_square_order(x, from_order, to_order):
        if from_order == to_order:
            return x
        if {from_order, to_order} != {"model", "lc0"}:
            raise RuntimeError(f"Unsupported square order conversion: {from_order} -> {to_order}")

        batch_size = x.shape[0]
        squares = x.reshape(batch_size, 8, 8, 8, 8)
        squares = squares[:, torch.arange(7, -1, -1, device=x.device), :, :, :]
        squares = squares[:, :, :, torch.arange(7, -1, -1, device=x.device), :]
        return squares.reshape(batch_size, 64, 64)

    @staticmethod
    def _copy_linear_from_onnx(linear, weight_2d, bias_1d):
        weight = np.array(weight_2d.T, copy=True)
        bias = np.array(bias_1d, copy=True)
        linear.weight.data.copy_(torch.from_numpy(weight).to(linear.weight.dtype))
        linear.bias.data.copy_(torch.from_numpy(bias).to(linear.bias.dtype))

    @staticmethod
    def _load_policy_tensors_from_onnx(onnx_path):
        try:
            import onnx
            from onnx import numpy_helper
        except Exception as exc:
            raise RuntimeError("LC0 move head requires onnx package. Install with: pip install onnx") from exc

        onnx_model = onnx.load(str(onnx_path))
        initializer_map = {
            init.name: numpy_helper.to_array(init).astype(np.float32, copy=False)
            for init in onnx_model.graph.initializer
        }

        def find_node(node_name):
            target = node_name.lstrip("/")
            for node in onnx_model.graph.node:
                if node.name == node_name or node.name.lstrip("/") == target:
                    return node
            raise RuntimeError(f"Could not find ONNX node '{node_name}' in {onnx_path}.")

        def matmul_weight(node_name):
            node = find_node(node_name)
            if len(node.input) < 2:
                raise RuntimeError(f"Node '{node_name}' does not have MatMul weight input.")
            weight_name = node.input[1]
            if weight_name not in initializer_map:
                raise RuntimeError(f"Initializer '{weight_name}' for node '{node_name}' not found.")
            return initializer_map[weight_name]

        def add_bias(node_name):
            node = find_node(node_name)
            if len(node.input) < 2:
                raise RuntimeError(f"Node '{node_name}' does not have Add bias input.")
            bias_name = node.input[1]
            if bias_name not in initializer_map:
                raise RuntimeError(f"Initializer '{bias_name}' for node '{node_name}' not found.")
            bias = initializer_map[bias_name]
            if bias.ndim != 1:
                bias = bias.reshape(-1)
            return bias

        def mul_scale(node_name):
            node = find_node(node_name)
            if len(node.input) < 2:
                raise RuntimeError(f"Node '{node_name}' does not have Mul scale input.")
            for input_name in node.input:
                if input_name in initializer_map:
                    arr = initializer_map[input_name]
                    return float(np.asarray(arr, dtype=np.float32).reshape(-1)[0])
            raise RuntimeError(f"Could not find scalar initializer for node '{node_name}'.")

        return (
            matmul_weight("policy/dense1/matmul"),
            add_bias("policy/dense1/add"),
            matmul_weight("policy/Q/matmul"),
            add_bias("policy/Q/add"),
            matmul_weight("policy/K/matmul"),
            add_bias("policy/K/add"),
            mul_scale("policy/scale"),
            matmul_weight("policy/promotion/matmul"),
        )

    def _compute_policy_logits_and_promotion_offsets(self, x):
        if x.ndim != 3 or x.shape[1] != 64:
            raise RuntimeError(f"Expected LC0 move-head input [B,64,d], got {tuple(x.shape)}.")

        x = self._reorder_b64d_square_order(x, from_order="model", to_order="lc0")
        if x.shape[-1] != self.lc0_emb_size:
            raise RuntimeError(f"LC0 move-head expected feature dim {self.lc0_emb_size}, got {x.shape[-1]}.")

        batch_size = x.size(0)
        x_dense = self.activation(self.dense(x))
        q = self.q(x_dense)
        k = self.k(x_dense)
        policy_scale = torch.matmul(q, k.transpose(-1, -2)) * self.policy_scale.to(dtype=q.dtype)

        keys_last_8 = k[:, 56:64, :]
        promotion_raw = torch.matmul(keys_last_8, self.promotion_matmul_w.to(dtype=k.dtype))
        promotion_raw = promotion_raw.transpose(1, 2)
        promotion_offsets = promotion_raw[:, :3, :] + promotion_raw[:, 3:4, :]

        policy_scale = self._reorder_b6464_square_order(policy_scale, from_order="lc0", to_order="model")
        return policy_scale.reshape(batch_size, -1), promotion_offsets

    def forward_with_promotion_offsets(self, x):
        return self._compute_policy_logits_and_promotion_offsets(x)

    def forward(self, x):
        logits, _ = self._compute_policy_logits_and_promotion_offsets(x)
        return logits


class ValueHeadModel(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.dense = nn.Linear(emb_size, 32)
        torch.nn.init.xavier_normal_(self.dense.weight)
        self.activation = nn.Mish()
        self.dense_2 = nn.Linear(32 * 64, 128)
        torch.nn.init.xavier_normal_(self.dense_2.weight)
        self.out_linear = nn.Linear(128, 1)
        torch.nn.init.xavier_normal_(self.out_linear.weight)

    def forward(self, x):
        x = self.activation(self.dense(x))
        x = x.view(x.size(0), 64 * 32)
        x = self.activation(self.dense_2(x))
        return self.out_linear(x)


class TrainableLc0PerSquareEmbedder(nn.Module):
    def __init__(self, onnx_path, embedding_tensor_name="/encoder14/ln2/betas", square_order="model"):
        super().__init__()
        try:
            import onnx
            from onnx2torch import convert
        except Exception as exc:
            raise RuntimeError(
                "trainable_torch LC0 backend requires onnx and onnx2torch. Install with: pip install onnx onnx2torch"
            ) from exc

        self.embedding_tensor_name = embedding_tensor_name
        self.pre_last_embedding_tensor_name = "/encoder13/ln2/betas"
        self.square_order = square_order
        self.supports_pre_last_embedding = True
        self.output_wdl_name = "/output/wdl"
        self.output_wdl_name_alt = "output/wdl"

        onnx_model = onnx.load(str(onnx_path))
        if not any(o.name == self.pre_last_embedding_tensor_name for o in onnx_model.graph.output):
            onnx_model.graph.output.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        self.pre_last_embedding_tensor_name,
                        onnx.TensorProto.FLOAT,
                        None,
                    )
                ]
            )
        self.model_output_names = [o.name for o in onnx_model.graph.output]
        self.model = convert(onnx_model)
        self.promoted_buffer_count = 0
        if sum(1 for _ in self.model.parameters()) == 0:
            self.promoted_buffer_count = self._promote_float_buffers_to_parameters(self.model)
        self.embedding_dim = self._infer_embedding_dim()

    @staticmethod
    def _promote_float_buffers_to_parameters(root_module):
        promoted = 0
        skip_names = {"running_mean", "running_var", "num_batches_tracked"}
        for module in root_module.modules():
            for name, buf in list(module.named_buffers(recurse=False)):
                if name in skip_names:
                    continue
                if not torch.is_tensor(buf) or not torch.is_floating_point(buf):
                    continue
                delattr(module, name)
                module.register_parameter(name, nn.Parameter(buf.detach().clone(), requires_grad=True))
                promoted += 1
        return promoted

    def _outputs_to_named_map(self, outputs):
        if isinstance(outputs, dict):
            return outputs
        if isinstance(outputs, (list, tuple)):
            return {
                name: outputs[idx]
                for idx, name in enumerate(self.model_output_names)
                if idx < len(outputs)
            }
        return {}

    def _extract_named_embedding_tensor(self, outputs, batch_size, tensor_name):
        named_outputs = self._outputs_to_named_map(outputs)
        if tensor_name in named_outputs and torch.is_tensor(named_outputs[tensor_name]):
            return self._to_b64d_tensor(named_outputs[tensor_name], batch_size=batch_size, square_order=self.square_order)

        if isinstance(outputs, dict):
            if tensor_name in outputs and torch.is_tensor(outputs[tensor_name]):
                return self._to_b64d_tensor(outputs[tensor_name], batch_size=batch_size, square_order=self.square_order)
            for value in outputs.values():
                if torch.is_tensor(value):
                    try:
                        return self._to_b64d_tensor(value, batch_size=batch_size, square_order=self.square_order)
                    except RuntimeError:
                        continue

        if torch.is_tensor(outputs):
            return self._to_b64d_tensor(outputs, batch_size=batch_size, square_order=self.square_order)

        if isinstance(outputs, (list, tuple)):
            for value in outputs:
                if torch.is_tensor(value):
                    try:
                        return self._to_b64d_tensor(value, batch_size=batch_size, square_order=self.square_order)
                    except RuntimeError:
                        continue

        raise RuntimeError("Could not extract Lc0 embedding tensor from trainable model outputs.")

    def _extract_embedding_tensor(self, outputs, batch_size):
        return self._extract_named_embedding_tensor(outputs, batch_size=batch_size, tensor_name=self.embedding_tensor_name)

    def _infer_embedding_dim(self):
        sample = torch.zeros(1, 112, 8, 8, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(sample)
        emb = self._extract_embedding_tensor(outputs, batch_size=1)
        return int(emb.shape[-1])

    def _extract_wdl_tensor(self, outputs, batch_size):
        return _extract_lc0_wdl_tensor(
            outputs,
            batch_size=batch_size,
            preferred_keys=(self.output_wdl_name, self.output_wdl_name_alt),
        )

    @staticmethod
    def _to_b64d_tensor(emb, batch_size, square_order):
        if emb.ndim == 2:
            if emb.shape[0] == batch_size * 64:
                emb = emb.reshape(batch_size, 64, emb.shape[1])
            elif emb.shape[0] == batch_size and emb.shape[1] % 64 == 0:
                emb = emb.reshape(batch_size, 64, emb.shape[1] // 64)
            else:
                raise RuntimeError(
                    f"Cannot convert embedding shape {tuple(emb.shape)} to [B,64,d] for B={batch_size}."
                )
        elif emb.ndim == 3:
            if not (emb.shape[0] == batch_size and emb.shape[1] == 64):
                raise RuntimeError(f"Expected [B,64,d], got {tuple(emb.shape)} for B={batch_size}.")
        else:
            raise RuntimeError(f"Unsupported embedding rank {emb.ndim} for shape {tuple(emb.shape)}.")

        if square_order == "model":
            emb = emb.reshape(batch_size, 8, 8, emb.shape[2])[:, torch.arange(7, -1, -1, device=emb.device), :, :].reshape(
                batch_size, 64, emb.shape[2]
            )
        elif square_order != "lc0":
            raise RuntimeError(f"Unsupported square_order: {square_order}")

        return emb

    def forward(self, board, lc0_meta=None, return_wdl=False, return_pre_last=False):
        planes = _board_to_lc0_planes_with_history_fill(board, lc0_meta=lc0_meta)
        planes_tensor = torch.from_numpy(planes).to(device=board.device, dtype=torch.float32)
        outputs = self.model(planes_tensor)
        emb = self._extract_embedding_tensor(outputs, batch_size=planes_tensor.shape[0]).to(device=board.device)

        pre_last_emb = None
        if return_pre_last:
            pre_last_emb = self._extract_named_embedding_tensor(
                outputs,
                batch_size=planes_tensor.shape[0],
                tensor_name=self.pre_last_embedding_tensor_name,
            ).to(device=board.device)

        if return_wdl:
            wdl = self._extract_wdl_tensor(outputs, batch_size=planes_tensor.shape[0]).to(device=board.device)
            wdl_probs = _to_lc0_wdl_probs_tensor(wdl)
            if return_pre_last:
                return emb, wdl_probs, pre_last_emb
            return emb, wdl_probs

        if return_pre_last:
            return emb, pre_last_emb

        return emb


class Lc0LastEncoderBlockFromConvertedModel(nn.Module):
    def __init__(self, source_graph_module, input_node_name="encoder13_ln2_betas", output_node_name="encoder14_ln2_betas"):
        super().__init__()
        self.input_node_name = input_node_name
        self.output_node_name = output_node_name
        self.graph = copy.deepcopy(source_graph_module.graph)

        required = self._collect_required_node_names(
            self.graph,
            input_node_name=self.input_node_name,
            output_node_name=self.output_node_name,
        )
        self.exec_node_names = [
            node.name
            for node in self.graph.nodes
            if (node.name in required) and (node.op != "output") and (node.name != self.input_node_name)
        ]

        required_get_attr_targets = []
        required_call_module_targets = []
        for node in self.graph.nodes:
            if node.name not in required:
                continue
            if node.op == "get_attr":
                required_get_attr_targets.append(node.target)
            elif node.op == "call_module":
                required_call_module_targets.append(node.target)

        self.call_modules = nn.ModuleDict()
        self._module_key_by_target = {}
        for idx, target in enumerate(sorted(set(required_call_module_targets))):
            try:
                source_module = source_graph_module.get_submodule(target)
                copied_module = copy.deepcopy(source_module)
            except Exception:
                continue
            key = f"m{idx}"
            self.call_modules[key] = copied_module
            self._module_key_by_target[target] = key

        self._parameter_key_by_target = {}
        self._buffer_key_by_target = {}
        self._const_attr_by_target = {}
        for target in sorted(set(required_get_attr_targets)):
            try:
                param = source_graph_module.get_parameter(target)
            except Exception:
                param = None
            if param is not None:
                key = f"p{len(self._parameter_key_by_target)}"
                self.register_parameter(key, nn.Parameter(param.detach().clone(), requires_grad=True))
                self._parameter_key_by_target[target] = key
                continue

            try:
                buf = source_graph_module.get_buffer(target)
            except Exception:
                buf = None
            if buf is not None:
                key = f"b{len(self._buffer_key_by_target)}"
                self.register_buffer(key, buf.detach().clone())
                self._buffer_key_by_target[target] = key
                continue

            attr = self._fetch_attr(source_graph_module, target)
            if torch.is_tensor(attr):
                key = f"b{len(self._buffer_key_by_target)}"
                self.register_buffer(key, attr.detach().clone())
                self._buffer_key_by_target[target] = key
            else:
                self._const_attr_by_target[target] = copy.deepcopy(attr)

    @staticmethod
    def _collect_required_node_names(graph, input_node_name, output_node_name):
        nodes_by_name = {node.name: node for node in graph.nodes}
        if input_node_name not in nodes_by_name:
            raise RuntimeError(f"Could not find LC0 input node '{input_node_name}' in converted graph.")
        if output_node_name not in nodes_by_name:
            raise RuntimeError(f"Could not find LC0 output node '{output_node_name}' in converted graph.")

        required = set()
        stack = [nodes_by_name[output_node_name]]

        def _push_arg(arg):
            if isinstance(arg, torch.fx.Node):
                if arg.name in nodes_by_name:
                    stack.append(nodes_by_name[arg.name])
                return
            if isinstance(arg, (tuple, list)):
                for item in arg:
                    _push_arg(item)
                return
            if isinstance(arg, dict):
                for item in arg.values():
                    _push_arg(item)

        while stack:
            node = stack.pop()
            if node.name in required:
                continue
            required.add(node.name)
            if node.name == input_node_name:
                continue
            _push_arg(node.args)
            _push_arg(node.kwargs)

        return required

    @staticmethod
    def _fetch_attr(module, target):
        attr = module
        for item in target.split("."):
            attr = getattr(attr, item)
        return attr

    def _fetch_copied_attr(self, target):
        if target in self._parameter_key_by_target:
            return getattr(self, self._parameter_key_by_target[target])
        if target in self._buffer_key_by_target:
            return getattr(self, self._buffer_key_by_target[target])
        if target in self._const_attr_by_target:
            return self._const_attr_by_target[target]
        raise RuntimeError(f"Missing copied get_attr target '{target}' in LC0 last-block runner.")

    @staticmethod
    def _resolve_arg(arg, env):
        if isinstance(arg, torch.fx.Node):
            return env[arg.name]
        if isinstance(arg, tuple):
            return tuple(Lc0LastEncoderBlockFromConvertedModel._resolve_arg(x, env) for x in arg)
        if isinstance(arg, list):
            return [Lc0LastEncoderBlockFromConvertedModel._resolve_arg(x, env) for x in arg]
        if isinstance(arg, dict):
            return {key: Lc0LastEncoderBlockFromConvertedModel._resolve_arg(value, env) for key, value in arg.items()}
        return arg

    def forward(self, x):
        env = {self.input_node_name: x}
        node_by_name = {node.name: node for node in self.graph.nodes}

        for node_name in self.exec_node_names:
            node = node_by_name[node_name]
            if node.op == "get_attr":
                env[node.name] = self._fetch_copied_attr(node.target)
                continue

            args = self._resolve_arg(node.args, env)
            kwargs = self._resolve_arg(node.kwargs, env)
            if node.op == "call_module":
                module_key = self._module_key_by_target.get(node.target)
                if module_key is None:
                    raise RuntimeError(f"Missing copied call_module target '{node.target}' in LC0 last-block runner.")
                env[node.name] = self.call_modules[module_key](*args, **kwargs)
            elif node.op == "call_function":
                env[node.name] = node.target(*args, **kwargs)
            elif node.op == "call_method":
                env[node.name] = getattr(args[0], node.target)(*args[1:], **kwargs)
            elif node.op == "placeholder":
                continue
            else:
                raise RuntimeError(f"Unsupported FX node op in LC0 last-block runner: {node.op}")

        if self.output_node_name not in env:
            raise RuntimeError(f"LC0 last-block runner did not produce node '{self.output_node_name}'.")
        return env[self.output_node_name]


class Model(nn.Module):
    def __init__(self, emb_size, lc0_embedding_onnx_path, lc0_embedding_tensor_name="/encoder14/ln2/betas", lc0_square_order="model"):
        super().__init__()
        if not lc0_embedding_onnx_path:
            raise ValueError("lc0_embedding_onnx_path is required for the paper-repo model")

        self.y_lc0_embedder = TrainableLc0PerSquareEmbedder(
            onnx_path=lc0_embedding_onnx_path,
            embedding_tensor_name=lc0_embedding_tensor_name,
            square_order=lc0_square_order,
        )
        if getattr(self.y_lc0_embedder, "promoted_buffer_count", 0) > 0:
            print(f"Promoted {self.y_lc0_embedder.promoted_buffer_count} LC0 buffers to trainable parameters.")
        for param in self.y_lc0_embedder.parameters():
            param.requires_grad = True

        self.lc0_input_proj = nn.Linear(self.y_lc0_embedder.embedding_dim, emb_size)
        nn.init.xavier_normal_(self.lc0_input_proj.weight)
        self.lc0_input_norm = nn.LayerNorm(emb_size, eps=1e-6)

        self.move_head_model = Lc0MoveHeadModel(onnx_path=lc0_embedding_onnx_path)
        if self.y_lc0_embedder.embedding_dim != self.move_head_model.lc0_emb_size:
            raise RuntimeError(
                "LC0 embedder and move-head dimensions differ in the supported paper-repo path: "
                f"{self.y_lc0_embedder.embedding_dim} vs {self.move_head_model.lc0_emb_size}."
            )

        self.value_head_model = ValueHeadModel(emb_size)

    def get_y_init(self, lc0_emb):
        return self.lc0_input_norm(self.lc0_input_proj(lc0_emb))

    def forward(self, board, accumulation_step=None, lc0_meta=None, return_lc0_wdl=False, return_lc0_pre_last=False):
        del accumulation_step

        lc0_wdl_probs = None
        lc0_pre_last_emb = None
        if return_lc0_wdl and return_lc0_pre_last:
            lc0_emb, lc0_wdl_probs, lc0_pre_last_emb = self.y_lc0_embedder(
                board,
                lc0_meta=lc0_meta,
                return_wdl=True,
                return_pre_last=True,
            )
        elif return_lc0_wdl:
            lc0_emb, lc0_wdl_probs = self.y_lc0_embedder(board, lc0_meta=lc0_meta, return_wdl=True)
        elif return_lc0_pre_last:
            lc0_emb, lc0_pre_last_emb = self.y_lc0_embedder(board, lc0_meta=lc0_meta, return_pre_last=True)
        else:
            lc0_emb = self.y_lc0_embedder(board, lc0_meta=lc0_meta)

        y_init = self.get_y_init(lc0_emb)
        moves_initial, promotion_offsets_initial = self.move_head_model.forward_with_promotion_offsets(lc0_emb)
        value = self.value_head_model(y_init)

        aux_outputs = {
            "all_move_logits": [moves_initial],
            "initial_move_logits": moves_initial,
            "promotion_offsets": promotion_offsets_initial,
            "all_promotion_offsets": [promotion_offsets_initial],
            "initial_promotion_offsets": promotion_offsets_initial,
        }
        if lc0_wdl_probs is not None:
            aux_outputs["lc0_wdl_probs"] = lc0_wdl_probs
        if lc0_pre_last_emb is not None:
            aux_outputs["lc0_pre_last_embedding"] = lc0_pre_last_emb
        return moves_initial, value, aux_outputs
